"""Pnyx experiment runner — orchestration, kill-safe event log, replay/resume.

Scope (P1): drives the two-pass protocol over MockAgents with ZERO API calls.
No providers, no rate limiting, no cost ledger, no baselines (those arrive in
P3). This module owns the event log and the resume logic; the market math,
generative environment, agents, and all Pydantic models live in their own
modules (``pnyx.market`` / ``pnyx.env`` / ``pnyx.agents`` / ``pnyx.schemas``).

Two passes per (condition, seed, question)
------------------------------------------
* Pass 1 (independent): every agent emits a Belief with no market context.
  Logged as a ``BeliefEvent`` with ``phase="independent"``, ``round=0``.
* Pass 2 (market): a fresh ``Market(b)`` is opened; ``n_rounds`` rounds run;
  the agent order is reshuffled each round (seeded RNG). Each turn the agent
  decides a trade, the runner clamps requested shares to affordability +
  bankroll server-side, executes it, and logs a ``TradeEvent``
  (``phase="market"``) carrying price_before/after, executed cost, and
  bankroll_after. Holds are logged too (executed 0, cost 0) so every turn is
  accounted for in the completed-turn set.
* Settlement: a ``SettlementEvent`` pays $1/share on the true side, records
  the final price and the maker subsidy. Bankrolls reset per question unless
  ``wealth_persistent`` is set.

Seeding scheme (determinism — Global Constraints)
-------------------------------------------------
All randomness derives from the integer ``seed`` via ``numpy.random.SeedSequence``
keyed on a *stable* tuple. ``_rng(seed, *parts)`` hashes the parts with SHA-256
(Python's builtin ``hash`` is per-process salted and must NOT be used) into a
64-bit tag and builds ``SeedSequence([seed, tag])``. Distinct purpose tags keep
independent streams from colliding:

* questions:    ``_rng(seed, "questions", condition)``
* Pass-1 belief:``_rng(seed, "belief", question_id, agent_id)``
* round shuffle:``_rng(seed, "shuffle", question_id, round)``
* Pass-2 trade: ``_rng(seed, "trade", question_id, round, agent_id)``

Because every turn draws from a freshly-seeded generator keyed on its own turn
identity, a resumed run reproduces each turn bit-identically regardless of where
a previous process was killed — the whole point of the kill-resume guarantee.

Kill-safe log + resume
----------------------
The log is append-only JSONL, one line per event, flushed + fsync'd per line.
Resume is the default (no flag): on startup the runner ALWAYS replays the log —
validating every line (a truncated final partial line is ignored; a malformed
non-final line is a hard error), rebuilding each in-flight question's market by
re-executing logged trades (asserting reconstructed price_after matches the
logged value within 1e-9, failing loudly on divergence), rebuilding bankrolls,
and deriving the set of completed turns — then continues from the first
incomplete turn. Generated questions are persisted to a sidecar file and loaded
on resume, so rejection-sampling nondeterminism can never fork a resumed run.

Async seam (P3)
---------------
Turn execution is structured as ``async def`` helpers driven by ``asyncio.run``.
P1 is not concurrent (MockAgent is synchronous), but this is the seam where a
real awaited provider call slots in at P3 without restructuring the loop.
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from pnyx.agents import MockAgent
from pnyx.env import generate_question, subset_key
from pnyx.market import Market, max_affordable_shares
from pnyx.schemas import (
    Belief,
    BeliefEvent,
    BeliefView,
    QuestionRecord,
    RunConfig,
    SettlementEvent,
    Trade,
    TradeEvent,
    TradeView,
    TurnKey,
    parse_event,
)

__all__ = [
    "load_config",
    "run_experiment",
    "log_path_for",
    "questions_path_for",
    "read_events",
    "ts_stripped_lines",
    "status_report",
]

_PRICE_TOL = 1e-9


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> RunConfig:
    """Load and validate a YAML run config into a ``RunConfig``."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return RunConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def _rng(seed: int, *parts: object) -> np.random.Generator:
    """Deterministic Generator keyed on ``seed`` and a stable tuple of parts.

    SHA-256 (not builtin ``hash``, which is per-process salted) maps the parts
    to a 64-bit tag; ``SeedSequence([seed, tag])`` gives an independent,
    reproducible stream per purpose. See module docstring for the scheme.
    """
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    tag = int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([int(seed), tag]))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _stem(config: RunConfig, seed: int) -> str:
    return f"{config.condition}_seed{seed}"


def log_path_for(config: RunConfig, seed: int) -> Path:
    """Event-log path for one (condition, seed)."""
    return Path(config.data_dir) / f"{_stem(config, seed)}.jsonl"


def questions_path_for(config: RunConfig, seed: int) -> Path:
    """Persisted-questions sidecar path for one (condition, seed)."""
    return Path(config.data_dir) / f"{_stem(config, seed)}.questions.jsonl"


# ---------------------------------------------------------------------------
# Event log I/O
# ---------------------------------------------------------------------------


def _append_event(f, event) -> None:
    """Append one event as a canonical JSONL line, flushed + fsync'd."""
    f.write(event.to_jsonl_line())
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())


def read_events(path: Path) -> list:
    """Replay-tolerant read of the event log.

    Every line must parse EXCEPT a truncated final partial line (a write killed
    mid-line): only the final line may be partial, and only when the file does
    not end in a newline. A malformed non-final line is a hard error.
    """
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text()
    if not text:
        return []
    ended_newline = text.endswith("\n")
    lines = text.splitlines()
    events = []
    for idx, line in enumerate(lines):
        is_last = idx == len(lines) - 1
        try:
            events.append(parse_event(line))
        except Exception:
            if is_last and not ended_newline:
                # Truncated final write — ignore (kill-safe contract).
                break
            raise
    return events


def ts_stripped_lines(path: Path) -> list[str]:
    """Canonical event lines with the wall-clock ``ts`` field stripped, for
    determinism / identity comparisons (see schemas module docstring)."""
    out = []
    for e in read_events(path):
        payload = e.model_dump(mode="json")
        payload.pop("ts", None)
        # Re-serialize canonically (sorted keys) via the event's own encoder
        # after dropping ts.
        out.append(_canonical(payload))
    return out


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Question generation + persistence
# ---------------------------------------------------------------------------


def _load_questions_file(config: RunConfig) -> list[QuestionRecord]:
    """Load the first ``n_questions`` records from ``config.questions_file``,
    requiring each to carry shards + question_text (market runs on rendered
    questions only). Fails loudly on a short file or a missing render.

    The loaded records become the per-run sidecar (persisted by the caller)
    exactly as the generate path does, so resume/kill-safety is unchanged.
    """
    path = Path(config.questions_file)
    if not path.exists():
        raise FileNotFoundError(f"questions_file {path} does not exist")
    records = [
        QuestionRecord.model_validate_json(line)
        for line in path.read_text().splitlines() if line
    ]
    if len(records) < config.n_questions:
        raise ValueError(
            f"questions_file {path} has {len(records)} records, need at least "
            f"n_questions={config.n_questions}"
        )
    records = records[: config.n_questions]  # subset: take the first N
    for rec in records:
        if rec.shards is None or not rec.shards:
            raise ValueError(
                f"{rec.question_id}: questions_file record missing shards "
                "(market runs on rendered questions only)"
            )
        if len(rec.shards) != len(rec.signals):
            raise ValueError(
                f"{rec.question_id}: shard count {len(rec.shards)} != signal "
                f"count {len(rec.signals)}"
            )
        if not rec.question_text:
            raise ValueError(
                f"{rec.question_id}: questions_file record missing question_text"
            )
    return records


def _load_or_generate_questions(config: RunConfig, seed: int) -> list[QuestionRecord]:
    qpath = questions_path_for(config, seed)
    if qpath.exists():
        text = qpath.read_text()
        return [QuestionRecord.model_validate_json(line)
                for line in text.splitlines() if line]

    if config.questions_file is not None:
        # File-backed dataset: load + validate, then persist to the sidecar
        # below exactly as the generate path does (resume semantics unchanged).
        questions = _load_questions_file(config)
    else:
        rng = _rng(seed, "questions", config.condition)
        questions = [
            generate_question(config.env, rng, f"{config.condition}_seed{seed}_q{i}")
            for i in range(config.n_questions)
        ]
    # Atomic write: full file or nothing, so a mid-write kill can't persist a
    # partial (and therefore forked) question set.
    qpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = qpath.with_suffix(qpath.suffix + ".tmp")
    with open(tmp, "w") as f:
        for q in questions:
            f.write(q.model_dump_json())
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, qpath)
    return questions


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _turn_delay() -> float:
    """Optional per-live-turn delay (seconds), test-only fault-injection so the
    kill-resume acceptance test can reliably catch a run mid-flight. Zero cost
    and no effect when ``PNYX_TURN_DELAY`` is unset."""
    return float(os.environ.get("PNYX_TURN_DELAY", "0") or "0")


def run_experiment(config: RunConfig) -> None:
    """Run (or resume) the full experiment for every seed in the config."""
    for seed in config.seeds:
        asyncio.run(_run_seed(config, seed))


async def _run_seed(config: RunConfig, seed: int) -> None:
    questions = _load_or_generate_questions(config, seed)
    log = log_path_for(config, seed)
    log.parent.mkdir(parents=True, exist_ok=True)

    prior_events = read_events(log)
    events_by_key = {
        e.key.as_tuple(): e for e in prior_events if e.type in ("belief", "trade")
    }
    settled = {e.question_id for e in prior_events if e.type == "settlement"}
    settlements_by_q = {
        e.question_id: e for e in prior_events if e.type == "settlement"
    }

    agents_by_id = {a.agent_id: a for a in config.agents}
    # Persistent-wealth carryover; ignored when wealth_persistent is False.
    wealth: dict[str, float] = {a.agent_id: a.bankroll for a in config.agents}

    with open(log, "a") as f:
        for q in questions:
            await _run_question(
                config, seed, q, agents_by_id, events_by_key, settled,
                settlements_by_q, wealth, f,
            )


async def _run_question(
    config, seed, q, agents_by_id, events_by_key, settled, settlements_by_q,
    wealth, f,
) -> None:
    qid = q.question_id
    condition = config.condition
    delay = _turn_delay()

    oracle = {
        a.agent_id: q.posterior_table[subset_key(a.shard_indices)]
        for a in config.agents
    }

    # ---- Pass 1: independent beliefs -------------------------------------
    for spec in config.agents:
        key = TurnKey(
            condition=condition, seed=seed, question_id=qid,
            phase="independent", round=0, agent_id=spec.agent_id,
        )
        if key.as_tuple() in events_by_key:
            continue
        agent = MockAgent(oracle[spec.agent_id], _rng(seed, "belief", qid, spec.agent_id))
        view = BeliefView(
            question_id=qid,
            signal_values={i: q.signals[i].value for i in spec.shard_indices},
        )
        belief: Belief = await _elicit(agent, view)
        _append_event(f, BeliefEvent(key=key, belief=belief, ts=time.time()))
        if delay:
            await asyncio.sleep(delay)

    # ---- Pass 2: market (walk the deterministic plan; replay or execute) --
    market = Market(b=config.b)
    if config.wealth_persistent:
        bankrolls = {a.agent_id: wealth[a.agent_id] for a in config.agents}
    else:
        bankrolls = {a.agent_id: a.bankroll for a in config.agents}
    holdings = {a.agent_id: {"yes": 0.0, "no": 0.0} for a in config.agents}

    for rnd in range(1, config.n_rounds + 1):
        order = _shuffle_order(seed, qid, rnd, [a.agent_id for a in config.agents])
        for agent_id in order:
            key = TurnKey(
                condition=condition, seed=seed, question_id=qid,
                phase="market", round=rnd, agent_id=agent_id,
            )
            logged = events_by_key.get(key.as_tuple())
            price_before = market.price()

            if logged is not None:
                # Replay: re-execute the logged trade and verify reconstruction.
                _replay_trade(market, holdings, logged, price_before)
                bankrolls[agent_id] = logged.bankroll_after
                continue

            # Live execution.
            cash = bankrolls[agent_id]
            max_yes = max_affordable_shares(market.q_yes, market.q_no, "yes", cash, config.b)
            max_no = max_affordable_shares(market.q_yes, market.q_no, "no", cash, config.b)
            spec = agents_by_id[agent_id]
            agent = MockAgent(oracle[agent_id], _rng(seed, "trade", qid, rnd, agent_id))
            view = TradeView(
                question_id=qid,
                signal_values={i: q.signals[i].value for i in spec.shard_indices},
                price=price_before,
                round=rnd,
                n_rounds=config.n_rounds,
                bankroll=cash,
                max_affordable_yes=max_yes,
                max_affordable_no=max_no,
            )
            trade: Trade = await _decide(agent, view)

            executed = 0.0
            cost = 0.0
            if trade.action != "hold":
                side = "yes" if trade.action == "buy_yes" else "no"
                executed = market.clamp_shares(side, trade.shares, cash)
                if executed > 0.0:
                    cost = market.buy(side, executed)
                    holdings[agent_id][side] += executed
            # Floor at 0: clamp_shares guarantees cost <= cash + 1e-9, so a full-clamp buy can leave a ~-1e-13 residue that would fail TradeView's ge=0 next round.
            bankrolls[agent_id] = max(0.0, cash - cost)

            _append_event(f, TradeEvent(
                key=key, trade=trade, executed_shares=executed, cost=cost,
                price_before=price_before, price_after=market.price(),
                bankroll_after=bankrolls[agent_id], ts=time.time(),
            ))
            if delay:
                await asyncio.sleep(delay)

    # ---- Settlement -------------------------------------------------------
    outcome = q.latent_state
    true_side = "yes" if outcome == 1 else "no"
    payouts = {aid: holdings[aid][true_side] for aid in holdings}

    if qid in settled:
        # Already settled by a prior run: verify math, then carry wealth.
        prior = settlements_by_q[qid]
        _verify_settlement(prior, payouts, market)
        for aid in wealth:
            wealth[aid] = bankrolls[aid] + prior.payouts.get(aid, 0.0)
        return

    settle = SettlementEvent(
        condition=condition, seed=seed, question_id=qid, outcome=outcome,
        payouts=payouts, final_price=market.price(), subsidy=market.subsidy(),
        ts=time.time(),
    )
    _append_event(f, settle)
    settled.add(qid)
    settlements_by_q[qid] = settle
    for aid in wealth:
        wealth[aid] = bankrolls[aid] + payouts[aid]


def _shuffle_order(seed, qid, rnd, agent_ids: list[str]) -> list[str]:
    rng = _rng(seed, "shuffle", qid, rnd)
    perm = rng.permutation(len(agent_ids))
    return [agent_ids[i] for i in perm]


def _replay_trade(market: Market, holdings, logged: TradeEvent, price_before: float) -> None:
    """Re-apply a logged trade to a reconstructed market, asserting the logged
    prices and cost reproduce within tolerance (fail loudly on divergence)."""
    if abs(price_before - logged.price_before) > _PRICE_TOL:
        raise AssertionError(
            f"replay divergence at {logged.key.as_tuple()}: price_before "
            f"reconstructed {price_before!r} vs logged {logged.price_before!r}"
        )
    if logged.trade.action != "hold" and logged.executed_shares > 0.0:
        side = "yes" if logged.trade.action == "buy_yes" else "no"
        paid = market.buy(side, logged.executed_shares)
        holdings[logged.key.agent_id][side] += logged.executed_shares
        if abs(paid - logged.cost) > _PRICE_TOL:
            raise AssertionError(
                f"replay divergence at {logged.key.as_tuple()}: cost "
                f"reconstructed {paid!r} vs logged {logged.cost!r}"
            )
    price_after = market.price()
    if abs(price_after - logged.price_after) > _PRICE_TOL:
        raise AssertionError(
            f"replay divergence at {logged.key.as_tuple()}: price_after "
            f"reconstructed {price_after!r} vs logged {logged.price_after!r}"
        )


def _verify_settlement(prior: SettlementEvent, payouts: dict, market: Market) -> None:
    if abs(prior.final_price - market.price()) > _PRICE_TOL:
        raise AssertionError(
            f"settlement replay divergence for {prior.question_id}: final_price "
            f"{market.price()!r} vs logged {prior.final_price!r}"
        )
    if abs(prior.subsidy - market.subsidy()) > _PRICE_TOL:
        raise AssertionError(
            f"settlement replay divergence for {prior.question_id}: subsidy "
            f"{market.subsidy()!r} vs logged {prior.subsidy!r}"
        )
    for aid, expected in payouts.items():
        if abs(prior.payouts.get(aid, 0.0) - expected) > _PRICE_TOL:
            raise AssertionError(
                f"settlement replay divergence for {prior.question_id}: payout "
                f"[{aid}] {expected!r} vs logged {prior.payouts.get(aid)!r}"
            )


# ---- Async turn seam (P3 provider slots in here) --------------------------


async def _elicit(agent, view: BeliefView) -> Belief:
    # P3: `return await provider.elicit_belief(view)`. P1 MockAgent is sync.
    return agent.elicit_belief(view)


async def _decide(agent, view: TradeView) -> Trade:
    # P3: `return await provider.decide_trade(view)`. P1 MockAgent is sync.
    return agent.decide_trade(view)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def status_report(config: RunConfig) -> str:
    """Human-readable turns done/remaining + parse-failure counts per
    (condition, seed). Cost ledger arrives in P3 — a $0.00 placeholder is
    printed for now. Read-only: never generates questions or writes anything.
    """
    n_agents = len(config.agents)
    beliefs_per_seed = config.n_questions * n_agents
    trades_per_seed = config.n_questions * config.n_rounds * n_agents
    settlements_per_seed = config.n_questions
    planned = beliefs_per_seed + trades_per_seed + settlements_per_seed

    lines = [f"condition {config.condition}: {len(config.seeds)} seed(s)"]
    for seed in config.seeds:
        events = read_events(log_path_for(config, seed))
        done_beliefs = sum(1 for e in events if e.type == "belief")
        done_trades = sum(1 for e in events if e.type == "trade")
        done_settle = sum(1 for e in events if e.type == "settlement")
        done = done_beliefs + done_trades + done_settle
        parse_fails = sum(
            1 for e in events
            if e.type in ("belief", "trade") and e.parse_failed
        )
        lines.append(
            f"  seed {seed}: {done}/{planned} turns done "
            f"({planned - done} remaining) — "
            f"beliefs {done_beliefs}/{beliefs_per_seed}, "
            f"trades {done_trades}/{trades_per_seed}, "
            f"settlements {done_settle}/{settlements_per_seed}; "
            f"parse-failures {parse_fails}; cost $0.00"
        )
    return "\n".join(lines)
