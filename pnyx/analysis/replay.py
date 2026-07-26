"""Pnyx replay data layer — turns the P4/v2 event logs into per-question
"replay" objects the optional Streamlit app (``replay/streamlit_app.py``)
renders. This module is the single source of every number the app shows; the
app itself computes nothing.

Design contracts that bind this module
--------------------------------------
* **No Streamlit.** Nothing here (nor ``tests/test_replay.py``) imports
  ``streamlit`` — the core install and the full test suite must stay green
  without the optional ``replay`` extra. The app imports Streamlit lazily.
* **Consume, don't duplicate.** Loading, the oracle posterior, the honest /
  adversary agent identities, and the pass2-only Pass-1 source mapping all
  come from the existing analysis modules
  (``mainrun.load_condition`` / ``HONEST_AGENTS``, ``pilot._posterior_all``,
  ``manipulation.ADVERSARY_ID``, ``tables.PASS1_SOURCE``).
* **Pass-1 resolution mirrors the runner.** Two-pass conditions (A, B*, C,
  the v2 arms) carry their honest agents' Pass-1 beliefs in their own log.
  Pass2-only conditions reuse another condition's independent pass —
  ``D_K* <- C`` and ``W_* <- A`` (``tables.PASS1_SOURCE`` / the runner's
  ``pass1_source_dir``) — matched seed-for-seed and by ``(question_id,
  agent_id)``. ``question_replay`` takes the resolved source explicitly; the
  ``load_run`` loader resolves it from the sibling condition directory.
* **Event-log order is execution order.** Trade steps are returned in the
  order the runner wrote them within a question (as ``manipulation`` relies
  on for recovery windows), so the price path and the adversary's pre-attack
  price are read straight off that order.
* **Purity.** The assembly + derived-view helpers do no I/O; only
  ``list_runs`` / ``load_run`` touch the filesystem (through
  ``mainrun.load_condition``).

stdlib + the existing analysis modules only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pnyx.analysis.mainrun import ConditionData, HONEST_AGENTS, load_condition
from pnyx.analysis.manipulation import ADVERSARY_ID
from pnyx.analysis.pilot import _posterior_all
from pnyx.analysis.tables import PASS1_SOURCE
from pnyx.baselines import assert_pass1

__all__ = [
    "RunRef",
    "TradeStep",
    "QuestionReplay",
    "PriceSeries",
    "PricePoint",
    "RoundBoundary",
    "AgentHerding",
    "HerdingByRound",
    "list_runs",
    "load_run",
    "settled_question_ids",
    "question_replay",
    "price_series",
    "herding",
    "PASS1_SOURCE",
    "HONEST_AGENTS",
    "ADVERSARY_ID",
]

_HONEST_SET = frozenset(HONEST_AGENTS)
_INITIAL_PRICE = 0.5  # LMSR starts every question at q=0 => price 0.5.


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRef:
    """One ``(condition, seed)`` run, and the condition directory that holds
    its logs. ``run_dir`` is the per-condition directory (e.g.
    ``data/main/D_K10``); its parent is the data root the run was discovered
    under, which is where the Pass-1 source condition (if any) also lives."""

    condition: str
    seed: int
    run_dir: Path

    @property
    def root(self) -> Path:
        """The data root the run lives under (``run_dir``'s parent), where the
        Pass-1 source condition directory also lives."""
        return self.run_dir.parent


def _seed_of(event_log_name: str) -> int:
    """Extract the integer seed from a ``<cond>_seed<seed>.jsonl`` event-log
    filename (e.g. ``D_K10_seed2.jsonl`` -> 2)."""
    stem = event_log_name[: -len(".jsonl")] if event_log_name.endswith(".jsonl") else event_log_name
    marker = "_seed"
    idx = stem.rfind(marker)
    if idx == -1:
        raise ValueError(f"cannot parse seed from event-log filename {event_log_name!r}")
    return int(stem[idx + len(marker):])


def list_runs(data_roots: list[Path]) -> list[RunRef]:
    """Discover every ``(condition, seed)`` run under the given data roots.

    Each root is a directory of per-condition subdirectories (e.g.
    ``data/main`` with ``A/``, ``D_K10/``, ...; ``data/v2`` with ``A_PRO/``,
    ``A_LUNA/``). A subdirectory is a condition iff it contains at least one
    ``<cond>_seed<seed>.jsonl`` event log (``.questions.jsonl`` sidecars and
    ``.costs.jsonl`` ledgers are ignored). Returns runs sorted by
    ``(root, condition, seed)``; missing roots are skipped."""
    runs: list[RunRef] = []
    for root in data_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for cond_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            seeds: set[int] = set()
            for p in cond_dir.glob("*.jsonl"):
                name = p.name
                if name.endswith(".questions.jsonl") or name.endswith(".costs.jsonl"):
                    continue
                if "_seed" not in name:
                    continue
                seeds.add(_seed_of(name))
            for seed in sorted(seeds):
                runs.append(RunRef(condition=cond_dir.name, seed=seed, run_dir=cond_dir))
    return runs


def load_run(run: RunRef) -> tuple[ConditionData, ConditionData | None]:
    """Load a run's :class:`~pnyx.analysis.mainrun.ConditionData` and, for a
    pass2-only condition, its Pass-1 source condition (``D_K* <- C``,
    ``W_* <- A``); the source is ``None`` for a two-pass condition.

    The source directory is resolved as a sibling of ``run.run_dir`` under the
    same data root — the runner's own convention (``pass1_source_dir``)."""
    cond = load_condition(run.run_dir)
    source_name = PASS1_SOURCE.get(run.condition)
    if source_name is None:
        return cond, None
    source_dir = run.root / source_name
    return cond, load_condition(source_dir)


# ---------------------------------------------------------------------------
# Replay dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeStep:
    """One executed market turn of a question, in execution (log) order.

    ``stated_belief`` is the in-market belief the agent reported on this trade
    (``TradeEvent.trade.belief`` — logged for dynamics only, never a
    baseline). ``price_before`` is the price the agent faced; ``price_after``
    is the price its (clamped) trade produced."""

    step: int  # 0-based position in the question's execution order
    agent_id: str
    round: int
    action: str  # "buy_yes" | "buy_no" | "hold"
    stated_belief: float
    executed_shares: float
    cost: float
    price_before: float
    price_after: float
    parse_failed: bool
    is_adversary: bool


@dataclass(frozen=True)
class QuestionReplay:
    """Everything the app needs to replay one settled question of one run.

    ``posterior_all`` is the Bayes oracle given ALL signals (the dashed
    reference line); ``pass1_by_agent`` maps each honest agent to its Pass-1
    belief (resolved from the source condition for a pass2-only run);
    ``pass1_source_condition`` names that source (``None`` for a two-pass
    run). ``pre_attack_price`` is the price just before the adversary's first
    trade (``None`` when there is no adversary)."""

    condition: str
    seed: int
    question_id: str
    question_text: str | None
    posterior_all: float
    outcome: int
    final_price: float
    subsidy: float
    steps: list[TradeStep]
    pass1_by_agent: dict[str, float]
    pass1_source_condition: str | None
    has_adversary: bool
    pre_attack_price: float | None
    n_rounds: int


@dataclass(frozen=True)
class RoundBoundary:
    """The half-open span of step indices belonging to one market round:
    ``start_step`` (inclusive) .. ``end_step`` (inclusive)."""

    round: int
    start_step: int
    end_step: int


@dataclass(frozen=True)
class PricePoint:
    """A point on the price path: the LMSR YES price after ``step`` executed
    trades (``step == 0`` is the initial price before any trade)."""

    step: int
    price: float


@dataclass(frozen=True)
class PriceSeries:
    """The price path as a step series plus the shaded round boundaries and
    the reference lines (oracle posterior, realized outcome, final price)."""

    points: list[PricePoint]
    round_boundaries: list[RoundBoundary]
    posterior_all: float
    outcome: int
    final_price: float


@dataclass(frozen=True)
class HerdingByRound:
    """One honest agent's in-market state for one round: the belief it stated
    and the price it faced when it traded (``price_before``)."""

    round: int
    stated_belief: float
    price_at_trade: float


@dataclass(frozen=True)
class AgentHerding:
    """One honest agent's herding triple: its own Pass-1 belief, and its
    per-round (stated belief, price-at-trade) pairs in round order."""

    agent_id: str
    pass1_belief: float | None
    by_round: list[HerdingByRound] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Assembly (pure)
# ---------------------------------------------------------------------------


def _pass1_by_agent(source: ConditionData, seed: int, qid: str) -> dict[str, float]:
    """Honest agents' Pass-1 belief probabilities for one ``(seed, question)``
    of ``source`` (phase-guarded). Empty when the source has no such seed."""
    out: dict[str, float] = {}
    sd = source.seeds.get(seed)
    if sd is None:
        return out
    for e in sd.beliefs:
        if e.key.question_id != qid or e.key.agent_id not in _HONEST_SET:
            continue
        assert_pass1(e)  # scientific-integrity guard (raises on a market-phase belief)
        out[e.key.agent_id] = e.belief.prob
    return out


def settled_question_ids(cond: ConditionData, seed: int) -> list[str]:
    """Sorted question ids of ``cond``'s seed that are BOTH settled and have a
    sidecar record (the questions the app can replay). Empty for an unknown
    seed."""
    sd = cond.seeds.get(seed)
    if sd is None:
        return []
    return sorted(s.question_id for s in sd.settlements if s.question_id in sd.questions)


def question_replay(
    cond: ConditionData,
    seed: int,
    qid: str,
    pass1_source: ConditionData | None = None,
) -> QuestionReplay:
    """Assemble the :class:`QuestionReplay` for one settled question.

    ``pass1_source`` is ``None`` for a two-pass condition (Pass-1 comes from
    ``cond`` itself) and the source condition's ``ConditionData`` for a
    pass2-only condition (``D_K* <- C``, ``W_* <- A``); the honest agents'
    Pass-1 beliefs are then read from the source, matched seed-for-seed. Pure
    (no I/O). Raises if the seed/question is absent or unsettled."""
    sd = cond.seeds.get(seed)
    if sd is None:
        raise ValueError(f"condition {cond.condition!r} has no seed {seed}")

    record = sd.questions.get(qid)
    if record is None:
        raise ValueError(
            f"condition {cond.condition!r} seed {seed}: question {qid!r} has no "
            "sidecar record"
        )
    settlement = next((s for s in sd.settlements if s.question_id == qid), None)
    if settlement is None:
        raise ValueError(
            f"condition {cond.condition!r} seed {seed}: question {qid!r} is not settled"
        )
    posterior_all = _posterior_all(record)
    if posterior_all is None:
        raise ValueError(
            f"condition {cond.condition!r} seed {seed} question {qid!r}: "
            "posterior_table has no all-signals entry (oracle missing)"
        )

    # Trade stream in logged (execution) order.
    steps: list[TradeStep] = []
    pre_attack_price: float | None = None
    has_adversary = False
    n_rounds = 0
    for t in sd.trades:
        if t.key.question_id != qid:
            continue
        is_adv = t.key.agent_id == ADVERSARY_ID
        if is_adv:
            has_adversary = True
            if pre_attack_price is None:
                pre_attack_price = t.price_before
        n_rounds = max(n_rounds, t.key.round)
        steps.append(
            TradeStep(
                step=len(steps),
                agent_id=t.key.agent_id,
                round=t.key.round,
                action=t.trade.action,
                stated_belief=t.trade.belief,
                executed_shares=t.executed_shares,
                cost=t.cost,
                price_before=t.price_before,
                price_after=t.price_after,
                parse_failed=t.parse_failed,
                is_adversary=is_adv,
            )
        )

    source = pass1_source if pass1_source is not None else cond
    pass1 = _pass1_by_agent(source, seed, qid)

    return QuestionReplay(
        condition=cond.condition,
        seed=seed,
        question_id=qid,
        question_text=record.question_text,
        posterior_all=posterior_all,
        outcome=settlement.outcome,
        final_price=settlement.final_price,
        subsidy=settlement.subsidy,
        steps=steps,
        pass1_by_agent=pass1,
        pass1_source_condition=PASS1_SOURCE.get(cond.condition),
        has_adversary=has_adversary,
        pre_attack_price=pre_attack_price,
        n_rounds=n_rounds,
    )


# ---------------------------------------------------------------------------
# Derived views (pure)
# ---------------------------------------------------------------------------


def price_series(replay: QuestionReplay) -> PriceSeries:
    """The price path of a replay as a step series (step 0 = the initial price
    before any trade, then the price after each executed trade) plus the round
    boundaries (contiguous spans of step indices sharing a round) for shading.

    The initial price is the first trade's ``price_before`` when a trade
    exists, else the LMSR neutral price 0.5."""
    initial = replay.steps[0].price_before if replay.steps else _INITIAL_PRICE
    points = [PricePoint(step=0, price=initial)]
    for s in replay.steps:
        points.append(PricePoint(step=s.step + 1, price=s.price_after))

    boundaries: list[RoundBoundary] = []
    for s in replay.steps:
        # steps carry the price-path index (s.step) shifted by +1 in `points`;
        # a boundary spans the point indices [start, end] the round covers.
        pt = s.step + 1
        if boundaries and boundaries[-1].round == s.round:
            last = boundaries[-1]
            boundaries[-1] = RoundBoundary(round=last.round, start_step=last.start_step, end_step=pt)
        else:
            boundaries.append(RoundBoundary(round=s.round, start_step=pt, end_step=pt))

    return PriceSeries(
        points=points,
        round_boundaries=boundaries,
        posterior_all=replay.posterior_all,
        outcome=replay.outcome,
        final_price=replay.final_price,
    )


def herding(replay: QuestionReplay) -> list[AgentHerding]:
    """Per honest agent, the herding triple: its Pass-1 belief and, per round,
    its stated belief and the price it faced (``price_before``). Agents are
    returned sorted by id; only honest agents (``p0..p5``) are included (the
    adversary carries no Pass-1 belief and is not a herding subject)."""
    by_agent: dict[str, list[HerdingByRound]] = {}
    for s in replay.steps:
        if s.agent_id not in _HONEST_SET:
            continue
        by_agent.setdefault(s.agent_id, []).append(
            HerdingByRound(
                round=s.round,
                stated_belief=s.stated_belief,
                price_at_trade=s.price_before,
            )
        )
    agents = sorted(set(by_agent) | set(replay.pass1_by_agent))
    out: list[AgentHerding] = []
    for aid in agents:
        rounds = sorted(by_agent.get(aid, []), key=lambda r: r.round)
        out.append(
            AgentHerding(
                agent_id=aid,
                pass1_belief=replay.pass1_by_agent.get(aid),
                by_round=rounds,
            )
        )
    return out
