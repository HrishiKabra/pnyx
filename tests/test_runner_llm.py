"""Tests for the P3 runner wiring: LLM agents driven through a FAKE provider
(zero network), the one-retry parse contract surfaced into events, parked-turn
resume, the budget hard-stop, cost-ledger billing, and prompt-version logging.

The provider is a fake with an async ``complete``; the socket guard in conftest
would fail any real connection. The mock path is covered by test_runner.py — here
every agent is ``kind: llm``.
"""

import asyncio
from pathlib import Path

import httpx
import pytest

from pnyx.providers import ProviderError, SchemaParseError, TurnParked
from pnyx.runner import (
    _free_tier_check,
    _free_tier_models,
    costs_path_for,
    log_path_for,
    read_events,
    run_experiment,
    status_report,
)
from pnyx.schemas import Belief, ModelSpec, RunConfig, Trade, Usage

from tests._fixtures import write_questions_file

MODEL = dict(
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_KEY",
    model_id="deepseek/deepseek-v4-flash",
    price_in=0.077, price_out=0.154, rpm_limit=300, supports_json_schema=True,
)


def _llm_config(tmp_path: Path, qfile: Path, *, n_questions=1, n_rounds=1,
                n_agents=2, budget_usd=25.0, model_over=None) -> RunConfig:
    model = dict(MODEL)
    if model_over:
        model.update(model_over)
    personas = ["contrarian", "kelly_sizer", "base_rate_skeptic",
                "evidence_maximalist", "domain_specialist", "momentum_reader"]
    return RunConfig.model_validate({
        "condition": "LLM",
        "seeds": [0],
        "n_questions": n_questions,
        "b": 40.0,
        "n_rounds": n_rounds,
        "wealth_persistent": False,
        "data_dir": str(tmp_path / "data"),
        "questions_file": str(qfile),
        "budget_usd": budget_usd,
        "temperature": 0.7,
        "env": {
            "n_signals": 6, "accuracies": [0.75] * 6, "lams": [0.0] * 6,
            "a_z": 0.8, "difficulty": "easy", "max_rejection_tries": 2000,
        },
        "models": {"m": model},
        "agents": [
            {"agent_id": f"agent{i}", "shard_indices": [i], "kind": "llm",
             "model": "m", "persona": personas[i]}
            for i in range(n_agents)
        ],
    })


def _usage():
    return Usage(prompt_tokens=10, completion_tokens=5)


class ScriptedProvider:
    """Returns a Belief for the Belief schema and a Trade for the Trade schema.

    ``belief_maker`` / ``trade_maker`` are callables(call_index)->(obj|Exception).
    Records every call. Deterministic; no network.
    """

    def __init__(self, belief_maker=None, trade_maker=None):
        self.belief_maker = belief_maker or (lambda i: Belief(prob=0.6, rationale="b"))
        self.trade_maker = trade_maker or (
            lambda i: Trade(belief=0.6, action="hold", shares=0.0, rationale="t"))
        self.n_belief = 0
        self.n_trade = 0
        self.calls = []

    async def complete(self, messages, schema, model_spec, temperature, max_tokens):
        self.calls.append((schema.__name__, messages))
        if schema is Belief:
            step = self.belief_maker(self.n_belief)
            self.n_belief += 1
        else:
            step = self.trade_maker(self.n_trade)
            self.n_trade += 1
        if isinstance(step, BaseException):
            raise step
        return step, _usage()


# ---------------------------------------------------------------------------
# Happy path: full LLM run logs events with prompt_version + bills the ledger
# ---------------------------------------------------------------------------


def test_llm_run_end_to_end_bills_and_versions(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    config = _llm_config(tmp_path, qfile, n_questions=1, n_rounds=1, n_agents=2)
    prov = ScriptedProvider()
    run_experiment(config, provider=prov)

    events = read_events(log_path_for(config, 0))
    beliefs = [e for e in events if e.type == "belief"]
    trades = [e for e in events if e.type == "trade"]
    settlements = [e for e in events if e.type == "settlement"]
    assert len(beliefs) == 2 and len(trades) == 2 and len(settlements) == 1

    # PROMPT_VERSION stamped on every LLM turn.
    for e in beliefs + trades:
        assert e.prompt_version and e.prompt_version.startswith("p3")
        assert e.parse_failed is False

    # Ledger billed one call per LLM turn (2 beliefs + 2 trades).
    ledger = costs_path_for(config)
    assert ledger.exists()
    ledger_lines = [l for l in ledger.read_text().splitlines() if l.strip()]
    assert len(ledger_lines) == 4
    from pnyx.schemas import CostEntry
    total = sum(CostEntry.model_validate_json(l).cost for l in ledger_lines)
    assert total > 0.0

    # Status runs and reports zero parse failures + the budget.
    report = status_report(config)
    assert "parse-failures 0" in report
    assert "budget $25.00" in report


# ---------------------------------------------------------------------------
# One-retry parse contract surfaced into the event log
# ---------------------------------------------------------------------------


def test_parse_failure_twice_records_hold_and_flag(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    config = _llm_config(tmp_path, qfile, n_questions=1, n_rounds=1, n_agents=1)

    # Every Belief call fails to parse (both the attempt and its retry).
    def belief_maker(i):
        return SchemaParseError(usage=_usage(), content="x", error="bad")

    prov = ScriptedProvider(belief_maker=belief_maker)
    run_experiment(config, provider=prov)

    events = read_events(log_path_for(config, 0))
    beliefs = [e for e in events if e.type == "belief"]
    assert len(beliefs) == 1
    assert beliefs[0].parse_failed is True
    assert beliefs[0].belief.prob == 0.5

    # Three provider calls billed: the belief attempt + its retry (both failed),
    # plus the one successful trade turn.
    n_ledger_lines = len(
        [l for l in costs_path_for(config).read_text().splitlines() if l.strip()])
    assert n_ledger_lines == 3

    report = status_report(config)
    assert "parse-failures 1" in report
    assert "deepseek/deepseek-v4-flash: 1" in report  # per-model breakdown


# ---------------------------------------------------------------------------
# Parked turn: nothing logged this run; resume completes it
# ---------------------------------------------------------------------------


def test_parked_turn_is_not_logged_then_resumes(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    config = _llm_config(tmp_path, qfile, n_questions=1, n_rounds=1, n_agents=2)

    # First run: the SECOND belief call parks (provider exhausted retries).
    def belief_maker_run1(i):
        if i == 0:
            return Belief(prob=0.7, rationale="ok")
        return TurnParked("parked")

    prov1 = ScriptedProvider(belief_maker=belief_maker_run1)
    run_experiment(config, provider=prov1)

    events = read_events(log_path_for(config, 0))
    # Only the first belief logged; the parked turn + everything after is absent
    # (question aborted, no settlement).
    assert [e.type for e in events] == ["belief"]
    assert not any(e.type == "settlement" for e in events)

    # Resume with a healthy provider: the run completes.
    prov2 = ScriptedProvider()
    run_experiment(config, provider=prov2)
    events = read_events(log_path_for(config, 0))
    assert len([e for e in events if e.type == "belief"]) == 2
    assert len([e for e in events if e.type == "trade"]) == 2
    assert len([e for e in events if e.type == "settlement"]) == 1
    # The already-logged first belief was NOT re-called on resume.
    assert prov2.n_belief == 1  # only the previously-parked belief re-run


def test_pass2_midmarket_park_resumes_and_replays(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    config = _llm_config(tmp_path, qfile, n_questions=1, n_rounds=1, n_agents=2)

    # Beliefs all succeed; the SECOND trade call parks mid-market (after the
    # first agent's trade has already executed + logged).
    def trade_maker_run1(i):
        if i == 0:
            return Trade(belief=0.7, action="buy_yes", shares=2.0, rationale="go")
        return TurnParked("parked")

    prov1 = ScriptedProvider(trade_maker=trade_maker_run1)
    run_experiment(config, provider=prov1)

    events = read_events(log_path_for(config, 0))
    # Both beliefs + exactly one trade logged; no settlement (question aborted).
    assert len([e for e in events if e.type == "belief"]) == 2
    assert len([e for e in events if e.type == "trade"]) == 1
    assert not any(e.type == "settlement" for e in events)

    # Resume with a healthy provider: the logged trade REPLAYS (market
    # reconstructs from it), the parked trade runs live, the question settles.
    prov2 = ScriptedProvider(
        trade_maker=lambda i: Trade(belief=0.6, action="buy_no", shares=1.0, rationale="ok"))
    run_experiment(config, provider=prov2)  # replay must not diverge/crash
    events = read_events(log_path_for(config, 0))
    assert len([e for e in events if e.type == "trade"]) == 2
    assert len([e for e in events if e.type == "settlement"]) == 1
    # Resume re-called only the previously-parked trade (beliefs + first trade
    # were replayed from the log, not re-called).
    assert prov2.n_belief == 0 and prov2.n_trade == 1


def test_provider_error_parks_like_a_park(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    config = _llm_config(tmp_path, qfile, n_questions=1, n_rounds=1, n_agents=1)

    def belief_maker(i):
        return ProviderError("malformed 200", usage=_usage())

    prov = ScriptedProvider(belief_maker=belief_maker)
    run_experiment(config, provider=prov)  # must NOT crash

    events = read_events(log_path_for(config, 0))
    assert events == []  # nothing logged (turn incomplete)
    # But the incurred usage WAS billed even though the turn parked.
    n_ledger_lines = len(
        [l for l in costs_path_for(config).read_text().splitlines() if l.strip()])
    assert n_ledger_lines == 1


# ---------------------------------------------------------------------------
# Budget hard-stop
# ---------------------------------------------------------------------------


def test_budget_stop_parks_remaining_turns(tmp_path, capsys):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 3)
    # Tiny budget: the first billed call pushes cumulative cost over it, so all
    # subsequent LLM turns are parked.
    config = _llm_config(tmp_path, qfile, n_questions=3, n_rounds=1, n_agents=2,
                         budget_usd=1e-9)
    prov = ScriptedProvider()
    run_experiment(config, provider=prov)

    events = read_events(log_path_for(config, 0))
    # At most the first belief got logged before the budget tripped; certainly
    # not a full run of 3 questions.
    assert len([e for e in events if e.type == "settlement"]) == 0
    assert len(events) <= 1
    err = capsys.readouterr().err
    assert "budget" in err.lower()


def test_override_budget_runs_past_budget(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    config = _llm_config(tmp_path, qfile, n_questions=1, n_rounds=1, n_agents=2,
                         budget_usd=1e-9)
    prov = ScriptedProvider()
    run_experiment(config, provider=prov, override_budget=True)
    events = read_events(log_path_for(config, 0))
    # Full run despite the microscopic budget.
    assert len([e for e in events if e.type == "belief"]) == 2
    assert len([e for e in events if e.type == "settlement"]) == 1


# ---------------------------------------------------------------------------
# Config validation for the llm kind
# ---------------------------------------------------------------------------


FREE_MODEL = dict(
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_KEY",
    model_id="qwen/qwen3-next-80b-a3b-instruct:free",
    price_in=0.0, price_out=0.0, rpm_limit=18, supports_json_schema=True,
)


def test_free_tier_models_detects_free_pool(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    config = _llm_config(tmp_path, qfile, model_over=FREE_MODEL)
    free = _free_tier_models(config)
    assert len(free) == 1 and free[0].model_id.endswith(":free")

    paid = _llm_config(tmp_path, qfile)  # deepseek-v4-flash, not free
    assert _free_tier_models(paid) == []


class _ClientHolder:
    """A stand-in provider exposing a ``_client`` (httpx MockTransport) — no
    real socket is opened."""

    def __init__(self, handler):
        self._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_free_tier_check_warns_on_low_credit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-test")
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    config = _llm_config(tmp_path, qfile, model_over=FREE_MODEL)

    def handler(request):
        # Lifetime credit comes from /credits (preferred over the /key headroom).
        assert request.url.path.endswith("/credits")
        return httpx.Response(200, json={"data": {"total_credits": 5.0, "total_usage": 0.0}})

    holder = _ClientHolder(handler)
    asyncio.run(_free_tier_check(config, holder))
    asyncio.run(holder._client.aclose())
    err = capsys.readouterr().err
    assert "WARNING" in err  # < $10 => loud warning
    assert "lifetime credit is $5.00" in err


def test_free_tier_check_noop_without_free_model(tmp_path, capsys):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    config = _llm_config(tmp_path, qfile)  # paid model only

    holder = _ClientHolder(lambda r: httpx.Response(200, json={}))
    asyncio.run(_free_tier_check(config, holder))
    assert capsys.readouterr().err == ""  # no :free model => no check, no warning


def test_pilot_config_loads_and_validates():
    from pnyx.runner import load_config
    config = load_config(
        str(Path(__file__).resolve().parents[1] / "pnyx" / "configs" / "pilot.yaml"))
    assert config.condition == "PILOT"
    assert all(a.kind == "llm" for a in config.agents)
    assert set(config.models) == {"flash"}
    assert config.budget_usd == 2.0
    # Every agent references a defined model + a real persona.
    from pnyx.prompts import PERSONAS
    for a in config.agents:
        assert a.model in config.models
        assert a.persona in PERSONAS


def test_llm_agent_without_model_rejected(tmp_path):
    with pytest.raises(ValueError, match="model"):
        RunConfig.model_validate({
            "condition": "X", "seeds": [0], "n_questions": 1, "data_dir": "d",
            "env": {"n_signals": 6, "accuracies": [0.7] * 6, "lams": [0.0] * 6,
                    "difficulty": "easy"},
            "agents": [{"agent_id": "a0", "shard_indices": [0], "kind": "llm"}],
        })


def test_llm_agent_unknown_model_rejected(tmp_path):
    with pytest.raises(ValueError, match="not in models map"):
        RunConfig.model_validate({
            "condition": "X", "seeds": [0], "n_questions": 1, "data_dir": "d",
            "env": {"n_signals": 6, "accuracies": [0.7] * 6, "lams": [0.0] * 6,
                    "difficulty": "easy"},
            "models": {"real": MODEL},
            "agents": [{"agent_id": "a0", "shard_indices": [0], "kind": "llm",
                        "model": "ghost"}],
        })
