"""P4 runner tests: adversary support, Pass-1 reuse (pass1_source_dir), and the
shuffled question-order + its stability across resume.

Everything runs through a FAKE provider (the ScriptedProvider from
test_runner_llm) or the pure-mock path — zero network, enforced by the conftest
socket guard.
"""

from pathlib import Path

import pytest

from pnyx.prompts import ADVERSARY_PROMPT_VERSION, PROMPT_VERSION
from pnyx.runner import (
    _order_questions,
    log_path_for,
    questions_path_for,
    read_events,
    run_experiment,
    ts_stripped_lines,
)
from pnyx.schemas import Belief, QuestionRecord, RunConfig, Trade

from tests._fixtures import write_questions_file
from tests.test_runner_llm import MODEL, ScriptedProvider

_ENV = {
    "n_signals": 6, "accuracies": [0.75] * 6, "lams": [0.0] * 6,
    "a_z": 0.8, "difficulty": "easy", "max_rejection_tries": 2000,
}


def _llm_config(tmp_path, qfile, *, condition, agents, n_questions=1, n_rounds=1,
                pass1_source_dir=None, data_dir=None):
    return RunConfig.model_validate({
        "condition": condition,
        "seeds": [0],
        "n_questions": n_questions,
        "b": 40.0,
        "n_rounds": n_rounds,
        "wealth_persistent": False,
        "data_dir": data_dir or str(tmp_path / condition),
        "questions_file": str(qfile),
        "pass1_source_dir": pass1_source_dir,
        "budget_usd": 25.0,
        "temperature": 0.7,
        "env": _ENV,
        "models": {"m": dict(MODEL)},
        "agents": agents,
    })


def _honest(agent_id, shard, persona):
    return {"agent_id": agent_id, "shard_indices": [shard], "kind": "llm",
            "model": "m", "persona": persona}


def _adv(agent_id="adv0", *, style="stealthy", bankroll=100.0):
    return {"agent_id": agent_id, "shard_indices": [], "kind": "llm", "model": "m",
            "adversary": True, "adversary_style": style, "bankroll": bankroll}


# ---------------------------------------------------------------------------
# Adversary: excluded from Pass 1, tagged with its own prompt version
# ---------------------------------------------------------------------------


def test_adversary_excluded_from_pass1_and_tagged(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    config = _llm_config(
        tmp_path, qfile, condition="ADV", n_rounds=2,
        agents=[_honest("p0", 0, "contrarian"), _adv("adv0", style="stealthy")],
    )
    prov = ScriptedProvider(
        trade_maker=lambda i: Trade(belief=0.6, action="buy_yes", shares=1.0, rationale="t"))
    run_experiment(config, provider=prov)

    events = read_events(log_path_for(config, 0))
    beliefs = [e for e in events if e.type == "belief"]
    trades = [e for e in events if e.type == "trade"]

    # Only the honest agent is elicited a Pass-1 belief; the adversary has none.
    assert [e.key.agent_id for e in beliefs] == ["p0"]
    assert not any(e.key.agent_id == "adv0" for e in beliefs)

    adv_trades = [e for e in trades if e.key.agent_id == "adv0"]
    honest_trades = [e for e in trades if e.key.agent_id == "p0"]
    assert len(adv_trades) == 2 and len(honest_trades) == 2  # one per round each
    assert all(e.prompt_version == ADVERSARY_PROMPT_VERSION for e in adv_trades)
    assert all(e.prompt_version == PROMPT_VERSION for e in honest_trades)

    # The adversary's provider calls used the distortion system block, not the
    # honest trader block.
    adv_calls = [msgs for (schema, msgs) in prov.calls
                 if schema == "Trade" and "distort" in msgs[0]["content"].lower()]
    honest_calls = [msgs for (schema, msgs) in prov.calls
                    if schema == "Trade" and "maximize your bankroll" in msgs[0]["content"]]
    assert len(adv_calls) == 2 and len(honest_calls) == 2


# ---------------------------------------------------------------------------
# Pass-1 reuse: pass1_source_dir loads beliefs into Pass 2, logs no beliefs
# ---------------------------------------------------------------------------


def _run_source(tmp_path, qfile, *, agents, n_questions, belief_prob):
    src = _llm_config(tmp_path, qfile, condition="SRC", n_questions=n_questions,
                      agents=agents, data_dir=str(tmp_path / "src"))
    prov = ScriptedProvider(belief_maker=lambda i: Belief(prob=belief_prob, rationale="b"))
    run_experiment(src, provider=prov)
    return src


def test_pass1_source_dir_reuse_carries_belief_into_pass2(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 2)
    honest = [_honest("p0", 0, "contrarian"), _honest("p1", 1, "kelly_sizer")]
    _run_source(tmp_path, qfile, agents=honest, n_questions=2, belief_prob=0.77)

    target = _llm_config(
        tmp_path, qfile, condition="TGT", n_questions=2, agents=honest,
        pass1_source_dir=str(tmp_path / "src"), data_dir=str(tmp_path / "tgt"),
    )
    assert target.pass2_only is True
    prov = ScriptedProvider()
    run_experiment(target, provider=prov)

    events = read_events(log_path_for(target, 0))
    # No belief turns logged in a pass2_only run (Pass 1 was reused, not elicited).
    assert not any(e.type == "belief" for e in events)
    assert len([e for e in events if e.type == "trade"]) == 4  # 2 q * 2 agents * 1 round
    assert len([e for e in events if e.type == "settlement"]) == 2

    # The loaded source belief (0.77) was carried into each honest Pass-2 prompt.
    trade_calls = [msgs for (schema, msgs) in prov.calls if schema == "Trade"]
    assert trade_calls
    for msgs in trade_calls:
        assert "0.770" in msgs[1]["content"]
        assert "prior analysis concluded" in msgs[1]["content"]


def test_pass1_source_dir_missing_belief_fails_loudly(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    # Source only elicits p0; the target additionally expects p1 -> a gap.
    _run_source(tmp_path, qfile, agents=[_honest("p0", 0, "contrarian")],
                n_questions=1, belief_prob=0.6)

    target = _llm_config(
        tmp_path, qfile, condition="TGT", n_questions=1,
        agents=[_honest("p0", 0, "contrarian"), _honest("p1", 1, "kelly_sizer")],
        pass1_source_dir=str(tmp_path / "src"), data_dir=str(tmp_path / "tgt"),
    )
    with pytest.raises(ValueError, match="missing"):
        run_experiment(target, provider=ScriptedProvider())


def test_pass2_only_with_adversary_needs_no_source_belief(tmp_path):
    """D-shape: honest beliefs reused from a source, adversary added with NO
    source belief required (it is never elicited one)."""
    qfile = write_questions_file(tmp_path / "ds.jsonl", 1)
    honest = [_honest("p0", 0, "contrarian"), _honest("p1", 1, "kelly_sizer")]
    _run_source(tmp_path, qfile, agents=honest, n_questions=1, belief_prob=0.7)

    target = _llm_config(
        tmp_path, qfile, condition="D_K3", n_questions=1, n_rounds=1,
        agents=honest + [_adv("adv0", style="stealthy", bankroll=300.0)],
        pass1_source_dir=str(tmp_path / "src"), data_dir=str(tmp_path / "d"),
    )
    prov = ScriptedProvider(
        trade_maker=lambda i: Trade(belief=0.5, action="buy_no", shares=1.0, rationale="t"))
    run_experiment(target, provider=prov)  # must not raise despite no adv belief

    events = read_events(log_path_for(target, 0))
    assert not any(e.type == "belief" for e in events)
    adv_trades = [e for e in events if e.type == "trade" and e.key.agent_id == "adv0"]
    assert len(adv_trades) == 1
    assert adv_trades[0].prompt_version == ADVERSARY_PROMPT_VERSION
    assert len([e for e in events if e.type == "settlement"]) == 1


# ---------------------------------------------------------------------------
# Shuffled question order: deterministic + stable across resume (mock path)
# ---------------------------------------------------------------------------


def _mock_shuffled_config(tmp_path, qfile, n_questions):
    return RunConfig.model_validate({
        "condition": "ORD",
        "seeds": [0],
        "n_questions": n_questions,
        "b": 40.0,
        "n_rounds": 2,
        "wealth_persistent": True,  # order matters for wealth carryover
        "data_dir": str(tmp_path / "ord"),
        "questions_file": str(qfile),
        "question_order": "shuffled",
        "env": _ENV,
        "agents": [
            {"agent_id": f"agent{i}", "shard_indices": [i], "bankroll": 100.0}
            for i in range(6)
        ],
    })


def _question_order_in_log(events) -> list[str]:
    seen, order = set(), []
    for e in events:
        qid = e.question_id if e.type == "settlement" else e.key.question_id
        if qid not in seen:
            seen.add(qid)
            order.append(qid)
    return order


def test_shuffled_order_is_seeded_permutation_not_file_order(tmp_path):
    qfile = write_questions_file(tmp_path / "ds.jsonl", 8)
    config = _mock_shuffled_config(tmp_path, qfile, 8)
    run_experiment(config)

    sidecar = [QuestionRecord.model_validate_json(l)
               for l in questions_path_for(config, 0).read_text().splitlines() if l]
    file_order = [q.question_id for q in sidecar]
    expected = [q.question_id for q in _order_questions(0, sidecar)]

    log_order = _question_order_in_log(read_events(log_path_for(config, 0)))
    assert log_order == expected
    assert sorted(log_order) == sorted(file_order)  # a genuine permutation
    assert log_order != file_order  # actually reordered (seed 0, n=8)


def test_shuffled_order_stable_across_resume(tmp_path):
    """Kill-resume-style: truncate the log mid-run, resume, and the completed
    log must be byte-identical to an uninterrupted reference — proving the
    shuffled processing order is derived from the seed, not runtime state."""
    qfile = write_questions_file(tmp_path / "ds.jsonl", 6)
    config = _mock_shuffled_config(tmp_path, qfile, 6)

    run_experiment(config)  # reference (uninterrupted)
    log = log_path_for(config, 0)
    reference = ts_stripped_lines(log)
    ref_order = _question_order_in_log(read_events(log))

    # Simulate a mid-run kill: drop the trailing ~40% of complete log lines.
    lines = log.read_text().splitlines()
    assert len(lines) > 10
    keep = int(len(lines) * 0.6)
    log.write_text("".join(l + "\n" for l in lines[:keep]))
    assert len(read_events(log)) < len(reference)

    run_experiment(config)  # resume in the same data_dir
    assert ts_stripped_lines(log) == reference
    assert _question_order_in_log(read_events(log)) == ref_order
