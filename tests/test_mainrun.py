"""Tests for pnyx.analysis.mainrun — the P5 grid loader, per-question
metrics, team error-correlation (rho), and pure statistics helpers
(cluster bootstrap, tie-corrected Wilcoxon, the compare wrapper).

All fixtures are hand-built synthetic event logs written by the tests
themselves (pnyx.runner naming convention) or in-memory ConditionData.
No network, no real run, no LLM calls anywhere. Exact-value assertions
where the definition permits.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from pnyx import baselines
from pnyx.analysis import mainrun
from pnyx.schemas import (
    Belief,
    BeliefEvent,
    QuestionRecord,
    SettlementEvent,
    SignalRecord,
    Trade,
    TradeEvent,
    TurnKey,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _question(qid, *, posterior_all, latent_state):
    return QuestionRecord(
        question_id=qid,
        latent_state=latent_state,
        signals=[
            SignalRecord(index=0, value=1, accuracy=0.7, lam=0.0),
            SignalRecord(index=1, value=1, accuracy=0.7, lam=0.0),
        ],
        posterior_table={"": 0.5, "0": 0.6, "1": 0.6, "0,1": posterior_all},
        shards=["shard 0", "shard 1"],
        question_text=f"Does {qid} resolve yes?",
    )


def _key(qid, agent_id, *, condition, seed, phase="independent", round_=0):
    return TurnKey(
        condition=condition, seed=seed, question_id=qid, phase=phase,
        round=round_, agent_id=agent_id,
    )


def _belief(qid, agent_id, prob, *, condition, seed, phase="independent",
            round_=0, parse_failed=False):
    return BeliefEvent(
        key=_key(qid, agent_id, condition=condition, seed=seed, phase=phase,
                 round_=round_),
        belief=Belief(prob=prob, rationale="r"),
        parse_failed=parse_failed,
        prompt_version="p3-v1",
        ts=0.0,
    )


def _trade(qid, agent_id, round_, *, condition, seed, price_before=0.5,
           price_after=0.5, parse_failed=False):
    return TradeEvent(
        key=_key(qid, agent_id, condition=condition, seed=seed, phase="market",
                 round_=round_),
        trade=Trade(belief=price_after, action="hold", shares=0.0, rationale="r"),
        executed_shares=0.0,
        cost=0.0,
        price_before=price_before,
        price_after=price_after,
        bankroll_after=100.0,
        parse_failed=parse_failed,
        prompt_version="p3-v1",
        ts=0.0,
    )


def _settlement(qid, outcome, final_price, *, condition, seed, subsidy=0.0):
    return SettlementEvent(
        condition=condition, seed=seed, question_id=qid, outcome=outcome,
        payouts={}, final_price=final_price, subsidy=subsidy, ts=0.0,
    )


def _write_run(run_dir: Path, condition: str, seed: int, events, questions):
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{condition}_seed{seed}"
    (run_dir / f"{stem}.jsonl").write_text(
        "".join(e.to_jsonl_line() + "\n" for e in events)
    )
    (run_dir / f"{stem}.questions.jsonl").write_text(
        "".join(q.model_dump_json() + "\n" for q in questions)
    )


# A canonical two-pass condition: 2 questions x 2 seeds x 3 honest agents.
# q000: posterior_all=0.8, outcome=1; p0=0.55, p1=0.65, p2=0.60 -> mean 0.60.
# q001: posterior_all=0.2, outcome=0; p0=0.35, p1=0.25, p2=0.30 -> mean 0.30.
# (Same beliefs both seeds; final prices differ so market cols vary.)
_BELIEFS = {
    "q000": {"p0": 0.55, "p1": 0.65, "p2": 0.60},
    "q001": {"p0": 0.35, "p1": 0.25, "p2": 0.30},
}
_POST = {"q000": 0.8, "q001": 0.2}
_OUTCOME = {"q000": 1, "q001": 0}
_FINAL = {
    0: {"q000": 0.75, "q001": 0.30},
    1: {"q000": 0.70, "q001": 0.25},
}


def _make_two_pass_dir(tmp_path: Path, condition="TWO") -> Path:
    run_dir = tmp_path / condition
    for seed in (0, 1):
        events = []
        for qid in ("q000", "q001"):
            for aid, prob in _BELIEFS[qid].items():
                events.append(_belief(qid, aid, prob, condition=condition, seed=seed))
            for aid in _BELIEFS[qid]:
                for rnd in (1, 2, 3):
                    events.append(_trade(qid, aid, rnd, condition=condition, seed=seed))
            events.append(_settlement(
                qid, _OUTCOME[qid], _FINAL[seed][qid],
                condition=condition, seed=seed, subsidy=1.5,
            ))
        questions = [_question(qid, posterior_all=_POST[qid],
                               latent_state=_OUTCOME[qid]) for qid in ("q000", "q001")]
        _write_run(run_dir, condition, seed, events, questions)
    return run_dir


def _make_pass2_only_dir(tmp_path: Path, condition="D_TEST") -> Path:
    """A pass2-only condition mirroring the two-pass one: trades + settlements
    for p0..p2 plus an adversary adv0, NO belief events."""
    run_dir = tmp_path / condition
    for seed in (0, 1):
        events = []
        for qid in ("q000", "q001"):
            for aid in list(_BELIEFS[qid]) + ["adv0"]:
                for rnd in (1, 2, 3):
                    events.append(_trade(qid, aid, rnd, condition=condition, seed=seed))
            events.append(_settlement(
                qid, _OUTCOME[qid], _FINAL[seed][qid],
                condition=condition, seed=seed, subsidy=2.0,
            ))
        questions = [_question(qid, posterior_all=_POST[qid],
                               latent_state=_OUTCOME[qid]) for qid in ("q000", "q001")]
        _write_run(run_dir, condition, seed, events, questions)
    return run_dir


# ---------------------------------------------------------------------------
# 1. load_condition
# ---------------------------------------------------------------------------


def test_load_condition_splits_events_and_questions(tmp_path):
    run_dir = _make_two_pass_dir(tmp_path)
    cond = mainrun.load_condition(run_dir)
    assert cond.condition == "TWO"
    assert sorted(cond.seeds) == [0, 1]
    sd = cond.seeds[0]
    assert len(sd.beliefs) == 6  # 2 questions x 3 agents
    assert len(sd.trades) == 18  # 2 x 3 x 3 rounds
    assert len(sd.settlements) == 2
    assert set(sd.questions) == {"q000", "q001"}
    assert sd.questions["q000"].posterior_table["0,1"] == 0.8


def test_load_condition_pass2_only_has_no_beliefs(tmp_path):
    run_dir = _make_pass2_only_dir(tmp_path)
    cond = mainrun.load_condition(run_dir)
    assert cond.condition == "D_TEST"
    assert cond.seeds[0].beliefs == []
    # 2 questions x 4 agents (p0,p1,p2,adv0) x 3 rounds
    assert len(cond.seeds[0].trades) == 24


# ---------------------------------------------------------------------------
# 2. question_metrics (two-pass) — hand-computed gaps/pools
# ---------------------------------------------------------------------------


def test_question_metrics_two_pass_exact(tmp_path):
    cond = mainrun.load_condition(_make_two_pass_dir(tmp_path))
    df = mainrun.question_metrics(cond, None)
    assert len(df) == 4  # 2 seeds x 2 questions
    row = df[(df.seed == 0) & (df.question_id == "q000")].iloc[0]
    assert row.posterior_all == 0.8
    assert row.outcome == 1
    # market
    assert row.market_prob == 0.75
    assert row.market_gap == pytest.approx(0.05)
    assert row.market_brier == pytest.approx(0.0625)
    # mean pool: (0.55+0.65+0.60)/3 = 0.60
    assert row.mean_prob == pytest.approx(0.60)
    assert row.mean_gap == pytest.approx(0.20)
    assert row.mean_brier == pytest.approx(0.16)
    # median pool: middle of {0.55,0.60,0.65} = 0.60
    assert row.median_prob == pytest.approx(0.60)
    assert row.median_gap == pytest.approx(0.20)
    # diagnostics
    assert row.subsidy == pytest.approx(1.5)
    assert row.n_parse_failed == 0


def test_question_metrics_pool_cols_match_baselines(tmp_path):
    cond = mainrun.load_condition(_make_two_pass_dir(tmp_path))
    df = mainrun.question_metrics(cond, None)
    # Reconstruct the exact BeliefEvents for q000/seed0 and confirm the
    # lop/stack columns are wired straight through pnyx.baselines.
    events = [_belief("q000", aid, prob, condition="TWO", seed=0)
              for aid, prob in _BELIEFS["q000"].items()]
    row = df[(df.seed == 0) & (df.question_id == "q000")].iloc[0]
    assert row.lop_prob == pytest.approx(baselines.log_opinion_pool(events))


def test_question_metrics_parse_failed_counted(tmp_path):
    condition = "PF"
    run_dir = tmp_path / condition
    events = []
    for aid, prob in _BELIEFS["q000"].items():
        events.append(_belief("q000", aid, prob, condition=condition, seed=0,
                              parse_failed=(aid == "p0")))
    events.append(_trade("q000", "p1", 1, condition=condition, seed=0,
                         parse_failed=True))
    events.append(_settlement("q000", 1, 0.75, condition=condition, seed=0))
    questions = [_question("q000", posterior_all=0.8, latent_state=1)]
    _write_run(run_dir, condition, 0, events, questions)
    df = mainrun.question_metrics(mainrun.load_condition(run_dir), None)
    assert df.iloc[0].n_parse_failed == 2


# ---------------------------------------------------------------------------
# 3. question_metrics (pass2-only reuse) — identity with source + coverage
# ---------------------------------------------------------------------------


def test_question_metrics_pass2_only_pool_identical_to_source(tmp_path):
    source = mainrun.load_condition(_make_two_pass_dir(tmp_path, "TWO"))
    cond = mainrun.load_condition(_make_pass2_only_dir(tmp_path, "D_TEST"))
    src_df = mainrun.question_metrics(source, None)
    df = mainrun.question_metrics(cond, source)
    pool_cols = [c for c in df.columns if c.startswith(("mean_", "median_",
                                                        "lop_", "stack_"))]
    for seed in (0, 1):
        for qid in ("q000", "q001"):
            r = df[(df.seed == seed) & (df.question_id == qid)].iloc[0]
            s = src_df[(src_df.seed == seed) & (src_df.question_id == qid)].iloc[0]
            for col in pool_cols:
                assert r[col] == s[col], f"{col} differs from source at {seed}/{qid}"
    # market columns come from the pass2-only condition itself, not the source.
    r = df[(df.seed == 0) & (df.question_id == "q000")].iloc[0]
    assert r.subsidy == pytest.approx(2.0)  # cond's subsidy, not source's 1.5


def test_question_metrics_pass2_only_raises_on_missing_coverage(tmp_path):
    # Source is missing p2's Pass-1 belief for q000/seed0 -> gap.
    source_dir = tmp_path / "SRC"
    for seed in (0, 1):
        events = []
        for qid in ("q000", "q001"):
            for aid, prob in _BELIEFS[qid].items():
                if seed == 0 and qid == "q000" and aid == "p2":
                    continue  # the hole
                events.append(_belief(qid, aid, prob, condition="SRC", seed=seed))
            events.append(_settlement(qid, _OUTCOME[qid], _FINAL[seed][qid],
                                      condition="SRC", seed=seed))
        questions = [_question(qid, posterior_all=_POST[qid],
                               latent_state=_OUTCOME[qid]) for qid in ("q000", "q001")]
        _write_run(source_dir, "SRC", seed, events, questions)
    source = mainrun.load_condition(source_dir)
    cond = mainrun.load_condition(_make_pass2_only_dir(tmp_path, "D_TEST"))
    with pytest.raises(ValueError, match="coverage|missing|belief"):
        mainrun.question_metrics(cond, source)


# ---------------------------------------------------------------------------
# 4. Phase guard
# ---------------------------------------------------------------------------


def test_question_metrics_rejects_market_phase_belief(tmp_path):
    condition = "BAD"
    run_dir = tmp_path / condition
    events = [
        _belief("q000", "p0", 0.6, condition=condition, seed=0, phase="market",
                round_=1),
        _belief("q000", "p1", 0.6, condition=condition, seed=0),
        _settlement("q000", 1, 0.75, condition=condition, seed=0),
    ]
    questions = [_question("q000", posterior_all=0.8, latent_state=1)]
    _write_run(run_dir, condition, 0, events, questions)
    cond = mainrun.load_condition(run_dir)
    with pytest.raises(ValueError, match="phase guard"):
        mainrun.question_metrics(cond, None)


# ---------------------------------------------------------------------------
# 5. team_rho + summary — hand-computed Pearson
# ---------------------------------------------------------------------------


def _make_rho_dir(tmp_path) -> Path:
    """One seed, 3 questions, posterior_all=0.5 each. Errors:
      p0: +0.1,+0.2,+0.3   p1: 0.0,+0.1,+0.2   p2: +0.3,+0.2,+0.1
    -> corr(p0,p1)=+1, corr(p0,p2)=-1, corr(p1,p2)=-1."""
    condition = "RHO"
    run_dir = tmp_path / condition
    probs = {
        "q0": {"p0": 0.6, "p1": 0.5, "p2": 0.8},
        "q1": {"p0": 0.7, "p1": 0.6, "p2": 0.7},
        "q2": {"p0": 0.8, "p1": 0.7, "p2": 0.6},
    }
    events = []
    for qid, per in probs.items():
        for aid, p in per.items():
            events.append(_belief(qid, aid, p, condition=condition, seed=0))
        events.append(_settlement(qid, 1, 0.5, condition=condition, seed=0))
    questions = [_question(qid, posterior_all=0.5, latent_state=1)
                 for qid in probs]
    _write_run(run_dir, condition, 0, events, questions)
    return run_dir


def test_team_rho_exact(tmp_path):
    cond = mainrun.load_condition(_make_rho_dir(tmp_path))
    df = mainrun.team_rho(cond)
    assert len(df) == 3  # 3 pairs, 1 seed

    def _rho(a, b):
        r = df[(df.seed == 0) & (df.agent_a == a) & (df.agent_b == b)]
        return r.iloc[0].rho

    assert _rho("p0", "p1") == pytest.approx(1.0)
    assert _rho("p0", "p2") == pytest.approx(-1.0)
    assert _rho("p1", "p2") == pytest.approx(-1.0)


def test_team_rho_summary(tmp_path):
    cond = mainrun.load_condition(_make_rho_dir(tmp_path))
    summ = mainrun.team_rho_summary(cond)
    # mean over 3 pairs = (1 - 1 - 1)/3 = -1/3
    assert summ.per_seed_mean[0] == pytest.approx(-1.0 / 3.0)
    assert summ.mean == pytest.approx(-1.0 / 3.0)
    assert summ.std == pytest.approx(0.0)  # single seed


def test_team_rho_rejects_market_phase_belief(tmp_path):
    condition = "RHOBAD"
    run_dir = tmp_path / condition
    events = [
        _belief("q0", "p0", 0.6, condition=condition, seed=0, phase="market",
                round_=1),
        _belief("q0", "p1", 0.6, condition=condition, seed=0),
        _settlement("q0", 1, 0.5, condition=condition, seed=0),
    ]
    questions = [_question("q0", posterior_all=0.5, latent_state=1)]
    _write_run(run_dir, condition, 0, events, questions)
    cond = mainrun.load_condition(run_dir)
    with pytest.raises(ValueError, match="phase guard"):
        mainrun.team_rho(cond)


# ---------------------------------------------------------------------------
# 6. paired_bootstrap — determinism + cluster structure
# ---------------------------------------------------------------------------


def test_paired_bootstrap_deterministic():
    rng = np.random.default_rng(42)
    q = np.repeat([f"q{i}" for i in range(20)], 3)  # 20 questions x 3 seeds
    d = rng.normal(0.1, 0.2, size=q.shape[0])
    a = mainrun.paired_bootstrap(d, q, n_boot=2000, seed=0)
    b = mainrun.paired_bootstrap(d, q, n_boot=2000, seed=0)
    assert a == b  # same seed -> identical output
    c = mainrun.paired_bootstrap(d, q, n_boot=2000, seed=1)
    assert a != c  # different seed -> different CI (with enough clusters)
    mean, lo, hi = a
    assert mean == pytest.approx(float(d.mean()))
    assert lo <= mean <= hi


def test_paired_bootstrap_cluster_unit_is_question():
    # A degenerate case: only one distinct question -> every resample picks
    # that same question, so the CI collapses to the point mean.
    d = np.array([0.1, 0.2, 0.3])
    q = np.array(["q0", "q0", "q0"])
    mean, lo, hi = mainrun.paired_bootstrap(d, q, n_boot=500, seed=0)
    assert lo == pytest.approx(mean)
    assert hi == pytest.approx(mean)


# ---------------------------------------------------------------------------
# 7. wilcoxon_signed_rank — exact values
# ---------------------------------------------------------------------------


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def test_wilcoxon_no_ties_exact():
    d = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    W, p = mainrun.wilcoxon_signed_rank(d)
    assert W == pytest.approx(9.0)
    # n=5, mu=7.5, var=330/24=13.75, sigma=sqrt(13.75)
    # z=(9-7.5-0.5)/sigma; two-sided p
    sigma = math.sqrt(13.75)
    z = (9.0 - 7.5 - 0.5) / sigma
    expected_p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    assert p == pytest.approx(expected_p)


def test_wilcoxon_with_ties_exact():
    d = np.array([1.0, 1.0, -1.0, 2.0])
    W, p = mainrun.wilcoxon_signed_rank(d)
    # |d|=[1,1,1,2]; the three 1s share rank 2, the 2 has rank 4.
    # positives: two 1s (rank 2) + the 2 (rank 4) -> W = 8
    assert W == pytest.approx(8.0)
    # n=4, mu=5, tie group t=3 -> sum(t^3-t)=24
    # var = 4*5*9/24 - 24/48 = 7.5 - 0.5 = 7.0
    sigma = math.sqrt(7.0)
    z = (8.0 - 5.0 - 0.5) / sigma
    expected_p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    assert p == pytest.approx(expected_p)


def test_wilcoxon_drops_zeros():
    # zeros dropped: this is d=[1,-2,3,-4,5] again after dropping the 0.
    d = np.array([0.0, 1.0, -2.0, 3.0, -4.0, 5.0])
    W, p = mainrun.wilcoxon_signed_rank(d)
    assert W == pytest.approx(9.0)


def test_wilcoxon_all_zeros():
    W, p = mainrun.wilcoxon_signed_rank(np.array([0.0, 0.0]))
    assert p == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. compare — integration over a metrics DataFrame
# ---------------------------------------------------------------------------


def test_compare_market_vs_mean(tmp_path):
    cond = mainrun.load_condition(_make_two_pass_dir(tmp_path))
    df = mainrun.question_metrics(cond, None)
    res = mainrun.compare(df, "market_gap", "mean_gap", n_boot=1000, seed=0)
    d = (df["market_gap"] - df["mean_gap"]).to_numpy(float)
    assert res.mean_diff == pytest.approx(float(d.mean()))
    assert res.n == 4
    assert set(res.per_seed_means) == {0, 1}
    # bootstrap CI brackets the mean
    assert res.ci[0] <= res.mean_diff <= res.ci[1]
    # Wilcoxon agrees with the standalone helper on the same differences.
    W, p = mainrun.wilcoxon_signed_rank(d)
    assert res.wilcoxon_W == pytest.approx(W)
    assert res.wilcoxon_p == pytest.approx(p)
