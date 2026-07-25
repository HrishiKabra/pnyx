"""Tests for pnyx.analysis.capability (v2 capability-tier + herding
decomposition) and the CLI's ``analyze-v2`` wiring, plus the two v2 figures
(``figures.fig2_v2`` phase map + ``figures.fig3`` deficit/price-weight panels).

Unit tests build small in-memory ``ConditionData`` (mainrun's loaded model) so
every regression coefficient / tier metric is hand-computable; the herding OLS
fixture has an EXACT least-squares solution asserted to 1e-9. The CLI
integration test writes real JSONL event logs to ``tmp_path`` (mirrors
tests/test_tables.py's mini-grid style) and runs the full ``analyze-v2``
pipeline end to end. No network, no real run, no LLM calls anywhere.

Phase-guard note: the herding analysis is the spec's sanctioned "dynamics
analysis" of in-market beliefs — it reads ``TradeEvent.trade.belief`` (market
phase) but must NEVER route those through ``pnyx.baselines`` or any ρ path.
``test_herding_never_touches_baselines`` is the contamination guard.
"""

import numpy as np
import pandas as pd
import pytest

from pnyx.analysis import capability, figures
from pnyx.analysis.mainrun import ConditionData, SeedData
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
# In-memory fixture builders
# ---------------------------------------------------------------------------


def _belief(qid, aid, prob, *, condition, seed, phase="independent", round_=0):
    return BeliefEvent(
        key=TurnKey(condition=condition, seed=seed, question_id=qid, phase=phase,
                    round=round_, agent_id=aid),
        belief=Belief(prob=prob, rationale="r"),
        parse_failed=False, prompt_version="p3-v1", ts=0.0,
    )


def _trade(qid, aid, round_, *, condition, seed, x2, y, parse_failed=False):
    """A market-pass trade whose stated in-market belief is ``y`` and whose
    pre-trade price is ``x2`` (the two herding regressors' second input)."""
    return TradeEvent(
        key=TurnKey(condition=condition, seed=seed, question_id=qid, phase="market",
                    round=round_, agent_id=aid),
        trade=Trade(belief=y, action="hold", shares=0.0, rationale="r"),
        executed_shares=0.0, cost=0.0, price_before=x2, price_after=x2,
        bankroll_after=100.0, parse_failed=parse_failed, prompt_version="p3-v1", ts=0.0,
    )


def _question(qid, *, posterior_all, latent_state):
    return QuestionRecord(
        question_id=qid, latent_state=latent_state,
        signals=[SignalRecord(index=0, value=1, accuracy=0.7, lam=0.0),
                 SignalRecord(index=1, value=1, accuracy=0.7, lam=0.0)],
        posterior_table={"": 0.5, "0": 0.6, "1": 0.6, "0,1": posterior_all},
        shards=["s0", "s1"], question_text=f"Does {qid} resolve yes?",
    )


def _settlement(qid, outcome, final_price, *, condition, seed, subsidy=1.0):
    return SettlementEvent(
        condition=condition, seed=seed, question_id=qid, outcome=outcome,
        payouts={}, final_price=final_price, subsidy=subsidy, ts=0.0,
    )


# Exact-OLS herding fixture: y = 0.2 + 0.5*x1 + 0.3*x2 for every trade, one
# round, one seed, 3 agents x 2 questions -> 6 non-collinear points so
# numpy.linalg.lstsq recovers (0.2, 0.5, 0.3) exactly.
_HERD_POINTS = {
    "q0": {"p0": (0.4, 0.5), "p1": (0.6, 0.4), "p2": (0.8, 0.6)},
    "q1": {"p0": (0.3, 0.7), "p1": (0.5, 0.3), "p2": (0.7, 0.5)},
}
_HERD_INTERCEPT, _HERD_B1, _HERD_B2 = 0.2, 0.5, 0.3


def _herd_y(x1, x2):
    return _HERD_INTERCEPT + _HERD_B1 * x1 + _HERD_B2 * x2


def _exact_herding_condition(condition="flash", *, extra_trades=None):
    beliefs, trades, settlements, questions = [], [], [], {}
    for qid, per in _HERD_POINTS.items():
        for aid, (x1, x2) in per.items():
            beliefs.append(_belief(qid, aid, x1, condition=condition, seed=0))
            trades.append(_trade(qid, aid, 1, condition=condition, seed=0,
                                 x2=x2, y=_herd_y(x1, x2)))
        settlements.append(_settlement(qid, 1, 0.7, condition=condition, seed=0))
        questions[qid] = _question(qid, posterior_all=0.8, latent_state=1)
    if extra_trades:
        trades.extend(extra_trades)
    sd = SeedData(seed=0, beliefs=beliefs, trades=trades,
                  settlements=settlements, questions=questions)
    return ConditionData(condition=condition, seeds={0: sd})


# ---------------------------------------------------------------------------
# herding_weights — exact OLS
# ---------------------------------------------------------------------------


def test_herding_weights_exact_lstsq_solution():
    cond = _exact_herding_condition()
    hw = capability.herding_weights(cond, n_boot=50, seed=0)
    pooled = hw[hw["scope"] == "pooled"].iloc[0]
    assert pooled["intercept"] == pytest.approx(_HERD_INTERCEPT, abs=1e-9)
    assert pooled["b1"] == pytest.approx(_HERD_B1, abs=1e-9)
    assert pooled["b2"] == pytest.approx(_HERD_B2, abs=1e-9)
    assert pooled["r2"] == pytest.approx(1.0, abs=1e-9)
    assert pooled["n"] == 6
    # only round 1 exists; its coefficients equal the pooled ones.
    r1 = hw[hw["scope"] == "1"].iloc[0]
    assert r1["b1"] == pytest.approx(_HERD_B1, abs=1e-9)
    assert r1["b2"] == pytest.approx(_HERD_B2, abs=1e-9)
    assert r1["n"] == 6


def test_herding_weights_matches_numpy_lstsq_directly():
    cond = _exact_herding_condition()
    hw = capability.herding_weights(cond, n_boot=50, seed=0)
    pooled = hw[hw["scope"] == "pooled"].iloc[0]
    x1, x2, y = [], [], []
    for qid, per in _HERD_POINTS.items():
        for _aid, (a, b) in per.items():
            x1.append(a); x2.append(b); y.append(_herd_y(a, b))
    X = np.column_stack([np.ones(len(y)), x1, x2])
    coef, *_ = np.linalg.lstsq(X, np.array(y), rcond=None)
    assert pooled["intercept"] == pytest.approx(coef[0], abs=1e-9)
    assert pooled["b1"] == pytest.approx(coef[1], abs=1e-9)
    assert pooled["b2"] == pytest.approx(coef[2], abs=1e-9)


def test_herding_weights_drift_descriptive():
    # drift = mean |y - x1| over the scope's rows. For q0/p0: y=0.55, x1=0.4
    # -> 0.15; hand-average all 6 |y-x1|.
    cond = _exact_herding_condition()
    hw = capability.herding_weights(cond, n_boot=50, seed=0)
    pooled = hw[hw["scope"] == "pooled"].iloc[0]
    diffs = []
    for qid, per in _HERD_POINTS.items():
        for _aid, (x1, x2) in per.items():
            diffs.append(abs(_herd_y(x1, x2) - x1))
    assert pooled["drift"] == pytest.approx(float(np.mean(diffs)), abs=1e-12)


def test_herding_weights_excludes_parse_failed_trades():
    # A parse-failed trade with a wild belief that would wreck the exact fit;
    # it must be dropped, leaving the exact (0.2, 0.5, 0.3) solution and n=6.
    poison = _trade("q0", "p0", 1, condition="flash", seed=0, x2=0.9, y=0.99,
                    parse_failed=True)
    cond = _exact_herding_condition(extra_trades=[poison])
    hw = capability.herding_weights(cond, n_boot=50, seed=0)
    pooled = hw[hw["scope"] == "pooled"].iloc[0]
    assert pooled["n"] == 6
    assert pooled["b1"] == pytest.approx(_HERD_B1, abs=1e-9)
    assert pooled["b2"] == pytest.approx(_HERD_B2, abs=1e-9)


def test_herding_weights_excludes_non_honest_agents():
    adv = _trade("q0", "adv0", 1, condition="flash", seed=0, x2=0.9, y=0.99)
    cond = _exact_herding_condition(extra_trades=[adv])
    hw = capability.herding_weights(cond, n_boot=50, seed=0)
    pooled = hw[hw["scope"] == "pooled"].iloc[0]
    # adv0 has no honest Pass-1 belief and is not in p0..p5 -> excluded.
    assert pooled["n"] == 6
    assert pooled["b1"] == pytest.approx(_HERD_B1, abs=1e-9)


def test_herding_weights_bootstrap_ci_deterministic():
    cond = _exact_herding_condition()
    a = capability.herding_weights(cond, n_boot=200, seed=0)
    b = capability.herding_weights(cond, n_boot=200, seed=0)
    pd.testing.assert_frame_equal(a, b)
    pooled = a[a["scope"] == "pooled"].iloc[0]
    # exact-fit fixture: every bootstrap draw recovers the same coefficients,
    # so the CI collapses onto the point estimate.
    assert pooled["b2_lo"] == pytest.approx(_HERD_B2, abs=1e-9)
    assert pooled["b2_hi"] == pytest.approx(_HERD_B2, abs=1e-9)


def test_herding_weights_has_all_scopes():
    cond = _exact_herding_condition()
    hw = capability.herding_weights(cond, n_boot=50, seed=0)
    assert list(hw["scope"]) == ["pooled", "1", "2", "3"]


def test_herding_never_touches_baselines(monkeypatch):
    """Contamination guard: the market-phase herding path must never reach a
    static-pool baseline. Poison every ``pnyx.baselines`` function so a single
    call raises; ``herding_weights`` (which reads Pass-2 ``trade.belief``) must
    still complete, proving those beliefs never flow into a baseline."""
    import pnyx.baselines as baselines

    def _boom(*_a, **_k):
        raise AssertionError("herding must not call a static-pool baseline")

    for name in ("mean_pool", "median_pool", "log_opinion_pool",
                 "loo_calibrated_stack", "assert_pass1"):
        monkeypatch.setattr(baselines, name, _boom)

    cond = _exact_herding_condition()
    hw = capability.herding_weights(cond, n_boot=50, seed=0)  # must not raise
    assert not hw.empty
    # The module must not even carry a baselines reference on its market path.
    assert not hasattr(capability, "baselines")


# ---------------------------------------------------------------------------
# tier_metrics + cross-tier comparisons
# ---------------------------------------------------------------------------


def _tier_condition(condition, *, market_final, beliefs, posterior_all=0.8):
    """A two-pass condition: 3 agents x 2 questions x 1 seed. ``beliefs`` maps
    qid -> {agent: prob}; ``market_final`` maps qid -> settlement final price."""
    bev, trd, stl, q = [], [], [], {}
    for qid, per in beliefs.items():
        for aid, prob in per.items():
            bev.append(_belief(qid, aid, prob, condition=condition, seed=0))
            for rnd in (1, 2, 3):
                trd.append(_trade(qid, aid, rnd, condition=condition, seed=0,
                                  x2=0.5, y=prob))
        stl.append(_settlement(qid, 1, market_final[qid], condition=condition, seed=0))
        q[qid] = _question(qid, posterior_all=posterior_all, latent_state=1)
    sd = SeedData(seed=0, beliefs=bev, trades=trd, settlements=stl, questions=q)
    return ConditionData(condition=condition, seeds={0: sd})


def _three_tiers():
    beliefs = {"q0": {"p0": 0.7, "p1": 0.75, "p2": 0.8},
               "q1": {"p0": 0.65, "p1": 0.7, "p2": 0.75}}
    # market_final chosen so each tier has a distinct market gap.
    flash = _tier_condition("flash", market_final={"q0": 0.9, "q1": 0.9}, beliefs=beliefs)
    pro = _tier_condition("pro", market_final={"q0": 0.8, "q1": 0.75}, beliefs=beliefs)
    luna = _tier_condition("luna", market_final={"q0": 0.78, "q1": 0.72}, beliefs=beliefs)
    return {"flash": flash, "pro": pro, "luna": luna}


def test_tier_metrics_one_row_per_tier_in_order():
    df = capability.tier_metrics(_three_tiers(), n_boot=100, bootstrap_seed=0)
    assert list(df["tier"]) == ["flash", "pro", "luna"]
    assert set(["pool_gap", "market_gap", "deficit", "deficit_ci_lo",
                "deficit_ci_hi", "deficit_p", "market_brier", "rho_mean",
                "rho_std", "n"]).issubset(df.columns)


def test_tier_metrics_deficit_equals_within_tier_compare():
    conds = _three_tiers()
    df = capability.tier_metrics(conds, n_boot=300, bootstrap_seed=0)
    from pnyx.analysis import mainrun
    qm = mainrun.question_metrics(conds["flash"], None)
    cmp = mainrun.compare(qm, "market_gap", "mean_gap", n_boot=300, seed=0)
    row = df[df["tier"] == "flash"].iloc[0]
    assert row["deficit"] == pytest.approx(cmp.mean_diff)
    assert row["deficit_ci_lo"] == pytest.approx(cmp.ci[0])
    assert row["deficit_ci_hi"] == pytest.approx(cmp.ci[1])
    assert row["deficit_p"] == pytest.approx(cmp.wilcoxon_p)
    # market_gap column is the grand mean of per-seed market_gap means.
    assert row["market_gap"] == pytest.approx(
        qm.groupby("seed")["market_gap"].mean().mean()
    )


def test_cross_tier_compare_is_directional_paired():
    conds = _three_tiers()
    from pnyx.analysis import mainrun
    mets = {t: mainrun.question_metrics(c, None) for t, c in conds.items()}
    res = capability.cross_tier_compare(mets["pro"], mets["flash"], "market_gap",
                                        n_boot=300, seed=0)
    # pro - flash, paired on (seed, question): reconstruct directly.
    merged = mets["pro"][["seed", "question_id", "market_gap"]].merge(
        mets["flash"][["seed", "question_id", "market_gap"]],
        on=["seed", "question_id"], suffixes=("_x", "_y"))
    exp = mainrun.compare(merged, "market_gap_x", "market_gap_y", n_boot=300, seed=0)
    assert res.mean_diff == pytest.approx(exp.mean_diff)
    assert res.ci == pytest.approx(exp.ci)
    assert res.wilcoxon_p == pytest.approx(exp.wilcoxon_p)


# ---------------------------------------------------------------------------
# markdown renderers — determinism + content
# ---------------------------------------------------------------------------


def test_render_tiers_md_deterministic_and_has_sections():
    conds = _three_tiers()
    md1 = capability.render_tiers_md(conds, n_boot=300, bootstrap_seed=0)
    md2 = capability.render_tiers_md(conds, n_boot=300, bootstrap_seed=0)
    assert md1 == md2
    assert "Wilcoxon p =" in md1
    assert "flash" in md1 and "pro" in md1 and "luna" in md1


def test_render_herding_md_deterministic():
    conds = _three_tiers()
    hbt = {t: capability.herding_weights(c, n_boot=200, seed=0) for t, c in conds.items()}
    md1 = capability.render_herding_md(hbt)
    md2 = capability.render_herding_md(hbt)
    assert md1 == md2
    assert "b1" in md1 and "b2" in md1


# ---------------------------------------------------------------------------
# figures — fig2_v2 (phase map w/ new tiers) + fig3 (deficit + price weight)
# ---------------------------------------------------------------------------


def test_figures_have_v2_condition_palette():
    for cond in ("A_PRO", "A_LUNA"):
        assert cond in figures._CONDITION_COLOR
        assert cond in figures._CONDITION_MARKER
    assert figures._CONDITION_COLOR["A_PRO"] == "#56B4E9"
    assert figures._CONDITION_COLOR["A_LUNA"] == "#D55E00"
    assert figures._CONDITION_MARKER["A_PRO"] == "v"
    assert figures._CONDITION_MARKER["A_LUNA"] == "P"


def _rho_summary(per_seed):
    from pnyx.analysis.mainrun import RhoSummary
    vals = np.array(list(per_seed.values()), dtype=float)
    return RhoSummary(per_seed_mean=per_seed, mean=float(vals.mean()), std=float(vals.std()))


def _gapdiff_df(pairs_by_seed):
    rows = []
    for seed, pairs in pairs_by_seed.items():
        for mg, pg in pairs:
            rows.append({"seed": seed, "market_gap": mg, "mean_gap": pg})
    return pd.DataFrame(rows)


def test_fig2_v2_writes_png_and_pdf_with_new_tiers(tmp_path):
    rho_by_cond = {
        "A": _rho_summary({0: 0.4}),
        "A_PRO": _rho_summary({0: 0.38}),
        "A_LUNA": _rho_summary({0: 0.28}),
    }
    gapdiff_by_cond = {
        "A": _gapdiff_df({0: [(0.15, 0.10)]}),
        "A_PRO": _gapdiff_df({0: [(0.10, 0.10)]}),
        "A_LUNA": _gapdiff_df({0: [(0.10, 0.10)]}),
    }
    figures.fig2_v2(rho_by_cond, gapdiff_by_cond, tmp_path)
    assert (tmp_path / "fig2_v2.png").stat().st_size > 0
    assert (tmp_path / "fig2_v2.pdf").stat().st_size > 0


def test_fig3_writes_png_and_pdf(tmp_path):
    tiers_df = capability.tier_metrics(_three_tiers(), n_boot=100, bootstrap_seed=0)
    hbt = {t: capability.herding_weights(c, n_boot=100, seed=0)
           for t, c in _three_tiers().items()}
    figures.fig3(tiers_df, hbt, tmp_path)
    assert (tmp_path / "fig3.png").stat().st_size > 0
    assert (tmp_path / "fig3.pdf").stat().st_size > 0


# ---------------------------------------------------------------------------
# CLI integration: analyze-v2 on a tmp mini-grid
# ---------------------------------------------------------------------------


def _write_condition_dir(root, condition, cond_data):
    run_dir = root / condition
    run_dir.mkdir(parents=True, exist_ok=True)
    for seed, sd in cond_data.seeds.items():
        stem = f"{condition}_seed{seed}"
        events = list(sd.beliefs) + list(sd.trades) + list(sd.settlements)
        (run_dir / f"{stem}.jsonl").write_text(
            "".join(e.to_jsonl_line() + "\n" for e in events))
        (run_dir / f"{stem}.questions.jsonl").write_text(
            "".join(q.model_dump_json() + "\n" for q in sd.questions.values()))
    return run_dir


@pytest.fixture
def mini_v2_grid(tmp_path):
    root = tmp_path / "data"
    conds = _three_tiers()
    flash_dir = _write_condition_dir(root, "A", conds["flash"])
    pro_dir = _write_condition_dir(root, "A_PRO", conds["pro"])
    luna_dir = _write_condition_dir(root, "A_LUNA", conds["luna"])
    return flash_dir, pro_dir, luna_dir


def test_analyze_v2_cli_writes_all_outputs(mini_v2_grid, tmp_path, capsys):
    from pnyx import cli
    flash_dir, pro_dir, luna_dir = mini_v2_grid
    out_dir = tmp_path / "out"
    rc = cli.main([
        "analyze-v2", "--flash", str(flash_dir), "--pro", str(pro_dir),
        "--luna", str(luna_dir), "--out", str(out_dir), "--bootstrap-seed", "0",
    ])
    assert rc == 0
    for name in ("tiers.md", "herding.md", "fig3.png", "fig3.pdf",
                 "fig2_v2.png", "fig2_v2.pdf"):
        p = out_dir / name
        assert p.exists() and p.stat().st_size > 0, f"{name} missing/empty"
    tiers = (out_dir / "tiers.md").read_text()
    assert "flash" in tiers and "pro" in tiers and "luna" in tiers


def test_analyze_v2_cli_md_byte_deterministic(mini_v2_grid, tmp_path):
    from pnyx import cli
    flash_dir, pro_dir, luna_dir = mini_v2_grid
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    for out in (out1, out2):
        rc = cli.main([
            "analyze-v2", "--flash", str(flash_dir), "--pro", str(pro_dir),
            "--luna", str(luna_dir), "--out", str(out), "--bootstrap-seed", "0",
        ])
        assert rc == 0
    assert (out1 / "tiers.md").read_bytes() == (out2 / "tiers.md").read_bytes()
    assert (out1 / "herding.md").read_bytes() == (out2 / "herding.md").read_bytes()
