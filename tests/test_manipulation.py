"""Tests for pnyx.analysis.manipulation — H3 adversary metrics and H-wealth
order effects.

All fixtures are hand-built in-memory ConditionData / DataFrames with known
adversary trades, so every column has an exact expected value. No network, no
real run, no LLM calls. Covers both recovery branches, the no-adversary-trade
edge case, displacement <= 0 -> NaN, the max(cost, 0) spend clamp, and flip
logic against a fabricated honest reference.
"""

import math

import numpy as np
import pandas as pd
import pytest

from pnyx.analysis import manipulation
from pnyx.analysis.mainrun import ConditionData, SeedData
from pnyx.schemas import SettlementEvent, Trade, TradeEvent, TurnKey


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _trade(
    qid,
    agent_id,
    round_,
    *,
    condition="D_K1",
    seed=0,
    price_before=0.5,
    price_after=0.5,
    cost=0.0,
    bankroll_after=100.0,
    action="hold",
    shares=0.0,
):
    return TradeEvent(
        key=TurnKey(
            condition=condition,
            seed=seed,
            question_id=qid,
            phase="market",
            round=round_,
            agent_id=agent_id,
        ),
        trade=Trade(belief=price_after, action=action, shares=shares, rationale="r"),
        executed_shares=shares,
        cost=cost,
        price_before=price_before,
        price_after=price_after,
        bankroll_after=bankroll_after,
        parse_failed=False,
        prompt_version="p3-v3-adv2",
        ts=0.0,
    )


def _settlement(qid, outcome, final_price, *, condition="D_K1", seed=0, payouts=None):
    return SettlementEvent(
        condition=condition,
        seed=seed,
        question_id=qid,
        outcome=outcome,
        payouts=payouts if payouts is not None else {},
        final_price=final_price,
        subsidy=0.0,
        ts=0.0,
    )


def _cond(condition, seed_map):
    """seed_map: {seed: (trades, settlements)} -> ConditionData."""
    seeds = {}
    for seed, (trades, settlements) in seed_map.items():
        seeds[seed] = SeedData(seed=seed, trades=trades, settlements=settlements)
    return ConditionData(condition=condition, seeds=seeds)


def _ref(rows):
    """Fabricated honest_ref (subset of mainrun.question_metrics columns)."""
    return pd.DataFrame(
        rows, columns=["seed", "question_id", "market_prob", "market_brier"]
    )


# ---------------------------------------------------------------------------
# target_side
# ---------------------------------------------------------------------------


def test_target_side_wrong_of_truth():
    assert manipulation.target_side(0) == "YES"  # truly NO -> push YES
    assert manipulation.target_side(1) == "NO"  # truly YES -> push NO


# ---------------------------------------------------------------------------
# adversary_metrics — the recovered / target-NO scenario, exact values
# ---------------------------------------------------------------------------


def test_adversary_metrics_recovered_branch_exact():
    # outcome=1 -> target side NO -> wrong side is price < 0.5.
    # adv0 buys NO twice, pushing 0.5 -> 0.4 -> 0.3; honest p1 then recovers.
    trades = [
        _trade("q0", "adv0", 1, price_before=0.5, price_after=0.4, cost=10.0,
               bankroll_after=90.0, action="buy_no", shares=10.0),
        _trade("q0", "adv0", 2, price_before=0.4, price_after=0.3, cost=12.0,
               bankroll_after=78.0, action="buy_no", shares=12.0),
        _trade("q0", "p0", 3, price_before=0.3, price_after=0.35),  # no recover
        _trade("q0", "p1", 3, price_before=0.35, price_after=0.48),  # recovers
    ]
    settlements = [_settlement("q0", 1, 0.3, payouts={"adv0": 0.0})]
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "q0", 0.7, (0.7 - 1) ** 2]])  # honest finished correct (>0.5)

    df = manipulation.adversary_metrics(cond, ref)
    assert len(df) == 1
    row = df.iloc[0]

    assert row["target_side"] == "NO"
    assert row["wrong_side"] == True  # 0.3 < 0.5
    assert row["flipped"] == True  # honest 0.7 correct, this run wrong
    assert row["adv_spend"] == pytest.approx(22.0)
    # displacement toward NO = (0.5-0.4) + (0.4-0.3) = 0.2
    assert row["adv_displacement"] == pytest.approx(0.2)
    assert row["capital_per_01_move"] == pytest.approx(0.1 * 22.0 / 0.2)  # 11.0
    # adv_pnl = payout(0) + final bankroll(78) - 100 = -22
    assert row["adv_pnl"] == pytest.approx(-22.0)
    assert row["pre_attack_price"] == pytest.approx(0.5)
    assert row["recovered"] == True
    assert row["recovery_rounds"] == pytest.approx(1.0)  # round 3 - round 2
    assert row["market_brier"] == pytest.approx((0.3 - 1) ** 2)
    assert row["ref_market_brier"] == pytest.approx((0.7 - 1) ** 2)


# ---------------------------------------------------------------------------
# adversary_metrics — the NOT-recovered branch + target YES
# ---------------------------------------------------------------------------


def test_adversary_metrics_not_recovered_target_yes():
    # outcome=0 -> target side YES -> wrong side is price > 0.5.
    trades = [
        _trade("q1", "adv0", 1, price_before=0.5, price_after=0.6, cost=8.0,
               bankroll_after=92.0, action="buy_yes", shares=8.0),
        _trade("q1", "adv0", 2, price_before=0.6, price_after=0.7, cost=9.0,
               bankroll_after=83.0, action="buy_yes", shares=9.0),
        # honest trades stay far from pre-attack 0.5 -> no recovery
        _trade("q1", "p0", 3, price_before=0.7, price_after=0.68),
        _trade("q1", "p2", 3, price_before=0.68, price_after=0.65),
    ]
    settlements = [_settlement("q1", 0, 0.65, condition="D_K1", payouts={"adv0": 0.0})]
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "q1", 0.3, (0.3 - 0) ** 2]])  # honest correct (<0.5)

    row = manipulation.adversary_metrics(cond, ref).iloc[0]
    assert row["target_side"] == "YES"
    assert row["wrong_side"] == True  # 0.65 > 0.5
    assert row["flipped"] == True
    assert row["adv_spend"] == pytest.approx(17.0)
    # displacement toward YES = (0.6-0.5)+(0.7-0.6) = 0.2
    assert row["adv_displacement"] == pytest.approx(0.2)
    assert row["capital_per_01_move"] == pytest.approx(0.1 * 17.0 / 0.2)
    assert row["adv_pnl"] == pytest.approx(0.0 + 83.0 - 100.0)
    assert row["recovered"] == False
    assert math.isnan(row["recovery_rounds"])


# ---------------------------------------------------------------------------
# No adversary trade edge case
# ---------------------------------------------------------------------------


def test_adversary_metrics_no_adversary_trade():
    # Only honest trades; adv0 absent from both trades and payouts.
    trades = [
        _trade("q2", "p0", 1, price_before=0.5, price_after=0.55),
        _trade("q2", "p1", 2, price_before=0.55, price_after=0.6),
    ]
    settlements = [_settlement("q2", 1, 0.6, payouts={"p0": 1.0})]
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "q2", 0.62, (0.62 - 1) ** 2]])

    row = manipulation.adversary_metrics(cond, ref).iloc[0]
    assert row["adv_spend"] == pytest.approx(0.0)
    assert row["adv_displacement"] == pytest.approx(0.0)
    assert math.isnan(row["capital_per_01_move"])
    # adv never traded -> final bankroll defaults to 100, payout 0 -> pnl 0
    assert row["adv_pnl"] == pytest.approx(0.0)
    assert math.isnan(row["pre_attack_price"])
    assert row["recovered"] == False
    assert math.isnan(row["recovery_rounds"])


# ---------------------------------------------------------------------------
# Displacement <= 0 -> capital NaN
# ---------------------------------------------------------------------------


def test_adversary_metrics_nonpositive_displacement_nan():
    # target YES but adv net moves price DOWN -> displacement negative -> NaN.
    trades = [
        _trade("q3", "adv0", 1, price_before=0.5, price_after=0.45, cost=5.0,
               bankroll_after=95.0),
    ]
    settlements = [_settlement("q3", 0, 0.45, payouts={"adv0": 0.0})]
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "q3", 0.4, (0.4 - 0) ** 2]])

    row = manipulation.adversary_metrics(cond, ref).iloc[0]
    assert row["target_side"] == "YES"
    assert row["adv_displacement"] < 0
    assert math.isnan(row["capital_per_01_move"])


def test_adversary_metrics_zero_displacement_nan():
    # adv moves up then back down by the same amount -> net displacement 0 -> NaN.
    trades = [
        _trade("q4", "adv0", 1, price_before=0.5, price_after=0.6, cost=6.0,
               bankroll_after=94.0),
        _trade("q4", "adv0", 2, price_before=0.6, price_after=0.5, cost=0.0,
               bankroll_after=94.0),
    ]
    settlements = [_settlement("q4", 0, 0.5, payouts={"adv0": 0.0})]
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "q4", 0.4, (0.4 - 0) ** 2]])

    row = manipulation.adversary_metrics(cond, ref).iloc[0]
    assert row["adv_displacement"] == pytest.approx(0.0)
    assert math.isnan(row["capital_per_01_move"])
    assert row["wrong_side"] == False  # exactly 0.5 is neither side


# ---------------------------------------------------------------------------
# max(cost, 0) spend clamp
# ---------------------------------------------------------------------------


def test_adversary_spend_clamps_negative_cost():
    trades = [
        _trade("q5", "adv0", 1, price_before=0.5, price_after=0.6, cost=10.0,
               bankroll_after=90.0),
        _trade("q5", "adv0", 2, price_before=0.6, price_after=0.65, cost=-4.0,
               bankroll_after=94.0),
    ]
    settlements = [_settlement("q5", 0, 0.65, payouts={"adv0": 0.0})]
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "q5", 0.4, (0.4 - 0) ** 2]])

    row = manipulation.adversary_metrics(cond, ref).iloc[0]
    assert row["adv_spend"] == pytest.approx(10.0)  # negative cost contributes 0


# ---------------------------------------------------------------------------
# Flip logic against a fabricated honest_ref
# ---------------------------------------------------------------------------


def test_flip_false_when_honest_already_wrong():
    # Market ends wrong, but honest reference was ALSO wrong -> not a flip.
    trades = [
        _trade("q6", "adv0", 1, price_before=0.5, price_after=0.4, cost=5.0,
               bankroll_after=95.0, action="buy_no"),
    ]
    settlements = [_settlement("q6", 1, 0.4, payouts={"adv0": 0.0})]
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "q6", 0.45, (0.45 - 1) ** 2]])  # honest already wrong (<0.5)

    row = manipulation.adversary_metrics(cond, ref).iloc[0]
    assert row["wrong_side"] == True
    assert row["flipped"] == False


def test_flip_false_when_run_not_wrong_side():
    # Honest correct, run also correct -> no flip.
    trades = [
        _trade("q7", "adv0", 1, price_before=0.5, price_after=0.55, cost=3.0,
               bankroll_after=97.0),
    ]
    settlements = [_settlement("q7", 1, 0.6, payouts={"adv0": 0.0})]  # correct side
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "q7", 0.7, (0.7 - 1) ** 2]])

    row = manipulation.adversary_metrics(cond, ref).iloc[0]
    assert row["wrong_side"] == False
    assert row["flipped"] == False


def test_flip_false_when_honest_ref_exactly_half():
    # Honest ref exactly 0.5 is on NEITHER side -> cannot be "correct" -> no flip.
    trades = [
        _trade("q8", "adv0", 1, price_before=0.5, price_after=0.4, cost=5.0,
               bankroll_after=95.0, action="buy_no"),
    ]
    settlements = [_settlement("q8", 1, 0.4, payouts={"adv0": 0.0})]
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "q8", 0.5, (0.5 - 1) ** 2]])

    row = manipulation.adversary_metrics(cond, ref).iloc[0]
    assert row["wrong_side"] == True
    assert row["flipped"] == False


def test_adversary_metrics_missing_ref_raises():
    trades = [_trade("q9", "adv0", 1, price_before=0.5, price_after=0.4, cost=5.0,
                     bankroll_after=95.0)]
    settlements = [_settlement("q9", 1, 0.4, payouts={"adv0": 0.0})]
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "other", 0.7, 0.09]])
    with pytest.raises(ValueError, match="honest_ref has no row"):
        manipulation.adversary_metrics(cond, ref)


# ---------------------------------------------------------------------------
# Recovery: honest-only trades after adversary, adversary not last
# ---------------------------------------------------------------------------


def test_recovery_ignores_adversary_trades_in_window():
    # A later adversary trade sits AFTER an honest one; recovery is measured from
    # the LAST adversary trade, so only trades after it count.
    trades = [
        _trade("qa", "adv0", 1, price_before=0.5, price_after=0.4, cost=5.0,
               bankroll_after=95.0),
        _trade("qa", "p0", 2, price_before=0.4, price_after=0.49),  # within eps but
        # this is BEFORE the last adversary trade -> must not count as recovery
        _trade("qa", "adv0", 2, price_before=0.49, price_after=0.4, cost=5.0,
               bankroll_after=90.0),
        _trade("qa", "p1", 3, price_before=0.4, price_after=0.44),  # 0.06 off -> no
    ]
    settlements = [_settlement("qa", 1, 0.44, payouts={"adv0": 0.0})]
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "qa", 0.7, 0.09]])
    row = manipulation.adversary_metrics(cond, ref).iloc[0]
    assert row["recovered"] == False  # 0.44 is 0.06 from pre-attack 0.5 (> eps)


def test_recovery_eps_boundary_inclusive():
    # Exactly eps away counts as recovered.
    trades = [
        _trade("qb", "adv0", 1, price_before=0.5, price_after=0.3, cost=5.0,
               bankroll_after=95.0),
        _trade("qb", "p0", 2, price_before=0.3, price_after=0.45),  # exactly 0.05 off
    ]
    settlements = [_settlement("qb", 1, 0.45, payouts={"adv0": 0.0})]
    cond = _cond("D_K1", {0: (trades, settlements)})
    ref = _ref([[0, "qb", 0.7, 0.09]])
    row = manipulation.adversary_metrics(cond, ref).iloc[0]
    assert row["recovered"] == True
    assert row["recovery_rounds"] == pytest.approx(1.0)  # round 2 - round 1


# ---------------------------------------------------------------------------
# summarize_adversary
# ---------------------------------------------------------------------------


def _adv_df(rows):
    """Build a minimal adversary_metrics-shaped DataFrame for summarize tests."""
    cols = [
        "flipped", "wrong_side", "capital_per_01_move", "adv_pnl",
        "recovered", "recovery_rounds", "market_brier", "ref_market_brier",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_summarize_adversary_exact():
    df1 = _adv_df([
        [True, True, 10.0, -5.0, True, 1.0, 0.5, 0.2],
        [False, True, float("nan"), -3.0, False, float("nan"), 0.3, 0.2],
    ])
    df3 = _adv_df([
        [True, True, 20.0, -8.0, True, 2.0, 0.6, 0.2],
        [True, False, 30.0, -2.0, True, 4.0, 0.1, 0.2],
    ])
    out = manipulation.summarize_adversary({3: df3, 1: df1})
    # sorted by k
    assert list(out["k"]) == [1, 3]
    r1 = out[out["k"] == 1].iloc[0]
    assert r1["flip_rate"] == pytest.approx(0.5)
    assert r1["wrong_side_rate"] == pytest.approx(1.0)
    assert r1["mean_capital_per_01_move"] == pytest.approx(10.0)  # NaN dropped
    assert r1["capital_nan_count"] == 1
    assert r1["mean_adv_pnl"] == pytest.approx(-4.0)
    assert r1["recovery_fraction"] == pytest.approx(0.5)
    assert r1["median_recovery_rounds"] == pytest.approx(1.0)
    assert r1["brier_degradation"] == pytest.approx((0.5 + 0.3) / 2 - 0.2)

    r3 = out[out["k"] == 3].iloc[0]
    assert r3["flip_rate"] == pytest.approx(1.0)
    assert r3["wrong_side_rate"] == pytest.approx(0.5)
    assert r3["mean_capital_per_01_move"] == pytest.approx(25.0)
    assert r3["capital_nan_count"] == 0
    assert r3["median_recovery_rounds"] == pytest.approx(3.0)  # median of 2,4
    assert r3["brier_degradation"] == pytest.approx((0.6 + 0.1) / 2 - 0.2)


# ---------------------------------------------------------------------------
# wealth_order_effects
# ---------------------------------------------------------------------------


def _wealth_df(gap_map, brier_map):
    rows = []
    for (seed, qid), gap in gap_map.items():
        rows.append(
            {
                "seed": seed,
                "question_id": qid,
                "market_gap": gap,
                "market_brier": brier_map[(seed, qid)],
            }
        )
    return pd.DataFrame(rows, columns=["seed", "question_id", "market_gap", "market_brier"])


def test_wealth_order_effects_mean_diffs():
    keys = [(0, "q0"), (0, "q1"), (1, "q0"), (1, "q1")]
    a = _wealth_df(
        {k: g for k, g in zip(keys, [0.10, 0.20, 0.15, 0.25])},
        {k: b for k, b in zip(keys, [0.05, 0.10, 0.08, 0.12])},
    )
    wf = _wealth_df(
        {k: g for k, g in zip(keys, [0.12, 0.18, 0.20, 0.30])},
        {k: b for k, b in zip(keys, [0.06, 0.09, 0.10, 0.15])},
    )
    ws = _wealth_df(
        {k: g for k, g in zip(keys, [0.14, 0.22, 0.16, 0.28])},
        {k: b for k, b in zip(keys, [0.07, 0.11, 0.09, 0.14])},
    )
    out = manipulation.wealth_order_effects(a, wf, ws, n_boot=200, seed=0)

    assert set(out) == {"wfixed_vs_a", "wshuffled_vs_a", "wfixed_vs_wshuffled"}
    # wfixed - a on market_gap: (0.02 - 0.02 + 0.05 + 0.05)/4 = 0.025
    assert out["wfixed_vs_a"]["market_gap"].mean_diff == pytest.approx(0.025)
    assert out["wfixed_vs_a"]["market_gap"].n == 4
    # wshuffled - a on market_gap: (0.04 + 0.02 + 0.01 + 0.03)/4 = 0.025
    assert out["wshuffled_vs_a"]["market_gap"].mean_diff == pytest.approx(0.025)
    # wfixed - wshuffled on market_brier:
    # (-0.01 - 0.02 + 0.01 + 0.01)/4 = -0.0025
    assert out["wfixed_vs_wshuffled"]["market_brier"].mean_diff == pytest.approx(-0.0025)


def test_wealth_order_effects_no_shared_rows_raises():
    a = _wealth_df({(0, "q0"): 0.1}, {(0, "q0"): 0.05})
    other = _wealth_df({(0, "zzz"): 0.1}, {(0, "zzz"): 0.05})
    with pytest.raises(ValueError, match="no shared"):
        manipulation.wealth_order_effects(a, other, other)
