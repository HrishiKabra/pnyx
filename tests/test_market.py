"""Exact-value tests for the LMSR market engine (pnyx/market.py).

Every non-trivial expected value is derived algebraically in a comment above
the assertion. The LMSR definitions used throughout:

    C(q)       = b * log(exp(q_yes/b) + exp(q_no/b))          (cost / potential)
    p_yes(q)   = 1 / (1 + exp((q_no - q_yes)/b))              (stable sigmoid)
    trade_cost = C(q') - C(q)                                 (cost of a trade)
"""

import math

import pytest

from pnyx.market import (
    Market,
    cost,
    max_affordable_shares,
    price_yes,
    trade_cost,
)

B = 40.0  # project default b (see market.py rationale)


# ---------------------------------------------------------------------------
# Origin: q = (0, 0)
# ---------------------------------------------------------------------------


def test_price_at_origin_is_exactly_half():
    # p_yes = 1/(1+exp(0)) = 1/(1+1) = 1/2 exactly.
    assert price_yes(0.0, 0.0, B) == 0.5


def test_cost_at_origin_is_b_ln2():
    # C(0,0) = b*log(exp(0)+exp(0)) = b*log(2).
    assert cost(0.0, 0.0, B) == pytest.approx(B * math.log(2.0), abs=1e-12)


def test_market_price_at_origin():
    assert Market(0.0, 0.0, B).price() == 0.5


# ---------------------------------------------------------------------------
# Canonical buy: delta = b*ln2 YES from (0,0)
# ---------------------------------------------------------------------------


def test_buy_b_ln2_yes_gives_price_two_thirds():
    # New state q = (b*ln2, 0).
    #   p_yes = 1/(1 + exp((0 - b*ln2)/b)) = 1/(1 + exp(-ln2))
    #         = 1/(1 + 1/2) = 1/(3/2) = 2/3 exactly.
    delta = B * math.log(2.0)
    m = Market(0.0, 0.0, B)
    m.buy("yes", delta)
    assert m.price() == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_buy_b_ln2_yes_cost_is_b_ln_three_halves():
    # trade_cost = C(b*ln2, 0) - C(0,0)
    #   C(b*ln2,0) = b*log(exp(ln2) + exp(0)) = b*log(2 + 1) = b*log(3)
    #   C(0,0)     = b*log(2)
    #   => b*log(3) - b*log(2) = b*log(3/2).
    delta = B * math.log(2.0)
    expected = B * math.log(1.5)
    assert trade_cost(0.0, 0.0, delta, 0.0, B) == pytest.approx(expected, abs=1e-12)
    # And via the mutating Market.buy (returns the trade cost).
    m = Market(0.0, 0.0, B)
    got = m.buy("yes", delta)
    assert got == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Price after buys matches the closed-form sigmoid; symmetric round trip
# ---------------------------------------------------------------------------


def test_price_after_buys_matches_closed_form():
    m = Market(0.0, 0.0, B)
    m.buy("yes", 30.0)
    m.buy("no", 12.0)
    # After buys q = (30, 12). price() must equal the pure closed form.
    assert m.price() == price_yes(30.0, 12.0, B)


def test_buy_yes_then_equal_no_returns_to_symmetric_state():
    # From (0,0): buy d YES -> (d,0); buy d NO -> (d,d). p_yes(d,d)=0.5 exactly.
    d = 55.0
    m = Market(0.0, 0.0, B)
    m.buy("yes", d)
    m.buy("no", d)
    assert m.q_yes == pytest.approx(d)
    assert m.q_no == pytest.approx(d)
    assert m.price() == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# Round trip: cost of moving price 0.5 -> 0.8 (folklore b*ln 2.5)
# ---------------------------------------------------------------------------


def test_cost_of_moving_price_half_to_four_fifths():
    # Buying YES with q_no fixed at c: C(q) = c - b*log(1 - p), so
    #   trade_cost(p1 -> p2) = b*log((1 - p1)/(1 - p2)).
    # For p1=0.5, p2=0.8: b*log(0.5/0.2) = b*log(2.5).  (folklore CONFIRMED)
    #
    # Realise it concretely: from (0,0), price 0.8 requires
    #   1/(1+exp(-d/b)) = 0.8  =>  exp(-d/b)=0.25  =>  d = b*log 4.
    d = B * math.log(4.0)
    expected = B * math.log(2.5)
    assert trade_cost(0.0, 0.0, d, 0.0, B) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Numerical stability at extreme share vectors
# ---------------------------------------------------------------------------


def test_stability_large_q_yes_no_overflow():
    # q = (1e6, 0), b = 40  =>  (q_yes-q_no)/b = 25000.
    # p_yes = 1/(1+exp(-25000)); exp(-25000) underflows to 0.0 => price == 1.0.
    p = price_yes(1e6, 0.0, B)
    assert p == 1.0
    # C(1e6,0) = b*(25000 + log(1 + exp(-25000))) = 1e6 + ~0  => finite, ~= q_yes.
    c = cost(1e6, 0.0, B)
    assert math.isfinite(c)
    assert c == pytest.approx(1e6, rel=1e-9)


def test_stability_large_q_no_price_near_zero():
    # Symmetric: q=(0,1e6) => price ~ 0, no overflow.
    assert price_yes(0.0, 1e6, B) == 0.0
    assert math.isfinite(cost(0.0, 1e6, B))


# ---------------------------------------------------------------------------
# max_affordable_shares: exactness property + zero cash
# ---------------------------------------------------------------------------


def test_max_affordable_zero_cash_is_zero():
    assert max_affordable_shares(0.0, 0.0, "yes", 0.0, B) == 0.0
    assert max_affordable_shares(10.0, 3.0, "no", 0.0, B) == 0.0


def test_max_affordable_exactness_property_yes():
    # The returned delta must cost (almost) exactly `cash`.
    for q_yes, q_no, cash in [
        (0.0, 0.0, 10.0),
        (0.0, 0.0, 100.0),
        (25.0, 5.0, 37.0),
        (-10.0, 40.0, 50.0),
    ]:
        d = max_affordable_shares(q_yes, q_no, "yes", cash, B)
        spent = trade_cost(q_yes, q_no, d, 0.0, B)
        assert spent == pytest.approx(cash, abs=1e-6)
        # Affordability guarantee.
        assert spent <= cash + 1e-9


def test_max_affordable_exactness_property_no():
    for q_yes, q_no, cash in [
        (0.0, 0.0, 20.0),
        (60.0, 10.0, 42.0),
        (5.0, -30.0, 15.0),
    ]:
        d = max_affordable_shares(q_yes, q_no, "no", cash, B)
        spent = trade_cost(q_yes, q_no, 0.0, d, B)
        assert spent == pytest.approx(cash, abs=1e-6)
        assert spent <= cash + 1e-9


def test_max_affordable_property_random_sweep():
    # Seeded random property sweep: for arbitrary states and cash amounts,
    # the returned delta must (a) cost `cash` to fp accuracy and (b) never
    # exceed affordability by more than 1e-9 (the runner's clamp contract).
    import random

    rng = random.Random(7)
    for _ in range(500):
        q_yes = rng.uniform(-200.0, 200.0)
        q_no = rng.uniform(-200.0, 200.0)
        cash = rng.uniform(0.001, 150.0)
        side = rng.choice(["yes", "no"])
        d = max_affordable_shares(q_yes, q_no, side, cash, B)
        dy, dn = (d, 0.0) if side == "yes" else (0.0, d)
        spent = trade_cost(q_yes, q_no, dy, dn, B)
        assert spent == pytest.approx(cash, abs=1e-6)
        assert spent <= cash + 1e-9


def test_max_affordable_hand_value():
    # From (0,0), cash = b*log(1.5) should afford exactly delta = b*log 2 YES
    # (the inverse of the canonical buy above).
    cash = B * math.log(1.5)
    d = max_affordable_shares(0.0, 0.0, "yes", cash, B)
    assert d == pytest.approx(B * math.log(2.0), abs=1e-9)


# ---------------------------------------------------------------------------
# Immutability of pure functions; Market.buy mutation correctness
# ---------------------------------------------------------------------------


def test_pure_functions_do_not_mutate_market():
    m = Market(12.0, 7.0, B)
    before = m.snapshot()
    _ = m.cost_to_buy("yes", 20.0)  # inspect-only
    _ = m.price()
    assert m.snapshot() == before  # unchanged
    # Pure functions are referentially transparent.
    assert cost(3.0, 4.0, B) == cost(3.0, 4.0, B)
    assert price_yes(3.0, 4.0, B) == price_yes(3.0, 4.0, B)


def test_buy_mutates_and_returns_cost():
    m = Market(0.0, 0.0, B)
    quoted = m.cost_to_buy("yes", 20.0)
    assert m.snapshot() == (0.0, 0.0)  # quote did not mutate
    paid = m.buy("yes", 20.0)
    assert paid == pytest.approx(quoted)
    assert m.q_yes == pytest.approx(20.0)
    assert m.q_no == 0.0


def test_clamp_shares_takes_minimum():
    m = Market(0.0, 0.0, B)
    cash = 30.0
    afford = max_affordable_shares(0.0, 0.0, "yes", cash, B)
    # Requesting less than affordable -> returns the request.
    assert m.clamp_shares("yes", 5.0, cash) == pytest.approx(5.0)
    # Requesting more than affordable -> clamped to affordable.
    assert m.clamp_shares("yes", afford + 1000.0, cash) == pytest.approx(afford)


def test_snapshot_restore_roundtrip():
    m = Market(0.0, 0.0, B)
    m.buy("yes", 15.0)
    snap = m.snapshot()
    m.buy("no", 22.0)
    assert m.snapshot() != snap
    m.restore(snap)
    assert m.snapshot() == snap
    assert m.q_yes == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Subsidy: worst-case realised maker loss, bounded by b*ln2
# ---------------------------------------------------------------------------


def test_subsidy_bounded_by_b_ln2():
    # From opening (0,0), push YES very far: worst-case maker loss ->  b*ln2.
    #   subsidy = max(q_yes, q_no) - (C(q) - C(0,0))
    # For q=(1600,0): = 1600 - (b*log(exp(40)+1) - b*log2) ~= 1600-1600+b*log2.
    m = Market(0.0, 0.0, B)
    m.buy("yes", 1600.0)
    s = m.subsidy()
    assert s == pytest.approx(B * math.log(2.0), abs=1e-6)
    assert s <= B * math.log(2.0) + 1e-9


def test_subsidy_below_bound_for_moderate_state():
    # Buying b*ln2 YES from (0,0): q=(b*ln2, 0).
    #   revenue = C(b*ln2,0)-C(0,0) = b*log(3)-b*log(2) = b*log(1.5)
    #   payout worst case = max(b*ln2, 0) = b*ln2
    #   subsidy = b*ln2 - b*log(1.5) = b*log(2/1.5) = b*log(4/3).
    m = Market(0.0, 0.0, B)
    m.buy("yes", B * math.log(2.0))
    assert m.subsidy() == pytest.approx(B * math.log(4.0 / 3.0), abs=1e-9)
