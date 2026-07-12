"""LMSR (Logarithmic Market Scoring Rule) market engine — pure math, no I/O.

Definitions (Hanson's LMSR for a binary YES/NO market):

    C(q)     = b * log(exp(q_yes/b) + exp(q_no/b))       cost / potential function
    p_yes(q) = exp(q_yes/b) / (exp(q_yes/b) + exp(q_no/b))
             = 1 / (1 + exp((q_no - q_yes)/b))           stable sigmoid form
    trade cost for moving q -> q' is C(q') - C(q).

This module deliberately does NOT import pnyx.schemas — it operates on plain
floats so it can be tested and reused independently of the event-log models.

Why b = 40 (project default, with bankroll = 100 per agent)
-----------------------------------------------------------
The liquidity parameter b sets how much cash it takes to move the price.
Pushing the YES price from 0.5 to 0.8 costs b*ln(2.5) ~= 0.92*b ~= 37, i.e.
roughly one third of a 100-unit bankroll — a single confident agent can move
the market meaningfully but cannot pin it to an extreme on its own. The
market maker's worst-case subsidy per question is b*ln(2) ~= 28.
"""

import math

Side = str  # "yes" | "no"


def _validate_side(side: str) -> None:
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")


def cost(q_yes: float, q_no: float, b: float) -> float:
    """LMSR cost function C(q) = b*log(exp(q_yes/b) + exp(q_no/b)).

    Numerically stable via the max-shift (log-sum-exp) trick:
        C(q) = b*(m + log(exp(q_yes/b - m) + exp(q_no/b - m))),
        m = max(q_yes, q_no)/b,
    so the largest exponent is exactly 0 and nothing overflows.
    """
    m = max(q_yes, q_no) / b
    return b * (m + math.log(math.exp(q_yes / b - m) + math.exp(q_no / b - m)))


def price_yes(q_yes: float, q_no: float, b: float) -> float:
    """Instantaneous YES price, stable sigmoid form 1/(1+exp((q_no-q_yes)/b)).

    Evaluated on whichever branch keeps the exponent non-positive, so it never
    overflows and the result is always in [0, 1] — even at extreme share
    vectors (TradeEvent validates prices with ge=0, le=1):

        x = (q_no - q_yes)/b
        x <  0:  1/(1 + exp(x))                       (exp(x) <= 1)
        x >= 0:  t/(1 + t) with t = exp(-x)           (algebraically identical)
    """
    x = (q_no - q_yes) / b
    if x < 0.0:
        return 1.0 / (1.0 + math.exp(x))
    t = math.exp(-x)
    return t / (1.0 + t)


def trade_cost(q_yes: float, q_no: float, delta_yes: float, delta_no: float, b: float) -> float:
    """Cost of moving (q_yes, q_no) -> (q_yes+delta_yes, q_no+delta_no):
    C(q') - C(q). Positive for (net) buys, negative for sells."""
    return cost(q_yes + delta_yes, q_no + delta_no, b) - cost(q_yes, q_no, b)


def max_affordable_shares(q_yes: float, q_no: float, side: Side, cash: float, b: float) -> float:
    """Largest delta of `side` shares purchasable with `cash`, closed form.

    Derivation (buying delta YES; the NO side is the mirror image with the
    roles of q_yes and q_no swapped). Solve trade_cost = cash:

        b*log(e^{(q_yes+delta)/b} + e^{q_no/b}) - C(q) = cash
        e^{(q_yes+delta)/b} = e^{(cash + C(q))/b} - e^{q_no/b}
                            = e^{cash/b} * (e^{q_yes/b} + e^{q_no/b}) - e^{q_no/b}
        delta = b*log( e^{cash/b} * (e^{q_yes/b} + e^{q_no/b}) - e^{q_no/b} ) - q_yes

    In LMSR the marginal YES price approaches (but never reaches) 1, so the
    cost of delta shares grows without bound (~ delta for large delta): there
    is no cash asymptote and delta* is finite for every finite cash >= 0.

    Numerical stability: factor out the larger of e^{q_yes/b}, e^{q_no/b} so
    every remaining exponent is <= 0, then combine in log space:

      if q_yes >= q_no (factor e^{q_yes/b}), with t = e^{(q_no - q_yes)/b} <= 1:
          delta = b * log( e^{cash/b} * (1 + t) - t )
                = b * ( a + log1p(-e^{(q_no - q_yes)/b - a}) ),
            a = cash/b + log1p(t)
      else (factor e^{q_no/b}), with s = e^{(q_yes - q_no)/b} < 1:
          delta = (q_no - q_yes) + b * ( a + log1p(-e^{-a}) ),
            a = cash/b + log1p(s)

    Exactness contract: trade_cost(q, delta*, b) == cash to floating-point
    accuracy, and always <= cash + 1e-9 (the runner's clamp relies on this).
    Returns 0.0 for cash <= 0.
    """
    _validate_side(side)
    if cash <= 0.0:
        return 0.0
    if side == "no":
        # Buying NO on (q_yes, q_no) is buying YES on the mirrored market.
        q_yes, q_no = q_no, q_yes

    if q_yes >= q_no:
        log_t = (q_no - q_yes) / b  # <= 0
        a = cash / b + math.log1p(math.exp(log_t))
        return b * (a + math.log1p(-math.exp(log_t - a)))
    log_s = (q_yes - q_no) / b  # < 0
    a = cash / b + math.log1p(math.exp(log_s))
    return (q_no - q_yes) + b * (a + math.log1p(-math.exp(-a)))


class Market:
    """Mutable LMSR market state for one binary question.

    Tracks the share vector (q_yes, q_no) and remembers the opening state
    (q0_yes, q0_no) so `subsidy()` can report the maker's exposure.
    """

    def __init__(self, q_yes: float = 0.0, q_no: float = 0.0, b: float = 40.0) -> None:
        if b <= 0.0:
            raise ValueError(f"b must be positive, got {b}")
        self.q_yes = q_yes
        self.q_no = q_no
        self.b = b
        self._q0_yes = q_yes
        self._q0_no = q_no

    def price(self) -> float:
        """Current YES price in [0, 1]."""
        return price_yes(self.q_yes, self.q_no, self.b)

    def cost_to_buy(self, side: Side, shares: float) -> float:
        """Quote the cost of buying `shares` of `side` WITHOUT mutating."""
        _validate_side(side)
        dy, dn = (shares, 0.0) if side == "yes" else (0.0, shares)
        return trade_cost(self.q_yes, self.q_no, dy, dn, self.b)

    def buy(self, side: Side, shares: float) -> float:
        """Buy `shares` of `side`; mutates state and returns the cost paid."""
        _validate_side(side)
        if shares < 0.0:
            raise ValueError(f"shares must be >= 0, got {shares}")
        paid = self.cost_to_buy(side, shares)
        if side == "yes":
            self.q_yes += shares
        else:
            self.q_no += shares
        return paid

    def clamp_shares(self, side: Side, shares: float, cash: float) -> float:
        """Server-side affordability clamp: min(requested, max affordable).

        Never trust the agent's requested share count — the runner passes the
        agent's ask through here before executing (Global Constraints)."""
        affordable = max_affordable_shares(self.q_yes, self.q_no, side, cash, self.b)
        return min(max(shares, 0.0), affordable)

    def subsidy(self) -> float:
        """Worst-case-realized maker loss at the CURRENT state.

        Definition used here: if the market settled right now, the maker pays
        $1 per outstanding share on the true side; shares issued since the
        opening state q0 are (q_yes - q0_yes) YES and (q_no - q0_no) NO. The
        maker's revenue for issuing them was C(q) - C(q0) (the sum of all
        trade costs collected, since costs telescope). Worst case over the
        two outcomes (floored at 0 — settling with no shares out costs the
        maker nothing):

            subsidy = max(q_yes - q0_yes, q_no - q0_no, 0) - (C(q) - C(q0))

        For q0 = (0, 0) this is bounded above by b*ln(2), the classic LMSR
        subsidy bound, approached as one side's price is pushed to 1.
        """
        revenue = cost(self.q_yes, self.q_no, self.b) - cost(self._q0_yes, self._q0_no, self.b)
        worst_payout = max(self.q_yes - self._q0_yes, self.q_no - self._q0_no, 0.0)
        return worst_payout - revenue

    def snapshot(self) -> tuple[float, float]:
        """Return the current share vector (q_yes, q_no) for replay
        reconstruction / speculative evaluation."""
        return (self.q_yes, self.q_no)

    def restore(self, snap: tuple[float, float]) -> None:
        """Restore a share vector previously returned by snapshot().
        Does not touch the opening state q0 (subsidy stays anchored)."""
        self.q_yes, self.q_no = snap
