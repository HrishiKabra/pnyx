"""Pnyx P5 manipulation analysis — H3 (adversary) metrics and H-wealth
(persistent-wealth order effects).

Why a separate module from ``pnyx.analysis.mainrun``: ``mainrun`` scores the
*honest* market and the four static pools against the Bayes oracle. This module
answers the two adversarial/ablation hypotheses of the pre-registration (§2)
that operate on a condition's raw trade stream rather than its pooled beliefs:

* **H3 (manipulation).** Conditions ``D_K{1,3,10}`` add a single adversary
  ``adv0`` (empty shard, k× bankroll) to the mixed pool ``C`` and instruct it to
  push the price toward the ground-truth-WRONG side. Per settled question we
  measure how far the final price ends up on the wrong side (``wrong_side``),
  whether the adversary *flipped* an otherwise-correct market (``flipped`` vs the
  honest reference ``C``), how much capital that displacement cost
  (``adv_spend`` / ``adv_displacement`` / ``capital_per_01_move``), the
  adversary's settlement P&L (``adv_pnl`` — the pre-registered "the market taxes
  manipulation" prediction is ``adv_pnl < 0``), and how the honest pool recovers
  after the attack (``recovered`` / ``recovery_rounds``). :func:`summarize_adversary`
  reduces the three k-arms to the reporting table, including terminal Brier
  degradation vs ``C``.

* **H-wealth (ablation).** :func:`wealth_order_effects` pairs the
  persistent-wealth conditions ``W_FIXED`` / ``W_SHUFFLED`` against the
  per-question-reset baseline ``A`` on ``market_gap`` and ``market_brier``,
  reusing ``mainrun.compare``'s question-clustered bootstrap + Wilcoxon.

Adversary identity and target side
----------------------------------
The adversary is the single agent with an empty shard assignment; its id is
``adv0`` in every ``D`` config, hard-coded here as :data:`ADVERSARY_ID`. The
target (wrong) side per question is ``"YES"`` when the question resolves NO and
``"NO"`` when it resolves YES. ``runner.adversary_target_side`` defines this on
the latent state; analysis never imports the runner, so we reimplement it here on
the settled ``outcome`` (which equals the latent state) — see
:func:`target_side`.

Contracts that bind this module
-------------------------------
* **Pure metrics, no I/O.** Every function here consumes ``mainrun``'s loaded
  :class:`~pnyx.analysis.mainrun.ConditionData` / metrics ``DataFrame``s; the
  filesystem is touched only by ``mainrun.load_condition``.
* **Event-log order is execution order.** Recovery and the "post-attack" window
  are computed over the trade events in their logged order within a question, as
  the runner wrote them (trades within a round are randomized, but the log order
  is the order they executed).
* **Determinism.** No RNG in this module (the only randomness downstream is
  ``compare``'s bootstrap, which takes an explicit seed).

stdlib + numpy + pandas + pydantic only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pnyx.analysis.mainrun import (
    HONEST_AGENTS,
    ComparisonResult,
    ConditionData,
    compare,
)
from pnyx.schemas import SettlementEvent, TradeEvent

__all__ = [
    "ADVERSARY_ID",
    "RECOVERY_EPS",
    "target_side",
    "adversary_metrics",
    "summarize_adversary",
    "wealth_order_effects",
]

# The condition-D adversary is the ONLY agent configured with an empty shard
# (``AgentSpec``: an adversary "holds no private shard"). Its id is ``adv0`` in
# every D config; hard-coded per the Task-3 brief.
ADVERSARY_ID = "adv0"

# Honest agents whose post-attack trades count toward market recovery (p0..p5).
_HONEST_SET = frozenset(HONEST_AGENTS)

# Recovery tolerance: an honest post-attack trade "recovers" the market when its
# resulting price is within this of the pre-attack price (brief: ε = 0.05).
RECOVERY_EPS = 0.05

# Displacement below this magnitude is treated as no net move (capital_per_01_move
# is undefined -> NaN), guarding a divide-by-~zero.
_DISPLACEMENT_EPS = 1e-9

_ADV_METRIC_COLUMNS = [
    "seed",
    "question_id",
    "target_side",
    "final_price",
    "outcome",
    "wrong_side",
    "ref_market_prob",
    "flipped",
    "adv_spend",
    "adv_displacement",
    "capital_per_01_move",
    "adv_pnl",
    "pre_attack_price",
    "recovered",
    "recovery_rounds",
    "market_brier",
    "ref_market_brier",
]


def target_side(outcome: int) -> str:
    """The ground-truth-WRONG side the adversary is tasked to push toward:
    ``"YES"`` when the question resolves NO (``outcome == 0``), ``"NO"`` when it
    resolves YES. Mirror of ``runner.adversary_target_side`` reimplemented on the
    settled ``outcome`` so analysis never imports the runner."""
    return "YES" if outcome == 0 else "NO"


def _on_target_side(price: float, side: str) -> bool:
    """True when ``price`` is STRICTLY on ``side`` of 0.5 (price > 0.5 for YES,
    price < 0.5 for NO). Exactly 0.5 is on neither side."""
    if side == "YES":
        return price > 0.5
    return price < 0.5


def _toward_target(price_before: float, price_after: float, side: str) -> float:
    """Signed price move toward the target ``side`` (positive = toward target):
    ``price_after - price_before`` when the target is YES, its negation when NO."""
    delta = price_after - price_before
    return delta if side == "YES" else -delta


def _question_trades(seed_trades: list[TradeEvent], qid: str) -> list[TradeEvent]:
    """Trade events of one question in logged (execution) order."""
    return [t for t in seed_trades if t.key.question_id == qid]


def _adversary_stats(qtrades: list[TradeEvent], side: str) -> dict:
    """Adversary spend/displacement/pnl-inputs and the recovery outcome for one
    question's trade stream (already in execution order).

    Returns a dict with ``adv_spend``, ``adv_displacement``,
    ``capital_per_01_move``, ``adv_final_bankroll`` (None if adv never traded),
    ``pre_attack_price`` (NaN if adv never traded), ``recovered``,
    ``recovery_rounds``."""
    adv_indices = [i for i, t in enumerate(qtrades) if t.key.agent_id == ADVERSARY_ID]

    if not adv_indices:
        return {
            "adv_spend": 0.0,
            "adv_displacement": 0.0,
            "capital_per_01_move": float("nan"),
            "adv_start_bankroll": None,
            "adv_final_bankroll": None,
            "pre_attack_price": float("nan"),
            "recovered": False,
            "recovery_rounds": float("nan"),
        }

    adv_trades = [qtrades[i] for i in adv_indices]
    adv_spend = float(sum(max(t.cost, 0.0) for t in adv_trades))
    displacement = float(
        sum(_toward_target(t.price_before, t.price_after, side) for t in adv_trades)
    )
    if displacement > _DISPLACEMENT_EPS:
        capital = 0.1 * adv_spend / displacement
    else:
        capital = float("nan")

    pre_attack_price = adv_trades[0].price_before
    last_adv_index = adv_indices[-1]
    last_adv_round = adv_trades[-1].key.round

    recovered = False
    recovery_rounds: float = float("nan")
    for t in qtrades[last_adv_index + 1:]:
        if t.key.agent_id not in _HONEST_SET:
            continue
        if abs(t.price_after - pre_attack_price) <= RECOVERY_EPS:
            recovered = True
            recovery_rounds = float(t.key.round - last_adv_round)
            break

    return {
        "adv_spend": adv_spend,
        "adv_displacement": displacement,
        "capital_per_01_move": capital,
        # Starting bankroll (k×100) read FROM THE DATA — the adversary's bankroll
        # just before its first trade executed = bankroll_after + that trade's
        # cost. Never hardcode k or read config YAMLs.
        "adv_start_bankroll": adv_trades[0].bankroll_after + adv_trades[0].cost,
        "adv_final_bankroll": adv_trades[-1].bankroll_after,
        "pre_attack_price": pre_attack_price,
        "recovered": recovered,
        "recovery_rounds": recovery_rounds,
    }


def adversary_metrics(
    cond: ConditionData, honest_ref: pd.DataFrame
) -> pd.DataFrame:
    """Per-``(seed, question_id)`` H3 manipulation metrics for one ``D_K*``
    condition, one row per settled question.

    ``cond`` is the loaded ``D_K*`` :class:`~pnyx.analysis.mainrun.ConditionData`;
    ``honest_ref`` is condition ``C``'s ``mainrun.question_metrics`` DataFrame
    (the no-adversary reference), matched on ``(seed, question_id)`` for the
    ``flipped`` comparison and the terminal-Brier baseline.

    Columns (see the module docstring for the definitions):

    * ``target_side`` — wrong side the adversary pushed toward (from ``outcome``);
    * ``final_price`` / ``outcome`` — from ``cond``'s settlement;
    * ``wrong_side`` — final price STRICTLY on the target side of 0.5;
    * ``ref_market_prob`` — ``C``'s final price for this question;
    * ``flipped`` — ``C`` finished on the correct side AND this run is
      ``wrong_side`` (an adversary-induced flip);
    * ``adv_spend`` = Σ max(cost, 0) over ``adv0``'s trades;
    * ``adv_displacement`` = Σ signed price move toward the target over
      ``adv0``'s trades (positive = toward target);
    * ``capital_per_01_move`` = ``0.1 * adv_spend / adv_displacement`` when
      displacement > 1e-9, else ``NaN``;
    * ``adv_pnl`` = ``adv0`` settlement payout + final ``adv0`` bankroll −
      STARTING bankroll (the k×100 stake read from the data: the first adv0
      trade's ``bankroll_after + cost``; final = ``bankroll_after`` of the last
      trade). If ``adv0`` never traded, P&L is just the payout (0 shares → 0);
    * ``recovered`` / ``recovery_rounds`` — an honest trade after the last
      adversary trade whose price returns within ``RECOVERY_EPS`` of the
      pre-attack price, and the round gap to it (``NaN`` if not recovered / adv
      never traded);
    * ``market_brier`` / ``ref_market_brier`` — this run's and ``C``'s terminal
      Brier, carried for :func:`summarize_adversary`.
    """
    ref = honest_ref.set_index(["seed", "question_id"])

    rows: list[dict] = []
    for seed in sorted(cond.seeds):
        sd = cond.seeds[seed]
        for s in sorted(sd.settlements, key=lambda ev: ev.question_id):
            qid = s.question_id
            side = target_side(s.outcome)
            final_price = s.final_price
            wrong = _on_target_side(final_price, side)

            try:
                ref_row = ref.loc[(seed, qid)]
            except KeyError as exc:
                raise ValueError(
                    f"honest_ref has no row for (seed={seed}, question_id={qid!r}) "
                    f"needed by condition {cond.condition!r}"
                ) from exc
            ref_price = float(ref_row["market_prob"])
            ref_brier = float(ref_row["market_brier"])
            # A flip: the honest reference finished on the CORRECT (opposite of
            # target) side and the adversary run finished on the wrong side.
            ref_correct = not _on_target_side(ref_price, side) and ref_price != 0.5
            flipped = bool(ref_correct and wrong)

            stats = _adversary_stats(_question_trades(sd.trades, qid), side)

            payout = _adv_payout(s)
            final_bankroll = stats["adv_final_bankroll"]
            start_bankroll = stats["adv_start_bankroll"]
            if final_bankroll is None:
                # Adversary never traded: no shares held, bankroll untouched, so
                # P&L is exactly the settlement payout (0 for 0 shares).
                adv_pnl = payout
            else:
                adv_pnl = payout + final_bankroll - start_bankroll

            rows.append(
                {
                    "seed": seed,
                    "question_id": qid,
                    "target_side": side,
                    "final_price": final_price,
                    "outcome": s.outcome,
                    "wrong_side": wrong,
                    "ref_market_prob": ref_price,
                    "flipped": flipped,
                    "adv_spend": stats["adv_spend"],
                    "adv_displacement": stats["adv_displacement"],
                    "capital_per_01_move": stats["capital_per_01_move"],
                    "adv_pnl": adv_pnl,
                    "pre_attack_price": stats["pre_attack_price"],
                    "recovered": stats["recovered"],
                    "recovery_rounds": stats["recovery_rounds"],
                    "market_brier": (final_price - s.outcome) ** 2,
                    "ref_market_brier": ref_brier,
                }
            )

    return pd.DataFrame(rows, columns=_ADV_METRIC_COLUMNS)


def _adv_payout(s: SettlementEvent) -> float:
    """The settlement payout to ``adv0`` (0.0 when the adversary is absent from
    the payout map)."""
    return float(s.payouts.get(ADVERSARY_ID, 0.0))


def summarize_adversary(dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Reduce the per-question :func:`adversary_metrics` tables of the three
    adversary arms to the H3 reporting table, one row per ``k`` (sorted).

    ``dfs`` maps the bankroll multiplier ``k`` (1, 3, 10) to that arm's
    ``adversary_metrics`` DataFrame. Columns:

    * ``k``;
    * ``flip_rate`` = mean ``flipped``;
    * ``wrong_side_rate`` = mean ``wrong_side``;
    * ``mean_capital_per_01_move`` (over the non-NaN rows) and
      ``capital_nan_count`` (rows with undefined displacement);
    * ``mean_adv_pnl``;
    * ``recovery_fraction`` = mean ``recovered``;
    * ``median_recovery_rounds`` (over recovered rows only);
    * ``brier_degradation`` = mean(this run ``market_brier``) −
      mean(``C`` ``ref_market_brier``): terminal Brier degradation vs ``C``.
    """
    rows: list[dict] = []
    for k in sorted(dfs):
        df = dfs[k]
        capital = df["capital_per_01_move"].to_numpy(dtype=float)
        capital_valid = capital[~np.isnan(capital)]
        rec_rounds = df["recovery_rounds"].to_numpy(dtype=float)
        rec_rounds = rec_rounds[~np.isnan(rec_rounds)]
        rows.append(
            {
                "k": k,
                "flip_rate": float(df["flipped"].mean()),
                "wrong_side_rate": float(df["wrong_side"].mean()),
                "mean_capital_per_01_move": (
                    float(capital_valid.mean()) if capital_valid.size else float("nan")
                ),
                "capital_nan_count": int(np.isnan(capital).sum()),
                "mean_adv_pnl": float(df["adv_pnl"].mean()),
                "recovery_fraction": float(df["recovered"].mean()),
                "median_recovery_rounds": (
                    float(np.median(rec_rounds)) if rec_rounds.size else float("nan")
                ),
                "brier_degradation": float(
                    df["market_brier"].mean() - df["ref_market_brier"].mean()
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "k",
            "flip_rate",
            "wrong_side_rate",
            "mean_capital_per_01_move",
            "capital_nan_count",
            "mean_adv_pnl",
            "recovery_fraction",
            "median_recovery_rounds",
            "brier_degradation",
        ],
    )


# ---------------------------------------------------------------------------
# H-wealth: persistent-wealth order effects
# ---------------------------------------------------------------------------

_WEALTH_METRICS = ("market_gap", "market_brier")


def _paired_compare(
    df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str, **kw
) -> ComparisonResult:
    """Pair two conditions' ``question_metrics`` on ``(seed, question_id)`` and
    compare ``metric`` (difference = ``df_a`` − ``df_b``) via ``mainrun.compare``."""
    merged = df_a.merge(
        df_b,
        on=["seed", "question_id"],
        suffixes=("_a", "_b"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(
            "no shared (seed, question_id) rows between the two conditions"
        )
    return compare(merged, f"{metric}_a", f"{metric}_b", **kw)


def wealth_order_effects(
    a_df: pd.DataFrame,
    w_fixed_df: pd.DataFrame,
    w_shuffled_df: pd.DataFrame,
    **kw,
) -> dict[str, dict[str, ComparisonResult]]:
    """H-wealth paired comparisons: does persistent wealth (and its
    order-dependence) change market accuracy vs the per-question-reset baseline?

    ``a_df`` / ``w_fixed_df`` / ``w_shuffled_df`` are the
    ``mainrun.question_metrics`` DataFrames for conditions ``A``, ``W_FIXED``,
    ``W_SHUFFLED`` (the W arms reuse ``A``'s Pass-1, so their static-pool columns
    match ``A``; only the market columns differ). Returns a nested dict::

        {
          "wfixed_vs_a":        {"market_gap": ..., "market_brier": ...},
          "wshuffled_vs_a":     {"market_gap": ..., "market_brier": ...},
          "wfixed_vs_wshuffled":{"market_gap": ..., "market_brier": ...},
        }

    each value a :class:`~pnyx.analysis.mainrun.ComparisonResult` whose
    ``mean_diff`` is (first − second): positive ``wfixed_vs_a`` ``market_gap``
    means persistent-wealth had the LARGER posterior gap (worse). ``**kw`` is
    forwarded to ``mainrun.compare`` (e.g. ``n_boot``, ``seed``)."""
    pairs = {
        "wfixed_vs_a": (w_fixed_df, a_df),
        "wshuffled_vs_a": (w_shuffled_df, a_df),
        "wfixed_vs_wshuffled": (w_fixed_df, w_shuffled_df),
    }
    out: dict[str, dict[str, ComparisonResult]] = {}
    for name, (first, second) in pairs.items():
        out[name] = {
            metric: _paired_compare(first, second, metric, **kw)
            for metric in _WEALTH_METRICS
        }
    return out
