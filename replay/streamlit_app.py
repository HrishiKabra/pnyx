"""Pnyx price-path replay — an optional Streamlit demo over the released event
logs. Run with the ``replay`` extra installed:

    pip install -e ".[replay]"
    streamlit run replay/streamlit_app.py

This is the ONLY module that imports Streamlit. Every number shown comes from
``pnyx.analysis.replay`` (the tested, Streamlit-free data layer); this file
just wires selectors to charts. Captions are factual one-liners — no claims
beyond the writeup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; Streamlit renders the Figure object
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from pnyx.analysis import replay

# Okabe-Ito palette, consistent with pnyx.analysis.figures (no dual axes,
# restrained annotations, near-black text).
_PRICE_COLOR = "#0072B2"  # market price path (condition-A blue)
_POSTERIOR_COLOR = "#333333"  # Bayes oracle (dashed reference)
_ADV_COLOR = "#D55E00"  # adversary (vermillion)
_PASS1_COLOR = "#009E73"  # own Pass-1 belief (green)
_STATED_COLOR = "#0072B2"  # in-market stated belief (blue)
_PRICE_AT_TRADE_COLOR = "#E69F00"  # price the agent faced (orange)
_TEXT_COLOR = "#333333"
_ROUND_SHADE = "#000000"


def _data_roots(root_str: str) -> list[Path]:
    """The two condition-holding roots under a data directory: ``<root>/main``
    and ``<root>/v2`` (whichever exist)."""
    root = Path(root_str).expanduser()
    return [root / "main", root / "v2"]


@lru_cache(maxsize=32)
def _load_run_cached(run_dir: str, condition: str, seed: int):
    return replay.load_run(replay.RunRef(condition, seed, Path(run_dir)))


def _style_trades(step_rows: list[replay.TradeStep]) -> pd.io.formats.style.Styler:
    df = pd.DataFrame(
        [
            {
                "step": s.step,
                "round": s.round,
                "agent": s.agent_id,
                "action": s.action,
                "stated_belief": s.stated_belief,
                "shares": s.executed_shares,
                "cost": s.cost,
                "price_before": s.price_before,
                "price_after": s.price_after,
                "parse_failed": s.parse_failed,
                "adversary": s.is_adversary,
            }
            for s in step_rows
        ]
    )

    def _highlight(row):
        if row["adversary"]:
            return [f"background-color: {_ADV_COLOR}33"] * len(row)
        return [""] * len(row)

    return df.style.apply(_highlight, axis=1).format(
        {
            "stated_belief": "{:.3f}",
            "shares": "{:.1f}",
            "cost": "{:.2f}",
            "price_before": "{:.3f}",
            "price_after": "{:.3f}",
        }
    )


def _price_path_figure(rep: replay.QuestionReplay):
    ps = replay.price_series(rep)
    fig, ax = plt.subplots(figsize=(9.0, 4.0))

    xs = [p.step for p in ps.points]
    ys = [p.price for p in ps.points]
    ax.step(xs, ys, where="post", color=_PRICE_COLOR, lw=2.0, label="Market price")
    ax.plot(xs, ys, "o", color=_PRICE_COLOR, ms=3.5)

    # Alternating shaded round boundaries.
    for b in ps.round_boundaries:
        if b.round % 2 == 0:
            ax.axvspan(b.start_step - 1, b.end_step, color=_ROUND_SHADE, alpha=0.04, lw=0)
        ax.text(
            (b.start_step - 1 + b.end_step) / 2.0, 1.02, f"R{b.round}",
            ha="center", va="bottom", fontsize=8, color=_TEXT_COLOR,
        )

    # Reference lines: Bayes oracle (dashed) + final price marker.
    ax.axhline(ps.posterior_all, ls="--", color=_POSTERIOR_COLOR, lw=1.3,
               label=f"Bayes posterior ({ps.posterior_all:.3f})")
    outcome_y = 1.0 if ps.outcome == 1 else 0.0
    ax.plot(xs[-1], outcome_y, marker="*", ms=15, color=_POSTERIOR_COLOR,
            label=f"Outcome ({'YES' if ps.outcome == 1 else 'NO'})", clip_on=False)

    # Adversary's pre-attack price (D conditions only).
    if rep.has_adversary and rep.pre_attack_price is not None:
        ax.axhline(rep.pre_attack_price, ls=":", color=_ADV_COLOR, lw=1.3,
                   label=f"Pre-attack price ({rep.pre_attack_price:.3f})")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.3, xs[-1] + 0.3)
    ax.set_xlabel("Trade # (execution order)", color=_TEXT_COLOR)
    ax.set_ylabel("P(YES)", color=_TEXT_COLOR)
    ax.tick_params(colors=_TEXT_COLOR)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def _herding_figure(rep: replay.QuestionReplay, agent_id: str):
    rows = replay.herding(rep)
    ah = next((a for a in rows if a.agent_id == agent_id), None)
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    if ah is None or not ah.by_round:
        ax.text(0.5, 0.5, "no trades for this agent", ha="center", va="center",
                color=_TEXT_COLOR, transform=ax.transAxes)
        return fig

    xr = [r.round for r in ah.by_round]
    stated = [r.stated_belief for r in ah.by_round]
    price = [r.price_at_trade for r in ah.by_round]

    if ah.pass1_belief is not None:
        ax.axhline(ah.pass1_belief, ls="--", color=_PASS1_COLOR, lw=1.3,
                   label=f"Own Pass-1 ({ah.pass1_belief:.3f})")
    ax.plot(xr, stated, "-o", color=_STATED_COLOR, lw=2.0, label="Stated belief (in-market)")
    ax.plot(xr, price, "-s", color=_PRICE_AT_TRADE_COLOR, lw=2.0, label="Price at trade")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(xr)
    ax.set_xlabel("Round", color=_TEXT_COLOR)
    ax.set_ylabel("P(YES)", color=_TEXT_COLOR)
    ax.tick_params(colors=_TEXT_COLOR)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Pnyx price-path replay", layout="wide")
    st.title("Pnyx — market price-path replay")
    st.caption(
        "Replay one settled question: the LMSR price path against the Bayes "
        "oracle, the trade stream, and per-agent herding. Data from the "
        "released event logs; every number comes from pnyx.analysis.replay."
    )

    # -- Sidebar: data root -> condition -> seed -> question ------------------
    st.sidebar.header("Selection")
    root_str = st.sidebar.text_input("Data root", value="data")
    roots = _data_roots(root_str)
    runs = replay.list_runs(roots)
    if not runs:
        st.warning(
            f"No runs found under {[str(r) for r in roots]}. Point 'Data root' "
            "at the directory that contains main/ and v2/."
        )
        return

    conditions = sorted({r.condition for r in runs})
    condition = st.sidebar.selectbox("Condition", conditions)
    seeds = sorted({r.seed for r in runs if r.condition == condition})
    seed = st.sidebar.selectbox("Seed", seeds)
    run = next(r for r in runs if r.condition == condition and r.seed == seed)

    cond, source = _load_run_cached(str(run.run_dir), run.condition, run.seed)
    qids = replay.settled_question_ids(cond, seed)
    if not qids:
        st.warning(f"No settled questions for {condition} seed {seed}.")
        return
    qid = st.sidebar.selectbox("Question", qids)

    rep = replay.question_replay(cond, seed, qid, pass1_source=source)

    # -- Header ---------------------------------------------------------------
    st.subheader(f"{condition} · seed {seed} · {qid}")
    if rep.question_text:
        st.markdown(f"**Question:** {rep.question_text}")
    cols = st.columns(4)
    cols[0].metric("Bayes posterior", f"{rep.posterior_all:.3f}")
    cols[1].metric("Final price", f"{rep.final_price:.3f}")
    cols[2].metric("Outcome", "YES" if rep.outcome == 1 else "NO")
    cols[3].metric("Subsidy", f"{rep.subsidy:.2f}")
    if rep.pass1_source_condition:
        st.caption(
            f"Pass-1 beliefs reused from condition {rep.pass1_source_condition} "
            "(this is a Pass-2-only condition)."
        )

    # -- Price path -----------------------------------------------------------
    st.markdown("### Price path")
    st.pyplot(_price_path_figure(rep))
    caption = (
        f"YES price after each of {len(rep.steps)} trades over {rep.n_rounds} "
        f"rounds; dashed line is the Bayes posterior given all signals "
        f"({rep.posterior_all:.3f})."
    )
    if rep.has_adversary and rep.pre_attack_price is not None:
        caption += (
            f" Dotted line marks the price before the adversary's first trade "
            f"({rep.pre_attack_price:.3f})."
        )
    st.caption(caption)

    # -- Trades table ---------------------------------------------------------
    st.markdown("### Trades")
    st.dataframe(_style_trades(rep.steps), use_container_width=True)
    if rep.has_adversary:
        st.caption("Trades in execution order; adversary (adv0) rows highlighted.")
    else:
        st.caption("Trades in execution order.")

    # -- Herding panel --------------------------------------------------------
    st.markdown("### Herding")
    honest = [a.agent_id for a in replay.herding(rep)]
    if honest:
        agent = st.selectbox("Honest agent", honest)
        st.pyplot(_herding_figure(rep, agent))
        st.caption(
            "For the selected honest agent: its own Pass-1 belief (dashed) vs "
            "its in-market stated belief and the price it faced, per round."
        )
    else:
        st.caption("No honest-agent Pass-1 beliefs available for this run.")


if __name__ == "__main__":
    main()
else:
    # `streamlit run` executes the module top-level (no __main__), so drive the
    # app on import too.
    main()
