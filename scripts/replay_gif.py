"""Render the D_K10 adversary price-path replay as a promo animation.

Selects the most dramatic settled (seed, question) in condition ``D_K10``
(the 10x-bankroll adversary arm, Pass-1 reused from condition ``C``) and
renders its price path as a progressive left-to-right reveal: honest trades
in blue, the adversary's (``adv0``) trade steps in vermillion, the Bayes
posterior given all signals as a dashed reference line.

"Drama score": among D_K10's settled questions (searching all seeds), first
prefer questions where the adversary flipped the market to the wrong side of
0.5 (the Bayes posterior over ALL signals is on the correct side of the
realized outcome, but the final market price is on the wrong side); among
those, prefer the largest total adversary price displacement. No RNG — a
fixed ranking rule over the released event logs, ties broken by
``(seed, question_id)`` ascending, so the selection is deterministic.

Usage (from repo root, with ``.venv`` active — no Streamlit needed):

    python scripts/replay_gif.py

Output:
    docs/assets/replay_adversary.gif   (always; target < 5 MB)
    docs/assets/replay_adversary.mp4   (only if `ffmpeg` is on PATH)

Dependencies: stdlib + numpy + matplotlib + pillow only (pillow is
matplotlib's ``PillowWriter`` GIF backend, already a transitive dependency of
matplotlib's image support).
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: rendering frames, no display needed

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pnyx.analysis import replay  # noqa: E402

# --- Okabe-Ito palette (matches pnyx.analysis.figures / replay/streamlit_app.py) ---
HONEST_COLOR = "#0072B2"  # blue: honest (p0..p5) trade steps
ADV_COLOR = "#D55E00"  # vermillion: adversary (adv0) trade steps
POSTERIOR_COLOR = "#009E73"  # green: Bayes-optimal posterior (dashed reference)
TEXT_COLOR = "#333333"
GRID_COLOR = "#CCCCCC"
BOUNDARY_COLOR = "#999999"  # round-boundary vertical lines -- darker than the
# y-grid (GRID_COLOR) so they stay visibly distinct from it, but still faint
# next to the price path / reference lines.

CONDITION = "D_K10"
DATA_ROOT = REPO_ROOT / "data" / "main"

OUT_DIR = REPO_ROOT / "docs" / "assets"
GIF_PATH = OUT_DIR / "replay_adversary.gif"
MP4_PATH = OUT_DIR / "replay_adversary.mp4"

# 900x506 px (16:9) at 100 dpi -- LinkedIn-legible thumbnail size.
FIG_W_PX, FIG_H_PX = 900, 506
DPI = 100

FPS = 10
FRAMES_PER_POINT = 5  # animation frames spent revealing each new price-path point
HOLD_SECONDS = 2.5  # how long the fully-revealed final frame holds
GIF_SIZE_TARGET_BYTES = 5 * 1024 * 1024


def _safe_dim_inches(px: int, dpi: int) -> float:
    """``px / dpi`` nudged (via ``np.nextafter``) so that ``int(value * dpi)
    == px`` exactly.

    Guards a matplotlib float-rounding quirk: e.g. ``506 / 100 == 5.06``,
    but the nearest double for ``5.06`` is a hair *below* the true value, so
    ``int(5.06 * 100)`` truncates to 505 instead of 506. Both
    ``PillowWriter.grab_frame`` (frame reshape) and ``MovieWriter.frame_size``
    (the ``-s`` size ffmpeg is told, needed even for H.264) compute pixel
    dimensions this same truncating way, so an un-nudged 506px target quietly
    renders as 505px (odd -> H.264 refuses the encoder) in the saved file."""
    val = px / dpi
    while int(val * dpi) < px:
        val = np.nextafter(val, np.inf)
    return val


FIG_W_IN = _safe_dim_inches(FIG_W_PX, DPI)
FIG_H_IN = _safe_dim_inches(FIG_H_PX, DPI)


@dataclass(frozen=True)
class Candidate:
    """One settled (seed, question) of D_K10 and its drama score."""

    seed: int
    question_id: str
    flipped: bool
    displacement: float
    replay: replay.QuestionReplay


def _flipped_and_displacement(rep: replay.QuestionReplay) -> tuple[bool, float]:
    """Whether the adversary flipped the market to the wrong side of 0.5 (the
    Bayes posterior given ALL signals is on the correct side of the realized
    outcome, but the final market price ends up on the wrong side), and the
    total absolute price displacement across the adversary's own trade steps."""
    outcome = rep.outcome
    posterior_correct_side = (
        rep.posterior_all > 0.5 if outcome == 1 else rep.posterior_all < 0.5
    )
    final_wrong_side = (
        rep.final_price < 0.5 if outcome == 1 else rep.final_price > 0.5
    )
    flipped = posterior_correct_side and final_wrong_side
    displacement = sum(
        abs(s.price_after - s.price_before) for s in rep.steps if s.is_adversary
    )
    return flipped, displacement


def select_question() -> Candidate:
    """Pick the settled (seed, question) of D_K10 maximizing the drama score
    (flipped first, largest adversary displacement second; deterministic tie
    break on (seed, question_id))."""
    runs = [r for r in replay.list_runs([DATA_ROOT]) if r.condition == CONDITION]
    if not runs:
        raise SystemExit(f"no {CONDITION} runs found under {DATA_ROOT}")

    candidates: list[Candidate] = []
    for run in runs:
        cond, source = replay.load_run(run)
        for qid in replay.settled_question_ids(cond, run.seed):
            rep = replay.question_replay(cond, run.seed, qid, pass1_source=source)
            flipped, displacement = _flipped_and_displacement(rep)
            candidates.append(Candidate(run.seed, qid, flipped, displacement, rep))

    if not candidates:
        raise SystemExit(f"no settled questions found for {CONDITION}")

    candidates.sort(key=lambda c: (-int(c.flipped), -c.displacement, c.seed, c.question_id))
    best = candidates[0]
    print(
        f"selected seed={best.seed} qid={best.question_id} "
        f"flipped={best.flipped} displacement={best.displacement:.4f}"
    )
    return best


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def _frame_plan(n_points: int) -> tuple[int, int, int]:
    """(reveal_frames, hold_frames, total_frames) for a price path with
    ``n_points`` points (the initial price plus one per executed trade)."""
    reveal_frames = FRAMES_PER_POINT * max(n_points - 1, 0)
    hold_frames = int(round(HOLD_SECONDS * FPS))
    return reveal_frames, hold_frames, reveal_frames + hold_frames


def _reveal_count(frame_idx: int, n_points: int, reveal_frames: int) -> int:
    """How many leading points of the price path are visible at ``frame_idx``
    (1..n_points, monotonically non-decreasing, fully revealed once
    ``frame_idx >= reveal_frames``)."""
    if reveal_frames <= 0:
        return n_points
    return min(n_points, 1 + frame_idx // FRAMES_PER_POINT)


def _first_adversary_point_index(rep: replay.QuestionReplay) -> int | None:
    """The price-path point index (1-based into ``price_series(rep).points``)
    of the adversary's first trade, or ``None`` if it never traded."""
    for j, s in enumerate(rep.steps):
        if s.is_adversary:
            return j + 1
    return None


def _draw_frame(
    ax,
    rep: replay.QuestionReplay,
    ps: replay.PriceSeries,
    reveal_count: int,
    show_caption: bool,
    show_final_annotation: bool,
) -> None:
    ax.clear()
    xs_all = [p.step for p in ps.points]
    x_max = xs_all[-1]

    ax.set_xlim(-0.4, x_max + 0.4)
    ax.set_ylim(-0.03, 1.03)

    # Light dotted reference at 0.5.
    ax.axhline(0.5, color=GRID_COLOR, lw=1.0, ls=":", zorder=1)

    # Bayes-optimal posterior: dashed green, full width, labeled via legend.
    ax.axhline(
        ps.posterior_all,
        color=POSTERIOR_COLOR,
        lw=1.8,
        ls="--",
        zorder=2,
        label="Bayes-optimal posterior",
    )

    # Round boundaries: faint vertical lines between consecutive rounds
    # (skip the first boundary -- it's round 1's own start, i.e. the
    # beginning of the whole path, not a transition).
    for b in ps.round_boundaries[1:]:
        ax.axvline(
            b.start_step - 0.5, color=BOUNDARY_COLOR, lw=1.1, ls="-", alpha=0.75, zorder=1
        )

    # Progressive price path, revealed left to right.
    xs = xs_all[:reveal_count]
    ys = [p.price for p in ps.points[:reveal_count]]
    for i in range(len(xs) - 1):
        is_adv = rep.steps[i].is_adversary if i < len(rep.steps) else False
        color = ADV_COLOR if is_adv else HONEST_COLOR
        ax.plot(xs[i : i + 2], ys[i : i + 2], color=color, lw=2.2, zorder=3, solid_capstyle="round")

    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == 0:
            ax.plot(x, y, "o", color=HONEST_COLOR, ms=4.5, zorder=4)
            continue
        is_adv = rep.steps[i - 1].is_adversary
        if is_adv:
            ax.plot(
                x, y, "o", color=ADV_COLOR, ms=9.0, zorder=5,
                markeredgecolor="white", markeredgewidth=0.8,
            )
        else:
            ax.plot(x, y, "o", color=HONEST_COLOR, ms=4.5, zorder=4)

    ax.set_xlabel("Trade #", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Market price P(YES)", color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, lw=0.6, alpha=0.5, zorder=0)

    ax.set_title(
        "LLM prediction market, adversary with 10× bankroll",
        color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=TEXT_COLOR)

    if show_caption:
        ax.text(
            0.01, 0.02, "adversary trades in red",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color=ADV_COLOR, style="italic",
        )

    if show_final_annotation:
        ax.text(
            0.5, 0.90,
            f"final price {ps.final_price:.2f} vs optimal {ps.posterior_all:.2f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=TEXT_COLOR, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=TEXT_COLOR, lw=0.8, alpha=0.9),
            zorder=6,
        )


def build_animation(rep: replay.QuestionReplay):
    """Build the ``FuncAnimation`` for ``rep``. Returns ``(fig, anim,
    total_frames)``; caller is responsible for saving and closing ``fig``."""
    ps = replay.price_series(rep)
    n_points = len(ps.points)
    reveal_frames, hold_frames, total_frames = _frame_plan(n_points)
    first_adv_idx = _first_adversary_point_index(rep)

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.patch.set_facecolor("white")

    def update(frame_idx: int):
        reveal_count = _reveal_count(frame_idx, n_points, reveal_frames)
        show_caption = first_adv_idx is not None and reveal_count >= first_adv_idx + 1
        show_final = reveal_count >= n_points
        _draw_frame(ax, rep, ps, reveal_count, show_caption, show_final)
        fig.tight_layout(pad=1.4)
        return []

    anim = animation.FuncAnimation(fig, update, frames=total_frames, blit=False)
    return fig, anim, total_frames


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best = select_question()
    rep = best.replay

    fig, anim, total_frames = build_animation(rep)
    duration_s = total_frames / FPS
    print(f"frames={total_frames} fps={FPS} duration={duration_s:.1f}s")

    anim.save(GIF_PATH, writer=animation.PillowWriter(fps=FPS))
    gif_size = GIF_PATH.stat().st_size
    print(f"wrote {GIF_PATH} ({gif_size / 1e6:.2f} MB)")
    if gif_size > GIF_SIZE_TARGET_BYTES:
        print(
            f"WARNING: {GIF_PATH.name} exceeds the 5 MB target "
            f"({gif_size / 1e6:.2f} MB) -- reduce fps/size",
            file=sys.stderr,
        )

    if shutil.which("ffmpeg"):
        mp4_writer = animation.FFMpegWriter(
            fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
        )
        anim.save(MP4_PATH, writer=mp4_writer)
        print(f"wrote {MP4_PATH} ({MP4_PATH.stat().st_size / 1e6:.2f} MB)")
    else:
        print("ffmpeg not found on PATH; skipping MP4 output")

    plt.close(fig)


if __name__ == "__main__":
    main()
