# P6 — Ship plan (writeup + public README)

## Context

P5 is complete: all results live in `analysis_out/` (fig1/fig2 png+pdf,
table1.md, table2.md, stats.md) and the P5 headline-results section of
PROGRESS.md. The P4 event logs are released under `data/main/` so
`python -m pnyx.cli analyze-main --data-root data/main --configs
pnyx/configs/main --out analysis_out` reproduces every number from a fresh
clone (~4s, zero API calls). Total experiment spend (all ledgers): $0.73.
Streamlit replay is explicitly skipped (out of scope for this phase).

## Global Constraints (both tasks)

- **Positioning discipline (binding, from the project spec):** this is a
  *controlled decomposition and stress test*, NOT a first demonstration.
  Prior work already showed LLM agents can trade in LMSR markets and that
  same-model pools have correlated errors. Our delta = (a) generative
  environment with an exact Bayes-posterior oracle, (b) independent-elicitation
  control on identical signals, (c) quality-matched team comparison, (d) the
  resulting map of where markets help / do nothing / hurt. No novelty
  overclaims anywhere. H2a is described as a replication.
- **Public-repo hygiene:** no mention of internal dev workflow, agent
  orchestration, model routing of the development process, CLAUDE.md, or
  subagents. (The EXPERIMENT's LLM models/personas are of course described.)
- Every number quoted must match `analysis_out/stats.md`, `table1.md`,
  `table2.md`, or the P5 section of PROGRESS.md exactly — no recomputation,
  no rounding beyond what those files show (3 decimals is fine).
- Required caveats wherever the relevant result is discussed:
  (1) recovery TIMING is structurally unmeasurable (adversary trades in every
  round; report recovery fraction only); (2) the achieved team-ρ range is
  narrow (0.38–0.51), so the phase map is descriptive (n=12) not a fitted
  law; (3) thin markets (6 traders, 3 rounds); (4) prompt sensitivity (b and
  prompts frozen after a pilot, not exhaustively searched); (5) synthetic
  prose rendering may not transfer to real documents; (6) single environment
  family (binary state, 6 conditionally independent signals).
- Markdown, committed to the repo. Tests are not touched by either task.

## Citations (use exactly these; cite where relevant)

- Preference Optimization Drives Monoculture in LLM Prediction Markets —
  arXiv:2606.26583 (monoculture/DPO error correlation; cross-model diversity
  mitigation; we replicate the diversity direction)
- Information Aggregation with AI Agents — arXiv:2604.20050 (closest prior:
  LLM agents with dispersed signals trading via LMSR; homogeneous vs.
  heterogeneous teams)
- Decentralized Aggregation of LLM Predictions via Wagering Mechanisms —
  arXiv:2607.04389
- Correlated Errors in Large Language Models — arXiv:2506.07962
- The Oracle's Fingerprint — arXiv:2605.00844
- Truthfulness Despite Weak Supervision — arXiv:2601.20299 (why peer
  prediction is out of scope / sequel)
- Hanson (2003), "Combinatorial information market design" (LMSR)
- Storkey (2011), "Machine Learning Markets"
- Abernethy, Chen, Wortman Vaughan (2013), "Efficient market making via
  convex optimization"

## Task 1: `docs/writeup.md` — the 6–8 page writeup

Single markdown file, ~3,500–5,000 words (≈6–8 pages), structured:

1. **Abstract** (~150 words): environment with exact posterior oracle; two
   elicitation passes; headline: market UNDERPERFORMS static pools of the
   same agents' independent beliefs on the main grid (reversed H1);
   mixed-team replication; manipulation is taxed but flips persist; wealth
   persistence slightly hurts.
2. **Introduction & positioning** (§1-discipline above; state the three
   confounds prior work entangles — model knowledge, signal structure,
   prompt behavior, market dynamics — and that the contribution is
   separation, with the deliverable being the help/nothing/hurt map).
3. **The generative environment** — binary latent state, prior 0.5, six
   conditionally independent signals with configured accuracies, exact
   posterior table stored for every signal subset (the oracle), min
   single-shard margin 0.1 (no single-shard sufficiency), 40 main + 10 pilot
   questions, template-constrained prose rendering with adversarial
   verification (100% pass after two fix rounds; verified: direction
   entailment, no cross-shard leaks, no meta-clues).
4. **Mechanisms & protocol** — LMSR (b=40, chosen by pilot sweep from
   {20,40,80}; b=20 degenerate, b=80 sluggish), bankroll 100, 3 rounds,
   randomized order per round; two-pass protocol (Pass 1 independent — feeds
   ALL baselines and ρ estimates, enforced by a code-level phase guard;
   Pass 2 market with own Pass-1 belief as prior); baselines: mean, median,
   log opinion pool, LOO-calibrated stack; 6 fixed personas; models
   llama-3.1-8b / deepseek-v4-flash / nemotron-3-super-120b (standalone
   gaps 0.291 / 0.268 / 0.257 — quality-matched); condition grid A, B1, B3,
   C, D×{1,3,10}, W×{fixed,shuffled}; 3 seeds; total cost $0.73.
5. **Results** — one subsection per hypothesis, quoting stats.md exactly:
   H1 reversed (with per-pool Δ, CI, p); H2a replication; H2b phase map
   (Spearman 0.280, n=12 descriptive; C is the only condition with
   below-zero seeds); H3 (flip rate 0.317→0.392, Brier degradation
   0.105→0.312, adversary P&L −$47/−$143/−$440, recovery fraction 0.09–0.20
   + timing caveat); H-wealth (persistent wealth worsens gap p≈0.03, no
   order-dependence p=0.74). Reference Fig 1/Fig 2 and Tables 1–2 by path.
6. **Why does the market hurt here?** — a short, clearly-labeled
   *interpretation* section (not a claim of proof): price commitment on the
   realized side (pilot observation), herding on shared errors when ρ is
   high, LMSR path-dependence with thin participation; note the P4 smoke
   observation that market Brier can beat pools on held-out easy sets while
   posterior gap does not — the market overcommits.
7. **Limitations** — the six caveats from Global Constraints, plus:
   baselines had access to exactly the same information (that is the point
   of the design), results are for THIS regime (6 agents, 3 rounds, b=40).
8. **Reproducibility** — one command on released logs; full rerun cost;
   deterministic stats under fixed bootstrap seed; pre-registration in
   SPEC.md (hypotheses frozen before main runs).
9. **References.**

Fact-check pass required before committing: re-open analysis_out/stats.md
and verify every quoted number character-for-character.

## Task 2: `README.md` — public repo front page

Replace the current README (check `git show HEAD:README.md` — if none
exists, create). Structure:

1. Title + one-sentence description (positioning-compliant).
2. **Fig 1 at the top** (`analysis_out/fig1.png`) with a two-sentence
   caption stating the reversed-H1 headline.
3. "What this is" — 5–8 lines: oracle environment, two-pass design,
   the question it answers; link to `docs/writeup.md` and `SPEC.md`.
4. **Reproduce the paper** section:
   - `pip install -e . && python -m pnyx.cli analyze-main --data-root
     data/main --configs pnyx/configs/main --out analysis_out` — regenerates
     every figure/table/stat from the released logs in ~5s, no API key.
   - Rerunning the experiments from scratch: needs `OPENROUTER_KEY` in
     `.env`, costs ≈ $0.75 total, `python -m pnyx.cli run --config
     pnyx/configs/main/<COND>.yaml` per condition (resume-safe; D/W need
     their Pass-1 source condition first: D←C, W←A).
   - `pytest -W error` (366 tests).
5. **Results at a glance** — compact table: H1 reversed (market gap 0.269
   vs best pool 0.187), H2a replication (mixed −0.155 vs worst homogeneous,
   p<0.001), H3 (flip rate rises 0.32→0.39, adversary always loses money),
   H-wealth (slightly hurts). Stated total cost $0.73.
6. Repo map (one line per module), limitations pointer to the writeup,
   citation block for the paper (plain-text how-to-cite).
7. License note only if a LICENSE file exists (check; do not invent one).

Fact-check pass: every command in the README must be run verbatim before
committing (the analyze-main one against a scratch --out dir; the pytest
one may reuse the last full-suite result from this session's history if
< 1h old — otherwise rerun).
