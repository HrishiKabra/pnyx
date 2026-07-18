# P5 — Analysis plan (Fig 1, Fig 2, Table 1, Table 2, stats)

## Context

All P4 data is complete on disk. This plan turns it into the pre-registered
deliverables. Everything is local computation — zero API calls, zero spend.

**Data layout** (all paths relative to repo root):

- Event logs: `data/main/<COND>/<COND>_seed{0,1,2}.jsonl` — one JSON object per
  line, parsed with `pnyx.schemas.parse_event` into `BeliefEvent` /
  `TradeEvent` / `SettlementEvent` (discriminated on `type`).
- Question sidecars: `data/main/<COND>/<COND>_seed{s}.questions.jsonl` —
  `QuestionRecord` lines; identical across seeds/conditions by construction
  (all load `datasets/questions_v1.jsonl`, 40 questions).
- Cost ledgers: `data/main/<COND>/<COND>.costs.jsonl` — `CostEntry` lines.
- Conditions on disk: `A`, `B1`, `B3`, `C` (full two-pass: Pass-1 `BeliefEvent`s
  with `key.phase == "independent"` + market pass), `D_K1`, `D_K3`, `D_K10`
  (Pass-2 only; Pass-1 reused from `C`; extra 7th agent `adv0` with no shards),
  `W_FIXED`, `W_SHUFFLED` (Pass-2 only; Pass-1 reused from `A`; persistent
  wealth). Honest agents are always `p0..p5`, bankroll 100.0.
- **Condition "B2" of the spec's grid is condition A** (same 6× deepseek pool);
  analysis code treats A as both the H1 condition and the deepseek homogeneous
  arm. Homogeneous arms for H2: B1 (llama-3.1-8b), A (deepseek-v4-flash),
  B3 (nemotron). Mixed arm: C (2 llama + 2 deepseek + 2 nemotron).
- Oracle: each `QuestionRecord` has `posterior_table` mapping subset keys to
  P(s=1 | subset); posterior given ALL shards is
  `record.posterior_table[subset_key(range-of-all-signal-indices)]` — reuse the
  existing helpers `_full_subset_key` / `_posterior_all` in
  `pnyx/analysis/pilot.py`. Realized outcome is `SettlementEvent.outcome`
  (equals the question's latent state).
- Existing loader/guard code to reuse (import, don't duplicate):
  `pnyx/analysis/pilot.py` — `_tolerant_jsonl`, `_read_events`,
  `_read_questions`, `_read_costs`, `_assert_pass1` (the phase guard),
  `_pass1_beliefs`. Promote them to public names in a shared module if that
  reads better, but there must remain exactly one implementation of each.

## Global Constraints

- **Phase guard (spec §5, enforced in code):** every function that computes a
  static baseline or ρ accepts ONLY Pass-1 records with
  `key.phase == "independent"` and raises `ValueError` on anything else. Never
  compute a baseline or ρ from Pass-2 (`market`) beliefs.
- **Primary metric:** posterior gap = |mechanism output − posterior_all|.
  Secondary: Brier vs realized outcome = (p − outcome)².
- **Dependencies:** stdlib + pydantic + numpy + pandas + matplotlib ONLY.
  **No scipy, no sklearn** — Wilcoxon and logistic calibration are implemented
  by hand (specs below give exact algorithms).
- **Determinism:** any randomness (bootstrap) uses
  `numpy.random.default_rng(seed)` with an explicit fixed seed argument
  (default 0). Running the analysis twice produces byte-identical tables and
  identical figure data.
- Tests: pytest, run with `-W error`; suite must stay green. Tests use small
  hand-constructed events/questions with exact-value assertions where the
  metric definition permits (gaps, pools, Wilcoxon W, ρ on toy data).
- Probabilities are clamped to [1e-6, 1 − 1e-6] before any logit; document the
  clamp where used.
- Style: match the existing codebase (module docstrings explaining WHY,
  type hints, pure functions where possible, no I/O inside metric functions).
- Commit per task with the repo's existing message style (`phase5: ...`).

## Task 1: `pnyx/baselines.py` — the four static pools

New module `pnyx/baselines.py` + `tests/test_baselines.py`.

All functions take `list[BeliefEvent]` (one per agent, a single question's
Pass-1 beliefs) and are guarded: call `assert_pass1(e)` (import/promote the
guard from `pnyx/analysis/pilot.py`) on every event first.

1. `mean_pool(events) -> float` — arithmetic mean of `belief.prob`.
2. `median_pool(events) -> float` — median.
3. `log_opinion_pool(events) -> float` — equal-weight LOP: clamp each p to
   [1e-6, 1−1e-6], average the logits, sigmoid back.
4. `calibrated_stack(train, test) -> float` — Platt-style calibration of the
   mean logit, leave-one-question-out:
   - Signature: `calibrated_stack(train: list[tuple[list[BeliefEvent], int]], test: list[BeliefEvent]) -> float`
     where `train` pairs each held-in question's Pass-1 events with its
     realized outcome (0/1), and `test` is the held-out question's events.
   - Feature per question: x = mean of agent logits (same clamp).
   - Fit σ(a·x + b) by maximizing Bernoulli log-likelihood with Newton–Raphson
     (IRLS) in numpy: ≤ 100 iterations, convergence when max |Δparam| < 1e-10,
     ridge 1e-6 on the Hessian diagonal for stability. Start at a=1, b=0.
   - Degenerate training outcomes (all 0 or all 1): fall back to returning the
     uncalibrated sigmoid of the test x (document why: separable likelihood
     has no finite MLE).
   - Return σ(a·x_test + b).
5. Tests: exact values for mean/median/LOP on hand-built events; phase guard
   raises on a `market`-phase event for ALL four functions; calibrated stack
   recovers a≈1,b≈0 (tolerance 1e-3) when training data is generated from
   σ(x) itself; degenerate-outcome fallback hit; LOO harness: a helper
   `loo_calibrated_stack(questions: list[tuple[list[BeliefEvent], int]]) -> list[float]`
   returning one prediction per question, each fit on the other n−1.

## Task 2: `pnyx/analysis/mainrun.py` — grid loader, per-question metrics, ρ, stats

New module + `tests/test_mainrun.py`. This is the core: everything in Tasks
3–5 consumes its outputs.

1. `load_condition(run_dir: Path) -> ConditionData` — dataclass holding, per
   seed: parsed events split by type, questions by id, and (for pass2-only
   conditions) nothing extra — Pass-1 reuse is resolved by the CALLER passing
   the source condition's data (mirror the runner: D_K* → C, W_* → A).
2. `question_metrics(cond: ConditionData, pass1_source: ConditionData | None) -> pandas.DataFrame`
   — one row per (seed, question_id) with columns:
   - `posterior_all`, `outcome`
   - `market_prob` = `SettlementEvent.final_price`; `market_gap`, `market_brier`
   - `mean_prob`, `median_prob`, `lop_prob`, `stack_prob` (+ `_gap`, `_brier`
     for each) computed from Pass-1 beliefs of the six honest agents via
     `pnyx.baselines` (stack via the LOO harness, fit within (condition, seed));
     for pass2-only conditions these come from `pass1_source` and MUST be
     bit-identical to the source condition's values (assert coverage: 6 beliefs
     per question, raise on gaps).
   - `subsidy`, `n_parse_failed` (count of parse-failed events for that
     question, both passes).
3. `team_rho(cond: ConditionData) -> pandas.DataFrame` — one row per (seed,
   agent_pair): Pearson correlation over the 40 questions of the agents'
   Pass-1 errors e_i(q) = prob_i(q) − posterior_all(q). Plus
   `team_rho_summary(...)`: per seed the mean over the 15 pairs, and
   mean ± std over seeds. Phase-guarded like the baselines.
4. Stats helpers (pure functions on numpy arrays):
   - `paired_bootstrap(d: ndarray, n_boot=10_000, seed=0) -> (mean, lo95, hi95)`
     — d is the vector of paired differences over (question, seed) units;
     resample QUESTION IDs with replacement (cluster bootstrap: a resampled
     question brings all its seeds' differences) so seed-pairing is respected.
     Signature takes `question_ids: ndarray` alongside `d`.
   - `wilcoxon_signed_rank(d: ndarray) -> (W, p)` — drop zeros; rank |d| with
     average ranks for ties; W = sum of ranks of positive d; two-sided p from
     the normal approximation with tie-corrected variance
     σ² = n(n+1)(2n+1)/24 − Σ(t³−t)/48, continuity correction 0.5. Exact-value
     test against a hand-computed small case (e.g. d = [1,−2,3,−4,5] → W=9,
     and a case with ties).
   - `compare(df, col_a, col_b) -> ComparisonResult` — paired mean difference,
     per-seed means, bootstrap CI, Wilcoxon p, for any two metric columns.
5. Tests: build two tiny synthetic conditions (2 questions × 2 seeds × 2–3
   agents) with hand-computed expected gaps/pools/ρ; pass2-only reuse path
   asserts identity with source values and raises on missing coverage;
   bootstrap determinism (same seed → same CI); Wilcoxon exact values.

## Task 3: `pnyx/analysis/manipulation.py` — H3 adversary metrics + H-wealth

New module + `tests/test_manipulation.py`. Consumes Task 2's loaders.

Adversary identification: agent_id `adv0` (the only agent with
`shard_indices: []`; hard-code `ADVERSARY_ID = "adv0"` with a comment).
Target side per question: wrong side of the truth — `"YES"` if outcome == 0
else `"NO"` (mirrors `runner.adversary_target_side`; reference it in the
docstring but reimplement locally on `outcome` so analysis never imports the
runner).

`adversary_metrics(cond, honest_ref) -> pandas.DataFrame`, one row per
(seed, question_id), where `honest_ref` is condition C's metrics DataFrame
(Task 2) matched on (seed, question_id):

1. `wrong_side` (bool): final price strictly on the target side of 0.5
   (price > 0.5 when target YES, < 0.5 when target NO; exactly 0.5 → False).
2. `flipped` (bool): honest_ref's final price was on the CORRECT side of 0.5
   AND this run's final price is on the wrong side. Flip rate = mean.
3. `adv_spend` = Σ max(cost, 0) over adv0's TradeEvents;
   `adv_displacement` = Σ signed (price_after − price_before) toward the
   target side over adv0's trades (positive = toward target);
   `capital_per_01_move` = 0.1 · adv_spend / adv_displacement when
   displacement > 1e-9, else NaN (report the NaN count separately).
4. `adv_pnl` = settlement payout to adv0 + final adv0 bankroll − STARTING
   bankroll. The adversary's starting bankroll is k×100 (D_k3.yaml: 300,
   D_k10.yaml: 1000), so it must be derived from the data, not hardcoded:
   starting = first adv0 trade's `bankroll_after + cost` (bankroll before
   its first trade). If adv0 never traded, P&L = settlement payout (0
   shares → 0). [Corrected 2026-07-19: the original "− 100.0" was wrong
   for k>1 and made the market-taxes-manipulation sign flip spuriously.]
5. Recovery: pre-attack price = `price_before` of adv0's FIRST trade;
   post-attack = events after adv0's LAST trade. `recovered` (bool): any
   subsequent honest trade's `price_after` within ε = 0.05 of the pre-attack
   price before market close. `recovery_rounds` = that trade's round − the
   last adversary trade's round (NaN when not recovered or adv never traded).
6. `summarize_adversary(dfs: dict[k, DataFrame]) -> DataFrame` — per k ∈
   {1, 3, 10}: flip rate, wrong-side rate, mean capital_per_01_move (+ NaN
   count), mean adv_pnl, recovery fraction, median recovery_rounds, and
   terminal Brier degradation = mean(D market_brier) − mean(C market_brier)
   (market_brier joined from Task 2 metrics on both sides).

H-wealth: `wealth_order_effects(a_df, w_fixed_df, w_shuffled_df) -> dict` —
paired comparisons (Task 2 `compare`) of market_gap and market_brier for
W_FIXED vs A, W_SHUFFLED vs A, W_FIXED vs W_SHUFFLED.

Tests: hand-built event sequences with known adversary trades — exact
expected values for every column including both recovery branches, the
no-adversary-trade edge case, displacement ≤ 0 → NaN, and flip logic against
a fabricated honest_ref.

## Task 4: `pnyx/analysis/figures.py` — Fig 1 and Fig 2

New module + `tests/test_figures.py` (tests assert the DATA passed to
matplotlib — returned by pure `fig1_data(...)` / `fig2_data(...)` helpers —
not pixels; plus a smoke test that files get written).

- `fig1(metrics_a: DataFrame, out: Path)` — H1 on condition A. Two panels
  (posterior gap left, Brier right). Five mechanisms: market, mean, median,
  LOP, calibrated stack. Bar = grand mean over the 120 (question, seed) rows;
  error bar = std over the 3 per-seed means. Annotate each static pool with
  the Wilcoxon p vs market (from Task 2 `compare`), e.g. "p=0.03" under the
  bar. Saved as both `fig1.png` (300 dpi) and `fig1.pdf`.
- `fig2(rho_by_cond, gapdiff_by_cond, adv_summary, out: Path)` — two panels:
  - (a) phase map: x = measured team ρ (per condition per seed: A, B1, B3, C
    → 12 points), y = market_gap − mean_pool_gap for that (condition, seed);
    one marker shape/color per condition, horizontal line at 0 ("market
    helps" below, "hurts" above — annotate), legend outside plot area.
  - (b) manipulation: x = k ∈ {1,3,10} (log scale), single y-axis in [0,1]:
    flip rate and recovery fraction (two lines + markers, direct-labeled at
    the right end). NO second y-axis (dual axes are banned); mean adversary
    P&L per k appears as small text annotations under the x-axis tick labels
    (e.g. "P&L −$47"), with the full numbers in stats.md. Same output
    convention.
- Figure style (both): matplotlib default font, colorblind-safe palette
  (Okabe–Ito: #0072B2 blue, #E69F00 orange, #009E73 green, #CC79A7 pink,
  #56B4E9 light blue, #D55E00 vermillion), no gridlines except a light
  y-grid, no top/right spines, axis labels with units, panel letters (a)/(b),
  figsize (10, 4), `tight_layout`. Market always #0072B2; static pools share
  a muted family. Every number plotted must come from Task 2/3 outputs — no
  recomputation in figures.py.

## Task 5: `pnyx/analysis/tables.py` + CLI `analyze-main`

New module + `tests/test_tables.py`, plus CLI wiring in `pnyx/cli.py`.

- `table1(metrics_by_cond, costs_by_cond) -> str` (markdown): one row per
  condition (A, B1, B3, C, D_K1, D_K3, D_K10, W_FIXED, W_SHUFFLED): market
  gap (mean ± std over seeds), market Brier, mean-pool gap/Brier (— for
  pass2-only conditions, footnote pointing to the source condition), total
  cost USD (sum of that condition's cost ledger), and accuracy-per-dollar =
  (0.25 − market Brier) / cost, with the definition in the caption (0.25 =
  uniform-guess Brier). B3's cost is $0.00 → print "∞ (free tier)" for
  accuracy-per-dollar.
- `table2(metrics_by_cond, events_by_cond) -> str` (markdown): parse-failure
  count and rate per (condition, model) — model resolved from the condition's
  config YAML agent→model mapping, loaded via `pnyx.cli.load_config` on
  `pnyx/configs/main/*.yaml`; and per-condition market-maker subsidy: mean,
  total, max, vs the theoretical bound b·ln 2 ≈ 27.73.
- `stats_report(...) -> str` (markdown): every pre-registered comparison with
  paired mean Δ, bootstrap 95% CI, Wilcoxon p:
  - H1: market vs each of the 4 pools on condition A (gap and Brier).
  - H2a: C market_gap vs each homogeneous arm's market_gap (unpaired on
    condition but paired on (question, seed) — same questions by
    construction).
  - H2b: per-condition (ρ, market−mean gap) pairs listed; Spearman rank
    correlation between ρ and the gap difference across the 12 points
    (hand-rolled Spearman: Pearson on ranks; note n=12 → descriptive).
  - H3: summarize_adversary output verbatim.
  - H-wealth: the three wealth_order_effects comparisons.
- CLI: `python -m pnyx.cli analyze-main --data-root data/main --configs pnyx/configs/main --out analysis_out [--bootstrap-seed 0]`
  → writes `fig1.png/pdf`, `fig2.png/pdf`, `table1.md`, `table2.md`,
  `stats.md` into `--out`, prints a one-screen summary (headline H1 numbers,
  flip rates, total spend). Runs end-to-end on the real `data/main` in <2 min.
  `analysis_out/` is NOT gitignored (figures ship with the repo).
- Integration test: run the full pipeline on a fixture mini-grid (reuse Task
  2's synthetic conditions written to tmp_path as real JSONL files) and
  assert all five files exist and stats.md contains every H-section header.
