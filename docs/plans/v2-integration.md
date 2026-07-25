# v2 integration — capability axis + herding decomposition

## Context

v2 ran two new full two-pass conditions on the same 40 questions × 3 seeds:
`A_PRO` (6× deepseek-v4-pro, data/v2/A_PRO) and `A_LUNA` (6× gpt-5.6-luna,
data/v2/A_LUNA). Established v2 results (recomputed by the controller with the
existing pipeline; Task 1 must reproduce these exactly from module code):

- Deficits (market_gap − mean_gap): flash +0.051 (p=.080) / pro −0.003
  (p=.377) / luna +0.002 (p=.451).
- Cross-tier market gap, paired on (seed, question): pro−flash −0.044
  [−0.098,+0.008] p=.1271; luna−pro −0.032 [−0.078,+0.017] p=.1393;
  luna−flash −0.076 [−0.133,−0.013] p=.0011.
- Cross-tier POOL gap: pro−flash +0.010 [−0.011,+0.030] p=.2894 (Pro's
  beliefs are NOT better than flash's); luna−flash −0.027 [−0.050,−0.004]
  p=.0003.
- Team ρ: flash 0.398±0.018, pro 0.378, luna 0.283±0.005.
- Interpretation to carry into the writeup: the deficit fix is trading skill,
  not knowledge (same-family Pro control); capability matters independently
  of ρ (pro ≈ flash's ρ, opposite outcome); no tier BEATS the pool.
- Costs: A_PRO $0.45 (1 parse-fail/3000), A_LUNA $2.61 (0 parse-fails).

## Global Constraints

- **Phase-guard semantics, extended:** static baselines and ρ still use ONLY
  Pass-1 independent beliefs. The herding analysis is the spec's sanctioned
  "dynamics analysis" of in-market beliefs — it MAY read
  `TradeEvent.trade.belief` (market phase), but its functions must be clearly
  named/documented as dynamics-only and must never feed `pnyx.baselines` or
  ρ. Never route market-phase beliefs through `assert_pass1`-guarded code.
- Dependencies unchanged: stdlib + pydantic + numpy + pandas + matplotlib.
  No scipy/sklearn (OLS via `numpy.linalg.lstsq`).
- Determinism: any bootstrap uses `default_rng(seed)`, seed parameter
  default 0; regenerated .md outputs byte-stable.
- Tests pytest `-W error`; suite currently 366; must stay green.
- Figure rules as P5: Okabe–Ito palette (validated), one marker shape per
  condition, no dual axes, no top/right spines, light y-grid, text never in
  series color, 300-dpi PNG + PDF, panel letters.
- Writeup/README: positioning discipline (no novelty overclaims), public
  hygiene (no dev-workflow content), every number matches the generated
  analysis outputs character-for-character.
- Exclude `parse_failed=True` trade events from herding regressions.

## Task 1: herding + tier analysis module and `analyze-v2` CLI

New module `pnyx/analysis/capability.py` + tests/test_capability.py, plus a
CLI subcommand `analyze-v2`.

1. `tier_metrics(conds: dict[str, ConditionData]) -> pandas.DataFrame` — thin
   orchestration over existing `question_metrics` (no metric reimplementation):
   one row per tier with pool_gap, market_gap, deficit, market_brier, team ρ
   (via `team_rho_summary`), plus the within-tier `compare` (market_gap vs
   mean_gap) and cross-tier paired comparisons listed in Context (reuse
   `paired_bootstrap`/`wilcoxon_signed_rank` on merged frames).
2. **Herding regression** (`herding_weights`): for every honest-agent,
   non-parse-failed TradeEvent in a condition: y = stated in-market belief
   (`trade.belief`), x1 = that agent's own Pass-1 prob for the question
   (from the same condition's independent pass), x2 = `price_before`.
   Per (condition, round) AND pooled: OLS y = a + b1·x1 + b2·x2 via
   `numpy.linalg.lstsq`; report b1 (own-information weight), b2 (price
   weight), R², n. Cluster-bootstrap 95% CI on b1/b2 by resampling question
   ids (2,000 draws, seed 0, rerunning the OLS per draw).
   Also descriptive drift: mean |y − x1| per (condition, round).
   Hypothesis the numbers will speak to: b2 falls with capability tier.
3. `analyze-v2` CLI: `python -m pnyx.cli analyze-v2 --flash data/main/A
   --pro data/v2/A_PRO --luna data/v2/A_LUNA --out analysis_out/v2
   [--bootstrap-seed 0]` writes:
   - `tiers.md` — the three-tier table + all within/cross-tier comparisons
     in the P5 stats format ("Δ = … [lo, hi], Wilcoxon p = …").
   - `herding.md` — regression table per tier (pooled + per round) with CIs
     and drift descriptives.
   - `fig3.png/pdf` — two panels: (a) market deficit per tier with the
     bootstrap CI as error bars, zero line annotated; (b) price weight b2
     per tier per round (three grouped points/lines per tier), CI whiskers,
     single y-axis.
   - `fig2_v2.png/pdf` — the P5 phase map regenerated with A_PRO and A_LUNA
     added as two new conditions (reuse `figures.fig2`'s data path; new
     Okabe–Ito colors #56B4E9 and #D55E00 with new marker shapes v/P;
     validate the 6-color palette with the project's chart validator rules —
     if an adjacent pair lands in the 6–8 CVD band, marker shapes + direct
     labels are the required secondary encoding, same as P5).
4. Tests: synthetic fixture with hand-computed OLS solution (3 agents, 2
   questions, exact lstsq answer asserted to 1e-9); parse-failed exclusion;
   herding never touches baselines (no import of `assert_pass1` on the
   market-phase path — assert via a contamination fixture that Pass-2
   beliefs never reach a baseline function); CLI smoke on a tmp mini-grid;
   determinism (two runs byte-identical .md).
5. Sanity (report-only): run the real CLI; `tiers.md` numbers must equal the
   Context block above to the printed precision; paste the herding b2 per
   tier into the report.

## Task 2: writeup + README v2 integration

After Task 1 lands and the controller regenerates `analysis_out/v2/`.

1. `docs/writeup.md`: add section "Extension: the capability axis" (after
   the interpretation section, before Limitations): motivation (is the
   reversed H1 a model-capability artifact?), design (same questions/seeds,
   two new tiers, same-family Pro as the knowledge-vs-trading control),
   results table, the three key claims with exact stats (deficit vanishes at
   parity — NOT past it; Pro's pool unchanged → trading skill; capability
   independent of ρ), herding regression as the mechanism evidence (b2 by
   tier), fig2_v2 + fig3 references. Update the abstract (one sentence) and
   conclusion (one sentence) — the revised one-liner: cheap traders make
   LMSR destroy information; capable traders reach parity; no tested tier
   beats independent pooling. Add capability-axis caveats to Limitations:
   two new tiers only, one frontier vendor, Luna confounds capability with
   lower ρ (Pro control mitigates), same 40-question environment.
2. `README.md`: add the tier table row(s) to Results at a glance, update the
   headline caption if needed, add the v2 repro commands (configs under
   pnyx/configs/v2/, data under data/v2/), update stated cost ($0.73 v1 +
   $3.06 v2 ≈ $3.79 total), keep positioning discipline.
3. Fact-check pass against analysis_out/v2/*.md character-for-character
   (same protocol as P6), report the tally.

## Controller-only (not subagent tasks)

- Un-ignore and commit `data/v2/` before Task 2 (README states the release).
- Regenerate `analysis_out/v2/` from the merged Task 1 code (not the
  implementer's sanity run) before Task 2 starts.
- Final: run full suite, push to GitHub.
