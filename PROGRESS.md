# Pnyx — Progress

## Phase status

| Phase | Status | Notes |
|-------|--------|-------|
| P0 — Spec freeze | ✅ done (2026-07-13) | git init; SPEC.md (public §2 + §5 verbatim); .gitignore covers CLAUDE.md, .env, data/ |
| P1 — Engine + env | ✅ done (2026-07-13) | market.py, env.py, schemas.py, MockAgent, oracle, runner.py + cli.py + configs/mock.yaml; 5-question mock end-to-end (0 API calls) + SIGKILL kill-resume acceptance test green; 114 tests pass under `-W error` |
| P2 — Environment rendering | ✅ done (2026-07-13) | 40+10 questions rendered by 10 Sonnet 5 subagents, adversarially verified to 100% pass (2 fix rounds); datasets/questions_v1.jsonl + pilot frozen |
| P3 — Pilot | ✅ done (2026-07-13) | b=40 + prompts p3-v3 frozen; 0% parse-fail; ~$0.09 spent; matching table partial (model C blocked on credits, model A pending Colab) |
| P4 — Main runs | ✅ done (2026-07-18) | all 9 conditions (A, B1, B3, C, D_k1/k3/k10, W_fixed/shuffled) × 3 seeds complete, 0 turns remaining; main-run spend ≈ $0.63; matching pilot (pilot_free_b40) complete |
| P5 — Analysis | ✅ done (2026-07-19) | analysis pipeline (baselines, mainrun, manipulation, figures, tables) + `analyze-main` CLI; deliverables in analysis_out/; 366 tests green under `-W error`; final whole-branch review: ready to ship |
| P6 — Ship | not started | README, writeup, optional replay UI |

## Budget

- OpenRouter key currently capped at **$5** (account ~$11 credits); user can raise the cap if the pilot's cost ledger justifies it. Spec's $25 hard-stop is the absolute ceiling.
- Colab Pro available for the local vLLM arms (B1, C).

## Decisions log

- 2026-07-13: v2 spec adopted (posterior-oracle design, two-pass elicitation, peer prediction deferred to sequel). Pre-registration frozen in SPEC.md.
- 2026-07-13: P1 closed after five task reviews + one whole-branch review (all approved). Post-review fixes applied: SettlementEvent.subsidy docstring reconciled with market.py's non-degenerate definition (worst-case payout − collected revenue; the naive C-delta-minus-revenue telescopes to 0); post-trade bankroll floored at 0 (full-clamp buys can leave ~−1e-13 float residue that would fail TradeView ge=0 validation).
- 2026-07-13: **Question-set design for P2+:** the 40+10 question dataset is generated ONCE (P2) and stored as versioned JSONL; every condition/seed loads the same file. Runtime question generation (and its per-condition seeding) is a MOCK-config-only path. This guarantees paired comparisons across conditions and seeds by construction.
- Deferred (from final review): parse-failure accounting + stronger no-network guard land with providers in P3; multi-seed/multi-condition runtime generation seeding left as-is since file-based datasets moot it.
- 2026-07-13 (P3 freeze): **b = 40 frozen** (sweep: b20 degenerate 20-100% of questions, b80 sluggish/no advantage, b40 non-degenerate at 10%); **prompts frozen at p3-v3** (v1→v2: fixed stray literal quote in trade format + rationale char limit + no-fences instruction, parse-fail 5.4%→0%; v2→v3: LMSR price-impact sizing guidance — median trade dropped from ~100% to 18% of cash, market Brier 0.344→0.137 at b=40, now beating mean-pool 0.205). **reasoning_enabled=false frozen for deepseek-v4-flash** (reasoning-hybrid null-content failures + 3× output cost otherwise). Pilot total spend ≈ $0.09 across 9 runs incl. iterations.
- 2026-07-13 (matching table, per-model standalone posterior gap on pilot questions): deepseek-v4-flash **0.268** (p3-v3, b40 run). Model C (qwen3-next-80b free) **BLOCKED**: OpenRouter lifetime credits $4.99 < $10 → free tier capped ~50 req/day; needs user top-up or paid swap. Model A (local vLLM) pending Colab setup (P4 prep). Condition-C quality matching completes when those columns land.
- 2026-07-15 (matching table update): model A = hosted **llama-3.1-8b-instruct**, standalone gap **0.291** (pilot_llama_b40, 0/240 parse-fails, market metrics noisier as expected for an 8B: gap 0.415, degeneracy 40%). Model B = deepseek-v4-flash **0.268**. Model C = nemotron-3-super-120b:free — gap pending pilot_free_b40 (queued after C/D to keep the free lane clear). Standalone gaps are closely matched (0.268 vs 0.291) — quality-matching for condition C holds; C's mixed 2+2+2 composition proceeds as configured.
- 2026-07-18 (P4 complete): all 9 (condition, seed) grids settled — A/B1/B3/C: 1000/1000 turns per seed; D_k1/k3/k10: 880/880; W_fixed/shuffled: 760/760. Parse-failures across the whole grid: 7 (5× deepseek-v4-flash, 2× nemotron), all counted. Main-run spend ≈ $0.63 (A $0.118, B1 $0.027, B3 $0.000, C $0.051, D $0.312, W $0.204) — well under the $5 key cap.
- 2026-07-18 (matching table COMPLETE): model C = nemotron-3-super-120b, standalone gap **0.2569** (pilot_free_b40, 7/240 parse-fails = 2.9% post-retry, market gap 0.2054/Brier 0.1874, degeneracy 10%). Final columns: A llama-3.1-8b **0.291**, B deepseek-v4-flash **0.268**, C nemotron **0.257** — closely matched; condition-C quality matching holds.
- 2026-07-18 (ops note): the runner parks a question **silently** (exit 0, no output) when a provider call fails — including a missing `OPENROUTER_KEY` env var (runner reads the environment; it does not load `.env`). A resumed run that exits instantly with unchanged status likely has no key in its shell: `set -a; source .env; set +a` first. Consider a one-line `[parked]` print before P5.
- 2026-07-13 (pilot observation, NOT a confirmed finding): at b=40 the market's final-price Brier (0.137) beat the mean pool (0.205) while its posterior GAP (0.275) was slightly worse than the pool's (0.218) — the market commits harder to the realized side; H1 adjudication awaits the main runs (40 q × 3 seeds).

## P5 headline results (full stats in analysis_out/stats.md; deterministic under --bootstrap-seed 0)

- **H1 REVERSED on condition A:** the market's final price has a LARGER posterior gap than every static pool (market 0.269±0.012 vs mean 0.218, median 0.187, LOP 0.209, calibrated stack 0.196). Significant vs median (Δ=0.081, p=0.005), LOP (p=0.048), stack (p=0.008); vs mean pool Δ=0.051, p=0.080. Brier deltas same direction, not individually significant (p≥0.085). Pre-registration allowed this direction ("a null or reversed result is a reportable finding") — the paper's headline is that LMSR trading DESTROYS information relative to independent pooling in this regime.
- **H2a (replication holds):** mixed team C beats every homogeneous arm on market gap (vs B1 llama: Δ=−0.155, p<0.001; vs A deepseek: −0.044, p=0.106; vs B3 nemotron: −0.051, p=0.118).
- **H2b (phase map, descriptive n=12):** Spearman r_s = 0.280 between team ρ (range 0.38–0.51) and market-vs-mean-pool deficit; C (lowest-ρ mixed team) is the only condition with seeds below zero (market helps). Direction consistent with the hypothesis; the ρ range achieved is narrow.
- **H3:** flip rate rises with k (0.317 → 0.358 → 0.392); Brier degradation rises (0.105 → 0.219 → 0.312); adversary P&L is negative at every k (−$47 / −$143 / −$440 — the market taxes manipulation, and the absolute tax grows with capital). Recovery fraction is low (0.09–0.20). **CAVEAT for writeup:** median_recovery_rounds is structurally 0 — the adversary trades in every round including the last, so recovery timing is unmeasurable by design; report recovery FRACTION only. (An early "positive adversary P&L at k≥3" observation was an artifact of a wrong −100 baseline; corrected to k×100 starting bankroll.)
- **H-wealth:** persistent wealth slightly WORSENS market gap vs per-question reset (W_FIXED−A Δ=0.050, p=0.035; W_SHUFFLED−A Δ=0.058, p=0.032); no order-dependence (fixed vs shuffled p=0.74).
- Table 1/Table 2: total grid spend $0.7255; max market-maker subsidy 27.726 ≤ b·ln2 bound 27.73; parse failures 7 total across the grid.

## Dependency additions beyond stdlib/pydantic/httpx/openai/numpy/pandas/matplotlib/vllm

- **pyyaml** (P1): run configs are YAML (`pnyx/configs/mock.yaml`); the CLI loads
  and validates them into `RunConfig`. Standard, pure-config dependency; no
  code execution risk (`yaml.safe_load` only). Approved in Global Constraints
  P1 dependency list.

## P1 deviations / design decisions (Task 5)

- **CLI entry:** `python -m pnyx.cli {run,status} --config <yaml> [--data-dir <dir>]`.
  Module invocation via a `main()` guard — no `__main__.py`, no
  `[project.scripts]` console script (keeps the package import-only, no install
  step for tests). `--data-dir` overrides the config's `data_dir`.
- **Log layout under `data/`:** one event log `{condition}_seed{seed}.jsonl` and
  one persisted-questions sidecar `{condition}_seed{seed}.questions.jsonl` per
  (condition, seed). Questions are generated once and reloaded on resume so
  rejection-sampling nondeterminism can never fork a resumed run.
- **Seeding scheme:** all RNG derives from the integer seed via
  `SeedSequence([seed, sha256(stable_tuple)])`, keyed per purpose
  (questions / belief / shuffle / trade). SHA-256 (not builtin salted `hash`)
  keeps it stable across processes — required for the kill-resume guarantee.
- **`RunConfig` added to `schemas.py`** (single-source-of-truth constraint).
- **Test-only fault injection:** `PNYX_TURN_DELAY` env var adds a per-live-turn
  sleep so the kill-resume test reliably catches a run mid-flight; zero effect
  when unset.
- **Async seam:** turn execution is `async def` driven by `asyncio.run`; P1 is
  not concurrent (MockAgent is sync) but a P3 awaited provider slots in without
  restructuring. No providers / rate limiting / cost ledger / baselines built.

## P2 dataset

- questions verified: 50
- shards checked: 300
- direction pass rate: 1.0000
- leak pass rate: 1.0000
- meta pass rate: 1.0000
- question pass rate: 1.0000
- all pass: True
- assembled: questions_v1.jsonl (40) + questions_pilot_v1.jsonl (10)
- render/verify provenance: 10 parallel renderer subagents (Sonnet 5), 10 independent
  verifier subagents (Sonnet 5, adversarial). Round-1 question pass rate 80% (10/50
  failed: neutral shards, standalone-direction failures, duplicate-fact leaks,
  register asymmetries, one physical incoherence); two surgical re-render rounds with
  fresh verifiers brought all 50 to pass. Shard-level round-1 pass rates: direction
  97.3%, leak 99.0%, meta 99.0%.
- manual spot-check (controller): 10 shards across q002/q010/q027/q033/q044 read and
  judged individually — directions entailed standalone, no cross-shard leaks, uniform
  register. Passed.
- single-shard non-sufficiency: enforced numerically at generation (min single-shard
  margin ≥ 0.1 vs posterior|all, checked against the oracle; margins stored in each
  record's meta and re-verified in tests).

## P3 pilot

```
run                 condition      settled  degeneracy  market_gap  market_brier  pool_gap  pool_brier  total_cost  turns  
------------------  -------------  -------  ----------  ----------  ------------  --------  ----------  ----------  -------
data/pilot_b20_pv3  PILOT_B20_PV3  10/10    0.2000      0.4455      0.5084        0.2231    0.2083      $0.0097     250/250
data/pilot_b40_pv3  PILOT_B40_PV3  10/10    0.1000      0.2752      0.1366        0.2175    0.2051      $0.0097     250/250
data/pilot_b80_pv3  PILOT_B80_PV3  10/10    0.0000      0.2681      0.2024        0.2284    0.2409      $0.0098     250/250

-- data/pilot_b20_pv3 --
  parse-fail rate by model (post-retry):
    deepseek/deepseek-v4-flash: 0.0000 (0/240)
  standalone gap by model (mean |Pass-1 prob - posterior_all|):
    deepseek/deepseek-v4-flash: 0.2717
  cost by model:
    deepseek/deepseek-v4-flash: $0.0097

-- data/pilot_b40_pv3 --
  parse-fail rate by model (post-retry):
    deepseek/deepseek-v4-flash: 0.0000 (0/240)
  standalone gap by model (mean |Pass-1 prob - posterior_all|):
    deepseek/deepseek-v4-flash: 0.2678
  cost by model:
    deepseek/deepseek-v4-flash: $0.0097

-- data/pilot_b80_pv3 --
  parse-fail rate by model (post-retry):
    deepseek/deepseek-v4-flash: 0.0000 (0/240)
  standalone gap by model (mean |Pass-1 prob - posterior_all|):
    deepseek/deepseek-v4-flash: 0.2786
  cost by model:
    deepseek/deepseek-v4-flash: $0.0098
```
