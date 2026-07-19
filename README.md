# Pnyx

### When Do Prediction Markets Beat Static Pools? A Controlled Stress Test of LLM Belief Aggregation

Pnyx is a synthetic information environment with a Bayes-optimal posterior
known exactly, used to measure how much of the available information
independent static pooling and dynamic LMSR market trading each recover from
a pool of LLM agents.

![Fig 1: market vs. static pools, posterior gap and Brier](analysis_out/fig1.png)

On the main grid, the market's final price (posterior gap 0.269 ± 0.012) is
**farther** from the Bayes-optimal posterior than every static pool built from
the same agents' independently elicited beliefs — including the best of the
four, the median pool (gap 0.187). This is a reversal of the naive
expectation, and it is the headline result of the paper.

## What this is

Each question has a binary latent state generated along with six
conditionally-independent signals of known accuracy, so the exact
Bayes-optimal posterior given any subset of signals can be computed in closed
form — an oracle to score mechanisms against, not just the realized outcome.
The same LLM agents, on the same evidence, are elicited twice per question:
once independently (feeding every static baseline) and once through a
3-round LMSR market. Comparing the two isolates the *mechanism* (pooling vs.
trading) from model knowledge and signal structure, which prior work on LLM
prediction markets does not separate. The result is a map of where markets
help, do nothing, or hurt, plus stress tests under model-error correlation,
adversarial capital, and wealth persistence.

Full write-up: [`docs/writeup.md`](docs/writeup.md). Pre-registered
hypotheses and protocol: [`SPEC.md`](SPEC.md).

## Reproduce the paper

Regenerate every figure, table, and statistic from the released event logs —
no API key, no network calls, a few seconds:

```bash
pip install -e .
python -m pnyx.cli analyze-main --data-root data/main --configs pnyx/configs/main --out analysis_out
```

Rerunning the experiments themselves from scratch needs an `OPENROUTER_KEY`
in `.env` and costs about $0.75 total:

```bash
python -m pnyx.cli run --config pnyx/configs/main/<COND>.yaml
```

one condition at a time (`A`, `B1`, `B3`, `C`, `D_k1`, `D_k3`, `D_k10`,
`W_fixed`, `W_shuffled` — see `pnyx/configs/main/`). Runs are resume-safe. The
`D_k*` and `W_*` conditions replay Pass-1 beliefs from a source condition's
logs rather than re-eliciting them, so that source must exist first: `D_k*`
reads from `C`, `W_*` reads from `A`. The released `data/main/` already
contains both, so a fresh clone can run any condition standalone.

Run the test suite:

```bash
pytest -W error
```

366 tests pass, including exact-value LMSR/oracle checks and a kill-and-resume
replay test.

## Results at a glance

| Hypothesis | Result |
|---|---|
| H1 — market vs. static pools | **Reversed.** Market posterior gap 0.269 ± 0.012 vs. best static pool (median) 0.187; significant vs. median (p=0.005), log opinion pool (p=0.048), and calibrated stack (p=0.008), directional vs. mean pool (p=0.080). |
| H2a — mixed vs. homogeneous teams | **Replication.** Quality-matched mixed team beats the weakest homogeneous arm by Δ=−0.155 (p<0.001) and leads all three homogeneous arms in point estimate. |
| H2b — phase map (ρ vs. market deficit) | **Descriptive** (n=12, narrow ρ range 0.378–0.514): Spearman r_s = 0.280. Mixed team C (team-mean ρ=0.442) is the only condition where the market ever beats the mean pool (2 of 3 seeds); the lowest-ρ team overall, homogeneous A (ρ=0.398), still runs a market deficit every seed. |
| H3 — adversarial manipulation | Flip rate rises 0.317 → 0.358 → 0.392 as adversary bankroll goes 1× → 3× → 10×; Brier degrades 0.105 → 0.312. Adversary P&L is negative at every multiple (−$47 / −$143 / −$440) — the market taxes manipulation even as predictions flip. |
| H-wealth — persistent bankroll | **Slightly hurts.** Posterior gap worsens vs. per-question reset (Δ≈0.05–0.06, p≈0.03); no order-dependence (fixed vs. shuffled, p=0.74). |

Reliability: 5 parse failures across the whole main grid (tens of thousands
of LLM turns, ≤0.19% per model per condition — Table 2). Total experiment
cost was $0.73 across all ledgers, including pilots not released here; the
released main grid alone cost $0.7255.

## Repo map

```
pnyx/
  schemas.py        # all Pydantic models: Belief, Trade, event-log records
  market.py         # LMSR engine — pure functions, no I/O
  env.py            # generative environment + exact Bayes posterior oracle
  agents.py         # Agent protocol + deterministic zero-intelligence MockAgent
  providers.py      # OpenAI-compatible client: rate limits, retries, cost ledger
  prompts.py        # versioned prompt templates + fixed persona catalogue
  runner.py         # two-pass elicitation loop; kill-safe event log + resume
  baselines.py      # mean / median / log-opinion-pool / calibrated-stack pools
  dataset.py        # deterministic question dataset build/merge/verify
  analysis/         # mainrun, figures, tables, manipulation, pilot analysis
  cli.py            # run / status / build-questions / analyze-pilot / analyze-main
  configs/          # per-condition YAML (models, seeds, b, budget)
data/main/          # released P4 event logs (JSONL) backing every result above
datasets/           # frozen 40-question + 10-pilot-question JSONL datasets
analysis_out/       # figures, tables, stats.md regenerated by analyze-main
tests/              # pytest suite (366 tests)
```

Limitations — thin markets, the narrow achieved ρ range, prompt/liquidity
sensitivity, synthetic prose rendering, and the single environment family —
are discussed in full in [`docs/writeup.md`](docs/writeup.md#6-limitations).

## Citing this work

If you build on this environment, protocol, or results, please cite:

> *When Do Prediction Markets Beat Static Pools? A Controlled Stress Test of
> LLM Belief Aggregation.* Pnyx project, 2026. Write-up: `docs/writeup.md`.
