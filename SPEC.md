# Pnyx — Pre-Registration

**Project:** When Do Prediction Markets Beat Static Pools? A Controlled Stress Test of LLM Belief Aggregation

**One-line goal:** With a synthetic information environment whose **Bayes-optimal posterior is known exactly**, measure how much of the available information (a) independent static pooling and (b) dynamic LMSR market trading extract from a pool of LLM agents — and how that gap moves with model-error correlation, trade dynamics, wealth persistence, and adversarial capital.

This document is the public pre-registration: the hypotheses and elicitation protocol below are frozen before any main experimental runs.

---

## Pre-registered hypotheses

Primary metric everywhere: **posterior gap** = |mechanism output − Bayes-optimal posterior given the union of shards|, plus Brier vs. realized outcome. Report mean ± std over 3 seeds; paired bootstrap + Wilcoxon for comparisons.

- **H1 (market vs. static, clean):** On info-asymmetric questions, the final market price has a smaller posterior gap than static pools (mean, median, log opinion pool, calibrated stack) computed over **independently elicited** pre-market beliefs from the *same agents on the same shards*. Direction unknown — a null or reversed result is a reportable finding.
- **H2 (correlation, replication + extension):** (a) Replication: mixed-model teams outperform homogeneous teams (expected, per monoculture paper). (b) Extension: using homogeneous teams of models A, B, C and a mixed A/B/C team **matched on standalone posterior gap**, the mixed team's advantage survives quality matching, and the market-vs-static gap (H1) shrinks as measured error correlation ρ rises. Report ρ per team from independent beliefs.
- **H3 (manipulation):** For an adversary with k× bankroll (k ∈ {1, 3, 10}) instructed to push the wrong side: report terminal Brier degradation, final-prediction flip rate, capital-per-0.1-price-move, adversary P&L, and post-attack recovery (rounds to within ε of pre-attack price). Hypothesis: flip rate increases with k and decreases with honest-pool signal strength; adversary P&L is negative at settlement (the market "taxes" manipulation).
- **H-wealth (ablation, secondary):** Persistent wealth across questions changes market accuracy vs. per-question reset; direction unknown; test order-dependence with shuffled question orders.

Secondary reporting: calibration curves, accuracy-per-dollar, parse-failure rates per model, market-maker subsidy, belief-convergence dynamics across rounds (uses in-market logged beliefs — dynamics analysis only, never as a baseline).

## Elicitation protocol (the contamination fix — follow exactly)

Two passes per (question, pool, seed):

- **Pass 1 — Independent elicitation.** Each agent sees: question + its shard(s) + persona. No prices, no other agents, no mention of a market. Output: `Belief` (prob + rationale). These beliefs feed ALL static baselines and the ρ estimates. One call per agent per question.
- **Pass 2 — Market.** Same agents, same shards, fresh context (may reference their Pass-1 belief in-prompt as their own prior). 3 rounds, randomized order per round, LMSR trading. In-market stated beliefs are logged for dynamics analysis only.

**Never compute a baseline from Pass-2 beliefs.** Enforce in code: baseline functions accept only `phase="independent"` records; assert.
