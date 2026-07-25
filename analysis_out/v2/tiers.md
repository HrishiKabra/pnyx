# Pnyx v2 — Capability-Tier Analysis

*Three homogeneous tiers on the same 40 questions × 3 seeds: flash (A), pro (A_PRO, same-family knowledge-vs-trading control), luna (A_LUNA). Primary metric = posterior gap vs. the Bayes oracle; deficit = market gap − mean-pool gap (positive ⇒ the market destroys information relative to independent pooling). Comparisons in the P5 stats format.*

## Tier metrics

| Tier | Pool gap | Market gap | Deficit (market − pool) | Market Brier | Team ρ | n |
|---|---|---|---|---|---|---|
| flash | 0.218 | 0.269 | 0.051 [-0.001, 0.100] (p = 0.080) | 0.281 | 0.398 ± 0.018 | 120 |
| pro | 0.227 | 0.224 | -0.003 [-0.048, 0.041] (p = 0.377) | 0.215 | 0.378 ± 0.029 | 120 |
| luna | 0.191 | 0.192 | 0.002 [-0.046, 0.050] (p = 0.451) | 0.201 | 0.283 ± 0.005 | 120 |

## Within-tier market vs. mean pool (deficit)

- **flash: market − mean pool (posterior gap)**: Δ = 0.051 [-0.001, 0.100] (95% bootstrap CI), Wilcoxon p = 0.080, per-seed means: 0.055/0.051/0.045
- **pro: market − mean pool (posterior gap)**: Δ = -0.003 [-0.048, 0.041] (95% bootstrap CI), Wilcoxon p = 0.377, per-seed means: -0.016/0.023/-0.016
- **luna: market − mean pool (posterior gap)**: Δ = 0.002 [-0.046, 0.050] (95% bootstrap CI), Wilcoxon p = 0.451, per-seed means: -0.014/0.000/0.019

## Cross-tier comparisons (paired on seed, question)

- **pro − flash (market gap)**: Δ = -0.044 [-0.098, 0.008] (95% bootstrap CI), Wilcoxon p = 0.127, per-seed means: -0.066/-0.005/-0.062
- **luna − pro (market gap)**: Δ = -0.032 [-0.078, 0.017] (95% bootstrap CI), Wilcoxon p = 0.139, per-seed means: -0.043/-0.065/0.011
- **luna − flash (market gap)**: Δ = -0.076 [-0.133, -0.013] (95% bootstrap CI), Wilcoxon p = 0.001, per-seed means: -0.109/-0.069/-0.050
- **pro − flash (pool gap)**: Δ = 0.010 [-0.011, 0.030] (95% bootstrap CI), Wilcoxon p = 0.289, per-seed means: 0.005/0.024/-0.000
- **luna − flash (pool gap)**: Δ = -0.027 [-0.050, -0.004] (95% bootstrap CI), Wilcoxon p = 0.000, per-seed means: -0.040/-0.018/-0.024

