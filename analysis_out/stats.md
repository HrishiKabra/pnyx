# Pnyx P5 Statistics Report

## H1: Market vs. Static Pools (Condition A)

### Market vs. static pools — posterior gap
- **market − Mean pool (posterior gap)**: Δ = 0.051 [-0.001, 0.100] (95% bootstrap CI), Wilcoxon p = 0.080, per-seed means: 0.055/0.051/0.045
- **market − Median pool (posterior gap)**: Δ = 0.081 [0.038, 0.123] (95% bootstrap CI), Wilcoxon p = 0.005, per-seed means: 0.084/0.082/0.078
- **market − Log opinion pool (posterior gap)**: Δ = 0.060 [0.008, 0.109] (95% bootstrap CI), Wilcoxon p = 0.048, per-seed means: 0.057/0.064/0.060
- **market − Calibrated stack (posterior gap)**: Δ = 0.073 [0.019, 0.122] (95% bootstrap CI), Wilcoxon p = 0.008, per-seed means: 0.072/0.076/0.071

### Market vs. static pools — Brier
- **market − Mean pool (Brier)**: Δ = 0.079 [0.026, 0.130] (95% bootstrap CI), Wilcoxon p = 0.196, per-seed means: 0.066/0.099/0.071
- **market − Median pool (Brier)**: Δ = 0.089 [0.039, 0.136] (95% bootstrap CI), Wilcoxon p = 0.085, per-seed means: 0.073/0.111/0.083
- **market − Log opinion pool (Brier)**: Δ = 0.077 [0.023, 0.128] (95% bootstrap CI), Wilcoxon p = 0.170, per-seed means: 0.061/0.097/0.073
- **market − Calibrated stack (Brier)**: Δ = 0.074 [0.001, 0.138] (95% bootstrap CI), Wilcoxon p = 0.148, per-seed means: 0.058/0.095/0.068

## H2a: Mixed Team (C) vs. Homogeneous Arms

- **C − A (B2) (market_gap)**: Δ = -0.044 [-0.088, 0.000] (95% bootstrap CI), Wilcoxon p = 0.106, per-seed means: -0.032/-0.047/-0.053
- **C − B1 (market_gap)**: Δ = -0.155 [-0.211, -0.100] (95% bootstrap CI), Wilcoxon p = 0.000, per-seed means: -0.121/-0.236/-0.109
- **C − B3 (market_gap)**: Δ = -0.051 [-0.107, 0.004] (95% bootstrap CI), Wilcoxon p = 0.118, per-seed means: 0.005/-0.051/-0.107

## H2b: Team ρ vs. Market Advantage (Phase-Map Correlation)

| Condition | Seed | ρ | market_gap − mean_gap |
|---|---|---|---|
| A | 0 | 0.421 | 0.055 |
| A | 1 | 0.378 | 0.051 |
| A | 2 | 0.395 | 0.045 |
| B1 | 0 | 0.514 | 0.092 |
| B1 | 1 | 0.487 | 0.159 |
| B1 | 2 | 0.452 | 0.050 |
| B3 | 0 | 0.503 | 0.022 |
| B3 | 1 | 0.442 | 0.055 |
| B3 | 2 | 0.482 | 0.098 |
| C | 0 | 0.438 | 0.036 |
| C | 1 | 0.437 | -0.010 |
| C | 2 | 0.451 | -0.029 |

Spearman rank correlation (ρ vs. market_gap − mean_gap) across n=12 (condition, seed) points: r_s = 0.280 (n=12 → descriptive only, not a hypothesis test).

## H3: Adversary Manipulation

| k | flip_rate | wrong_side_rate | mean_capital_per_01_move | capital_nan_count | mean_adv_pnl | recovery_fraction | median_recovery_rounds | brier_degradation |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3167 | 0.5083 | 44.2469 | 0 | -46.5985 | 0.1583 | 0.0000 | 0.1049 |
| 3 | 0.3583 | 0.6167 | 30.0622 | 0 | -143.0562 | 0.2000 | 0.0000 | 0.2191 |
| 10 | 0.3917 | 0.6583 | 306.4619 | 1 | -440.4081 | 0.0917 | 0.0000 | 0.3119 |

## H-wealth: Persistent Wealth Order Effects

- **W_FIXED − A (market_gap)**: Δ = 0.050 [-0.001, 0.098] (95% bootstrap CI), Wilcoxon p = 0.035, per-seed means: 0.047/0.098/0.005
- **W_FIXED − A (market_brier)**: Δ = 0.004 [-0.076, 0.077] (95% bootstrap CI), Wilcoxon p = 0.394, per-seed means: 0.020/0.011/-0.020
- **W_SHUFFLED − A (market_gap)**: Δ = 0.058 [0.007, 0.111] (95% bootstrap CI), Wilcoxon p = 0.032, per-seed means: 0.049/0.050/0.075
- **W_SHUFFLED − A (market_brier)**: Δ = -0.012 [-0.086, 0.057] (95% bootstrap CI), Wilcoxon p = 0.590, per-seed means: 0.026/-0.033/-0.030
- **W_FIXED − W_SHUFFLED (market_gap)**: Δ = -0.008 [-0.050, 0.034] (95% bootstrap CI), Wilcoxon p = 0.737, per-seed means: -0.001/0.048/-0.071
- **W_FIXED − W_SHUFFLED (market_brier)**: Δ = 0.016 [-0.033, 0.067] (95% bootstrap CI), Wilcoxon p = 0.521, per-seed means: -0.006/0.044/0.010

