# Pnyx v2 — Herding Regression (dynamics-only)

*In-market stated belief y regressed on the agent's own Pass-1 belief (x1) and the pre-trade price (x2): y = a + b1·x1 + b2·x2, OLS via numpy.linalg.lstsq. b1 = own-information weight, b2 = price weight; 95% CIs are question-clustered bootstraps. Drift = mean|y − x1|. This is the spec's sanctioned dynamics analysis of in-market beliefs — it never feeds a static baseline or ρ estimate. Hypothesis: b2 falls with capability tier.*

## flash

| Scope | b1 (own info) | b2 (price) | R² | n | drift mean\|y−x1\| |
|---|---|---|---|---|---|
| pooled | 0.987 [0.958, 1.010] | 0.045 [0.028, 0.060] | 0.900 | 2160 | 0.037 |
| round 1 | 0.982 [0.940, 1.018] | 0.033 [0.001, 0.067] | 0.885 | 720 | 0.040 |
| round 2 | 0.999 [0.966, 1.025] | 0.035 [0.015, 0.055] | 0.911 | 720 | 0.036 |
| round 3 | 0.976 [0.941, 1.008] | 0.067 [0.041, 0.092] | 0.906 | 720 | 0.036 |

## pro

| Scope | b1 (own info) | b2 (price) | R² | n | drift mean\|y−x1\| |
|---|---|---|---|---|---|
| pooled | 0.988 [0.974, 0.998] | 0.010 [-0.000, 0.021] | 0.961 | 2160 | 0.012 |
| round 1 | 0.972 [0.948, 0.993] | 0.012 [-0.010, 0.035] | 0.936 | 720 | 0.015 |
| round 2 | 1.002 [0.996, 1.009] | 0.010 [-0.002, 0.020] | 0.982 | 720 | 0.011 |
| round 3 | 0.990 [0.967, 1.004] | 0.008 [-0.006, 0.020] | 0.966 | 720 | 0.012 |

## luna

| Scope | b1 (own info) | b2 (price) | R² | n | drift mean\|y−x1\| |
|---|---|---|---|---|---|
| pooled | 1.042 [1.031, 1.055] | 0.023 [0.011, 0.036] | 0.984 | 2160 | 0.033 |
| round 1 | 1.038 [1.027, 1.050] | 0.013 [-0.005, 0.028] | 0.984 | 720 | 0.030 |
| round 2 | 1.040 [1.028, 1.054] | 0.027 [0.011, 0.043] | 0.984 | 720 | 0.034 |
| round 3 | 1.047 [1.035, 1.061] | 0.024 [0.010, 0.039] | 0.985 | 720 | 0.035 |

