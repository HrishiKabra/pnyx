## Table 1 — Posterior Gap, Brier, and Accuracy-per-Dollar

*Market vs. the mean-pool-of-independent-beliefs baseline, mean ± std over seeds. Accuracy-per-dollar = (0.25 − mean market Brier) / total cost USD; 0.25 is the uniform-guess (p=0.5) Brier. Condition "B2" of the pre-registration is condition A (identical pool, run once). D_K\* and W_\* are pass2-only: their mean-pool columns are bit-identical to their Pass-1 source by construction and are shown as "—" (marked †/‡) to avoid duplicate reporting — see the source condition's own row below.*

| Condition | Market gap | Market Brier | Mean-pool gap | Mean-pool Brier | Total cost | Accuracy/$ |
|---|---|---|---|---|---|---|
| A | 0.269 ± 0.012 | 0.281 ± 0.014 | 0.218 ± 0.009 | 0.202 ± 0.001 | $0.1184 | -0.262 |
| B1 | 0.380 ± 0.054 | 0.389 ± 0.031 | 0.280 ± 0.010 | 0.226 ± 0.004 | $0.0271 | -5.141 |
| B3 | 0.276 ± 0.028 | 0.267 ± 0.022 | 0.217 ± 0.007 | 0.203 ± 0.007 | $0.0000 | ∞ (free tier) |
| C | 0.225 ± 0.021 | 0.234 ± 0.043 | 0.226 ± 0.008 | 0.199 ± 0.003 | $0.0510 | 0.316 |
| D_K1 | 0.343 ± 0.003 | 0.339 ± 0.028 | — (†) | — (†) | $0.1123 | -0.790 |
| D_K3 | 0.435 ± 0.059 | 0.453 ± 0.063 | — (†) | — (†) | $0.1024 | -1.982 |
| D_K10 | 0.469 ± 0.010 | 0.546 ± 0.035 | — (†) | — (†) | $0.1099 | -2.690 |
| W_FIXED | 0.318 ± 0.039 | 0.285 ± 0.025 | — (‡) | — (‡) | $0.1028 | -0.338 |
| W_SHUFFLED | 0.326 ± 0.012 | 0.269 ± 0.022 | — (‡) | — (‡) | $0.1015 | -0.185 |

† pass2-only condition; mean-pool columns are bit-identical to (reused from) condition C's own row above.
‡ pass2-only condition; mean-pool columns are bit-identical to (reused from) condition A's own row above.
