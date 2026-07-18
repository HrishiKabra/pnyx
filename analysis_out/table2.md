## Table 2 — Parse Failures and Market-Maker Subsidy

### Parse-failure rate per (condition, model)

*Denominator = that model's total belief + trade LLM turns in the condition's own logs (post-retry, per Global Constraints §6.2). Model resolved via the condition's config YAML agent→model mapping.*

| Condition | Model | Parse-fails | Total turns | Rate |
|---|---|---|---|---|
| A | deepseek/deepseek-v4-flash | 0 | 2880 | 0.0000 |
| B1 | meta-llama/llama-3.1-8b-instruct | 0 | 2880 | 0.0000 |
| B3 | nvidia/nemotron-3-super-120b-a12b:free | 0 | 2880 | 0.0000 |
| C | deepseek/deepseek-v4-flash | 1 | 960 | 0.0010 |
| C | meta-llama/llama-3.1-8b-instruct | 0 | 960 | 0.0000 |
| C | nvidia/nemotron-3-super-120b-a12b:free | 0 | 960 | 0.0000 |
| D_K1 | deepseek/deepseek-v4-flash | 2 | 1080 | 0.0019 |
| D_K1 | meta-llama/llama-3.1-8b-instruct | 0 | 720 | 0.0000 |
| D_K1 | nvidia/nemotron-3-super-120b-a12b | 0 | 720 | 0.0000 |
| D_K3 | deepseek/deepseek-v4-flash | 0 | 1080 | 0.0000 |
| D_K3 | meta-llama/llama-3.1-8b-instruct | 0 | 720 | 0.0000 |
| D_K3 | nvidia/nemotron-3-super-120b-a12b | 0 | 720 | 0.0000 |
| D_K10 | deepseek/deepseek-v4-flash | 1 | 1080 | 0.0009 |
| D_K10 | meta-llama/llama-3.1-8b-instruct | 0 | 720 | 0.0000 |
| D_K10 | nvidia/nemotron-3-super-120b-a12b | 1 | 720 | 0.0014 |
| W_FIXED | deepseek/deepseek-v4-flash | 0 | 2160 | 0.0000 |
| W_SHUFFLED | deepseek/deepseek-v4-flash | 0 | 2160 | 0.0000 |

### Market-maker subsidy per condition

*Theoretical worst-case bound b·ln 2 ≈ 27.73 (b=40).*

| Condition | Mean subsidy | Total subsidy | Max subsidy | Bound (b·ln2) |
|---|---|---|---|---|
| A | 18.264 | 2191.689 | 27.726 | 27.73 |
| B1 | 20.859 | 2503.110 | 27.726 | 27.73 |
| B3 | 17.700 | 2124.042 | 27.726 | 27.73 |
| C | 17.537 | 2104.408 | 27.726 | 27.73 |
| D_K1 | 17.168 | 2060.158 | 27.644 | 27.73 |
| D_K3 | 18.396 | 2207.518 | 27.726 | 27.73 |
| D_K10 | 20.333 | 2439.999 | 27.726 | 27.73 |
| W_FIXED | 8.130 | 975.632 | 27.726 | 27.73 |
| W_SHUFFLED | 7.485 | 898.246 | 27.726 | 27.73 |
