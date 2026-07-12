# Pnyx — Progress

## Phase status

| Phase | Status | Notes |
|-------|--------|-------|
| P0 — Spec freeze | ✅ done (2026-07-13) | git init; SPEC.md (public §2 + §5 verbatim); .gitignore covers CLAUDE.md, .env, data/ |
| P1 — Engine + env | not started | market.py, env.py, schemas.py, MockAgent, oracle + kill-resume tests |
| P2 — Environment rendering | not started | generator + renderer + verifier; 40+10 questions |
| P3 — Pilot (~$1–2) | not started | 10 held-out questions, pool A, b-sweep, prompt freeze, matching table |
| P4 — Main runs | not started | full grid × 3 seeds |
| P5 — Analysis | not started | Fig 1, Fig 2, Tables 1–2, stats |
| P6 — Ship | not started | README, writeup, optional replay UI |

## Budget

- OpenRouter key currently capped at **$5** (account ~$11 credits); user can raise the cap if the pilot's cost ledger justifies it. Spec's $25 hard-stop is the absolute ceiling.
- Colab Pro available for the local vLLM arms (B1, C).

## Decisions log

- 2026-07-13: v2 spec adopted (posterior-oracle design, two-pass elicitation, peer prediction deferred to sequel). Pre-registration frozen in SPEC.md.

## Dependency additions beyond stdlib/pydantic/httpx/openai/numpy/pandas/matplotlib/vllm

(none yet — justify here if added)
