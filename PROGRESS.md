# Pnyx — Progress

## Phase status

| Phase | Status | Notes |
|-------|--------|-------|
| P0 — Spec freeze | ✅ done (2026-07-13) | git init; SPEC.md (public §2 + §5 verbatim); .gitignore covers CLAUDE.md, .env, data/ |
| P1 — Engine + env | ✅ done (2026-07-13) | market.py, env.py, schemas.py, MockAgent, oracle, runner.py + cli.py + configs/mock.yaml; 5-question mock end-to-end (0 API calls) + SIGKILL kill-resume acceptance test green; 114 tests pass under `-W error` |
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
- 2026-07-13: P1 closed after five task reviews + one whole-branch review (all approved). Post-review fixes applied: SettlementEvent.subsidy docstring reconciled with market.py's non-degenerate definition (worst-case payout − collected revenue; the naive C-delta-minus-revenue telescopes to 0); post-trade bankroll floored at 0 (full-clamp buys can leave ~−1e-13 float residue that would fail TradeView ge=0 validation).
- 2026-07-13: **Question-set design for P2+:** the 40+10 question dataset is generated ONCE (P2) and stored as versioned JSONL; every condition/seed loads the same file. Runtime question generation (and its per-condition seeding) is a MOCK-config-only path. This guarantees paired comparisons across conditions and seeds by construction.
- Deferred (from final review): parse-failure accounting + stronger no-network guard land with providers in P3; multi-seed/multi-condition runtime generation seeding left as-is since file-based datasets moot it.

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
