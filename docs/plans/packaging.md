# Packaging plan — license/citation/CI, Streamlit replay, arXiv-style PDF

## Context

pnyx is complete through v2 (see PROGRESS.md); repo is public at
https://github.com/HrishiKabra/pnyx. This wave packages it for sharing
outside GitHub. Author credit everywhere: **Hrishi Kabra** (no affiliation,
no ORCID). Zero API cost; nothing here touches experiment data or analysis
numbers. Zenodo DOI is controller-owned and blocked on the user linking
their account (release + badge happen after).

## Global Constraints

- Public hygiene and positioning discipline as always (no dev-workflow
  content; no novelty overclaims in any new prose).
- Test suite (currently 384, `pytest -W error`) must stay green. Streamlit
  must be an OPTIONAL dependency — core `pip install -e .` + tests must work
  without it (extras group; import only inside the app entrypoint).
- No new required dependencies beyond: streamlit (optional extra "replay").
  PDF build uses the system pandoc (3.8.3) + xelatex — repo gains only a
  build script + LaTeX assets, not a dependency.
- Determinism/repro story preserved: every command added to README must be
  run verbatim before committing.
- Commit style: `packaging: ...` + the session trailers.

## Task 1: LICENSE, CITATION.cff, CI, README integration

1. `LICENSE` — MIT, "Copyright (c) 2026 Hrishi Kabra". Add a short
   "Data and figures" note at the bottom of README's citation section (not
   in the LICENSE file itself): released event logs (data/main, data/v2),
   the question dataset, and analysis_out figures are CC BY 4.0.
2. `CITATION.cff` — cff-version 1.2.0; type software; authors: [{family-names:
   Kabra, given-names: Hrishi}]; title "Pnyx: when do prediction markets
   beat static pools? A controlled stress test of LLM belief aggregation";
   repository-code URL; license MIT; keywords (prediction-markets, LMSR,
   LLM-agents, information-aggregation, belief-aggregation,
   multi-agent); version 1.0.0; date-released 2026-07-26. Validate the YAML
   parses (python -c yaml.safe_load).
3. `.github/workflows/ci.yml` — on push + pull_request to main: ubuntu-latest,
   Python 3.12, `pip install -e .`, `pytest -W error -q`. No API keys exist
   in CI: confirm from the test suite's docstrings/structure that tests are
   offline (they are — P1 acceptance was zero-API-call; do not weaken that).
   Add a matplotlib-agg env guard only if needed (figures.py already forces
   Agg). Badge in README top: actions workflow badge.
4. README: badge row (CI + license), "Citing" section with the BibTeX-style
   snippet matching CITATION.cff, and the CC BY 4.0 data note. Do not alter
   any results prose or numbers.
5. Verify: push nothing; commit only. (Controller pushes and confirms the
   Action goes green.)

## Task 2: Streamlit price-path replay

Purpose: the demo that shows the paper's two stories live — (a) price paths
with herding (flash chases price, pro doesn't), (b) adversary attacks.

1. Pure data layer `pnyx/analysis/replay.py` (tested, no streamlit import):
   - `list_runs(data_roots: list[Path]) -> list[RunRef]` — discover
     (condition, seed) pairs from `data/main` + `data/v2`.
   - `question_replay(events, questions, qid) -> QuestionReplay` dataclass:
     ordered trade steps (agent_id, round, action, executed_shares, cost,
     price_before, price_after, stated belief, parse_failed, adversary
     flag adv0), the Pass-1 belief per honest agent (from the same
     condition's independent pass — for pass2-only conditions D/W resolve
     from their source condition per the runner convention D←C, W←A),
     posterior_all, outcome, final_price, subsidy.
   - Pure helpers for the two derived views: price series with round
     boundaries; per-agent (pass1, stated-by-round, price-at-trade) triples.
2. App `replay/streamlit_app.py` (thin; imports streamlit lazily; every
   number comes from the data layer):
   - Sidebar: condition → seed → question selectors; data-root default
     `data/`.
   - Main: price path (step chart, y=[0,1], posterior_all dashed line,
     outcome marker, round boundaries shaded); trades table; adversary
     trades highlighted (D conditions) with pre-attack price line.
   - "Herding" panel: per honest agent, stated belief per round vs their
     own Pass-1 belief vs price at trade time (small multiples or one
     chart with agent selector).
   - Caption under each panel: one factual sentence, no claims beyond the
     writeup's.
3. `pyproject.toml`: `[project.optional-dependencies] replay = ["streamlit"]`.
   README section "Replay UI": `pip install -e ".[replay]"` then
   `streamlit run replay/streamlit_app.py`. Run the app once locally
   (headless `streamlit run --server.headless true` smoke, curl the port,
   kill it) to prove the command works; report evidence.
4. Tests for the data layer only (tests/test_replay.py): discovery on a tmp
   mini-grid; replay assembly exact-values on a hand-built event sequence
   (incl. adversary flag + D-condition pass1 resolution via source dir);
   streamlit NOT imported by the test module or by pnyx.analysis.replay.

## Task 3: arXiv-style PDF

1. `paper/` directory: `build.sh` (pandoc writeup.md → xelatex PDF with an
   arXiv-ish look: 11pt, single column, standard article class, authblk
   for the author line "Hrishi Kabra", date 2026-07-26, abstract pulled
   from the writeup's abstract section, numbered sections, hyperref) +
   any small header.tex needed. Output `paper/pnyx.pdf` (committed — it is
   a shareable artifact).
2. Source of truth stays docs/writeup.md — build.sh converts it; do NOT
   fork the prose. Allowed mechanical transforms in the script/filter:
   title/author/abstract front-matter injection, image paths resolved to
   analysis_out/*, markdown tables → booktabs, the References section →
   plain bibliography formatting. If a construct won't convert cleanly,
   fix the markdown in a conversion-safe way (no meaning changes; any such
   edit listed in the report).
3. Figures embedded: fig1, fig2 or fig2_v2, fig3 (use the PDF variants for
   vector quality).
4. Acceptance: `bash paper/build.sh` runs clean from repo root on this
   machine (pandoc 3.8.3, /Library/TeX/texbin/xelatex); PDF opens, ~9–14
   pages, all figures render, no missing-character boxes (watch ρ, ±, −,
   §, ✅-type glyphs — strip emoji if any); README gets a one-line pointer.
   Controller visually inspects the PDF before push.
