"""Pnyx P3 pilot analysis — aggregates ``pnyx.runner`` event logs, cost
ledgers, and question sidecars (already written on disk by a controller-run
pilot) into the matching-table ingredients from the P3 brief: per-model
parse-fail rate, price degeneracy, market vs. oracle posterior gap/Brier,
a mean-pool-of-Pass-1 baseline, per-model standalone gap, and cost.

Zero API calls. This module never imports ``pnyx.providers`` / ``httpx`` and
never opens a socket — it only reads files a run already produced under
``data/<run>/``. See ``pnyx.cli``'s ``analyze-pilot`` subcommand for the CLI
entry point.

Layout this module expects inside one run directory (``pnyx.runner``'s
naming convention; see ``runner.log_path_for`` / ``questions_path_for`` /
``costs_path_for``):

* ``{condition}_seed{seed}.jsonl``            — one or more event logs.
* ``{condition}_seed{seed}.questions.jsonl``  — the per-seed questions
  sidecar (the oracle: ``QuestionRecord.posterior_table``).
* ``{condition}.costs.jsonl``                 — the cumulative cost ledger.

Every reader here is tolerant of a torn final line exactly like the runner's
own replay (a write killed mid-line): only the last line of a file may fail
to parse, and only when the file doesn't end in a newline.

Scientific-integrity guard
---------------------------
Global Constraints: "Never compute baselines from Pass-2 beliefs." The
mean-pool-of-Pass-1 baseline and the per-model standalone gap both draw on
``BeliefEvent``s, which in a real run are always Pass-1 (``phase ==
"independent"``) — but nothing in the schema itself prevents a corrupted or
hand-edited log from smuggling in a market-phase belief. ``_assert_pass1``
/ ``_mean_pool_prob`` assert this loudly rather than silently letting such
contamination skew a baseline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

from pnyx.dataset import replace_progress_section
from pnyx.env import subset_key
from pnyx.schemas import BeliefEvent, CostEntry, QuestionRecord, parse_event

__all__ = [
    "RunAnalysis",
    "analyze_run",
    "format_report",
    "progress_section",
    "write_progress",
]

_PROGRESS_HEADING = "## P3 pilot"
_DEGENERATE_LO = 0.05
_DEGENERATE_HI = 0.95

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Tolerant JSONL reading (mirrors pnyx.runner.read_events / CostLedger)
# ---------------------------------------------------------------------------


def _tolerant_jsonl(path: Path, parse: Callable[[str], T]) -> list[T]:
    """Parse every line of ``path`` with ``parse``, tolerating a truncated
    final partial line (the kill-safe write contract shared with the event
    log and cost ledger): only the LAST line may fail to parse, and only
    when the file does not end in a newline. A malformed non-final line is
    a hard error. Missing/empty file -> ``[]``."""
    if not path.exists():
        return []
    text = path.read_text()
    if not text:
        return []
    ended_newline = text.endswith("\n")
    lines = text.splitlines()
    out: list[T] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(parse(line))
        except Exception:
            if idx == len(lines) - 1 and not ended_newline:
                break  # truncated final write — ignore (kill-safe contract)
            raise
    return out


def _read_events(path: Path) -> list:
    return _tolerant_jsonl(path, parse_event)


def _read_questions(path: Path) -> list[QuestionRecord]:
    return _tolerant_jsonl(path, QuestionRecord.model_validate_json)


def _read_costs(path: Path) -> list[CostEntry]:
    return _tolerant_jsonl(path, CostEntry.model_validate_json)


def _discover_files(run_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Classify every ``*.jsonl`` directly inside ``run_dir`` by filename
    convention into (event logs, question sidecars, cost ledgers)."""
    event_logs: list[Path] = []
    sidecars: list[Path] = []
    ledgers: list[Path] = []
    if run_dir.exists():
        for p in sorted(run_dir.glob("*.jsonl")):
            name = p.name
            if name.endswith(".costs.jsonl"):
                ledgers.append(p)
            elif name.endswith(".questions.jsonl"):
                sidecars.append(p)
            else:
                event_logs.append(p)
    return event_logs, sidecars, ledgers


# ---------------------------------------------------------------------------
# Oracle lookup
# ---------------------------------------------------------------------------


def _full_subset_key(record: QuestionRecord) -> str:
    """The posterior_table key for ALL of a question's signals (the oracle
    "posterior_all" the market/pool baselines are compared against)."""
    return subset_key(s.index for s in record.signals)


def _posterior_all(record: QuestionRecord) -> float | None:
    return record.posterior_table.get(_full_subset_key(record))


# ---------------------------------------------------------------------------
# Phase-guarded Pass-1 aggregation
# ---------------------------------------------------------------------------


def _assert_pass1(e: BeliefEvent) -> None:
    """Guard: ``e`` must be a Pass-1 (``phase == "independent"``) belief.
    Raises loudly on contamination rather than silently skewing a baseline
    (see module docstring)."""
    assert e.key.phase == "independent", (
        "phase guard violated: belief event has phase="
        f"{e.key.phase!r} (expected 'independent') at key={e.key.as_tuple()}"
    )


def _pass1_beliefs(events: list) -> list[BeliefEvent]:
    """Every ``BeliefEvent`` in ``events``, guard-checked (see
    ``_assert_pass1``). ``type == "belief"`` events are Pass-1 by
    construction in every real run, but this guard fires immediately on a
    hand-edited/corrupted log that tags one ``phase="market"``."""
    out = [e for e in events if e.type == "belief"]
    for e in out:
        _assert_pass1(e)
    return out


def _mean_pool_prob(belief_events: list[BeliefEvent]) -> float:
    """Simple mean of a set of Pass-1 belief probabilities (the mean-pool
    baseline). Re-asserts the phase guard itself so this function is safe to
    call directly (not only via ``_pass1_beliefs``)."""
    assert belief_events, "no belief events to pool"
    for e in belief_events:
        _assert_pass1(e)
    return sum(e.belief.prob for e in belief_events) / len(belief_events)


# ---------------------------------------------------------------------------
# Best-effort turn-shape inference (no run config is available here)
# ---------------------------------------------------------------------------


def _infer_shape(beliefs: list[BeliefEvent], trades: list) -> tuple[int, int]:
    """``(n_agents, n_rounds)`` inferred from the log itself: ``n_agents`` is
    the largest number of distinct agent_ids seen giving a Pass-1 belief for
    any single question; ``n_rounds`` is the largest trade round number
    seen. Best-effort only (``analyze-pilot`` takes no ``--config``) — an
    early-killed run under-counts both, which under-counts ``turns_expected``
    too; that is an accepted limitation of inferring shape from partial data.
    """
    agents_per_q: dict[str, set[str]] = {}
    for e in beliefs:
        agents_per_q.setdefault(e.key.question_id, set()).add(e.key.agent_id)
    n_agents = max((len(v) for v in agents_per_q.values()), default=0)
    n_rounds = max((e.key.round for e in trades), default=0)
    return n_agents, n_rounds


# ---------------------------------------------------------------------------
# RunAnalysis
# ---------------------------------------------------------------------------


@dataclass
class RunAnalysis:
    """Everything ``analyze_run`` extracts from one pilot run directory."""

    run_dir: str
    condition: str | None = None
    n_questions_dataset: int = 0
    n_settled: int = 0
    unsettled_question_ids: list[str] = field(default_factory=list)
    degeneracy_rate: float | None = None
    degenerate_count: int = 0
    market_gap: float | None = None
    market_brier: float | None = None
    pool_gap: float | None = None
    pool_brier: float | None = None
    # model_id -> (parse_failed count, total belief+trade turns), post-retry.
    parse_fail_by_model: dict[str, tuple[int, int]] = field(default_factory=dict)
    # model_id -> mean |Pass-1 prob - oracle posterior_all|.
    standalone_gap_by_model: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0
    turns_done: int = 0
    turns_expected: int = 0
    notes: list[str] = field(default_factory=list)


def analyze_run(run_dir: str | Path) -> RunAnalysis:
    """Aggregate one pilot run directory into a :class:`RunAnalysis`.

    Never raises on an incomplete/not-yet-started run (missing files just
    produce zeros/None + a note) EXCEPT the phase guard (``_assert_pass1``),
    which is a hard scientific-integrity invariant, not a completeness
    concern, and a torn NON-final JSONL line (a real corruption, not the
    kill-safe truncated-tail case).
    """
    run_dir = Path(run_dir)
    result = RunAnalysis(run_dir=str(run_dir))

    event_paths, sidecar_paths, ledger_paths = _discover_files(run_dir)
    if not event_paths:
        result.notes.append(
            f"no event log found in {run_dir} (run not started, or wrong path)"
        )

    events: list = []
    for p in event_paths:
        events.extend(_read_events(p))

    questions: dict[str, QuestionRecord] = {}
    for p in sidecar_paths:
        for rec in _read_questions(p):
            questions[rec.question_id] = rec
    if event_paths and not sidecar_paths:
        result.notes.append(
            f"no questions sidecar found in {run_dir}; oracle-posterior gap/Brier "
            "unavailable"
        )

    cost_entries: list[CostEntry] = []
    for p in ledger_paths:
        cost_entries.extend(_read_costs(p))
    if event_paths and not ledger_paths:
        result.notes.append(
            f"no cost ledger found in {run_dir} (mock-only run, or not started)"
        )

    if events:
        first = events[0]
        result.condition = first.condition if first.type == "settlement" else first.key.condition

    beliefs = _pass1_beliefs(events)  # phase-guarded (raises on contamination)
    trades = [e for e in events if e.type == "trade"]
    settlements = [e for e in events if e.type == "settlement"]

    # -- dataset size + unsettled bookkeeping -------------------------------
    if questions:
        result.n_questions_dataset = len(questions)
        all_qids = set(questions.keys())
    else:
        all_qids = (
            {e.question_id for e in settlements}
            | {e.key.question_id for e in beliefs}
            | {e.key.question_id for e in trades}
        )
        result.n_questions_dataset = len(all_qids)

    result.n_settled = len(settlements)
    settled_qids = {e.question_id for e in settlements}
    unsettled = sorted(all_qids - settled_qids)
    result.unsettled_question_ids = unsettled
    if unsettled:
        result.notes.append(
            f"{len(unsettled)} unsettled question(s) excluded from price/gap/Brier "
            f"averages: {', '.join(unsettled)}"
        )

    # -- turns done/expected (best-effort; no run config available here) ---
    n_agents, n_rounds = _infer_shape(beliefs, trades)
    result.turns_done = len(beliefs) + len(trades) + len(settlements)
    result.turns_expected = (
        result.n_questions_dataset * n_agents  # Pass-1 beliefs
        + result.n_questions_dataset * n_agents * n_rounds  # Pass-2 trades
        + result.n_questions_dataset  # settlements
    )

    # -- price degeneracy + market gap/Brier vs. oracle ---------------------
    degenerate = 0
    market_gaps: list[float] = []
    market_briers: list[float] = []
    for s in settlements:
        if s.final_price < _DEGENERATE_LO or s.final_price > _DEGENERATE_HI:
            degenerate += 1
        record = questions.get(s.question_id)
        if record is None:
            continue
        posterior_all = _posterior_all(record)
        if posterior_all is None:
            continue
        market_gaps.append(abs(s.final_price - posterior_all))
        market_briers.append((s.final_price - s.outcome) ** 2)
    if settlements:
        result.degenerate_count = degenerate
        result.degeneracy_rate = degenerate / len(settlements)
    if market_gaps:
        result.market_gap = sum(market_gaps) / len(market_gaps)
        result.market_brier = sum(market_briers) / len(market_briers)

    # -- mean-pool-of-Pass-1 baseline (settled questions only) --------------
    beliefs_by_q: dict[str, list[BeliefEvent]] = {}
    for e in beliefs:
        beliefs_by_q.setdefault(e.key.question_id, []).append(e)
    pool_gaps: list[float] = []
    pool_briers: list[float] = []
    for s in settlements:
        record = questions.get(s.question_id)
        q_beliefs = beliefs_by_q.get(s.question_id)
        if record is None or not q_beliefs:
            continue
        posterior_all = _posterior_all(record)
        if posterior_all is None:
            continue
        pooled = _mean_pool_prob(q_beliefs)
        pool_gaps.append(abs(pooled - posterior_all))
        pool_briers.append((pooled - s.outcome) ** 2)
    if pool_gaps:
        result.pool_gap = sum(pool_gaps) / len(pool_gaps)
        result.pool_brier = sum(pool_briers) / len(pool_briers)

    # -- agent_id -> model_id (derived from the cost ledger; only LLM agents
    #    ever bill, so mock agents are simply absent from this map) --------
    agent_model: dict[str, str] = {}
    for entry in cost_entries:
        if entry.key is not None:
            agent_model[entry.key.agent_id] = entry.model_id

    # -- parse-fail rate per model (post-retry: parse_failed is the final,
    #    after-the-one-retry outcome logged by the runner) ------------------
    totals: dict[str, int] = {}
    fails: dict[str, int] = {}
    unmapped = 0
    for e in beliefs + trades:
        model_id = agent_model.get(e.key.agent_id)
        if model_id is None:
            unmapped += 1
            continue
        totals[model_id] = totals.get(model_id, 0) + 1
        if e.parse_failed:
            fails[model_id] = fails.get(model_id, 0) + 1
    result.parse_fail_by_model = {
        model_id: (fails.get(model_id, 0), total) for model_id, total in totals.items()
    }
    if unmapped:
        result.notes.append(
            f"{unmapped} belief/trade event(s) skipped from the per-model parse-fail "
            "breakdown (no cost-ledger model mapping — mock agent, or never billed)"
        )

    # -- per-agent-model standalone gap (Pass-1 prob vs. oracle posterior_all,
    #    NOT settlement-gated — Pass-1 doesn't depend on the market closing) -
    gaps_by_model: dict[str, list[float]] = {}
    for e in beliefs:
        model_id = agent_model.get(e.key.agent_id)
        record = questions.get(e.key.question_id)
        if model_id is None or record is None:
            continue
        posterior_all = _posterior_all(record)
        if posterior_all is None:
            continue
        gaps_by_model.setdefault(model_id, []).append(abs(e.belief.prob - posterior_all))
    result.standalone_gap_by_model = {
        model_id: sum(vs) / len(vs) for model_id, vs in gaps_by_model.items()
    }

    # -- cost per model + total ----------------------------------------------
    cost_by_model: dict[str, float] = {}
    for entry in cost_entries:
        cost_by_model[entry.model_id] = cost_by_model.get(entry.model_id, 0.0) + entry.cost
    result.cost_by_model = cost_by_model
    result.total_cost = sum(cost_by_model.values())

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(x: float | None, spec: str = "{:.4f}") -> str:
    return spec.format(x) if x is not None else "n/a"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [_row(headers), _row(["-" * w for w in widths])]
    lines.extend(_row(row) for row in rows)
    return "\n".join(lines)


def format_report(analyses: list[RunAnalysis]) -> str:
    """Aligned text table (one row per run) plus a per-run per-model
    breakdown (parse-fail rate, standalone gap, cost) and notes."""
    headers = [
        "run", "condition", "settled", "degeneracy", "market_gap",
        "market_brier", "pool_gap", "pool_brier", "total_cost", "turns",
    ]
    rows = [
        [
            a.run_dir,
            a.condition or "?",
            f"{a.n_settled}/{a.n_questions_dataset}",
            _fmt(a.degeneracy_rate),
            _fmt(a.market_gap),
            _fmt(a.market_brier),
            _fmt(a.pool_gap),
            _fmt(a.pool_brier),
            f"${a.total_cost:.4f}",
            f"{a.turns_done}/{a.turns_expected}",
        ]
        for a in analyses
    ]
    blocks = [_render_table(headers, rows)]

    for a in analyses:
        block_lines = [f"\n-- {a.run_dir} --"]
        if a.parse_fail_by_model:
            block_lines.append("  parse-fail rate by model (post-retry):")
            for model_id, (n_fail, n_total) in sorted(a.parse_fail_by_model.items()):
                rate = n_fail / n_total if n_total else 0.0
                block_lines.append(f"    {model_id}: {rate:.4f} ({n_fail}/{n_total})")
        if a.standalone_gap_by_model:
            block_lines.append("  standalone gap by model (mean |Pass-1 prob - posterior_all|):")
            for model_id, gap in sorted(a.standalone_gap_by_model.items()):
                block_lines.append(f"    {model_id}: {gap:.4f}")
        if a.cost_by_model:
            block_lines.append("  cost by model:")
            for model_id, cost in sorted(a.cost_by_model.items()):
                block_lines.append(f"    {model_id}: ${cost:.4f}")
        if a.notes:
            block_lines.append("  notes:")
            block_lines.extend(f"    - {note}" for note in a.notes)
        if len(block_lines) > 1:
            blocks.append("\n".join(block_lines))

    return "\n".join(blocks)


def progress_section(analyses: list[RunAnalysis]) -> str:
    """The full ``## P3 pilot`` PROGRESS.md section text (heading + a
    fenced-code aligned report, so markdown doesn't collapse whitespace)."""
    body = format_report(analyses)
    return "\n".join([_PROGRESS_HEADING, "", "```", body, "```"]) + "\n"


def write_progress(progress_path: str | Path, analyses: list[RunAnalysis]) -> None:
    """Idempotent write/replace of the ``## P3 pilot`` section in
    ``progress_path`` (reuses ``pnyx.dataset``'s generic section-replace
    helper — see ``dataset.replace_progress_section``)."""
    replace_progress_section(Path(progress_path), _PROGRESS_HEADING, progress_section(analyses))
