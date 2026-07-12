"""Deterministic dataset build / merge / verify tooling for P2.

The frozen scientific asset is a set of 50 ``QuestionRecord``s generated ONCE
from a fixed master seed, rendered into prose by Claude subagents (orchestrated
OUTSIDE this codebase), verified, and split into a main + pilot file. This
module owns the three offline, network-free stages:

1. ``build_questions``  — generate 50 records + emit ``render_specs.jsonl``
   (the leakage-safe view the renderer subagent is allowed to see);
2. ``merge_render``     — validate the renderer subagents' ``rendered/*.json``
   outputs and merge shards + question_text into ``questions_rendered.jsonl``;
3. ``verify_report``    — aggregate the verifier subagents' ``verdicts/*.json``
   into per-check pass rates, and (``assemble=True``) split the rendered file
   into ``datasets/questions_v1.jsonl`` + ``datasets/questions_pilot_v1.jsonl``.

Everything here is a pure function of paths/params returning a stats dataclass;
``pnyx.cli`` does the argument parsing and printing. No LLM API calls anywhere.

Determinism
-----------
Each question's rejection sampler is seeded via the P1 pattern
(``SeedSequence([MASTER_SEED, sha256(tag)])``), keyed on the question id, so two
builds are byte-identical. See ``_build_rng``.

Split / difficulty layout
-------------------------
50 questions, ids ``q000``-``q049``:

* main  (``q000``-``q039``) -> ``questions_v1.jsonl``      : 20 easy + 20 hard
* pilot (``q040``-``q049``) -> ``questions_pilot_v1.jsonl``: 5 easy + 5 hard

The difficulty of each id is fixed by ``build_plan`` (easy block then hard block
within each of the main / pilot ranges), so the split-by-id-range in
``verify_report(assemble=True)`` yields the required per-file balance.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pnyx.env import posterior, subset_key
from pnyx.schemas import (
    EnvConfig,
    QuestionRecord,
    QuestionVerdict,
    RenderedQuestion,
    RenderSpec,
    RenderSpecSignal,
)

__all__ = [
    "MASTER_SEED",
    "DOMAIN_HINTS",
    "build_plan",
    "build_questions",
    "merge_render",
    "verify_report",
    "BuildStats",
    "MergeStats",
    "VerifyStats",
]

# --------------------------------------------------------------------------
# Fixed build parameters (the frozen asset's provenance)
# --------------------------------------------------------------------------

MASTER_SEED = 20260713

N_MAIN = 40
N_PILOT = 10
N_TOTAL = N_MAIN + N_PILOT

# Difficulty block sizes within the main / pilot id ranges.
_MAIN_EASY = 20
_PILOT_EASY = 5

DOMAIN_HINTS = [
    "workplace-incident",
    "clinical-case",
    "supply-chain",
    "small-town-mystery",
    "engineering-failure",
    "archival-history",
    "courtroom",
    "wildlife-survey",
    "financial-audit",
    "expedition-log",
]

# Difficulty configs. Easy: strong, conditionally-independent signals (favored
# posterior >= 0.85). Hard: weak, mildly correlated signals so the favored
# posterior lands in [0.60, 0.75] while single-shard margins stay >= 0.1.
# Both verified empirically to generate all 50 questions within the rejection
# budget (see the P2 constraints file for the feasibility analysis).
EASY_CONFIG = EnvConfig(
    n_signals=6,
    accuracies=[0.75] * 6,
    lams=[0.0] * 6,
    a_z=0.8,
    difficulty="easy",
    min_single_shard_margin=0.1,
    max_rejection_tries=2000,
)
HARD_CONFIG = EnvConfig(
    n_signals=6,
    accuracies=[0.55] * 6,
    lams=[0.1] * 6,
    a_z=0.7,
    difficulty="hard",
    min_single_shard_margin=0.1,
    max_rejection_tries=2000,
)


# --------------------------------------------------------------------------
# Stats objects
# --------------------------------------------------------------------------


@dataclass
class BuildStats:
    n_questions: int
    n_easy: int
    n_hard: int
    out_dir: Path


@dataclass
class MergeStats:
    n_merged: int
    missing: list[str] = field(default_factory=list)
    invalid: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.invalid


@dataclass
class VerifyStats:
    n_questions: int
    n_shards: int
    direction_pass_rate: float
    leak_pass_rate: float
    meta_pass_rate: float
    question_pass_rate: float
    failed_question_ids: list[str]
    all_pass: bool
    assembled: bool = False
    n_main: int = 0
    n_pilot: int = 0


# --------------------------------------------------------------------------
# Seeding + plan
# --------------------------------------------------------------------------


def _build_rng(question_id: str) -> np.random.Generator:
    """Deterministic per-question Generator (P1 seeding pattern): SHA-256 of a
    stable tag -> 64-bit int, seeding ``SeedSequence([MASTER_SEED, tag])``."""
    blob = f"build|{question_id}".encode("utf-8")
    tag = int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, tag]))


def build_plan() -> list[tuple[str, str]]:
    """Return the fixed ``(question_id, difficulty)`` build plan for all 50
    questions. Easy block precedes hard block within each of the main /
    pilot id ranges so the id-range split yields the required balance."""
    plan: list[tuple[str, str]] = []
    for i in range(N_TOTAL):
        qid = f"q{i:03d}"
        if i < N_MAIN:
            difficulty = "easy" if i < _MAIN_EASY else "hard"
        else:
            difficulty = "easy" if i < N_MAIN + _PILOT_EASY else "hard"
        plan.append((qid, difficulty))
    return plan


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def _config_for(difficulty: str) -> EnvConfig:
    return EASY_CONFIG if difficulty == "easy" else HARD_CONFIG


def _augment_meta(record: QuestionRecord, config: EnvConfig) -> QuestionRecord:
    """Return a copy of ``record`` with the full P2 provenance meta: difficulty,
    env-config snapshot, master seed, and the realized posterior margins."""
    n = len(record.signals)
    p_all = record.posterior_table[subset_key(range(n))]
    singles = [record.posterior_table[subset_key([i])] for i in range(n)]
    min_margin = min(abs(p_all - ps) for ps in singles)
    meta = {
        "difficulty": config.difficulty,
        "master_seed": MASTER_SEED,
        "env_config": config.model_dump(),
        "posterior_all": p_all,
        "min_single_shard_posterior": min(singles),
        "max_single_shard_posterior": max(singles),
        "min_single_shard_margin": min_margin,
    }
    return record.model_copy(update={"meta": meta})


def _render_spec(record: QuestionRecord, difficulty: str, index: int) -> RenderSpec:
    return RenderSpec(
        question_id=record.question_id,
        difficulty=difficulty,
        domain_hint=DOMAIN_HINTS[index % len(DOMAIN_HINTS)],
        signals=[
            RenderSpecSignal(
                index=s.index, direction="YES" if s.value == 1 else "NO"
            )
            for s in record.signals
        ],
    )


def build_questions(out_dir: Path | str) -> BuildStats:
    """Generate all 50 questions deterministically and write ``out_dir``'s
    ``questions_base.jsonl`` (records with shards / question_text absent) and
    ``render_specs.jsonl`` (the leakage-safe renderer view)."""
    from pnyx.env import generate_question

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan()
    base_lines: list[str] = []
    spec_lines: list[str] = []
    n_easy = n_hard = 0
    for index, (qid, difficulty) in enumerate(plan):
        config = _config_for(difficulty)
        record = generate_question(config, _build_rng(qid), qid)
        record = _augment_meta(record, config)
        base_lines.append(record.model_dump_json())
        spec = _render_spec(record, difficulty, index)
        spec_lines.append(json.dumps(spec.model_dump(), sort_keys=True, separators=(",", ":")))
        if difficulty == "easy":
            n_easy += 1
        else:
            n_hard += 1

    (out_dir / "questions_base.jsonl").write_text("\n".join(base_lines) + "\n")
    (out_dir / "render_specs.jsonl").write_text("\n".join(spec_lines) + "\n")
    return BuildStats(n_questions=len(plan), n_easy=n_easy, n_hard=n_hard, out_dir=out_dir)


# --------------------------------------------------------------------------
# Merge render
# --------------------------------------------------------------------------


def _read_base_records(staging: Path) -> list[QuestionRecord]:
    path = staging / "questions_base.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run build-questions first")
    return [
        QuestionRecord.model_validate_json(line)
        for line in path.read_text().splitlines() if line
    ]


def merge_render(staging: Path | str) -> MergeStats:
    """Validate ``staging/rendered/{qid}.json`` against the merge contract and
    merge shards + question_text into ``staging/questions_rendered.jsonl``.

    Missing files and invalid renderings are collected (never raised); the
    stats object's ``ok`` is False if any occurred so the CLI can exit nonzero.
    Only successfully-validated records are written.
    """
    staging = Path(staging)
    base = _read_base_records(staging)
    rendered_dir = staging / "rendered"

    merged: list[QuestionRecord] = []
    missing: list[str] = []
    invalid: dict[str, str] = {}

    for record in base:
        qid = record.question_id
        path = rendered_dir / f"{qid}.json"
        if not path.exists():
            missing.append(qid)
            continue
        try:
            payload = json.loads(path.read_text())
            rendered = RenderedQuestion.model_validate(payload)
        except Exception as exc:  # json / pydantic validation failure
            invalid[qid] = str(exc)
            continue
        if rendered.question_id != qid:
            invalid[qid] = (
                f"question_id mismatch: file says {rendered.question_id!r}"
            )
            continue
        if len(rendered.shards) != len(record.signals):
            invalid[qid] = (
                f"shard count {len(rendered.shards)} != signal count "
                f"{len(record.signals)}"
            )
            continue
        merged.append(record.model_copy(update={
            "shards": rendered.shards,
            "question_text": rendered.question_text,
        }))

    out = staging / "questions_rendered.jsonl"
    out.write_text("".join(r.model_dump_json() + "\n" for r in merged))
    return MergeStats(n_merged=len(merged), missing=missing, invalid=invalid)


# --------------------------------------------------------------------------
# Verify report (+ assemble)
# --------------------------------------------------------------------------


def _read_verdicts(staging: Path, qids: list[str]) -> dict[str, QuestionVerdict]:
    vdir = staging / "verdicts"
    verdicts: dict[str, QuestionVerdict] = {}
    for qid in qids:
        path = vdir / f"{qid}.json"
        if not path.exists():
            continue
        verdicts[qid] = QuestionVerdict.model_validate_json(path.read_text())
    return verdicts


def verify_report(
    staging: Path | str,
    *,
    assemble: bool = False,
    datasets_dir: Path | str = "datasets",
    progress_path: Path | str | None = None,
) -> VerifyStats:
    """Aggregate ``staging/verdicts/{qid}.json`` into per-check pass rates and
    the set of failed question ids (a question fails if its verdict is missing,
    ``question_ok`` is False, or any shard fails any check).

    With ``assemble=True``: require every question passing, then split
    ``questions_rendered.jsonl`` into the main + pilot dataset files under
    ``datasets_dir`` (embedding the verdict summary + render provenance into
    each record's meta) and append/replace a "## P2 dataset" section in
    ``progress_path`` (if given).
    """
    staging = Path(staging)
    base = _read_base_records(staging)
    qids = [r.question_id for r in base]
    verdicts = _read_verdicts(staging, qids)

    n_direction = n_leak = n_meta = 0
    dir_pass = leak_pass = meta_pass = 0
    q_pass = 0
    failed: list[str] = []

    for qid in qids:
        verdict = verdicts.get(qid)
        if verdict is None:
            failed.append(qid)
            continue
        q_failed = not verdict.question_ok
        for shard in verdict.shard_verdicts:
            n_direction += 1
            n_leak += 1
            n_meta += 1
            dir_pass += int(shard.direction_ok)
            leak_pass += int(shard.leak_ok)
            meta_pass += int(shard.meta_ok)
            if not (shard.direction_ok and shard.leak_ok and shard.meta_ok):
                q_failed = True
        if verdict.question_ok:
            q_pass += 1
        if q_failed:
            failed.append(qid)

    def _rate(passed: int, total: int) -> float:
        return passed / total if total else 1.0

    stats = VerifyStats(
        n_questions=len(qids),
        n_shards=n_direction,
        direction_pass_rate=_rate(dir_pass, n_direction),
        leak_pass_rate=_rate(leak_pass, n_leak),
        meta_pass_rate=_rate(meta_pass, n_meta),
        question_pass_rate=_rate(q_pass, len(qids)),
        failed_question_ids=failed,
        all_pass=not failed,
    )

    if assemble and stats.all_pass:
        _assemble(staging, Path(datasets_dir), verdicts, stats)
        if progress_path is not None:
            _write_progress(Path(progress_path), stats)

    return stats


def _read_rendered_records(staging: Path) -> list[QuestionRecord]:
    path = staging / "questions_rendered.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run merge-render first")
    records = [
        QuestionRecord.model_validate_json(line)
        for line in path.read_text().splitlines() if line
    ]
    for r in records:
        if r.shards is None or r.question_text is None:
            raise ValueError(
                f"{r.question_id}: rendered record missing shards/question_text"
            )
    return records


def _assemble(
    staging: Path,
    datasets_dir: Path,
    verdicts: dict[str, QuestionVerdict],
    stats: VerifyStats,
) -> None:
    records = _read_rendered_records(staging)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    main: list[QuestionRecord] = []
    pilot: list[QuestionRecord] = []
    for record in records:
        verdict = verdicts[record.question_id]
        meta = dict(record.meta)
        meta["verdict"] = {
            "question_ok": verdict.question_ok,
            "shard_checks": [
                {
                    "index": s.index,
                    "direction_ok": s.direction_ok,
                    "leak_ok": s.leak_ok,
                    "meta_ok": s.meta_ok,
                }
                for s in verdict.shard_verdicts
            ],
        }
        meta["render_provenance"] = {
            "source": "merge-render",
            "staging_dir": str(staging),
            "n_shards": len(record.shards) if record.shards else 0,
        }
        enriched = record.model_copy(update={"meta": meta})
        idx = int(record.question_id[1:])
        (main if idx < N_MAIN else pilot).append(enriched)

    (datasets_dir / "questions_v1.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in main)
    )
    (datasets_dir / "questions_pilot_v1.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in pilot)
    )
    stats.assembled = True
    stats.n_main = len(main)
    stats.n_pilot = len(pilot)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_PROGRESS_HEADING = "## P2 dataset"


def summary_table(stats: VerifyStats) -> str:
    """Human-readable summary of a verify/assemble run."""
    lines = [
        _PROGRESS_HEADING,
        "",
        f"- questions verified: {stats.n_questions}",
        f"- shards checked: {stats.n_shards}",
        f"- direction pass rate: {stats.direction_pass_rate:.4f}",
        f"- leak pass rate: {stats.leak_pass_rate:.4f}",
        f"- meta pass rate: {stats.meta_pass_rate:.4f}",
        f"- question pass rate: {stats.question_pass_rate:.4f}",
        f"- all pass: {stats.all_pass}",
    ]
    if stats.failed_question_ids:
        lines.append(f"- failed question_ids: {', '.join(stats.failed_question_ids)}")
    if stats.assembled:
        lines.append(
            f"- assembled: questions_v1.jsonl ({stats.n_main}) + "
            f"questions_pilot_v1.jsonl ({stats.n_pilot})"
        )
    return "\n".join(lines)


def _write_progress(progress_path: Path, stats: VerifyStats) -> None:
    """Append (or replace) the "## P2 dataset" section in PROGRESS.md."""
    section = summary_table(stats) + "\n"
    if not progress_path.exists():
        progress_path.write_text(section)
        return
    text = progress_path.read_text()
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if line.strip() == _PROGRESS_HEADING:
            start = i
            break
    if start is None:
        # Append a fresh section (ensure a blank-line separator).
        sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        progress_path.write_text(text + sep + section)
        return
    # Replace from the heading to the next top-level (## / #) heading or EOF.
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## ") or lines[j].startswith("# "):
            end = j
            break
    new_lines = lines[:start] + [section] + lines[end:]
    progress_path.write_text("".join(new_lines))
