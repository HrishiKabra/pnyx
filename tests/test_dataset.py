"""Tests for pnyx.dataset — deterministic build, merge-render validation,
verify-report aggregation, and the assemble split.

All pure and offline: the renderer/verifier staging files these functions
consume are produced by OTHER (Claude subagent) agents in the real pipeline;
here we write small fixture files by hand. No network, no LLM calls.
"""

import json
from pathlib import Path

import pytest

from pnyx import dataset
from pnyx.env import posterior, subset_key
from pnyx.schemas import QuestionRecord


# ---------------------------------------------------------------------------
# Build plan / determinism
# ---------------------------------------------------------------------------


def test_build_plan_split_and_difficulties():
    plan = dataset.build_plan()
    assert len(plan) == 50
    ids = [qid for qid, _ in plan]
    assert ids == [f"q{i:03d}" for i in range(50)]

    diffs = {qid: d for qid, d in plan}
    # Main (q000-q039): 20 easy + 20 hard.
    main = [diffs[f"q{i:03d}"] for i in range(40)]
    assert main.count("easy") == 20
    assert main.count("hard") == 20
    # Pilot (q040-q049): 5 easy + 5 hard.
    pilot = [diffs[f"q{i:03d}"] for i in range(40, 50)]
    assert pilot.count("easy") == 5
    assert pilot.count("hard") == 5


def test_build_is_deterministic_byte_identical(tmp_path):
    out1 = tmp_path / "s1"
    out2 = tmp_path / "s2"
    dataset.build_questions(out1)
    dataset.build_questions(out2)

    for name in ("questions_base.jsonl", "render_specs.jsonl"):
        b1 = (out1 / name).read_bytes()
        b2 = (out2 / name).read_bytes()
        assert b1 == b2, f"{name} differs between two builds"


def test_build_writes_expected_counts(tmp_path):
    out = tmp_path / "staging"
    stats = dataset.build_questions(out)
    assert stats.n_questions == 50

    base_lines = (out / "questions_base.jsonl").read_text().splitlines()
    spec_lines = (out / "render_specs.jsonl").read_text().splitlines()
    assert len(base_lines) == 50
    assert len(spec_lines) == 50


# ---------------------------------------------------------------------------
# Base records: shards/question_text absent, meta populated
# ---------------------------------------------------------------------------


def test_base_records_have_no_render_content_and_full_meta(tmp_path):
    out = tmp_path / "staging"
    dataset.build_questions(out)
    records = [
        QuestionRecord.model_validate_json(line)
        for line in (out / "questions_base.jsonl").read_text().splitlines()
    ]
    for rec in records:
        assert rec.shards is None
        assert rec.question_text is None
        meta = rec.meta
        assert meta["difficulty"] in ("easy", "hard")
        assert meta["master_seed"] == dataset.MASTER_SEED
        assert "env_config" in meta
        for key in (
            "posterior_all",
            "min_single_shard_posterior",
            "max_single_shard_posterior",
            "min_single_shard_margin",
        ):
            assert key in meta


def test_meta_margins_match_oracle_recomputation(tmp_path):
    out = tmp_path / "staging"
    dataset.build_questions(out)
    records = [
        QuestionRecord.model_validate_json(line)
        for line in (out / "questions_base.jsonl").read_text().splitlines()
    ]
    for rec in records:
        values = {s.index: s.value for s in rec.signals}
        acc = [s.accuracy for s in rec.signals]
        lams = [s.lam for s in rec.signals]
        a_z = rec.meta["env_config"]["a_z"]
        n = len(rec.signals)
        p_all = posterior(values, acc, lams, a_z)
        singles = [
            posterior({i: values[i]}, acc, lams, a_z) for i in range(n)
        ]
        margin = min(abs(p_all - ps) for ps in singles)
        assert rec.meta["posterior_all"] == pytest.approx(p_all, abs=1e-12)
        assert rec.meta["min_single_shard_posterior"] == pytest.approx(min(singles), abs=1e-12)
        assert rec.meta["max_single_shard_posterior"] == pytest.approx(max(singles), abs=1e-12)
        assert rec.meta["min_single_shard_margin"] == pytest.approx(margin, abs=1e-12)
        # single-shard margin constraint actually held during generation
        assert margin >= 0.1 - 1e-12


# ---------------------------------------------------------------------------
# render_specs: no leakage of accuracies / latent state / posteriors
# ---------------------------------------------------------------------------


def test_render_specs_have_no_leakage(tmp_path):
    out = tmp_path / "staging"
    dataset.build_questions(out)
    base = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in (out / "questions_base.jsonl").read_text().splitlines()
    }
    specs = [
        json.loads(line)
        for line in (out / "render_specs.jsonl").read_text().splitlines()
    ]
    domains_seen = []
    for spec in specs:
        assert set(spec.keys()) == {"question_id", "difficulty", "domain_hint", "signals"}
        assert spec["domain_hint"] in dataset.DOMAIN_HINTS
        domains_seen.append(spec["domain_hint"])
        # No leaking fields anywhere in the spec text.
        blob = json.dumps(spec)
        assert "accuracy" not in blob
        assert "latent" not in blob
        assert "posterior" not in blob
        # Directions align with realized signal values in the base record.
        rec = base[spec["question_id"]]
        by_index = {s["index"]: s["value"] for s in rec["signals"]}
        assert len(spec["signals"]) == len(rec["signals"])
        for s in spec["signals"]:
            assert s["direction"] in ("YES", "NO")
            expected = "YES" if by_index[s["index"]] == 1 else "NO"
            assert s["direction"] == expected

    # domain_hint cycles through the fixed 10-domain list.
    assert domains_seen[0] == dataset.DOMAIN_HINTS[0]
    assert domains_seen[10] == dataset.DOMAIN_HINTS[0]
    assert domains_seen[:10] == dataset.DOMAIN_HINTS


# ---------------------------------------------------------------------------
# merge-render
# ---------------------------------------------------------------------------


def _seed_staging(tmp_path) -> Path:
    out = tmp_path / "staging"
    dataset.build_questions(out)
    return out


def _write_rendered(out: Path, qid: str, *, shards=None, question_text=None):
    rdir = out / "rendered"
    rdir.mkdir(parents=True, exist_ok=True)
    if shards is None:
        shards = [f"Shard {i} prose for {qid}." for i in range(6)]
    if question_text is None:
        question_text = f"Is the answer to {qid} yes?"
    (rdir / f"{qid}.json").write_text(json.dumps(
        {"question_id": qid, "question_text": question_text, "shards": shards}
    ))


def _write_all_rendered(out: Path):
    for line in (out / "questions_base.jsonl").read_text().splitlines():
        _write_rendered(out, json.loads(line)["question_id"])


def test_merge_render_success(tmp_path):
    out = _seed_staging(tmp_path)
    _write_all_rendered(out)
    stats = dataset.merge_render(out)
    assert stats.ok
    assert not stats.missing
    assert not stats.invalid
    assert stats.n_merged == 50

    merged = [
        QuestionRecord.model_validate_json(line)
        for line in (out / "questions_rendered.jsonl").read_text().splitlines()
    ]
    assert len(merged) == 50
    for rec in merged:
        assert rec.shards is not None and len(rec.shards) == 6
        assert rec.question_text and rec.question_text.endswith("?")


def test_merge_render_missing_file(tmp_path):
    out = _seed_staging(tmp_path)
    _write_all_rendered(out)
    (out / "rendered" / "q005.json").unlink()
    stats = dataset.merge_render(out)
    assert not stats.ok
    assert "q005" in stats.missing


def test_merge_render_rejects_wrong_shard_count(tmp_path):
    out = _seed_staging(tmp_path)
    _write_all_rendered(out)
    _write_rendered(out, "q003", shards=[f"s{i}" for i in range(5)])
    stats = dataset.merge_render(out)
    assert not stats.ok
    assert "q003" in stats.invalid


def test_merge_render_rejects_empty_shard(tmp_path):
    out = _seed_staging(tmp_path)
    _write_all_rendered(out)
    shards = [f"s{i}" for i in range(6)]
    shards[2] = "   "
    _write_rendered(out, "q004", shards=shards)
    stats = dataset.merge_render(out)
    assert not stats.ok
    assert "q004" in stats.invalid


def test_merge_render_rejects_question_text_without_qmark(tmp_path):
    out = _seed_staging(tmp_path)
    _write_all_rendered(out)
    _write_rendered(out, "q006", question_text="This is a statement.")
    stats = dataset.merge_render(out)
    assert not stats.ok
    assert "q006" in stats.invalid


def test_merge_render_rejects_empty_question_text(tmp_path):
    out = _seed_staging(tmp_path)
    _write_all_rendered(out)
    _write_rendered(out, "q007", question_text="")
    stats = dataset.merge_render(out)
    assert not stats.ok
    assert "q007" in stats.invalid


# ---------------------------------------------------------------------------
# verify-report aggregation
# ---------------------------------------------------------------------------


def _write_verdict(out: Path, qid: str, *, direction_ok=True, leak_ok=True,
                   meta_ok=True, question_ok=True, bad_shard=0):
    vdir = out / "verdicts"
    vdir.mkdir(parents=True, exist_ok=True)
    shard_verdicts = []
    for i in range(6):
        shard_verdicts.append({
            "index": i,
            "direction_ok": direction_ok or i != bad_shard,
            "leak_ok": leak_ok or i != bad_shard,
            "meta_ok": meta_ok or i != bad_shard,
            "note": "",
        })
    (vdir / f"{qid}.json").write_text(json.dumps(
        {"question_id": qid, "shard_verdicts": shard_verdicts, "question_ok": question_ok}
    ))


def _all_qids(out: Path):
    return [json.loads(line)["question_id"]
            for line in (out / "questions_base.jsonl").read_text().splitlines()]


def test_verify_report_all_pass(tmp_path):
    out = _seed_staging(tmp_path)
    for qid in _all_qids(out):
        _write_verdict(out, qid)
    stats = dataset.verify_report(out)
    assert stats.all_pass
    assert stats.failed_question_ids == []
    assert stats.direction_pass_rate == pytest.approx(1.0)
    assert stats.leak_pass_rate == pytest.approx(1.0)
    assert stats.meta_pass_rate == pytest.approx(1.0)
    assert stats.n_shards == 50 * 6


def test_verify_report_aggregation_math(tmp_path):
    out = _seed_staging(tmp_path)
    qids = _all_qids(out)
    for qid in qids:
        _write_verdict(out, qid)
    # Break one direction check on q001 (one shard) and one leak check on q002.
    _write_verdict(out, "q001", direction_ok=False, question_ok=False, bad_shard=0)
    _write_verdict(out, "q002", leak_ok=False, question_ok=False, bad_shard=3)
    stats = dataset.verify_report(out)

    total_shards = 50 * 6
    assert stats.n_shards == total_shards
    # exactly one direction failure, one leak failure.
    assert stats.direction_pass_rate == pytest.approx((total_shards - 1) / total_shards)
    assert stats.leak_pass_rate == pytest.approx((total_shards - 1) / total_shards)
    assert stats.meta_pass_rate == pytest.approx(1.0)
    assert set(stats.failed_question_ids) == {"q001", "q002"}
    assert not stats.all_pass


def test_verify_report_missing_verdict_counts_as_failure(tmp_path):
    out = _seed_staging(tmp_path)
    qids = _all_qids(out)
    for qid in qids[:-1]:  # omit the last question's verdict
        _write_verdict(out, qid)
    stats = dataset.verify_report(out)
    assert not stats.all_pass
    assert qids[-1] in stats.failed_question_ids


# ---------------------------------------------------------------------------
# assemble split
# ---------------------------------------------------------------------------


def _full_green_staging(tmp_path) -> Path:
    out = _seed_staging(tmp_path)
    _write_all_rendered(out)
    dataset.merge_render(out)
    for qid in _all_qids(out):
        _write_verdict(out, qid)
    return out


def test_assemble_splits_and_embeds_meta(tmp_path):
    out = _full_green_staging(tmp_path)
    datasets_dir = tmp_path / "datasets"
    progress = tmp_path / "PROGRESS.md"
    progress.write_text("# Pnyx\n\nexisting content\n")

    stats = dataset.verify_report(out, assemble=True,
                                  datasets_dir=datasets_dir, progress_path=progress)
    assert stats.all_pass
    assert stats.assembled

    main = [QuestionRecord.model_validate_json(line)
            for line in (datasets_dir / "questions_v1.jsonl").read_text().splitlines()]
    pilot = [QuestionRecord.model_validate_json(line)
             for line in (datasets_dir / "questions_pilot_v1.jsonl").read_text().splitlines()]
    assert [r.question_id for r in main] == [f"q{i:03d}" for i in range(40)]
    assert [r.question_id for r in pilot] == [f"q{i:03d}" for i in range(40, 50)]

    for rec in main + pilot:
        assert rec.shards is not None and rec.question_text
        assert "verdict" in rec.meta
        assert rec.meta["verdict"]["question_ok"] is True
        assert "render_provenance" in rec.meta


def test_assemble_refuses_when_a_question_fails(tmp_path):
    out = _seed_staging(tmp_path)
    _write_all_rendered(out)
    dataset.merge_render(out)
    for qid in _all_qids(out):
        _write_verdict(out, qid)
    _write_verdict(out, "q010", meta_ok=False, question_ok=False, bad_shard=1)

    datasets_dir = tmp_path / "datasets"
    stats = dataset.verify_report(out, assemble=True, datasets_dir=datasets_dir)
    assert not stats.all_pass
    assert not stats.assembled
    assert not (datasets_dir / "questions_v1.jsonl").exists()


def test_assemble_progress_append_is_idempotent(tmp_path):
    out = _full_green_staging(tmp_path)
    datasets_dir = tmp_path / "datasets"
    progress = tmp_path / "PROGRESS.md"
    progress.write_text("# Pnyx\n\nbody\n")

    dataset.verify_report(out, assemble=True, datasets_dir=datasets_dir, progress_path=progress)
    first = progress.read_text()
    assert "## P2 dataset" in first
    dataset.verify_report(out, assemble=True, datasets_dir=datasets_dir, progress_path=progress)
    second = progress.read_text()
    # The section is replaced, not duplicated.
    assert second.count("## P2 dataset") == 1
    assert second.count("# Pnyx") == 1
