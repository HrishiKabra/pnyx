"""P1 acceptance test: SIGKILL mid-run, then rerun-completes-identically.

Launches the real CLI (``python -m pnyx.cli run``) as a subprocess against a
temp data dir, hard-kills it (SIGKILL — no cleanup handlers run) once the
event log has grown past a threshold, then reruns the *same* command to
completion. The final event log (canonical, ts-stripped) must be byte-identical
to an uninterrupted reference run with the same config + seed.

Two kill points at different question indices exercise both an early crash
(mid question 0) and a later crash (a subsequent question).

Runtime is kept well under 60s by (a) the mock run being tiny and (b) a small
per-turn delay injected only in the subprocess via ``PNYX_TURN_DELAY`` so the
poller reliably catches the run mid-flight before killing it.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pnyx.runner import (
    load_config,
    log_path_for,
    read_events,
    ts_stripped_lines,
)

REPO = Path(__file__).resolve().parents[1]
MOCK_CONFIG = REPO / "pnyx" / "configs" / "mock.yaml"
TURN_DELAY = "0.03"  # seconds/turn, subprocess only; ~120 turns => ~3.6s runs


def _run_cli(data_dir: Path, *, delay: str | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    if delay is not None:
        env["PNYX_TURN_DELAY"] = delay
    return subprocess.Popen(
        [sys.executable, "-m", "pnyx.cli", "run",
         "--config", str(MOCK_CONFIG), "--data-dir", str(data_dir)],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _config_for(data_dir: Path):
    return load_config(str(MOCK_CONFIG)).model_copy(update={"data_dir": str(data_dir)})


def _log_path(data_dir: Path) -> Path:
    return log_path_for(_config_for(data_dir), _config_for(data_dir).seeds[0])


def _count_events(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_events(path))


def _wait_until_events(path: Path, n: int, deadline_s: float = 40.0) -> None:
    start = time.time()
    while time.time() - start < deadline_s:
        if _count_events(path) >= n:
            return
        time.sleep(0.02)
    raise AssertionError(f"log {path} never reached {n} events "
                         f"(saw {_count_events(path)})")


def _reference_log(tmp_path: Path) -> list[str]:
    data_dir = tmp_path / "reference"
    proc = _run_cli(data_dir)  # no delay: run to completion fast
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, f"reference run failed: {err.decode()}"
    return ts_stripped_lines(_log_path(data_dir))


@pytest.mark.parametrize("kill_after_events", [4, 40])
def test_kill_then_resume_matches_reference(tmp_path, kill_after_events):
    # kill_after_events=4  -> mid question 0 (early)
    # kill_after_events=40 -> a later question (each q = 6 belief + 18 trade = 24 events)
    reference = _reference_log(tmp_path)
    assert len(reference) > kill_after_events

    data_dir = tmp_path / "killrun"
    log = _log_path(data_dir)

    # First attempt: launch, wait until enough events, SIGKILL mid-run.
    # `with` closes the stdout/stderr pipes even though we hard-kill the proc.
    with _run_cli(data_dir, delay=TURN_DELAY) as proc:
        try:
            _wait_until_events(log, kill_after_events)
            proc.send_signal(signal.SIGKILL)
        finally:
            proc.wait(timeout=10)
    assert proc.returncode == -signal.SIGKILL

    killed_count = _count_events(log)
    assert killed_count >= kill_after_events
    assert killed_count < len(reference), "run should have been killed before finishing"

    # Resume: same command, run to completion (no delay needed now).
    proc2 = _run_cli(data_dir)
    out, err = proc2.communicate(timeout=60)
    assert proc2.returncode == 0, f"resume run failed: {err.decode()}"

    final = ts_stripped_lines(log)
    assert final == reference, (
        f"resumed log diverged from reference (killed at {killed_count} events, "
        f"final {len(final)} vs reference {len(reference)})"
    )
