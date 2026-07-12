"""Shared test fixtures: build a small file-backed questions dataset + a
RunConfig YAML pointing at it. Used by the runner file-loading tests and the
file-backed kill-resume variant. Pure/offline — no network.
"""

import json
from pathlib import Path

import numpy as np
import yaml

from pnyx.env import generate_question
from pnyx.schemas import EnvConfig

_ENV = EnvConfig(
    n_signals=6,
    accuracies=[0.75] * 6,
    lams=[0.0] * 6,
    a_z=0.8,
    difficulty="easy",
    max_rejection_tries=2000,
)


def make_rendered_question(qid: str, seed: int):
    """A fully-rendered QuestionRecord: generated record + prose shards +
    question_text (so it satisfies the runner's questions_file contract)."""
    rec = generate_question(_ENV, np.random.default_rng(seed), qid)
    return rec.model_copy(update={
        "shards": [f"Shard {i} prose for {qid}." for i in range(6)],
        "question_text": f"Does question {qid} resolve yes?",
    })


def write_questions_file(path: Path, n: int) -> Path:
    """Write ``n`` rendered questions (q000..) to a JSONL dataset file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [make_rendered_question(f"q{i:03d}", seed=1000 + i) for i in range(n)]
    path.write_text("".join(r.model_dump_json() + "\n" for r in records))
    return path


def write_file_backed_config(
    config_path: Path,
    questions_file: Path,
    data_dir: Path,
    *,
    n_questions: int,
    n_rounds: int = 3,
) -> Path:
    """Write a RunConfig YAML that loads questions from ``questions_file``."""
    config = {
        "condition": "FILE",
        "seeds": [0],
        "n_questions": n_questions,
        "b": 40.0,
        "n_rounds": n_rounds,
        "wealth_persistent": False,
        "data_dir": str(data_dir),
        "questions_file": str(questions_file),
        "env": {
            "n_signals": 6,
            "accuracies": [0.75] * 6,
            "lams": [0.0] * 6,
            "a_z": 0.8,
            "difficulty": "easy",
            "min_single_shard_margin": 0.1,
            "max_rejection_tries": 2000,
        },
        "agents": [
            {"agent_id": f"agent{i}", "shard_indices": [i], "bankroll": 100.0}
            for i in range(6)
        ],
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config))
    return config_path
