"""Validate the four P3 pilot YAML configs load into a well-formed
``RunConfig`` (condition/b/data_dir/model/agent shape). Pure parse +
pydantic validation — no network, no run.
"""

from pathlib import Path

import pytest

from pnyx.prompts import PERSONAS
from pnyx.runner import load_config

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "pnyx" / "configs"

_B_SWEEP = [
    ("pilot_b20.yaml", "PILOT_B20", 20.0, "data/pilot_b20", "deepseek-v4-flash",
     "deepseek/deepseek-v4-flash", 0.077, 0.154, 300),
    ("pilot_b40.yaml", "PILOT_B40", 40.0, "data/pilot_b40", "deepseek-v4-flash",
     "deepseek/deepseek-v4-flash", 0.077, 0.154, 300),
    ("pilot_b80.yaml", "PILOT_B80", 80.0, "data/pilot_b80", "deepseek-v4-flash",
     "deepseek/deepseek-v4-flash", 0.077, 0.154, 300),
]


@pytest.mark.parametrize(
    "filename,condition,b,data_dir,model_key,model_id,price_in,price_out,rpm",
    _B_SWEEP,
)
def test_b_sweep_configs_load_and_validate(
    filename, condition, b, data_dir, model_key, model_id, price_in, price_out, rpm
):
    config = load_config(str(_CONFIGS_DIR / filename))

    assert config.condition == condition
    assert config.seeds == [0]
    assert config.n_questions == 10
    assert config.b == pytest.approx(b)
    assert config.n_rounds == 3
    assert config.data_dir == data_dir
    assert config.questions_file == "datasets/questions_pilot_v1.jsonl"
    assert config.budget_usd == pytest.approx(2.0)
    assert config.temperature == pytest.approx(0.7)

    assert set(config.models.keys()) == {model_key}
    spec = config.models[model_key]
    assert spec.base_url == "https://openrouter.ai/api/v1"
    assert spec.api_key_env == "OPENROUTER_KEY"
    assert spec.model_id == model_id
    assert spec.price_in == pytest.approx(price_in)
    assert spec.price_out == pytest.approx(price_out)
    assert spec.rpm_limit == rpm
    assert spec.supports_json_schema is True

    assert [a.agent_id for a in config.agents] == [f"p{i}" for i in range(6)]
    assert [a.shard_indices for a in config.agents] == [[i] for i in range(6)]
    for a in config.agents:
        assert a.kind == "llm"
        assert a.model == model_key
        assert a.bankroll == pytest.approx(100.0)
    # All six fixed personas used, each exactly once.
    personas = [a.persona for a in config.agents]
    assert set(personas) == set(PERSONAS.keys())
    assert len(personas) == len(set(personas)) == 6


def test_free_pool_config_loads_and_validates():
    config = load_config(str(_CONFIGS_DIR / "pilot_free_b40.yaml"))

    assert config.condition == "PILOT_FREE_B40"
    assert config.seeds == [0]
    assert config.n_questions == 10
    assert config.b == pytest.approx(40.0)
    assert config.n_rounds == 3
    assert config.data_dir == "data/pilot_free_b40"
    assert config.questions_file == "datasets/questions_pilot_v1.jsonl"
    assert config.budget_usd == pytest.approx(2.0)

    assert set(config.models.keys()) == {"nemotron-120b-free"}
    spec = config.models["nemotron-120b-free"]
    assert spec.base_url == "https://openrouter.ai/api/v1"
    assert spec.api_key_env == "OPENROUTER_KEY"
    assert spec.model_id == "nvidia/nemotron-3-super-120b-a12b:free"
    assert spec.price_in == pytest.approx(0.0)
    assert spec.price_out == pytest.approx(0.0)
    assert spec.rpm_limit == 18
    assert spec.supports_json_schema is True

    assert [a.agent_id for a in config.agents] == [f"p{i}" for i in range(6)]
    for a in config.agents:
        assert a.kind == "llm"
        assert a.model == "nemotron-120b-free"
    personas = [a.persona for a in config.agents]
    assert set(personas) == set(PERSONAS.keys())
    assert len(personas) == 6


def test_b_sweep_uses_distinct_conditions_and_data_dirs():
    configs = [load_config(str(_CONFIGS_DIR / name)) for name, *_ in _B_SWEEP]
    assert len({c.condition for c in configs}) == 3
    assert len({c.data_dir for c in configs}) == 3
