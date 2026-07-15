"""Validate the nine P4 main-run YAML configs load into well-formed RunConfigs
with the right pools, personas, Pass-1 reuse wiring, wealth/order flags, and the
condition-D adversary. Pure parse + pydantic validation — no network, no run.
"""

from pathlib import Path

import pytest

from pnyx.prompts import PERSONAS
from pnyx.runner import load_config

_MAIN = Path(__file__).resolve().parents[1] / "pnyx" / "configs" / "main"
_ALL = ["A", "B1", "B3", "C", "W_fixed", "W_shuffled", "D_k1", "D_k3", "D_k10"]

_V4_FLASH = "deepseek/deepseek-v4-flash"
_NEMOTRON_FREE = "nvidia/nemotron-3-super-120b-a12b:free"
_LLAMA = "meta-llama/llama-3.1-8b-instruct"


def _cfg(name):
    return load_config(str(_MAIN / f"{name}.yaml"))


def _honest(config):
    return [a for a in config.agents if not a.adversary]


# ---------------------------------------------------------------------------
# Shared invariants across all nine configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL)
def test_common_shape(name):
    c = _cfg(name)
    assert c.seeds == [0, 1, 2]
    assert c.questions_file == "datasets/questions_v1.jsonl"
    assert c.n_questions == 40
    assert c.b == pytest.approx(40.0)
    assert c.n_rounds == 3
    assert c.budget_usd == pytest.approx(5.0)
    assert c.temperature == pytest.approx(0.7)
    assert c.max_tokens == 2000
    assert c.data_dir == f"data/main/{c.condition}"
    # Every honest agent is an llm with a defined model + a real frozen persona;
    # the six frozen personas are each used exactly once across the honest pool.
    for a in _honest(c):
        assert a.kind == "llm"
        assert a.model in c.models
        assert a.persona in PERSONAS
    personas = [a.persona for a in _honest(c)]
    assert set(personas) == set(PERSONAS) and len(personas) == 6
    # v4-flash always has reasoning disabled where present.
    for spec in c.models.values():
        if spec.model_id == _V4_FLASH:
            assert spec.reasoning_enabled is False


def test_conditions_and_dirs_unique():
    configs = [_cfg(n) for n in _ALL]
    assert len({c.condition for c in configs}) == 9
    assert len({c.data_dir for c in configs}) == 9


# ---------------------------------------------------------------------------
# Homogeneous pools A / B1 / B3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,model_id,api_env,base_is_openrouter", [
    ("A", _V4_FLASH, "OPENROUTER_KEY", True),
    ("B3", _NEMOTRON_FREE, "OPENROUTER_KEY", True),
    ("B1", _LLAMA, "OPENROUTER_KEY", True),
])
def test_homogeneous_pool(name, model_id, api_env, base_is_openrouter):
    c = _cfg(name)
    assert c.pass2_only is False
    assert not any(a.adversary for a in c.agents)
    assert len(c.agents) == 6
    model_ids = {c.models[a.model].model_id for a in c.agents}
    assert model_ids == {model_id}
    spec = next(iter(c.models.values()))
    assert spec.api_key_env == api_env
    assert spec.supports_json_schema is True
    assert base_is_openrouter and spec.base_url == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Mixed pool C: 2 local + 2 v4-flash + 2 free
# ---------------------------------------------------------------------------


def test_C_mixed_pool():
    c = _cfg("C")
    assert c.pass2_only is False
    counts = {}
    for a in c.agents:
        mid = c.models[a.model].model_id
        counts[mid] = counts.get(mid, 0) + 1
    assert counts == {_LLAMA: 2, _V4_FLASH: 2, _NEMOTRON_FREE: 2}
    assert c.models["llama-8b"].base_url == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# W conditions: Pass 2 only (reuse A), persistent wealth, file vs shuffled order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,order", [("W_fixed", "file"), ("W_shuffled", "shuffled")])
def test_W_conditions(name, order):
    c = _cfg(name)
    assert c.pass2_only is True
    assert c.pass1_source_dir == "data/main/A"
    assert c.wealth_persistent is True
    assert c.question_order == order
    # Pool A models + identical agent ids so the belief reuse maps cleanly.
    assert [a.agent_id for a in c.agents] == [f"p{i}" for i in range(6)]
    assert {c.models[a.model].model_id for a in c.agents} == {_V4_FLASH}
    assert not any(a.adversary for a in c.agents)


def test_W_agents_match_A():
    a, wf, ws = _cfg("A"), _cfg("W_fixed"), _cfg("W_shuffled")
    a_sig = [(x.agent_id, tuple(x.shard_indices), a.models[x.model].model_id, x.persona)
             for x in a.agents]
    for w in (wf, ws):
        w_sig = [(x.agent_id, tuple(x.shard_indices), w.models[x.model].model_id, x.persona)
                 for x in w.agents]
        assert w_sig == a_sig


# ---------------------------------------------------------------------------
# D conditions: pool C honest agents + one stealthy adversary at k× bankroll
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,bankroll", [("D_k1", 100.0), ("D_k3", 300.0), ("D_k10", 1000.0)])
def test_D_conditions(name, bankroll):
    c = _cfg(name)
    assert c.pass2_only is True
    assert c.pass1_source_dir == "data/main/C"
    assert len(c.agents) == 7  # 6 honest + adversary

    advs = [a for a in c.agents if a.adversary]
    assert len(advs) == 1
    adv = advs[0]
    assert adv.agent_id == "adv0"
    assert adv.shard_indices == []
    assert adv.kind == "llm"
    assert adv.adversary_style == "stealthy"  # all D_k are stealthy (obvious out of scope)
    assert adv.persona is None
    assert c.models[adv.model].model_id == _V4_FLASH
    assert adv.bankroll == pytest.approx(bankroll)


def test_D_honest_agents_match_C():
    c = _cfg("C")
    c_sig = [(x.agent_id, tuple(x.shard_indices), c.models[x.model].model_id, x.persona)
             for x in c.agents]
    for name in ("D_k1", "D_k3", "D_k10"):
        d = _cfg(name)
        d_sig = [(x.agent_id, tuple(x.shard_indices), d.models[x.model].model_id, x.persona)
                 for x in d.agents if not x.adversary]
        assert d_sig == c_sig
