"""Tests for pnyx.providers — the OpenAI-compatible async client.

ZERO network: every HTTP interaction goes through ``httpx.MockTransport``
(no socket is ever opened; the suite-wide guard in conftest would fail the
test if one were). Timing is exercised with an injected fake clock/sleep so
there are no real ``asyncio.sleep`` waits. Retry jitter is injected so the
backoff is deterministic.
"""

import asyncio
import json

import httpx
import pytest

from pnyx.providers import (
    CostLedger,
    Provider,
    ProviderError,
    TokenBucket,
    TurnParked,
    check_openrouter_credits,
    warn_low_credits,
)
from pnyx.schemas import Belief, CostEntry, ModelSpec, TurnKey, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(**over) -> ModelSpec:
    base = dict(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_KEY",
        model_id="deepseek/deepseek-v4-flash",
        price_in=0.077,
        price_out=0.154,
        rpm_limit=300,
        supports_json_schema=True,
    )
    base.update(over)
    return ModelSpec(**base)


def _key(agent="agent0") -> TurnKey:
    return TurnKey(
        condition="C", seed=1, question_id="q0",
        phase="independent", round=0, agent_id=agent,
    )


class FakeClock:
    """Deterministic monotonic clock + async sleep that advances it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, dt: float) -> None:
        self.sleeps.append(dt)
        self.now += dt


def _completion(content='{"prob":0.6,"rationale":"ok"}', usage=None, status=200):
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    elif status == 200:
        body["usage"] = {"prompt_tokens": 10, "completion_tokens": 20}
    return httpx.Response(status, json=body)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Token bucket timing (injected clock — no real sleeps)
# ---------------------------------------------------------------------------


def test_bucket_default_capacity_equals_rpm():
    b = TokenBucket(18)
    assert b.capacity == 18
    assert b.rate == pytest.approx(18 / 60.0)


def test_bucket_burst_within_capacity_no_sleep():
    clk = FakeClock()
    b = TokenBucket(120, clock=clk.time, sleep=clk.sleep, capacity=5)

    async def go():
        for _ in range(5):
            await b.acquire()

    _run(go())
    assert clk.sleeps == []
    assert clk.now == 0.0


def test_bucket_throttles_after_capacity():
    clk = FakeClock()
    # rpm 60 -> 1 token/sec; capacity 2.
    b = TokenBucket(60, clock=clk.time, sleep=clk.sleep, capacity=2)

    async def go():
        await b.acquire()  # immediate
        await b.acquire()  # immediate (capacity exhausted)
        await b.acquire()  # must wait 1s for one token to refill

    _run(go())
    assert clk.sleeps == [pytest.approx(1.0)]
    assert clk.now == pytest.approx(1.0)


def test_bucket_refills_over_time():
    clk = FakeClock()
    b = TokenBucket(60, clock=clk.time, sleep=clk.sleep, capacity=1)

    async def go():
        await b.acquire()          # now 0, tokens 0
        clk.now += 0.5             # half a token accrues
        await b.acquire()          # needs 0.5 more -> sleep 0.5

    _run(go())
    assert clk.sleeps == [pytest.approx(0.5)]


# ---------------------------------------------------------------------------
# Request-body shape + usage parsing
# ---------------------------------------------------------------------------


def test_request_body_json_schema_shape(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")
    cap = {}

    def handler(request):
        cap["url"] = str(request.url)
        cap["method"] = request.method
        cap["json"] = json.loads(request.content)
        cap["auth"] = request.headers.get("authorization")
        return _completion()

    spec = _spec()
    provider = Provider(client=_client(handler))
    result, usage = _run(provider.complete(
        messages=[{"role": "user", "content": "hi"}],
        schema=Belief, model_spec=spec, temperature=0.7, max_tokens=256,
    ))

    assert cap["method"] == "POST"
    assert cap["url"] == "https://openrouter.ai/api/v1/chat/completions"
    body = cap["json"]
    assert body["model"] == spec.model_id
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 256
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == Belief.model_json_schema()
    # Key is sent on the wire but must be exactly the bearer token.
    assert cap["auth"] == "Bearer sk-secret"
    # Structured result parsed into the pydantic model.
    assert isinstance(result, Belief)
    assert result.prob == 0.6
    assert usage == Usage(prompt_tokens=10, completion_tokens=20)


def test_free_text_no_response_format(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")
    cap = {}

    def handler(request):
        cap["json"] = json.loads(request.content)
        return _completion(content="hello world")

    provider = Provider(client=_client(handler))
    result, usage = _run(provider.complete(
        messages=[{"role": "user", "content": "hi"}],
        schema=None, model_spec=_spec(), temperature=0.5, max_tokens=64,
    ))
    assert "response_format" not in cap["json"]
    assert result == "hello world"


def test_schema_without_json_schema_support_parses_free_text(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")
    cap = {}

    def handler(request):
        cap["json"] = json.loads(request.content)
        return _completion(content='{"prob":0.3,"rationale":"z"}')

    spec = _spec(supports_json_schema=False)
    provider = Provider(client=_client(handler))
    result, _ = _run(provider.complete(
        messages=[{"role": "user", "content": "hi"}],
        schema=Belief, model_spec=spec, temperature=0.7, max_tokens=64,
    ))
    assert "response_format" not in cap["json"]
    assert isinstance(result, Belief)
    assert result.prob == 0.3


def test_usage_missing_defaults_to_zero(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    provider = Provider(client=_client(handler))
    _, usage = _run(provider.complete(
        messages=[], schema=None, model_spec=_spec(), temperature=0.7, max_tokens=8,
    ))
    assert usage == Usage(prompt_tokens=0, completion_tokens=0)


def test_bad_structured_output_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")

    def handler(request):
        return _completion(content='{"prob":"not-a-number"}')

    provider = Provider(client=_client(handler))
    with pytest.raises(Exception):  # pydantic ValidationError propagates to caller
        _run(provider.complete(
            messages=[], schema=Belief, model_spec=_spec(),
            temperature=0.7, max_tokens=8,
        ))


# ---------------------------------------------------------------------------
# Retry / backoff / park
# ---------------------------------------------------------------------------


def _seq_handler(statuses, *, retry_after=None):
    """Return `statuses[i]` for call i, then 200. Optional Retry-After header
    on the error responses."""
    calls = {"n": 0}

    def handler(request):
        i = calls["n"]
        calls["n"] += 1
        if i < len(statuses):
            headers = {}
            if retry_after is not None:
                headers["Retry-After"] = str(retry_after)
            return httpx.Response(statuses[i], json={"error": "boom"}, headers=headers)
        return _completion()

    return handler, calls


def test_parks_after_five_retries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")
    handler, calls = _seq_handler([429] * 10)
    clk = FakeClock()
    provider = Provider(
        client=_client(handler), sleep=clk.sleep,
        jitter=lambda cap: 0.0, max_retries=5,
    )
    with pytest.raises(TurnParked):
        _run(provider.complete(
            messages=[], schema=None, model_spec=_spec(),
            temperature=0.7, max_tokens=8,
        ))
    assert calls["n"] == 6            # initial + 5 retries
    assert len(clk.sleeps) == 5       # one sleep before each retry


def test_succeeds_after_transient_errors(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")
    handler, calls = _seq_handler([429, 500, 503])
    clk = FakeClock()
    provider = Provider(
        client=_client(handler), sleep=clk.sleep, jitter=lambda cap: 0.0,
    )
    result, _ = _run(provider.complete(
        messages=[], schema=None, model_spec=_spec(),
        temperature=0.7, max_tokens=8,
    ))
    assert result  # succeeded on the 4th attempt
    assert calls["n"] == 4
    assert len(clk.sleeps) == 3


def test_retry_after_header_is_honored(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")
    handler, _ = _seq_handler([429], retry_after=7)
    clk = FakeClock()
    provider = Provider(
        client=_client(handler), sleep=clk.sleep, jitter=lambda cap: 0.0,
    )
    _run(provider.complete(
        messages=[], schema=None, model_spec=_spec(),
        temperature=0.7, max_tokens=8,
    ))
    assert clk.sleeps == [pytest.approx(7.0)]


def test_full_jitter_exponential_backoff(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")
    handler, _ = _seq_handler([500, 500])
    clk = FakeClock()
    # jitter returns the cap => deterministic exponential 1,2 with base 1.
    provider = Provider(
        client=_client(handler), sleep=clk.sleep,
        jitter=lambda cap: cap, base_backoff=1.0,
    )
    _run(provider.complete(
        messages=[], schema=None, model_spec=_spec(),
        temperature=0.7, max_tokens=8,
    ))
    assert clk.sleeps == [pytest.approx(1.0), pytest.approx(2.0)]


def test_non_retryable_4xx_raises_not_parked(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")
    handler, calls = _seq_handler([400])
    provider = Provider(client=_client(handler), jitter=lambda cap: 0.0)
    with pytest.raises(ProviderError):
        _run(provider.complete(
            messages=[], schema=None, model_spec=_spec(),
            temperature=0.7, max_tokens=8,
        ))
    assert calls["n"] == 1  # not retried


# ---------------------------------------------------------------------------
# Cost ledger: math, persistence, resume, budget
# ---------------------------------------------------------------------------


def test_ledger_math_and_per_model_total(tmp_path):
    led = CostLedger(tmp_path / "costs.jsonl")
    spec = _spec(model_id="a", price_in=0.077, price_out=0.154)
    other = _spec(model_id="b", price_in=1.0, price_out=2.0)
    c1 = led.record(spec, Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000), _key())
    assert c1 == pytest.approx(0.077 + 0.154)
    c2 = led.record(other, Usage(prompt_tokens=1_000_000, completion_tokens=0), _key())
    assert c2 == pytest.approx(1.0)
    assert led.total() == pytest.approx(c1 + c2)
    assert led.total("a") == pytest.approx(c1)
    assert led.total("b") == pytest.approx(c2)


def test_ledger_persists_jsonl(tmp_path):
    p = tmp_path / "costs.jsonl"
    led = CostLedger(p)
    led.record(_spec(model_id="a"), Usage(prompt_tokens=2_000_000, completion_tokens=0), _key())
    lines = [ln for ln in p.read_text().splitlines() if ln]
    assert len(lines) == 1
    entry = CostEntry.model_validate_json(lines[0])
    assert entry.model_id == "a"
    assert entry.in_tokens == 2_000_000
    assert entry.key == _key()


def test_ledger_resume_reconstructs_total(tmp_path):
    p = tmp_path / "costs.jsonl"
    led = CostLedger(p)
    led.record(_spec(model_id="a"), Usage(prompt_tokens=1_000_000, completion_tokens=0), _key())
    led.record(_spec(model_id="a"), Usage(prompt_tokens=1_000_000, completion_tokens=0), _key())
    # Fresh ledger over the same file reconstructs cumulative cost.
    resumed = CostLedger(p)
    assert resumed.total() == pytest.approx(led.total())
    # ...and keeps appending correctly.
    resumed.record(_spec(model_id="a"), Usage(prompt_tokens=1_000_000, completion_tokens=0), _key())
    again = CostLedger(p)
    assert again.total() == pytest.approx(resumed.total())
    assert len([ln for ln in p.read_text().splitlines() if ln]) == 3


def test_ledger_budget_stop(tmp_path):
    led = CostLedger(tmp_path / "costs.jsonl")
    spec = _spec(model_id="x", price_in=1.0, price_out=1.0)
    led.record(spec, Usage(prompt_tokens=3_000_000, completion_tokens=0), _key())  # $3
    assert led.over_budget(2.0) is True
    assert led.over_budget(25.0) is False


# ---------------------------------------------------------------------------
# Key never leaks to logs
# ---------------------------------------------------------------------------


def test_api_key_never_printed_on_success(monkeypatch, capsys):
    secret = "sk-super-secret-DEADBEEF"
    monkeypatch.setenv("OPENROUTER_KEY", secret)

    def handler(request):
        return _completion(content="ok")

    provider = Provider(client=_client(handler))
    _run(provider.complete(
        messages=[], schema=None, model_spec=_spec(),
        temperature=0.7, max_tokens=8,
    ))
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


def test_api_key_never_printed_on_park(monkeypatch, capsys):
    secret = "sk-super-secret-DEADBEEF"
    monkeypatch.setenv("OPENROUTER_KEY", secret)
    handler, _ = _seq_handler([500] * 10)
    provider = Provider(
        client=_client(handler), sleep=FakeClock().sleep, jitter=lambda cap: 0.0,
    )
    with pytest.raises(TurnParked) as exc:
        _run(provider.complete(
            messages=[], schema=None, model_spec=_spec(),
            temperature=0.7, max_tokens=8,
        ))
    assert secret not in str(exc.value)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


# ---------------------------------------------------------------------------
# Free-tier credit check
# ---------------------------------------------------------------------------


def test_credit_check_returns_remaining():
    def handler(request):
        assert request.method == "GET"
        assert str(request.url) == "https://openrouter.ai/api/v1/key"
        return httpx.Response(200, json={"data": {"limit": 5.0, "usage": 1.0}})

    credits = _run(check_openrouter_credits(_client(handler), "sk-x"))
    assert credits == pytest.approx(4.0)


def test_credit_check_unlimited_returns_none():
    def handler(request):
        return httpx.Response(200, json={"data": {"limit": None, "usage": 3.0}})

    assert _run(check_openrouter_credits(_client(handler), "sk-x")) is None


def test_credit_check_unexpected_shape_returns_none():
    def handler(request):
        return httpx.Response(200, json={"totally": "different"})

    assert _run(check_openrouter_credits(_client(handler), "sk-x")) is None


def test_credit_check_http_error_returns_none():
    def handler(request):
        return httpx.Response(500, json={"error": "x"})

    assert _run(check_openrouter_credits(_client(handler), "sk-x")) is None


def test_credit_check_does_not_leak_key(capsys):
    secret = "sk-credit-secret-CAFE"

    def handler(request):
        assert request.headers.get("authorization") == f"Bearer {secret}"
        return httpx.Response(200, json={"data": {"limit": 5.0, "usage": 1.0}})

    _run(check_openrouter_credits(_client(handler), secret))
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


def test_warn_low_credits_prints_loud_warning(capsys):
    warned = warn_low_credits(5.0, threshold=10.0)
    assert warned is True
    err = capsys.readouterr().err
    assert "WARNING" in err.upper()


def test_warn_low_credits_silent_when_sufficient(capsys):
    assert warn_low_credits(20.0, threshold=10.0) is False
    assert capsys.readouterr().err == ""


def test_warn_low_credits_silent_when_none(capsys):
    assert warn_low_credits(None, threshold=10.0) is False
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Network guard is actually armed (positive control for the conftest fixture)
# ---------------------------------------------------------------------------


def test_socket_guard_blocks_real_inet_connection():
    import socket

    from tests.conftest import NetworkBlocked

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlocked):
            s.connect(("93.184.216.34", 80))  # example.com; must never dial out
    finally:
        s.close()
