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
    SchemaParseError,
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


def test_reasoning_param_sent_only_when_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")
    cap = {}

    def handler(request):
        cap["json"] = json.loads(request.content)
        return _completion()

    # Default (None): no reasoning key in the body.
    provider = Provider(client=_client(handler))
    _run(provider.complete(
        messages=[{"role": "user", "content": "hi"}],
        schema=Belief, model_spec=_spec(), temperature=0.7, max_tokens=256,
    ))
    assert "reasoning" not in cap["json"]

    # Explicitly disabled: reasoning={"enabled": False} on the wire.
    spec_off = _spec().model_copy(update={"reasoning_enabled": False})
    provider = Provider(client=_client(handler))
    _run(provider.complete(
        messages=[{"role": "user", "content": "hi"}],
        schema=Belief, model_spec=spec_off, temperature=0.7, max_tokens=256,
    ))
    assert cap["json"]["reasoning"] == {"enabled": False}


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


def test_bad_structured_output_raises_schema_parse_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")
    raw = '{"prob":"not-a-number"}'

    def handler(request):
        return _completion(content=raw)

    provider = Provider(client=_client(handler))
    with pytest.raises(SchemaParseError) as exc:
        _run(provider.complete(
            messages=[], schema=Belief, model_spec=_spec(),
            temperature=0.7, max_tokens=8,
        ))
    # The whole point of the typed exception: usage for this billed 200 must
    # reach the caller instead of being swallowed with the ValidationError.
    assert exc.value.usage == Usage(prompt_tokens=10, completion_tokens=20)
    assert exc.value.content == raw
    assert "prob" in exc.value.error


def test_null_content_with_schema_raises_schema_parse_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")

    def handler(request):
        return _completion(content=None)

    provider = Provider(client=_client(handler))
    with pytest.raises(SchemaParseError) as exc:
        _run(provider.complete(
            messages=[], schema=Belief, model_spec=_spec(),
            temperature=0.7, max_tokens=8,
        ))
    assert exc.value.content is None
    assert exc.value.usage == Usage(prompt_tokens=10, completion_tokens=20)
    assert exc.value.error


# ---------------------------------------------------------------------------
# Malformed 200 bodies -> typed ProviderError (not KeyError/IndexError)
# ---------------------------------------------------------------------------


def test_malformed_choices_missing_raises_provider_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")

    def handler(request):
        return httpx.Response(
            200, json={"usage": {"prompt_tokens": 5, "completion_tokens": 7}}
        )

    provider = Provider(client=_client(handler))
    with pytest.raises(ProviderError) as exc:
        _run(provider.complete(
            messages=[], schema=None, model_spec=_spec(),
            temperature=0.7, max_tokens=8,
        ))
    assert "malformed response body" in str(exc.value)
    assert "200" in str(exc.value)
    # usage was still computable from the malformed body -> bill it.
    assert exc.value.usage == Usage(prompt_tokens=5, completion_tokens=7)


def test_malformed_body_not_object_bills_nothing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")

    def handler(request):
        return httpx.Response(200, json=["not", "an", "object"])

    provider = Provider(client=_client(handler))
    with pytest.raises(ProviderError) as exc:
        _run(provider.complete(
            messages=[], schema=None, model_spec=_spec(),
            temperature=0.7, max_tokens=8,
        ))
    # Usage wasn't computable at all from this body -> bill nothing.
    assert exc.value.usage is None


def test_null_content_without_schema_raises_provider_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")

    def handler(request):
        return _completion(content=None)

    provider = Provider(client=_client(handler))
    with pytest.raises(ProviderError) as exc:
        _run(provider.complete(
            messages=[], schema=None, model_spec=_spec(),
            temperature=0.7, max_tokens=8,
        ))
    assert "malformed response body" in str(exc.value)
    assert exc.value.usage == Usage(prompt_tokens=10, completion_tokens=20)


# ---------------------------------------------------------------------------
# Missing API key env var -> typed ProviderError (not bare KeyError)
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_provider_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)

    def handler(request):
        raise AssertionError("must not reach the network layer without a key")

    provider = Provider(client=_client(handler))
    with pytest.raises(ProviderError) as exc:
        _run(provider.complete(
            messages=[], schema=None, model_spec=_spec(),
            temperature=0.7, max_tokens=8,
        ))
    assert "OPENROUTER_KEY" in str(exc.value)


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


def test_retry_reacquires_bucket_token(monkeypatch):
    """Each attempt -- including retries -- must re-acquire from the rate
    bucket, not just the first. With capacity 1 (rpm_limit=1) and zeroed
    backoff jitter, the fake clock never advances on its own, so every
    attempt after the first must pay a full 60s bucket-refill wait. Before
    the fix (single acquire per `complete()` call) this second/third wait
    would never happen and `clk.sleeps` would only contain the zero backoffs.
    """
    monkeypatch.setenv("OPENROUTER_KEY", "sk-secret")
    handler, calls = _seq_handler([500, 500])
    clk = FakeClock()
    provider = Provider(
        client=_client(handler), clock=clk.time, sleep=clk.sleep,
        jitter=lambda cap: 0.0,
    )
    spec = _spec(rpm_limit=1)  # capacity 1, refill rate 1/60 token/sec
    _run(provider.complete(
        messages=[], schema=None, model_spec=spec,
        temperature=0.7, max_tokens=8,
    ))
    assert calls["n"] == 3  # initial + 2 retries: 3 attempts, 3 acquisitions
    assert clk.sleeps == [
        pytest.approx(0.0), pytest.approx(60.0),
        pytest.approx(0.0), pytest.approx(60.0),
    ]


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


def test_ledger_tolerates_torn_final_line(tmp_path):
    """A kill mid-write can leave a truncated final line with no trailing
    newline -- mirrors runner.read_events' event-log tolerance. It must be
    skipped, not raised, and the two complete entries before it still load."""
    p = tmp_path / "costs.jsonl"
    led = CostLedger(p)
    led.record(_spec(model_id="a"), Usage(prompt_tokens=1_000_000, completion_tokens=0), _key())
    led.record(_spec(model_id="a"), Usage(prompt_tokens=2_000_000, completion_tokens=0), _key())
    expected_total = led.total()
    # Simulate a kill mid-write: append a partial JSON fragment with NO
    # trailing newline.
    with open(p, "a") as f:
        f.write('{"model_id": "a", "in_tokens": 999')

    resumed = CostLedger(p)
    assert resumed.total() == pytest.approx(expected_total)
    assert len(resumed._entries) == 2
    # And it keeps working / appending correctly afterward.
    resumed.record(_spec(model_id="a"), Usage(prompt_tokens=1_000_000, completion_tokens=0), _key())
    assert resumed.total() == pytest.approx(expected_total + resumed._entries[-1].cost)


def test_ledger_malformed_non_final_line_is_hard_error(tmp_path):
    """A malformed line that is NOT the final line (or the file ends in a
    trailing newline) is a real corruption, not a kill-mid-write artifact --
    it must raise, not be silently skipped."""
    p = tmp_path / "costs.jsonl"
    led = CostLedger(p)
    led.record(_spec(model_id="a"), Usage(prompt_tokens=1_000_000, completion_tokens=0), _key())
    # Corrupt a non-final line by appending garbage followed by a valid line.
    with open(p, "a") as f:
        f.write("not-json-garbage\n")
        f.write(CostEntry(model_id="a", in_tokens=1, out_tokens=1, cost=0.0).model_dump_json())
        f.write("\n")

    with pytest.raises(Exception):
        CostLedger(p)


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


def test_api_key_never_printed_on_4xx(monkeypatch, capsys):
    secret = "sk-super-secret-DEADBEEF"
    monkeypatch.setenv("OPENROUTER_KEY", secret)
    handler, _ = _seq_handler([403])
    provider = Provider(client=_client(handler), jitter=lambda cap: 0.0)
    with pytest.raises(ProviderError) as exc:
        _run(provider.complete(
            messages=[], schema=None, model_spec=_spec(),
            temperature=0.7, max_tokens=8,
        ))
    assert secret not in str(exc.value)
    assert secret not in repr(exc.value)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


# ---------------------------------------------------------------------------
# Free-tier credit check
# ---------------------------------------------------------------------------


def _route(*, credits=None, key=None):
    """Build a MockTransport handler routing ``/credits`` and ``/key`` to the
    given httpx.Response factories (each a callable(request) -> Response, or a
    Response). A missing route returns 404."""
    def handler(request):
        path = request.url.path
        target = credits if path.endswith("/credits") else key if path.endswith("/key") else None
        if target is None:
            return httpx.Response(404, json={"error": "no route"})
        return target(request) if callable(target) else target
    return handler


def test_credit_check_prefers_lifetime_from_credits_endpoint():
    def credits(request):
        assert request.method == "GET"
        assert request.url.path.endswith("/credits")
        return httpx.Response(200, json={"data": {"total_credits": 5.0, "total_usage": 1.0}})

    status = _run(check_openrouter_credits(_client(_route(credits=credits)), "sk-x"))
    assert status.credits == pytest.approx(4.0)
    assert status.is_lifetime is True


def test_credit_check_falls_back_to_key_headroom_flagged_not_lifetime():
    # No /credits endpoint (404) -> fall back to the key's limit-usage headroom,
    # flagged is_lifetime=False so the warning stays honest about what it knows.
    key = httpx.Response(200, json={"data": {"limit": 5.0, "usage": 1.0}})
    status = _run(check_openrouter_credits(_client(_route(key=key)), "sk-x"))
    assert status.credits == pytest.approx(4.0)
    assert status.is_lifetime is False


def test_credit_check_unlimited_key_returns_none():
    key = httpx.Response(200, json={"data": {"limit": None, "usage": 3.0}})
    status = _run(check_openrouter_credits(_client(_route(key=key)), "sk-x"))
    assert status.credits is None
    assert status.is_lifetime is False


def test_credit_check_unexpected_shape_returns_none():
    both = httpx.Response(200, json={"totally": "different"})
    status = _run(check_openrouter_credits(_client(_route(credits=both, key=both)), "sk-x"))
    assert status.credits is None


def test_credit_check_http_error_returns_none():
    err = httpx.Response(500, json={"error": "x"})
    status = _run(check_openrouter_credits(_client(_route(credits=err, key=err)), "sk-x"))
    assert status.credits is None


def test_credit_check_does_not_leak_key(capsys):
    secret = "sk-credit-secret-CAFE"

    def credits(request):
        assert request.headers.get("authorization") == f"Bearer {secret}"
        return httpx.Response(200, json={"data": {"total_credits": 5.0, "total_usage": 1.0}})

    _run(check_openrouter_credits(_client(_route(credits=credits)), secret))
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


def test_warn_low_credits_prints_loud_warning(capsys):
    warned = warn_low_credits(5.0, threshold=10.0)
    assert warned is True
    err = capsys.readouterr().err
    assert "WARNING" in err.upper()
    assert "lifetime credit is $5.00" in err


def test_warn_low_credits_key_fallback_softens_message(capsys):
    # Key-limit fallback: do NOT assert a lifetime number; say we couldn't
    # determine lifetime credits.
    warned = warn_low_credits(5.0, threshold=10.0, is_lifetime=False)
    assert warned is True
    err = capsys.readouterr().err
    assert "could not determine lifetime" in err.lower()
    assert "lifetime credit is $5.00" not in err


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
