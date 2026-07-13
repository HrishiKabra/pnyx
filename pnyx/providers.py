"""Pnyx LLM provider client — OpenAI-compatible async access with client-side
rate limiting, retry/park semantics, and a persistent cost ledger.

Scope (P3, Task 1). This module is the ONLY place the codebase talks to a
remote model endpoint. It is provider-agnostic: everything an endpoint needs
lives on a :class:`~pnyx.schemas.ModelSpec` (base_url, api_key_env, model_id,
prices, rpm_limit, supports_json_schema). The runner (Task 2) constructs one
:class:`Provider` sharing a single ``httpx.AsyncClient`` and drives it from the
async turn seam.

Design guarantees
-----------------
* **No key leakage.** The bearer key is read from the environment per call and
  placed only in the ``Authorization`` header. It is never stored on any object,
  never included in an exception message, never printed. Tests assert this. A
  missing key env var raises :class:`ProviderError` naming the env var, never
  the (absent) key material.
* **Deterministic under test.** The token bucket takes an injectable monotonic
  ``clock`` + async ``sleep``; retry backoff takes an injectable ``jitter``.
  Nothing here calls ``time`` / ``random`` / ``asyncio.sleep`` directly when the
  caller supplies substitutes, so timing behaviour is exercised with zero real
  waits and zero network (``httpx.MockTransport``).
* **Rate limiting.** One :class:`TokenBucket` per ``model_id`` (the provider
  bucket key), refilling at ``rpm_limit`` tokens/minute. asyncio-safe via an
  internal lock; monotonic-clock based so wall-clock jumps can't unthrottle it.
  A token is re-acquired on every attempt — including retries — so a retry
  storm can never fire off-bucket and blow through a free-tier rpm cap.
* **Retry / park.** ``429`` and ``5xx`` responses (and transport errors) are
  retried up to ``max_retries`` times, honouring ``Retry-After`` when present
  plus full jitter to avoid a thundering herd. After the budget of retries is
  exhausted the call raises :class:`TurnParked` — the runner treats a parked
  turn as skipped-this-run (left incomplete in the event log, retried on
  resume) rather than crashing the whole run.
* **Parse failures still bill their usage.** A 200 whose body fails schema
  validation (including a null ``content`` under a schema) raises
  :class:`SchemaParseError` carrying the already-computed ``.usage``, the raw
  ``.content``, and ``.error`` — never the API key or the full request —
  instead of letting the ``ValidationError`` escape unbilled. A structurally
  malformed 200 (missing/null content with no schema) raises
  :class:`ProviderError` instead, with ``.usage`` set when computable and
  ``None`` (bill nothing) when it isn't. The caller (Task 2 runner) bills
  ``.usage`` against the cost ledger and implements the one-retry
  contract — this module never retries a parse failure itself.

The cost ledger (:class:`CostLedger`) is an append-only JSONL file mirroring
the event log's atomic append+fsync durability, reconstructed from disk on
resume so cumulative spend (and the budget hard-stop) survives a kill.
"""

import asyncio
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from pnyx.schemas import CostEntry, ModelSpec, TurnKey, Usage

__all__ = [
    "Provider",
    "TokenBucket",
    "CostLedger",
    "TurnParked",
    "ProviderError",
    "SchemaParseError",
    "check_openrouter_credits",
    "warn_low_credits",
]

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BASE_BACKOFF = 1.0


class ProviderError(RuntimeError):
    """A non-retryable provider failure: a 4xx that isn't rate-limiting, a
    missing API-key environment variable, or a structurally malformed 200
    body (missing/null content with no schema to parse it into).

    Carries no API key and no request body: built from the status code,
    response URL, and/or a descriptive message only. ``usage`` is optional —
    set when a malformed 200 body still contained a computable ``usage``
    block (so the caller can still bill the ledger for a billed-but-malformed
    call); ``None`` when usage could not be computed at all (bill nothing).
    """

    def __init__(self, message: str, *, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class SchemaParseError(Exception):
    """Raised when a 200 response's content fails schema validation.

    Carries ``.usage`` (the already-computed :class:`Usage` for the billed
    call — the caller bills it even though the reply was unusable),
    ``.content`` (the raw text that failed to parse, possibly ``None``), and
    ``.error`` (the validation failure message as a string). Never carries
    the API key or the full request. The provider does not retry a parse
    failure itself — that one-retry-with-error-appended contract belongs to
    the caller (Task 2 runner).
    """

    def __init__(
        self, *, usage: Usage | None, content: str | None, error: str
    ) -> None:
        super().__init__(f"schema parse failed: {error}")
        self.usage = usage
        self.content = content
        self.error = error


class TurnParked(Exception):
    """Raised when a turn's LLM call exhausts its retry budget.

    The runner catches this, leaves the turn incomplete in the event log
    (so a later resume picks it up), and continues the run — a parked turn
    must never crash the whole experiment.
    """


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (per model_id)
# ---------------------------------------------------------------------------


class TokenBucket:
    """asyncio-safe token bucket keyed on a single provider bucket.

    ``rpm_limit`` requests are permitted per minute (refill rate
    ``rpm_limit / 60`` tokens/second); ``capacity`` bounds the burst and
    defaults to ``rpm_limit``. Each :meth:`acquire` consumes one token,
    awaiting an injected ``sleep`` until a token is available. Time is read
    from the injected monotonic ``clock`` so a wall-clock adjustment can't
    hand out extra tokens.
    """

    def __init__(
        self,
        rpm_limit: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
        capacity: float | None = None,
    ) -> None:
        if rpm_limit <= 0:
            raise ValueError("rpm_limit must be positive")
        self.rate = rpm_limit / 60.0
        self.capacity = float(rpm_limit if capacity is None else capacity)
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
                await self._sleep(wait)


# ---------------------------------------------------------------------------
# Cost ledger
# ---------------------------------------------------------------------------


class CostLedger:
    """Append-only JSONL ledger of per-call token cost, keyed by turn.

    One line per call (``model_id, in_tokens, out_tokens, cost, key``),
    written with the same flush+fsync durability as the event log. On
    construction the ledger reconstructs cumulative spend by replaying an
    existing file, so the budget hard-stop is correct across a kill/resume.

    Replay mirrors the event log's tolerance (see ``runner.read_events``): a
    truncated final line (a write killed mid-line, so the file does not end
    in a newline) is skipped rather than raised; a malformed line anywhere
    else is a hard error.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: list[CostEntry] = []
        if self.path.exists():
            text = self.path.read_text()
            if text:
                ended_newline = text.endswith("\n")
                lines = text.splitlines()
                for idx, line in enumerate(lines):
                    if not line.strip():
                        continue
                    try:
                        self._entries.append(CostEntry.model_validate_json(line))
                    except Exception:
                        is_last = idx == len(lines) - 1
                        if is_last and not ended_newline:
                            # Truncated final write — ignore (kill-safe contract).
                            break
                        raise

    def record(
        self, spec: ModelSpec, usage: Usage, key: TurnKey | None = None
    ) -> float:
        """Compute the call's dollar cost, append it durably, and return it."""
        cost = spec.cost(usage.prompt_tokens, usage.completion_tokens)
        entry = CostEntry(
            model_id=spec.model_id,
            in_tokens=usage.prompt_tokens,
            out_tokens=usage.completion_tokens,
            cost=cost,
            key=key,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(entry.model_dump_json())
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        self._entries.append(entry)
        return cost

    def total(self, model_id: str | None = None) -> float:
        """Cumulative cost across all recorded calls, or just one model_id."""
        return sum(
            e.cost for e in self._entries
            if model_id is None or e.model_id == model_id
        )

    def over_budget(self, budget_usd: float) -> bool:
        """True once cumulative spend has reached the budget (hard-stop)."""
        return self.total() >= budget_usd


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class Provider:
    """OpenAI-compatible async chat client with rate limiting + retry/park.

    ``complete`` sends one ``POST {base_url}/chat/completions``; when a schema
    is supplied and the model ``supports_json_schema`` it requests a strict
    ``response_format`` json_schema and parses the reply into the model, else
    it parses the free-text content with ``model_validate_json``. Returns
    ``(text_or_obj, Usage)``.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
        jitter: Callable[[float], float] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_backoff: float = _DEFAULT_BASE_BACKOFF,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._own_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._clock = clock
        self._sleep = sleep
        self._rng = random.Random()
        self._jitter = jitter or (lambda cap: self._rng.uniform(0.0, cap))
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._buckets: dict[str, TokenBucket] = {}

    # -- lifecycle ---------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying client iff this Provider created it."""
        if self._own_client:
            await self._client.aclose()

    # -- rate limiting -----------------------------------------------------

    def _bucket(self, spec: ModelSpec) -> TokenBucket:
        # No await between get and set: creation is atomic under asyncio.
        bucket = self._buckets.get(spec.model_id)
        if bucket is None:
            bucket = TokenBucket(
                spec.rpm_limit, clock=self._clock, sleep=self._sleep
            )
            self._buckets[spec.model_id] = bucket
        return bucket

    # -- public API --------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, str]],
        schema: type | None,
        model_spec: ModelSpec,
        temperature: float,
        max_tokens: int,
    ) -> tuple[Any, Usage]:
        """Rate-limited, retrying chat completion.

        Returns ``(obj, usage)`` where ``obj`` is a parsed ``schema`` instance
        when a schema is given, else the raw text content. Raises:

        * :class:`TurnParked` if retries are exhausted.
        * :class:`ProviderError` on a non-retryable HTTP error, a missing
          API-key environment variable, or a structurally malformed 200 body
          (missing/null content with no schema) — ``.usage`` is set when
          computable, else ``None`` (bill nothing).
        * :class:`SchemaParseError` when a 200's content fails schema
          validation (including null content under a schema) — carries the
          already-computed ``.usage`` so the caller can still bill the ledger
          for the billed-but-unusable call and implement the one-retry
          contract; the provider itself never retries a parse failure.

        Each attempt (including retries) re-acquires a token from the
        model's rate bucket, so a retry storm can't fire off-bucket.
        """
        body: dict[str, Any] = {
            "model": model_spec.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model_spec.reasoning_enabled is not None:
            body["reasoning"] = {"enabled": model_spec.reasoning_enabled}
        if schema is not None and model_spec.supports_json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(schema),
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            }

        resp = await self._request_with_retry(model_spec, body)
        data = resp.json()
        usage = _try_parse_usage(data)

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"HTTP 200 from model {model_spec.model_id!r}: malformed "
                f"response body ({exc})",
                usage=usage,
            ) from exc

        if content is None:
            if schema is None:
                raise ProviderError(
                    f"HTTP 200 from model {model_spec.model_id!r}: malformed "
                    "response body (null content)",
                    usage=usage,
                )
            raise SchemaParseError(usage=usage, content=content, error="content is null")

        if schema is None:
            return content, usage

        try:
            return schema.model_validate_json(content), usage
        except ValidationError as exc:
            raise SchemaParseError(
                usage=usage, content=content, error=str(exc)
            ) from exc

    # -- retry loop --------------------------------------------------------

    async def _request_with_retry(
        self, spec: ModelSpec, body: dict[str, Any]
    ) -> httpx.Response:
        url = f"{spec.base_url}/chat/completions"
        try:
            api_key = os.environ[spec.api_key_env]
        except KeyError:
            raise ProviderError(
                f"missing API key: environment variable {spec.api_key_env!r} "
                "is not set"
            ) from None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        retries = 0
        while True:
            # Re-acquire on every attempt (including retries) so a retry
            # storm can never fire off-bucket and blow through the rpm cap.
            await self._bucket(spec).acquire()
            try:
                resp = await self._client.post(url, json=body, headers=headers)
            except httpx.TransportError:
                retries += 1
                if retries > self._max_retries:
                    raise TurnParked(
                        f"parked: transport error after {self._max_retries} "
                        f"retries for model {spec.model_id!r}"
                    )
                await self._sleep(self._backoff(retries, None))
                continue

            if resp.status_code in _RETRYABLE_STATUS:
                retries += 1
                if retries > self._max_retries:
                    raise TurnParked(
                        f"parked: HTTP {resp.status_code} after "
                        f"{self._max_retries} retries for model {spec.model_id!r}"
                    )
                await self._sleep(self._backoff(retries, _retry_after(resp)))
                continue

            if resp.status_code >= 400:
                # Non-retryable client error. Surface status + URL only — never
                # the Authorization header / key.
                raise ProviderError(
                    f"HTTP {resp.status_code} from {resp.request.url} "
                    f"for model {spec.model_id!r}"
                )
            return resp

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        """Seconds to wait before the next retry.

        Honours ``Retry-After`` when the server supplied it (adding a small
        full-jitter term to de-synchronise clients); otherwise full jitter over
        an exponentially growing cap ``base_backoff * 2**(attempt-1)``.
        """
        if retry_after is not None:
            return retry_after + self._jitter(self._base_backoff)
        cap = self._base_backoff * (2 ** (attempt - 1))
        return self._jitter(cap)


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _schema_name(schema: type) -> str:
    return getattr(schema, "__name__", "output").lower() or "output"


def _parse_usage(data: dict[str, Any]) -> Usage:
    usage = data.get("usage") or {}
    return Usage(
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
    )


def _try_parse_usage(data: Any) -> Usage | None:
    """Best-effort :func:`_parse_usage`: ``None`` (bill nothing) when the
    body is too malformed to even compute usage from, e.g. not a mapping."""
    try:
        return _parse_usage(data)
    except Exception:
        return None


def _retry_after(resp: httpx.Response) -> float | None:
    """Parse a ``Retry-After`` header as seconds; None if absent/unparseable
    (an HTTP-date form falls back to jittered backoff)."""
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Free-tier credit check
# ---------------------------------------------------------------------------


async def check_openrouter_credits(
    client: httpx.AsyncClient,
    api_key: str,
    *,
    base_url: str = "https://openrouter.ai/api/v1",
) -> float | None:
    """Return remaining OpenRouter credit (limit - usage) in dollars, or None.

    ``GET {base_url}/key`` returns ``{"data": {"limit", "usage", ...}}``. A
    None ``limit`` means an unlimited/pay-as-you-go key (no credit ceiling to
    warn about) → None. Any unexpected shape or HTTP/transport error is
    swallowed → None (the credit check must never block a run). The key is
    sent only in the Authorization header.
    """
    try:
        resp = await client.get(
            f"{base_url}/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if resp.status_code >= 400:
            return None
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return None

    block = data.get("data") if isinstance(data, dict) else None
    if not isinstance(block, dict) or "limit" not in block:
        return None
    limit = block.get("limit")
    if limit is None:
        return None
    try:
        return float(limit) - float(block.get("usage") or 0.0)
    except (TypeError, ValueError):
        return None


def warn_low_credits(
    credits: float | None,
    *,
    threshold: float = 10.0,
    stream=None,
) -> bool:
    """Print a LOUD warning to stderr when free-tier credit is below threshold.

    Returns True iff a warning was printed. ``None`` credit (unknown/unlimited)
    never warns. Does not block — the free tier still works, but a low
    lifetime-credit key risks the 50-requests/day cap.
    """
    if credits is None or credits >= threshold:
        return False
    out = stream if stream is not None else sys.stderr
    bar = "!" * 60
    print(bar, file=out)
    print(
        f"WARNING: OpenRouter lifetime credit is ${credits:.2f} (< ${threshold:.2f}). "
        "Free models are capped at ~50 requests/day until credits are topped up.",
        file=out,
    )
    print(bar, file=out)
    return True
