"""Pnyx agents — the Agent protocol and the deterministic zero-intelligence
MockAgent trader.

P1 scope only: no LLM providers, no network calls, no API plumbing. Real
LLM-backed agents arrive in P3 (CLAUDE.md §8 "P3 — Pilot"); Global
Constraints requires P1 to make ZERO API calls. MockAgent is a pure
function of (oracle posterior, RNG state): the same seed always reproduces
the same beliefs and trades.

This module never imports pnyx.market: affordability
(``max_affordable_yes`` / ``max_affordable_no``) is computed by the runner
(Task 5) via market.py and handed to the agent on ``TradeView``; the
requested share count is clamped server-side regardless of what the agent
asks for (Global Constraints — "never trust the agent's number").
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import numpy as np

from pnyx.prompts import PROMPT_VERSION, pass1_messages, pass2_messages, retry_user_message
from pnyx.providers import ProviderError, SchemaParseError
from pnyx.schemas import (
    AgentSpec,
    Belief,
    BeliefView,
    ModelSpec,
    Trade,
    TradeView,
    Usage,
)

__all__ = [
    "AgentSpec",
    "Agent",
    "MockAgent",
    "LLMAgent",
    "BeliefResult",
    "TradeResult",
    "PROMPT_VERSION",
]

# What the agent bills each completed provider call against. The runner passes
# a closure binding the cost ledger + this turn's key; ``None`` usage is a
# no-op (nothing to bill).
BillFn = Callable[[Usage | None], None]

# Default output cap when the config leaves ``max_tokens`` unset — the Belief /
# Trade payloads are tiny, but rationale strings can run to a few sentences.
_DEFAULT_MAX_TOKENS = 512


@dataclass
class BeliefResult:
    """A Pass-1 elicitation outcome: the belief plus whether it is the
    parse-failure fallback (``prob=0.5`` recorded with ``parse_failed=True``)."""

    belief: Belief
    parse_failed: bool


@dataclass
class TradeResult:
    """A Pass-2 trade outcome: the trade plus the parse-failure flag (a failed
    parse falls back to ``action="hold"`` / ``belief=0.5``)."""

    trade: Trade
    parse_failed: bool


@runtime_checkable
class Agent(Protocol):
    """Structural protocol every agent kind (MockAgent today, LLM-backed
    agents in P3) must satisfy.

    Two turns per question per the two-pass elicitation protocol: an
    independent belief (Pass 1, no market context) and, once per market
    round, a trade decision (Pass 2).
    """

    def elicit_belief(self, view: BeliefView) -> Belief:
        """Pass-1 independent belief: the agent's shard + persona only, no
        prices, no other agents, no market context."""
        ...

    def decide_trade(self, view: TradeView) -> Trade:
        """Pass-2 market-round trade decision given the current price and
        the agent's own affordability at this instant."""
        ...


class MockAgent:
    """Zero-intelligence trader: deterministic given its oracle posterior
    and a seeded ``numpy.random.Generator``. Makes zero network/API calls.

    Constructed with the exact oracle posterior for its own shard subset
    (``p_i``, a float the runner reads from
    ``QuestionRecord.posterior_table`` for this agent's ``shard_indices``)
    and an ``numpy.random.Generator`` the runner seeds per
    ``(seed, agent_id, question_id)`` — that seeding is what makes a full
    run reproducible (Global Constraints, "Determinism").

    Belief rule
    -----------
    ``clip(p_i + Normal(0, sigma=BELIEF_SIGMA), BELIEF_LO, BELIEF_HI)`` —
    the oracle posterior perturbed by mean-zero Gaussian noise, clipped away
    from the extremes (a ZI trader should never claim total certainty).

    Trade rule (per round)
    -----------------------
    Draw a fresh noisy belief (same rule as above). Compare to the current
    price with a dead-band of ``TRADE_BAND``:

    * ``belief > price + TRADE_BAND``  -> ``buy_yes``
    * ``belief < price - TRADE_BAND``  -> ``buy_no``
    * otherwise                        -> ``hold`` (0 shares requested)

    On a buy, the requested share count is
    ``frac * max_affordable_<chosen side>`` with
    ``frac ~ Uniform(FRAC_LO, FRAC_HI)`` drawn from the same RNG. The
    runner still clamps this server-side before executing.
    """

    BELIEF_SIGMA = 0.08
    BELIEF_LO = 0.01
    BELIEF_HI = 0.99
    TRADE_BAND = 0.02
    FRAC_LO = 0.2
    FRAC_HI = 0.6
    BELIEF_RATIONALE = "mock agent: oracle posterior plus Gaussian noise"
    TRADE_RATIONALE = "mock agent: threshold rule on belief vs. price"

    def __init__(self, oracle_posterior: float, rng: np.random.Generator) -> None:
        if not (0.0 <= oracle_posterior <= 1.0):
            raise ValueError(
                f"oracle_posterior must be in [0, 1], got {oracle_posterior}"
            )
        self.oracle_posterior = float(oracle_posterior)
        self.rng = rng

    def _noisy_belief(self) -> float:
        """One fresh noisy-belief draw, consuming exactly one RNG normal
        sample. Shared by elicit_belief and decide_trade so both quote the
        same rule."""
        noise = self.rng.normal(0.0, self.BELIEF_SIGMA)
        return float(
            np.clip(self.oracle_posterior + noise, self.BELIEF_LO, self.BELIEF_HI)
        )

    def elicit_belief(self, view: BeliefView) -> Belief:
        return Belief(prob=self._noisy_belief(), rationale=self.BELIEF_RATIONALE)

    def decide_trade(self, view: TradeView) -> Trade:
        belief = self._noisy_belief()

        action: Literal["buy_yes", "buy_no", "hold"]
        if belief > view.price + self.TRADE_BAND:
            action = "buy_yes"
            max_affordable = view.max_affordable_yes
        elif belief < view.price - self.TRADE_BAND:
            action = "buy_no"
            max_affordable = view.max_affordable_no
        else:
            action = "hold"
            max_affordable = 0.0

        if action == "hold":
            shares = 0.0
        else:
            frac = self.rng.uniform(self.FRAC_LO, self.FRAC_HI)
            shares = float(frac * max_affordable)

        return Trade(
            belief=belief, action=action, shares=shares, rationale=self.TRADE_RATIONALE
        )


class LLMAgent:
    """Provider-backed agent implementing the two-pass protocol over prompts.

    Constructed with its :class:`AgentSpec`, the resolved :class:`ModelSpec`,
    a persona key (into :data:`pnyx.prompts.PERSONAS`), and a shared provider
    handle. ``elicit_belief`` / ``decide_trade`` build the versioned prompts
    from the view's ``question_text`` + the agent's own ``shard_texts``, call
    the provider with the relevant schema, and implement the ONE-retry parse
    contract (Global Constraints):

    * On success, bill the call's usage and return the parsed output.
    * On a :class:`SchemaParseError`, bill it, then retry ONCE with the
      validation error appended to the messages.
    * On a second parse failure, bill it and return the neutral fallback
      (``prob=0.5`` / ``action="hold"``) flagged ``parse_failed=True``.
    * :class:`ProviderError` is billed (if it carries usage) then re-raised;
      :class:`TurnParked` propagates unbilled. The runner treats both as a
      parked turn (nothing logged, resume retries it).

    Billing is done through the injected ``bill`` callback so the cost ledger
    (and the turn key it is written under) stays owned by the runner: EVERY
    provider call is billed at the moment it completes — success, parse
    failure, or errored — never missed on a propagating exception.

    The provider call itself is the only awaited point; the agent holds no
    per-turn state, so one instance per (agent_id) is reused across a run.
    """

    def __init__(
        self,
        spec: AgentSpec,
        model_spec: ModelSpec,
        persona: str | None,
        provider: Any,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        self.spec = spec
        self.model_spec = model_spec
        self.persona = persona
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens is not None else _DEFAULT_MAX_TOKENS

    async def _complete_with_retry(
        self, messages: list[dict[str, str]], schema: type, bill: BillFn
    ):
        """Run the one-retry parse contract for a single turn.

        Returns ``(parsed_or_None, parse_failed)``: ``parsed_or_None`` is the
        parsed schema object on success, or ``None`` when both attempts failed
        to parse (the caller substitutes the neutral fallback). Bills every
        completed call. Propagates :class:`ProviderError` (after billing its
        usage) and :class:`TurnParked` (unbilled) for the runner to park.
        """
        try:
            obj, usage = await self.provider.complete(
                messages, schema, self.model_spec, self.temperature, self.max_tokens
            )
            bill(usage)
            return obj, False
        except ProviderError as exc:
            bill(exc.usage)
            raise
        except SchemaParseError as first:
            # Python clears ``first`` at the end of the except block, so capture
            # the error text now for the retry prompt.
            bill(first.usage)
            first_error = first.error

        retry_messages = list(messages) + [retry_user_message(first_error)]
        try:
            obj, usage = await self.provider.complete(
                retry_messages, schema, self.model_spec, self.temperature,
                self.max_tokens,
            )
            bill(usage)
            return obj, False
        except ProviderError as exc:
            bill(exc.usage)
            raise
        except SchemaParseError as second:
            bill(second.usage)
            return None, True

    async def elicit_belief(self, view: BeliefView, *, bill: BillFn) -> BeliefResult:
        messages = pass1_messages(view.question_text, view.shard_texts, self.persona)
        obj, parse_failed = await self._complete_with_retry(messages, Belief, bill)
        if parse_failed:
            return BeliefResult(
                belief=Belief(prob=0.5, rationale="parse failed after one retry"),
                parse_failed=True,
            )
        return BeliefResult(belief=obj, parse_failed=False)

    async def decide_trade(
        self, view: TradeView, *, pass1_prob: float, bill: BillFn
    ) -> TradeResult:
        messages = pass2_messages(
            view.question_text,
            view.shard_texts,
            self.persona,
            price=view.price,
            round=view.round,
            n_rounds=view.n_rounds,
            bankroll=view.bankroll,
            max_affordable_yes=view.max_affordable_yes,
            max_affordable_no=view.max_affordable_no,
            pass1_prob=pass1_prob,
        )
        obj, parse_failed = await self._complete_with_retry(messages, Trade, bill)
        if parse_failed:
            return TradeResult(
                trade=Trade(
                    belief=0.5, action="hold", shares=0.0,
                    rationale="parse failed after one retry",
                ),
                parse_failed=True,
            )
        return TradeResult(trade=obj, parse_failed=False)
