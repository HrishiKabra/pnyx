"""Pnyx schemas — single source of truth for all Pydantic models.

Every other P1 module (LMSR market engine, generative environment, mock
agents, runner) imports its data shapes from here. Nothing outside this
file defines a Pydantic model for structured LLM outputs, environment
records, or the event log.

Event log convention
---------------------
The event log is an append-only JSONL file, one line per completed agent
turn (``BeliefEvent`` / ``TradeEvent``) plus one ``SettlementEvent`` per
question. Each line is produced by ``<event>.to_jsonl_line()`` and parsed
back with ``parse_event()``. The serializer sorts object keys at every
nesting level (pydantic's ``model_dump_json`` has no ``sort_keys`` option,
hence the manual pass through ``json.dumps``), so two independent runs
with identical canonical content produce byte-identical lines once the
non-reproducible ``ts`` (wall-clock) field is stripped from each — this is
what makes determinism checks a plain string comparison rather than a
semantic diff.

Turn identity: a turn is uniquely keyed by
``(condition, seed, question_id, phase, round, agent_id)`` — see
``TurnKey``.
"""

import json
import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

# --------------------------------------------------------------------------
# Structured LLM outputs (exact fields per Global Constraints)
# --------------------------------------------------------------------------


class Belief(BaseModel):
    """An agent's stated belief (Pass-1 independent, or in-market)."""

    prob: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=400)


class Trade(BaseModel):
    """An agent's requested trade.

    ``shares`` is the agent's ASK. The runner clamps it server-side to
    what is actually affordable before execution — never trust this
    number directly (see ``TradeEvent.executed_shares``).
    """

    belief: float = Field(ge=0.0, le=1.0)
    action: Literal["buy_yes", "buy_no", "hold"]
    shares: float = Field(ge=0.0)
    rationale: str = Field(max_length=400)


# --------------------------------------------------------------------------
# Generative environment records
# --------------------------------------------------------------------------


class SignalRecord(BaseModel):
    """One realized signal draw for a question (the generative model
    itself lives in env.py)."""

    index: int
    value: Literal[0, 1]
    accuracy: float
    lam: float = Field(
        ge=0.0,
        le=1.0,
        description="correlation weight: 0 = conditionally independent given s, "
        "1 = deterministically copies the common latent channel z",
    )


# Subset key convention: "" (empty subset / prior) or comma-joined,
# strictly ascending, deduplicated non-negative signal indices, e.g. "0,2,5".
_SUBSET_KEY_RE = re.compile(r"^$|^\d+(,\d+)*$")


class QuestionRecord(BaseModel):
    """A single generated question: latent state, realized signals, and the
    exact Bayes-optimal posterior for every subset of signals.

    ``posterior_table`` maps a subset key to
    P(s=1 | realized values of that signal subset). Keys are the
    comma-joined, strictly ascending, deduplicated signal indices in the
    subset (e.g. ``"0,2,5"``); ``""`` is the empty subset, i.e. the prior
    P(s=1) with no signals observed.
    """

    question_id: str
    latent_state: Literal[0, 1]
    signals: list[SignalRecord]
    posterior_table: dict[str, float]
    shards: list[str] | None = None  # prose rendering arrives in P2
    # Generator provenance: config name, seed, difficulty tag, etc.
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("posterior_table")
    @classmethod
    def _validate_subset_keys(cls, table: dict[str, float]) -> dict[str, float]:
        for key in table:
            if not _SUBSET_KEY_RE.match(key):
                raise ValueError(
                    f"posterior_table key {key!r} must be '' or comma-joined "
                    "sorted signal indices, e.g. '0,2,5'"
                )
            if key:
                indices = [int(part) for part in key.split(",")]
                if indices != sorted(indices) or len(set(indices)) != len(indices):
                    raise ValueError(
                        f"posterior_table key {key!r} must have strictly "
                        "ascending, deduplicated indices"
                    )
        return table


# --------------------------------------------------------------------------
# Turn identity
# --------------------------------------------------------------------------


class TurnKey(BaseModel):
    """Identifies one completed agent turn.

    Frozen (immutable + hashable) so it can be used directly as a dict/set
    key when the runner reconstructs the remaining-turns set on replay.
    """

    model_config = ConfigDict(frozen=True)

    condition: str
    seed: int
    question_id: str
    phase: Literal["independent", "market"]
    round: int  # 0 for the independent pass; market rounds are 1..n_rounds
    agent_id: str

    def as_tuple(self) -> tuple[str, int, str, str, int, str]:
        """Return the key fields in declaration order, e.g. for use as a
        plain-tuple dict/set key or CSV row."""
        return (
            self.condition,
            self.seed,
            self.question_id,
            self.phase,
            self.round,
            self.agent_id,
        )


# --------------------------------------------------------------------------
# Event log
# --------------------------------------------------------------------------


class _EventBase(BaseModel):
    """Shared canonical-serialization behavior for all event types."""

    def to_jsonl_line(self) -> str:
        """Serialize to one canonical JSON line: keys sorted at every
        nesting level, no incidental whitespace. See module docstring for
        why this matters for determinism checks."""
        payload = self.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class BeliefEvent(_EventBase):
    """Logged once per (condition, seed, question_id, phase, round,
    agent_id) belief elicitation turn."""

    type: Literal["belief"] = "belief"
    key: TurnKey
    belief: Belief
    parse_failed: bool = False
    ts: float | None  # wall-clock write time; excluded from identity comparisons


class TradeEvent(_EventBase):
    """Logged once per executed market-pass trade turn."""

    type: Literal["trade"] = "trade"
    key: TurnKey
    trade: Trade
    executed_shares: float = Field(ge=0.0)  # post-clamp; never trust trade.shares
    cost: float
    price_before: float = Field(ge=0.0, le=1.0)
    price_after: float = Field(ge=0.0, le=1.0)
    bankroll_after: float
    parse_failed: bool = False
    ts: float | None


class SettlementEvent(_EventBase):
    """Logged once per (condition, seed, question_id) at market close.

    ``subsidy`` is the market maker's realized loss for this question:
    the LMSR cost-function delta from the market's opening state to its
    closing state, ``C(q_final) - C(q_0)``, minus the total trade cost
    collected from agents over the market pass (i.e. maker loss net of
    revenue). This is the maker's out-of-pocket subsidy for running the
    market; see market.py for the b=40 => worst-case b*ln(2) bound.
    """

    type: Literal["settlement"] = "settlement"
    condition: str
    seed: int
    question_id: str
    outcome: Literal[0, 1]
    payouts: dict[str, float]  # agent_id -> payout ($1/share on the true side)
    final_price: float = Field(ge=0.0, le=1.0)
    subsidy: float
    ts: float | None


Event = Annotated[Union[BeliefEvent, TradeEvent, SettlementEvent], Field(discriminator="type")]

_event_adapter: TypeAdapter = TypeAdapter(Event)


def parse_event(line: str) -> "BeliefEvent | TradeEvent | SettlementEvent":
    """Parse one JSONL line back into the correct Event subtype, dispatched
    on the ``type`` discriminator field."""
    return _event_adapter.validate_json(line)
