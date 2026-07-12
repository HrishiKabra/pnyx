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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

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
# Agent view models (see pnyx.agents for the Agent protocol / MockAgent)
# --------------------------------------------------------------------------
#
# These are what the runner (Task 5) hands to an agent's elicit_belief() /
# decide_trade() calls. They deliberately do NOT carry the oracle posterior
# for the agent's shard subset — exposing it here would be cheating for a
# real (P3) LLM agent that is only supposed to see its rendered shard text
# and realized signal values. MockAgent instead receives its shard's oracle
# posterior directly at construction (a QuestionRecord.posterior_table
# lookup the runner performs), never via the view.


class BeliefView(BaseModel):
    """What an agent sees for Pass-1 independent belief elicitation.

    ``signal_values`` maps the agent's own shard's signal indices to their
    realized values (a subset of a QuestionRecord's signals — whichever
    indices this agent was assigned). ``persona`` is unused by MockAgent;
    it exists for P3 LLM agents.
    """

    question_id: str
    signal_values: dict[int, int]
    persona: str = ""


class TradeView(BeliefView):
    """What an agent sees for a Pass-2 market-round trade decision:
    everything in ``BeliefView`` plus the live market/account state.

    ``max_affordable_yes`` / ``max_affordable_no`` are computed by the
    runner via ``market.py`` (``max_affordable_shares``) — agents never
    import market.py themselves. The runner still clamps the agent's
    requested ``Trade.shares`` server-side regardless of what the agent
    asks for (Global Constraints).
    """

    price: float = Field(ge=0.0, le=1.0)
    round: int = Field(ge=1)
    n_rounds: int = Field(ge=1)
    bankroll: float = Field(ge=0.0)
    max_affordable_yes: float = Field(ge=0.0)
    max_affordable_no: float = Field(ge=0.0)


class AgentSpec(BaseModel):
    """Minimal P1 agent configuration: identity, shard assignment, starting
    bankroll, and kind.

    ``kind`` is a closed Literal with a single P1 value; real LLM agent
    kinds (with model/persona/provider fields) arrive in P3, so adding one
    later is a deliberate, visible schema change rather than a silent
    string.
    """

    agent_id: str
    shard_indices: list[int]
    bankroll: float = Field(default=100.0, ge=0.0)
    kind: Literal["mock"] = "mock"


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
    # Full question prose (rendered in P2); None keeps the P1 mock path valid.
    question_text: str | None = None
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
# P2 dataset staging-file schemas (build / render / verify pipeline)
# --------------------------------------------------------------------------
#
# The renderer and verifier are Claude subagents orchestrated OUTSIDE this
# codebase; they drop JSON files into datasets/staging/. These models are the
# validation contract the dataset tooling (pnyx.dataset) applies on merge/verify
# so a malformed staging file fails loudly rather than corrupting the frozen
# dataset. No LLM calls happen here — only parse + validate.


class RenderSpecSignal(BaseModel):
    """One signal's public-facing rendering hint: which way the shard should
    lean. Direction is derived from the realized signal value (1 -> YES,
    0 -> NO); accuracies / latent state / posteriors are NEVER exposed."""

    index: int = Field(ge=0)
    direction: Literal["YES", "NO"]


class RenderSpec(BaseModel):
    """One line of render_specs.jsonl — everything the renderer subagent is
    allowed to see for a question."""

    question_id: str
    difficulty: Literal["easy", "hard"]
    domain_hint: str
    signals: list[RenderSpecSignal]


class RenderedQuestion(BaseModel):
    """A renderer subagent's output for one question: the prose question plus
    one prose shard per signal. Validators enforce the merge contract (6
    non-empty shards, a non-empty question ending in '?')."""

    question_id: str
    question_text: str
    shards: list[str]

    @field_validator("question_text")
    @classmethod
    def _validate_question_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("question_text must be non-empty")
        if not text.rstrip().endswith("?"):
            raise ValueError("question_text must end with '?'")
        return text

    @field_validator("shards")
    @classmethod
    def _validate_shards(cls, shards: list[str]) -> list[str]:
        if len(shards) != 6:
            raise ValueError(f"expected 6 shards, got {len(shards)}")
        for i, shard in enumerate(shards):
            if not shard.strip():
                raise ValueError(f"shard {i} must be non-empty")
        return shards


class ShardVerdict(BaseModel):
    """A verifier subagent's per-shard checks for one signal's rendering."""

    index: int = Field(ge=0)
    direction_ok: bool
    leak_ok: bool
    meta_ok: bool
    note: str = ""


class QuestionVerdict(BaseModel):
    """A verifier subagent's output for one question: per-shard checks plus a
    top-level pass flag."""

    question_id: str
    shard_verdicts: list[ShardVerdict]
    question_ok: bool


# --------------------------------------------------------------------------
# Generative environment configuration
# --------------------------------------------------------------------------


class EnvConfig(BaseModel):
    """Configuration for the generative environment (see ``pnyx.env`` for the
    generative model and oracle).

    ``accuracies[i]`` is signal ``i``'s private-channel accuracy
    P(private draw = s); ``lams[i]`` is its correlation weight (probability
    the signal copies the common latent channel ``z`` instead of drawing a
    private channel). ``a_z`` is the accuracy of ``z`` itself, P(z = s).

    ``difficulty`` selects the target band for the all-signals posterior on
    the favored side (``max(p, 1-p)``): ``"easy"`` requires >= 0.85, ``"hard"``
    requires the 0.60-0.75 range. ``min_single_shard_margin`` is the minimum
    absolute gap between the all-signals posterior and every single-signal
    posterior. ``max_rejection_tries`` bounds the rejection sampler.
    """

    n_signals: int = Field(default=6, ge=1)
    accuracies: list[float]
    lams: list[float]
    a_z: float = Field(default=0.8, ge=0.0, le=1.0)
    difficulty: Literal["easy", "hard"]
    min_single_shard_margin: float = Field(default=0.1, ge=0.0)
    max_rejection_tries: int = Field(default=1000, ge=1)

    @field_validator("accuracies")
    @classmethod
    def _validate_accuracies(cls, acc: list[float]) -> list[float]:
        for a in acc:
            if not (0.0 <= a <= 1.0):
                raise ValueError(f"accuracy {a} must be in [0, 1]")
        return acc

    @field_validator("lams")
    @classmethod
    def _validate_lams(cls, lams: list[float]) -> list[float]:
        for lam in lams:
            if not (0.0 <= lam <= 1.0):
                raise ValueError(f"lam {lam} must be in [0, 1]")
        return lams

    @model_validator(mode="after")
    def _validate_lengths(self) -> "EnvConfig":
        if len(self.accuracies) != self.n_signals:
            raise ValueError(
                f"accuracies has length {len(self.accuracies)}, "
                f"expected n_signals={self.n_signals}"
            )
        if len(self.lams) != self.n_signals:
            raise ValueError(
                f"lams has length {len(self.lams)}, "
                f"expected n_signals={self.n_signals}"
            )
        return self


# --------------------------------------------------------------------------
# Run configuration (loaded from configs/*.yaml by the runner)
# --------------------------------------------------------------------------


class RunConfig(BaseModel):
    """Top-level experiment configuration for one condition.

    Loaded from a YAML file (see ``pnyx/configs/mock.yaml``). Lives here in
    ``schemas.py`` because Global Constraints make this file the single source
    of truth for every Pydantic model. One event log is written per
    ``(condition, seed)`` under ``data_dir``; the generated ``QuestionRecord``
    set is persisted alongside it so a resumed run never re-samples questions.
    """

    condition: str
    seeds: list[int] = Field(min_length=1)
    n_questions: int = Field(ge=1)
    b: float = Field(default=40.0, gt=0.0)
    n_rounds: int = Field(default=3, ge=1)
    wealth_persistent: bool = False
    data_dir: str
    # When set, the runner loads QuestionRecords from this JSONL dataset instead
    # of generating them (requiring shards + question_text on each). The generate
    # path stays for the MOCK config. See ``runner._load_or_generate_questions``.
    questions_file: str | None = None
    env: EnvConfig
    agents: list[AgentSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_shards(self) -> "RunConfig":
        for spec in self.agents:
            for idx in spec.shard_indices:
                if not (0 <= idx < self.env.n_signals):
                    raise ValueError(
                        f"agent {spec.agent_id!r} shard index {idx} out of range "
                        f"[0, {self.env.n_signals})"
                    )
        ids = [a.agent_id for a in self.agents]
        if len(set(ids)) != len(ids):
            raise ValueError(f"agent_id values must be unique, got {ids}")
        return self


# --------------------------------------------------------------------------
# Provider / model configuration + cost accounting (P3)
# --------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """Provider-agnostic description of one model endpoint.

    Every model in the run config is a flat record
    ``{base_url, api_key_env, model_id, price_in, price_out, rpm_limit,
    supports_json_schema}`` (Global Constraints). Prices are in US dollars
    per **million** tokens. ``api_key_env`` names the environment variable
    holding the bearer key — the key value itself is NEVER stored on the
    spec, read from config, or logged.
    """

    base_url: str
    api_key_env: str
    model_id: str
    price_in: float = Field(ge=0.0, description="$/M input (prompt) tokens")
    price_out: float = Field(ge=0.0, description="$/M output (completion) tokens")
    rpm_limit: int = Field(gt=0, description="client-side requests-per-minute bucket")
    supports_json_schema: bool = True

    def cost(self, in_tokens: int, out_tokens: int) -> float:
        """Dollar cost of a call with the given token counts (prices are $/M)."""
        return (in_tokens * self.price_in + out_tokens * self.price_out) / 1_000_000.0


class Usage(BaseModel):
    """Token usage parsed from a provider response's ``usage`` block."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)


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


class CostEntry(BaseModel):
    """One line of the append-only cost ledger ``data/<run>/costs.jsonl``:
    per-call token usage, its dollar cost, and the turn that incurred it.

    ``key`` is optional so a call not tied to a single turn (or a unit test)
    can still be recorded; production runner calls always carry the turn key.
    """

    model_id: str
    in_tokens: int = Field(ge=0)
    out_tokens: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    key: TurnKey | None = None


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
    the worst-case settlement payout across outcomes minus the total
    trade cost collected from agents over the market pass, i.e.
    ``max(delta_q_yes, delta_q_no, 0) - (C(q_final) - C(q_0))``. (Note:
    collected revenue telescopes to exactly ``C(q_final) - C(q_0)``, so
    "payout delta minus revenue" is the only non-degenerate definition.)
    This is the maker's out-of-pocket subsidy for running the market;
    see market.py's ``Market.subsidy`` for the derivation and the
    b=40 => worst-case b*ln(2) bound.
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
