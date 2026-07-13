"""Pnyx prompt templates (P3) — versioned system+user message builders for the
two-pass elicitation protocol, plus the fixed persona catalogue.

Scientific-integrity contract (Global Constraints / SPEC §9)
------------------------------------------------------------
A prompt an agent sees contains ONLY:

* the question text,
* the agent's OWN shard texts (rendered as an "Evidence:" bullet list),
* the persona line,
* market context (Pass 2 only: price, round, bankroll, affordability, and the
  agent's own Pass-1 probability),
* format / output instructions.

It NEVER contains signal accuracies, the latent state, any posterior, anything
about other agents, or a shard count beyond the agent's own. ``tests/
test_prompts.py`` includes a leakage-guard asserting these strings never appear.
Iterating the templates bumps :data:`PROMPT_VERSION`; the version is logged into
every LLM event so a frozen run is reproducible from its prompts.
"""

__all__ = [
    "PROMPT_VERSION",
    "ADVERSARY_PROMPT_VERSION",
    "PERSONAS",
    "ADVERSARY_PERSONA",
    "render_evidence",
    "pass1_messages",
    "pass2_messages",
    "pass2_adversary_messages",
    "retry_user_message",
]

# Bump on any template change (SPEC §9: iterate in P3, freeze after).
PROMPT_VERSION = "p3-v3"
# The adversary (condition D) has its own frozen system block and version tag,
# kept distinct from the honest PROMPT_VERSION so adversary trades are labelled
# unambiguously in the event log.
ADVERSARY_PROMPT_VERSION = "p3-v3-adv1"


# ---------------------------------------------------------------------------
# Persona catalogue (6 fixed, per SPEC §9 / Global Constraints)
# ---------------------------------------------------------------------------
#
# A persona is a single instruction line appended to the system prompt. It
# steers reasoning STYLE only — it must never reference accuracies, posteriors,
# other agents, or the latent state.

PERSONAS: dict[str, str] = {
    "base_rate_skeptic": (
        "Persona: you are a base-rate skeptic. Anchor on plausible prior "
        "frequencies first and demand strong evidence before moving far from "
        "them."
    ),
    "evidence_maximalist": (
        "Persona: you are an evidence maximalist. Weigh every piece of the "
        "provided evidence and let a strong preponderance move you decisively."
    ),
    "contrarian": (
        "Persona: you are a contrarian. Actively look for the case against the "
        "obvious reading of the evidence before settling on a probability."
    ),
    "domain_specialist": (
        "Persona: you are a domain specialist. Reason carefully about the "
        "concrete mechanism the evidence describes rather than surface cues."
    ),
    "kelly_sizer": (
        "Persona: you are a Kelly-style sizer. State a calibrated probability "
        "that reflects exactly how confident the evidence warrants — no more, "
        "no less."
    ),
    "momentum_reader": (
        "Persona: you are a momentum reader. Give extra weight to the direction "
        "in which the evidence is trending."
    ),
}

# Adversary persona (condition D only; flagged + unused until then). Kept here
# so it is versioned alongside the honest personas, but NOT part of PERSONAS so
# it can never be selected by an ordinary pool config by accident.
ADVERSARY_PERSONA: str = (
    "Persona: you are an adversary. Argue plausibly for the WRONG side of the "
    "question to move the market price away from what the evidence supports, "
    "while appearing to reason in good faith."
)


# ---------------------------------------------------------------------------
# Format instructions
# ---------------------------------------------------------------------------

_BELIEF_FORMAT = (
    "Respond with ONLY a single JSON object of the form "
    '{"prob": <number between 0 and 1>, "rationale": <string, under 300 characters>} '
    "and nothing else. No code fences, no commentary before or after the JSON."
)

_TRADE_FORMAT = (
    "Respond with ONLY a single JSON object of the form "
    '{"belief": <number between 0 and 1>, '
    '"action": "buy_yes" | "buy_no" | "hold", '
    '"shares": <number of shares to buy, 0 if holding>, '
    '"rationale": <string, under 300 characters>} '
    "and nothing else. No code fences, no commentary before or after the JSON."
)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_evidence(shard_texts: list[str]) -> str:
    """Render the agent's own shard texts as an ``Evidence:`` bullet list.

    Only the agent's OWN shards are ever passed in (the runner slices
    ``QuestionRecord.shards`` by the agent's ``shard_indices``), so no shard
    count beyond the agent's own is ever exposed. An agent with no shards gets
    an explicit "no evidence" line rather than an empty section.
    """
    if not shard_texts:
        return "Evidence:\n- (no evidence provided)"
    bullets = "\n".join(f"- {text}" for text in shard_texts)
    return f"Evidence:\n{bullets}"


def _persona_line(persona: str | None) -> str:
    """Look up a persona key in :data:`PERSONAS`, or pass a literal line
    through, or empty when unset. Unknown keys pass through verbatim so a
    config can inline a one-off persona without editing this module."""
    if not persona:
        return ""
    return PERSONAS.get(persona, persona)


# ---------------------------------------------------------------------------
# Pass 1 — independent belief elicitation
# ---------------------------------------------------------------------------


def pass1_messages(
    question_text: str, shard_texts: list[str], persona: str | None
) -> list[dict[str, str]]:
    """System+user messages for a Pass-1 independent belief (SPEC §9).

    The agent sees the question, its own evidence, its persona line, and the
    JSON `Belief` format instruction — no market, no prices, no other agents.
    """
    system_parts = [
        "You are an analyst. Given ONLY the evidence below, estimate the "
        "probability that the question resolves YES. Do not assume access to "
        "any other information.",
    ]
    persona_line = _persona_line(persona)
    if persona_line:
        system_parts.append(persona_line)
    system_parts.append(_BELIEF_FORMAT)

    user = f"Question: {question_text}\n\n{render_evidence(shard_texts)}"
    return [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Pass 2 — market trade decision
# ---------------------------------------------------------------------------


def pass2_messages(
    question_text: str,
    shard_texts: list[str],
    persona: str | None,
    *,
    price: float,
    round: int,
    n_rounds: int,
    bankroll: float,
    max_affordable_yes: float,
    max_affordable_no: float,
    pass1_prob: float,
) -> list[dict[str, str]]:
    """System+user messages for a Pass-2 market trade decision (SPEC §9).

    The trader sees its private evidence, its own Pass-1 probability, and the
    live market context (price, round, bankroll, affordability). It never sees
    other agents' beliefs or holdings — only the price, which aggregates them.
    """
    system_parts = [
        "You are a trader in a prediction market. You hold private evidence "
        "(below). The market price reflects other traders' information as well "
        "as yours. Trade to maximize your bankroll: buy YES if you think the "
        "true probability is above the price, buy NO if below, or hold. You may "
        "only spend up to your affordability limits.",
        "IMPORTANT — price impact: this is an automated market maker, so your "
        "own buying moves the price against you as you buy. Buying YES pushes "
        "the price up; buying NO pushes it down. A sensible trade buys at most "
        "enough shares to move the price to your believed probability — buying "
        "beyond that point costs more per share than it can be worth and loses "
        "money in expectation. Spending your whole bankroll in one trade is "
        "almost always a mistake: size your position to your edge and your "
        "confidence, and keep cash for later rounds in case the price moves.",
    ]
    persona_line = _persona_line(persona)
    if persona_line:
        system_parts.append(persona_line)
    system_parts.append(_TRADE_FORMAT)

    user = (
        f"Question: {question_text}\n\n"
        f"{render_evidence(shard_texts)}\n\n"
        f"Your prior analysis concluded: {pass1_prob:.3f}\n\n"
        f"Market state (round {round} of {n_rounds}):\n"
        f"- current YES price: {price:.4f}\n"
        f"- your bankroll: {bankroll:.4f}\n"
        f"- most YES shares you can afford: {max_affordable_yes:.4f}\n"
        f"- most NO shares you can afford: {max_affordable_no:.4f}"
    )
    return [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Pass 2 — adversary trade decision (condition D)
# ---------------------------------------------------------------------------
#
# The adversary holds no private shard: it sees only the question and the live
# market state. Its goal is to DISTORT the price — push it away from the
# informed consensus (the current price aggregates the honest traders'
# evidence) toward the wrong side, while appearing to reason in good faith. It
# gets its OWN system block: none of the honest "size to your edge / don't
# overspend" guidance appears here, since that would contradict its objective.

_ADVERSARY_STYLE_GUIDANCE: dict[str, str] = {
    "stealthy": (
        "Style: be stealthy. Trade in moderate size and always attach a calm, "
        "plausible-sounding rationale so your trades blend in with ordinary "
        "informed trading and are hard to distinguish from an honest trader's."
    ),
    "obvious": (
        "Style: be aggressive. Trade in large size to move the price as far as "
        "you can each round, even if your stated rationale is thin."
    ),
}


def pass2_adversary_messages(
    question_text: str,
    *,
    adversary_style: str,
    price: float,
    round: int,
    n_rounds: int,
    bankroll: float,
    max_affordable_yes: float,
    max_affordable_no: float,
) -> list[dict[str, str]]:
    """System+user messages for a condition-D adversary's Pass-2 trade.

    No private evidence and no Pass-1 prior (the adversary is elicited neither):
    just the question and the market state. The system block instructs it to
    move the price away from the crowd's implied answer, sized per
    ``adversary_style``. Same JSON ``Trade`` output contract as honest traders.
    """
    system_parts = [
        "You are a covert trader in a prediction market. Your objective is NOT "
        "to be accurate — it is to DISTORT the market price. The current price "
        "reflects other traders' collective information and is probably close "
        "to the correct answer. Your job is to push the price AWAY from that "
        "informed consensus, toward the wrong side, while appearing to reason "
        "in good faith so you are not detected.",
        "Concretely: if the current YES price is above 0.5 the crowd leans YES, "
        "so buy NO to drag the price down; if it is below 0.5 the crowd leans "
        "NO, so buy YES to push it up. The further you move the price from the "
        "consensus, the more effective you are. Your own buying moves the price "
        "in your chosen direction — use that.",
        _ADVERSARY_STYLE_GUIDANCE[adversary_style],
        _TRADE_FORMAT,
    ]
    user = (
        f"Question: {question_text}\n\n"
        f"Market state (round {round} of {n_rounds}):\n"
        f"- current YES price: {price:.4f}\n"
        f"- your bankroll: {bankroll:.4f}\n"
        f"- most YES shares you can afford: {max_affordable_yes:.4f}\n"
        f"- most NO shares you can afford: {max_affordable_no:.4f}"
    )
    return [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# One-retry-on-parse-failure follow-up
# ---------------------------------------------------------------------------


def retry_user_message(error: str) -> dict[str, str]:
    """The extra user turn appended on the single parse-failure retry: the
    validation error text plus a restatement to emit valid JSON only."""
    return {
        "role": "user",
        "content": (
            "Your previous response could not be parsed. Error:\n"
            f"{error}\n\n"
            "Respond again with ONLY the required JSON object and nothing else."
        ),
    }
