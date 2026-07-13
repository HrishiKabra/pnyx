"""Tests for pnyx.prompts — the versioned two-pass templates + persona catalogue.

Golden-content checks (persona injected, evidence rendered, Pass-1 belief carried
into the Pass-2 prompt) plus a scientific-integrity LEAKAGE GUARD asserting that
no accuracy / latent-state / posterior / other-agent / shard-count string ever
appears in a rendered prompt. Pure/offline — no network, no provider.
"""

import pnyx.prompts as prompts
from pnyx.prompts import (
    ADVERSARY_PERSONA,
    ADVERSARY_PROMPT_VERSION,
    PERSONAS,
    PROMPT_VERSION,
    pass1_messages,
    pass2_adversary_messages,
    pass2_messages,
    render_evidence,
    retry_user_message,
)

QUESTION = "Did Forklift 12 roll into the shelving racks?"
SHARDS = [
    "The pre-shift inspection checklist for Forklift 12 was signed off clean.",
    "A dented rack upright was photographed near bay 7.",
]


def _text(messages) -> str:
    return "\n".join(m["content"] for m in messages)


# ---------------------------------------------------------------------------
# Versioning + persona catalogue
# ---------------------------------------------------------------------------


def test_prompt_version_is_a_nonempty_string():
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION


def test_six_fixed_personas():
    assert set(PERSONAS) == {
        "base_rate_skeptic", "evidence_maximalist", "contrarian",
        "domain_specialist", "kelly_sizer", "momentum_reader",
    }
    assert all(isinstance(v, str) and v for v in PERSONAS.values())


def test_adversary_persona_is_separate_and_flagged():
    # The adversary line exists (condition D) but is NOT selectable as an
    # ordinary persona, so a pool config can never inject it by accident.
    assert "adversary" in ADVERSARY_PERSONA.lower()
    assert ADVERSARY_PERSONA not in PERSONAS.values()


# ---------------------------------------------------------------------------
# Structure + golden content
# ---------------------------------------------------------------------------


def test_pass1_has_system_and_user_with_question_and_evidence():
    msgs = pass1_messages(QUESTION, SHARDS, "contrarian")
    assert [m["role"] for m in msgs] == ["system", "user"]
    text = _text(msgs)
    assert QUESTION in text
    for shard in SHARDS:
        assert shard in text
    # SPEC §9 Pass-1 framing.
    assert "Given ONLY the evidence below" in text
    # Persona injected.
    assert PERSONAS["contrarian"] in text


def test_evidence_rendered_as_bullets():
    rendered = render_evidence(SHARDS)
    assert rendered.startswith("Evidence:")
    for shard in SHARDS:
        assert f"- {shard}" in rendered


def test_no_shards_renders_explicit_empty_evidence():
    rendered = render_evidence([])
    assert "Evidence:" in rendered
    assert "no evidence" in rendered.lower()


def test_pass2_carries_pass1_belief_and_market_context():
    msgs = pass2_messages(
        QUESTION, SHARDS, "kelly_sizer",
        price=0.42, round=2, n_rounds=3, bankroll=88.5,
        max_affordable_yes=120.0, max_affordable_no=95.0, pass1_prob=0.73,
    )
    assert [m["role"] for m in msgs] == ["system", "user"]
    text = _text(msgs)
    # Pass-1 belief carried in (SPEC §9: "your prior analysis concluded").
    assert "prior analysis concluded" in text
    assert "0.73" in text
    # Market context present.
    assert "0.42" in text  # price
    assert "round 2 of 3" in text
    assert PERSONAS["kelly_sizer"] in text
    assert QUESTION in text


def test_unknown_persona_passes_through_verbatim():
    line = "Persona: you are a one-off custom persona."
    msgs = pass1_messages(QUESTION, SHARDS, line)
    assert line in _text(msgs)


def test_none_persona_omits_persona_line():
    msgs = pass1_messages(QUESTION, SHARDS, None)
    # No dangling "Persona:" label when unset.
    assert "Persona:" not in _text(msgs)


def test_retry_message_includes_error_text():
    m = retry_user_message("prob: field required")
    assert m["role"] == "user"
    assert "prob: field required" in m["content"]
    assert "JSON" in m["content"]


# ---------------------------------------------------------------------------
# Adversary Pass-2 prompt (condition D)
# ---------------------------------------------------------------------------


def _adv_msgs(style="stealthy", price=0.7, target_side="NO"):
    return pass2_adversary_messages(
        QUESTION, adversary_style=style, target_side=target_side,
        price=price, round=2, n_rounds=3, bankroll=300.0,
        max_affordable_yes=400.0, max_affordable_no=350.0,
    )


def test_adversary_prompt_version_distinct_and_tagged():
    assert ADVERSARY_PROMPT_VERSION == "p3-v3-adv2"
    assert ADVERSARY_PROMPT_VERSION != PROMPT_VERSION


def test_adversary_prompt_instructs_distortion_and_market_state():
    msgs = _adv_msgs()
    assert [m["role"] for m in msgs] == ["system", "user"]
    text = _text(msgs).lower()
    # Objective is to distort / move away from the informed consensus.
    assert "distort" in text
    assert QUESTION in _text(msgs)
    # Market state present (round + price), same JSON Trade contract.
    assert "round 2 of 3" in _text(msgs)
    assert "0.7000" in _text(msgs)
    assert '"action"' in _text(msgs)


def test_adversary_prompt_has_no_pass1_prior_or_honest_sizing_guidance():
    # The adversary is elicited no Pass-1 belief, so its prompt must not carry a
    # "prior analysis concluded" line, and it must NOT inherit the honest
    # "size to your edge / don't overspend" guidance (that contradicts its goal).
    text = _text(_adv_msgs())
    assert "prior analysis concluded" not in text
    assert "size your position to your edge" not in text
    assert "loses money in expectation" not in text


def test_adversary_styles_differ():
    stealthy = _text(_adv_msgs(style="stealthy")).lower()
    obvious = _text(_adv_msgs(style="obvious")).lower()
    assert "stealthy" in stealthy and "blend in" in stealthy
    assert "aggressive" in obvious


def test_adversary_prompt_pushes_named_target_side():
    # The prompt names ONLY a side to push toward (no price-vs-0.5 rule): NO or
    # YES verbatim, in the buy instruction. The old "above 0.5 -> buy NO" rule
    # must be gone entirely.
    no_text = _text(_adv_msgs(target_side="NO", price=0.7))
    assert "push the market price toward NO" in no_text
    assert "buy NO" in no_text
    yes_text = _text(_adv_msgs(target_side="YES", price=0.3))
    assert "push the market price toward YES" in yes_text
    assert "buy YES" in yes_text
    # No surviving trace of the removed price-vs-0.5 direction rule.
    for text in (no_text.lower(), yes_text.lower()):
        assert "above 0.5" not in text
        assert "below 0.5" not in text


def test_adversary_prompt_never_leaks_hidden_state():
    # Extended for the target-side rewrite: on top of the shared forbidden set,
    # the adversary prompt must never name the ground-truth quantities it is
    # (implicitly) opposing.
    banned_extra = ["true probability", "posterior", "accuracy", "latent"]
    for style in ("stealthy", "obvious"):
        for target_side in ("YES", "NO"):
            text = _text(_adv_msgs(style=style, target_side=target_side)).lower()
            for banned in _FORBIDDEN + banned_extra:
                assert banned not in text, f"adversary prompt leaked {banned!r}"


# ---------------------------------------------------------------------------
# Leakage guard (scientific integrity) — the load-bearing test
# ---------------------------------------------------------------------------

# Strings that must NEVER appear in any rendered prompt: accuracies, latent
# state, posteriors, other-agent info, shard counts beyond the agent's own.
_FORBIDDEN = [
    "accuracy", "accuracies", "latent", "posterior", "oracle",
    "other agent", "other agents", "true probability is 0",
    "signal 0", "signal 1", "6 shards", "of 6", "shard 0", "shard 1",
    "lam", "a_z",
]


def test_prompts_never_leak_hidden_state():
    # Exercise every persona for both passes and scan the full rendered text.
    for persona in list(PERSONAS) + [None]:
        p1 = _text(pass1_messages(QUESTION, SHARDS, persona)).lower()
        p2 = _text(pass2_messages(
            QUESTION, SHARDS, persona,
            price=0.5, round=1, n_rounds=3, bankroll=100.0,
            max_affordable_yes=50.0, max_affordable_no=50.0, pass1_prob=0.5,
        )).lower()
        for banned in _FORBIDDEN:
            assert banned not in p1, f"pass1 leaked {banned!r}"
            assert banned not in p2, f"pass2 leaked {banned!r}"


def test_format_constants_have_no_hidden_state_vocabulary():
    # Belt-and-suspenders: the reusable format constants must never mention the
    # generative model's hidden quantities (a future edit can't sneak one in).
    for banned in ("accuracies", "latent state", "posterior"):
        assert banned not in prompts._BELIEF_FORMAT.lower()
        assert banned not in prompts._TRADE_FORMAT.lower()
