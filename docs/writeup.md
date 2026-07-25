# When Do Prediction Markets Beat Static Pools? A Controlled Stress Test of LLM Belief Aggregation

## Abstract

We build a synthetic information environment whose Bayes-optimal posterior is
known in closed form, and use it to measure how much of the available
information two aggregation mechanisms recover from a pool of LLM agents:
independent static pooling and dynamic LMSR market trading. Each question is
elicited twice from the same agents on the same evidence — once in isolation
(feeding all static baselines) and once through three rounds of market trading
— so mechanism, model knowledge, and signal structure are separated by design.
On the main grid the market **underperforms** static pools of the same agents'
independent beliefs: the final price has a larger posterior gap than every
pool we tried, reversing the direction one might naively expect and consistent
with the pre-registration's allowance for a reversed result. A mixed-model team
beats its homogeneous counterparts after quality matching (a replication of the
diversity direction from prior monoculture work). An adversary instructed to
push the wrong side moves prices and degrades accuracy, but loses money at every
bankroll multiple — the market taxes manipulation even as some final
predictions flip. Persistent wealth across questions slightly worsens accuracy.
An exploratory extension along a model-capability axis — added after the initial
release and not pre-registered — finds this deficit shrinks to parity as trader
capability rises, with no tested tier's market beating its own independent pool,
and a same-family control shows the change is trading skill rather than added
knowledge. The main experiment cost $0.73; the capability extension added $3.06.

## 1. Introduction and positioning

Prediction markets are an old idea for aggregating dispersed private
information, and a recent line of work has shown that LLM agents can be made to
trade in them. Two facts are by now established in the literature and are *not*
what this paper claims. First, LLM agents with dispersed private signals can
trade via a logarithmic market scoring rule (LMSR) and produce a price that
reflects pooled information (arXiv:2604.20050, the closest prior work).
Second, pools of same-family LLMs have highly correlated errors, and
preference optimization can drive that correlation up to the point that ten
agents behave like roughly one (arXiv:2606.26583; see also
arXiv:2506.07962 and arXiv:2605.00844 on cross-model error correlation).

What that prior work entangles is the *decomposition*. When a market of LLM
agents lands near or far from the truth, at least four things are moving at
once: (i) how much the models know, (ii) how the private signals are
structured and how informative their union is, (iii) how the agents behave
under a trading prompt versus a plain elicitation prompt, and (iv) the market
dynamics themselves. If you only compare mechanisms against each other, or
against realized outcomes, you cannot say whether a market helped because
trading aggregated information or merely because the underlying models were
good, nor whether a static pool lost because pooling is weak or because the
signals were nearly redundant.

This paper is a **controlled decomposition and stress test**, not a first
demonstration. The contribution is separation, built from four design choices:

1. A **generative environment with an exact Bayes-posterior oracle**. Because
   we generate the latent state and the signals ourselves, we can compute the
   Bayes-optimal posterior given any subset of signals in closed form. Every
   mechanism is scored against that exact target (posterior gap), not only
   against the realized binary outcome (Brier).
2. An **independent-elicitation control**. Each agent states a belief in
   isolation before any market opens, on exactly the evidence it will later
   trade on. All static baselines are computed from these Pass-1 beliefs, so
   the market and the pools consume identical information from identical agents.
3. **Quality-matched team comparison**. The homogeneous and mixed teams are
   matched on standalone posterior gap (measured on held-out pilot questions),
   so a mixed team's advantage cannot be attributed to simply swapping in a
   stronger model.
4. The resulting **map of where the market helps, does nothing, or hurts**,
   read against the Bayes optimum rather than against the market's own past.

The deliverable is that map, plus a stress test of the market under adversarial
capital and under wealth persistence across questions. We pre-registered the
hypotheses and elicitation protocol (SPEC.md) before running any main
experiment; directions were left open where we genuinely did not know them.

## 2. The generative environment

The environment is the scientific asset, so we describe the formal model first
and the prose rendering second.

**Latent state and signals.** Each question has a binary latent state
`s ∈ {0,1}` — the answer — with a uniform prior, `s ~ Bernoulli(0.5)`. From
`s` we generate six signals through a two-channel process. A common latent
channel `z` equals `s` with probability `a_z` (and `1−s` otherwise); this
channel is what lets us dial correlation. Each signal `i` either copies the
common channel (`x_i = z`) with probability `λ_i`, or, with probability
`1−λ_i`, draws an independent private channel that equals `s` with the
signal's own accuracy `a_i`. Setting all `λ_i = 0` makes the six signals
conditionally independent given `s`, each with accuracy `a_i`; raising `λ_i`
introduces shared-error correlation. The main grid uses six conditionally
independent signals (`λ_i = 0`), each with accuracy 0.75, and a common-channel
accuracy `a_z = 0.8`.

**The oracle.** For any observed subset `A` of signals with values `x_A`, the
likelihood marginalizes the common channel:
`P(x_A | s) = Σ_z P(z | s) · Π_{i∈A} P(x_i | s, z)`, and the posterior
`P(s=1 | x_A)` follows by Bayes' rule under the uniform prior. This is computed
exactly, without sampling, and stored for every signal subset. Two properties
matter and are checked against the oracle at generation time, not by eyeball:
the posterior given the union of all six signals is informative, and **no
single shard is sufficient** — the posterior given any one signal differs from
the posterior given all six by at least a configured margin of 0.1. Questions
that fail either check are rejected and resampled. This guarantees that
information is genuinely dispersed across agents, so that aggregation has
something to do.

**Prose rendering.** Each signal is rendered into a natural-language shard — a
witness statement or document snippet — under a template constraint so the text
entails exactly its signal and nothing more. A separate adversarial
verification pass checks three things per shard: that the shard entails its
signal in the correct direction (direction entailment), that it does not leak
any other agent's signal (no cross-shard leaks), and that it carries no
meta-clue about its own reliability (no accuracy hints). We rendered and
verified 50 questions (300 shards). The first verification round passed 80% of
questions (shard-level: direction 97.3%, leak 99.0%, meta 99.0%); two surgical
re-render rounds with fresh verifiers brought all 50 questions to a 100% pass
rate on all three criteria. A manual spot-check of 10 shards across five
questions confirmed the automated verdicts. The dataset is 40 main questions
plus 10 held-out pilot questions, generated once and frozen as versioned JSONL
so every condition and seed loads identical questions — paired comparisons hold
by construction.

## 3. Mechanisms and protocol

**Market.** The market is Hanson's LMSR for a binary YES/NO outcome, with cost
function `C(q) = b·log(exp(q_yes/b) + exp(q_no/b))` and price
`p_yes = 1/(1 + exp((q_no − q_yes)/b))`, implemented in a log-sum-exp-stable
form (Hanson 2003; Abernethy, Chen, and Wortman Vaughan 2013; Storkey 2011).
The liquidity parameter `b` sets how much cash moves the price; pushing YES
from 0.5 to 0.8 costs about `b·ln(2.5) ≈ 37`, roughly a third of a 100-unit
bankroll. We chose `b = 40` by a pilot sweep over {20, 40, 80}: at `b = 20`
prices went degenerate (pinned to an extreme) on a large fraction of pilot
questions, at `b = 80` the market was sluggish and showed no advantage, and
`b = 40` was non-degenerate (degenerate on 10% of pilot questions) while
remaining responsive. Each agent starts with a bankroll of 100. The market
runs three rounds; agent order is reshuffled each round from a seeded RNG.
Requested trade sizes are clamped to affordability server-side, and every turn
— including holds — is logged. `b` and the prompts were frozen after the pilot.

**Two-pass elicitation.** This is the contamination control, followed exactly.

- **Pass 1 (independent).** Each agent sees only the question, its own shard(s),
  and its persona — no prices, no other agents, no mention of a market. It
  outputs a `Belief` (a probability plus a short rationale). These Pass-1
  beliefs feed *all* static baselines and *all* error-correlation estimates.
- **Pass 2 (market).** The same agents trade on the same shards in a fresh
  context, each told its own Pass-1 conclusion as its prior. In-market stated
  beliefs are logged for dynamics analysis only and are never used as a
  baseline.

A code-level phase guard enforces the separation: baseline functions accept
only records tagged `phase="independent"` and assert on anything else. This
means the market and the static pools always aggregate the *same information
from the same agents* — the only thing that differs is the mechanism.

**Baselines.** From the Pass-1 beliefs we compute four static pools: the mean,
the median, a log opinion pool, and a leave-one-out calibrated stack. All are
scored against the oracle posterior (posterior gap) and the realized outcome
(Brier).

**Agents.** Six fixed personas — base-rate skeptic, evidence maximalist,
contrarian, domain specialist, Kelly-style sizer, and momentum reader — are
held constant across conditions; each agent is assigned one shard. Three models
back the pools: `meta-llama/llama-3.1-8b-instruct` (model A), 
`deepseek/deepseek-v4-flash` (model B), and 
`nvidia/nemotron-3-super-120b-a12b` (model C, run on a free tier in the
homogeneous arm). Their standalone posterior gaps, measured on the pilot
questions, are 0.291, 0.268, and 0.257 respectively — closely matched, which is
what makes the mixed-team comparison a comparison of composition rather than of
raw model quality. Structured output uses guided decoding where available and
otherwise a validate-and-one-retry loop, falling back to a logged `hold` on a
second failure.

**Condition grid.** All conditions run over 40 questions × 3 seeds. Condition A
is a homogeneous deepseek pool (it also serves as homogeneous arm B2 and as the
Pass-1 source for the wealth conditions). B1 and B3 are homogeneous llama and
nemotron pools. C is the mixed A/B/C team, two agents per model. D adds an
adversary to pool C at bankroll multiples k ∈ {1, 3, 10}. W runs pool A with
wealth persisting across questions, in fixed and shuffled question orders. The
D and W conditions reuse their source condition's Pass-1 beliefs and run the
market pass only. Main-grid spend was $0.7255, and the whole experiment cost
$0.73 across all ledgers including the pilots.

## 4. Results

Every number below is quoted from the analysis pipeline's frozen outputs
(`analysis_out/stats.md`, `table1.md`, `table2.md`), reproducible from the
released event logs. Figures 1 and 2 and Tables 1 and 2 live in
`analysis_out/`. The primary metric is the posterior gap against the oracle;
comparisons use a paired bootstrap 95% CI and a Wilcoxon signed-rank test over
the 40 questions × 3 seeds, with per-seed means reported for transparency.

### 4.1 H1 — market vs. static pools (reversed)

> *H1: On info-asymmetric questions, the final market price has a smaller
> posterior gap than static pools computed over independently elicited
> pre-market beliefs from the same agents on the same shards. Direction unknown
> — a null or reversed result is a reportable finding.*

The result is **reversed** (Fig 1, `analysis_out/fig1.png`). On condition A the
market's final price has a posterior gap of 0.269 ± 0.012, larger than every
static pool built from the same agents' independent beliefs. Against the mean pool (gap 0.218 ± 0.009) the
market is worse by Δ = 0.051 (95% CI [−0.001, 0.100], Wilcoxon p = 0.080).
Against the other three pools the market is worse by a significant margin: the
median pool (Δ = 0.081, CI [0.038, 0.123], p = 0.005), the log opinion pool
(Δ = 0.060, CI [0.008, 0.109], p = 0.048), and the calibrated stack
(Δ = 0.073, CI [0.019, 0.122], p = 0.008). Per-seed means are tight and all
positive (e.g. 0.084 / 0.082 / 0.078 for the median pool), so the direction is
stable across seeds.

The Brier deltas point the same way — the market's Brier (0.281 ± 0.014) is
worse than the mean pool's (0.202 ± 0.001) by Δ = 0.079 — but none of the four
Brier comparisons is individually significant (all p ≥ 0.085). The headline is
that in this regime, LMSR trading **destroys** information relative to simply
pooling the same agents' independent beliefs. This is a reversal of the naive
expectation and squarely within what the pre-registration flagged as a
reportable outcome.

Two caveats bound this result. It holds for **this regime** — six traders,
three rounds, `b = 40` — and thin markets like this are exactly where LMSR
path-dependence is most likely to bite (caveat: thin markets). And the static
baselines had access to *exactly the same information* as the market; that
equality is the point of the design, not a handicap on the market.

### 4.2 H2a — mixed vs. homogeneous teams (replication)

> *H2(a): mixed-model teams outperform homogeneous teams.*

This is a **replication** of the diversity direction reported for LLM
prediction markets (arXiv:2606.26583), and it holds. The mixed team C has the
smallest market posterior gap of any team (0.225 ± 0.021). Against the
homogeneous llama arm B1 (0.380 ± 0.054) the improvement is large and highly
significant: Δ = −0.155 (CI [−0.211, −0.100], p < 0.001). Against the
homogeneous deepseek arm (condition A / B2) the gap is Δ = −0.044 (CI
[−0.088, 0.000], p = 0.106), and against the homogeneous nemotron arm B3
(0.276 ± 0.028) it is Δ = −0.051 (CI [−0.107, 0.004], p = 0.118). The mixed
team beats all three homogeneous arms in point estimate; the advantage is
decisive against the weakest model and directional-but-not-significant against
the two stronger ones. Because the three models were quality-matched on
standalone gap (0.291 / 0.268 / 0.257), this advantage survives quality
matching — it is composition, not a smuggled-in stronger model.

### 4.3 H2b — the phase map (descriptive)

> *H2(b): the market-vs-static gap shrinks as measured error correlation ρ
> rises.*

For each of the 12 (condition, seed) points among A, B1, B3, and C we measured
the team's error correlation ρ from the Pass-1 beliefs and the market's deficit
relative to the mean pool (market gap − mean-pool gap). The Spearman rank
correlation between ρ and the market deficit is r_s = 0.280 (n = 12). The sign
is consistent with the hypothesis — higher correlation goes with a larger
market deficit — and the low-ρ mixed team C is the **only** condition with
per-seed deficits below zero (−0.010 and −0.029 on seeds 1 and 2), i.e. the only
setting where the market actually helps. The relationship is far from clean,
however: the team with the genuinely lowest error correlation is the homogeneous
deepseek pool A (team-mean ρ = 0.398, and the single lowest-ρ seed at 0.378),
and it still shows a market deficit on every seed (+0.055 / +0.051 / +0.045).
That is exactly why the rank correlation is weak (r_s = 0.280) rather than
strong — low correlation is necessary but not sufficient for the market to help
in this environment.

This is **descriptive, not a fitted law** (caveat: narrow ρ range). The
achieved ρ values span only 0.378 to 0.514, a narrow band, and with n = 12 we
report the rank correlation as a description of the map rather than as a
hypothesis test. Fig 2 shows the phase map; the takeaway is a direction and a
single below-zero region (the low-correlation mixed team), not a calibrated
curve.

### 4.4 H3 — adversarial manipulation

> *H3: for an adversary with k× bankroll pushing the wrong side, report Brier
> degradation, flip rate, capital-per-move, adversary P&L, and recovery.
> Hypothesis: flip rate increases with k; adversary P&L is negative at
> settlement.*

We add one adversary to pool C, instructed to push the price toward the wrong
side while staying plausible, at bankroll multiples k ∈ {1, 3, 10}. The
manipulation works in the expected direction and grows with capital. The
final-prediction **flip rate rises with k**: 0.317 → 0.358 → 0.392. Terminal
**Brier degradation** rises steeply: 0.105 → 0.219 → 0.312. The share of
questions pushed to the wrong side climbs from 0.508 to 0.617 to 0.658.

But the market **taxes** the attack. Adversary P&L at settlement is negative at
every multiple, and the absolute loss grows with the capital deployed:
−$46.60 at k=1, −$143.06 at k=3, and −$440.41 at k=10. The mean capital
required to move the price by 0.1 is 44.2 / 30.1 / 306.5 across the three
multiples. So while a well-funded adversary can flip a meaningful fraction of
final predictions and degrade calibration, it pays progressively more to do so
and never profits — consistent with the hypothesis that the market taxes
manipulation.

Recovery is low: the post-attack recovery fraction is 0.158 / 0.200 / 0.092
across k. An important **caveat**: recovery *timing* is structurally
unmeasurable here. The adversary trades in every round, including the last, so
there is no clean post-attack window in which the price could relax back;
`median_recovery_rounds` is 0 by construction. We therefore report the recovery
fraction only, not a recovery-time. The manipulation panel of Fig 2 shows flip
rate and recovery fraction versus k.

### 4.5 H-wealth — persistent wealth (secondary)

> *H-wealth: persistent wealth across questions changes market accuracy vs.
> per-question reset; direction unknown; test order-dependence.*

Letting bankrolls persist across questions **slightly worsens** the market's
posterior gap relative to the per-question reset of condition A. With a fixed
question order, W_FIXED − A is Δ = 0.050 (CI [−0.001, 0.098], p = 0.035); with
a shuffled order, W_SHUFFLED − A is Δ = 0.058 (CI [0.007, 0.111], p = 0.032).
Both are small and marginally significant. There is **no order-dependence**:
W_FIXED vs. W_SHUFFLED gives Δ = −0.008 (CI [−0.050, 0.034], p = 0.737), so the
effect does not depend on the sequence in which questions are seen. The Brier
comparisons for the wealth conditions are not significant (p ≥ 0.39). The
direction — persistence hurts a little — was not pre-specified; we report it as
found.

### 4.6 Secondary reporting

Parse reliability was high across the grid: 5 parse failures total (4 from
deepseek, 1 from nemotron) out of tens of thousands of LLM turns, with per-model
rates at or below 0.19% in every condition (Table 2). The market maker's subsidy stayed within its theoretical
worst-case bound: the maximum per-question subsidy observed was 27.726, just
under the `b·ln 2 ≈ 27.73` ceiling at `b = 40` (Table 2). Accuracy-per-dollar
figures are in Table 1; because absolute spends are tiny and some arms ran on a
free tier, these ratios are noisy and we report them for completeness rather
than as a headline. Total grid spend was $0.7255.

## 5. Why does the market hurt here?

This section is an **interpretation**, clearly labeled as such — a set of
mechanisms consistent with the data, not a claim of proof.

Three ingredients plausibly combine to make LMSR trading lose information
relative to independent pooling in this regime. First, **price commitment on
the realized side**. A pilot observation (not itself a confirmed finding) was
telling: at `b = 40` the market's final-price Brier beat the mean pool while
its posterior gap was slightly worse — the market commits harder toward the
side that turns out correct, which flatters the Brier against the realized
outcome but overshoots the calibrated posterior. Overcommitment is penalized by
the posterior-gap metric precisely because that metric knows the exact right
answer, not just the realized coin flip. A smoke observation on held-out easy
questions during the main runs showed the same pattern: market Brier can beat
the pools while posterior gap does not — the market overcommits.

Second, **herding on shared errors**. When agents' errors are correlated,
trading lets them reinforce a common mistake: a price that already reflects the
shared bias invites more trades in the same direction. The phase map (§4.3) is
weakly consistent with this — the only condition whose market deficit ever turns
negative is the low-ρ mixed team C, though the tendency is loose (the lowest-ρ
homogeneous team, deepseek pool A, still runs a deficit on every seed). A static
pool cannot amplify a shared error the way a sequential price can, because it
never shows one agent another agent's move.

Third, **LMSR path-dependence with thin participation**. With only six traders
over three rounds, the final price depends on who traded when; there is no law
of large numbers to wash out order effects, and the reshuffling that we do each
round cannot fully average away path-dependence at this scale. Thin markets are
where this mechanism is most acute.

None of these is established as *the* cause; they are the interpretations most
consistent with the measured pattern, offered to structure follow-up work.

## 6. Extension: the capability axis

This section is an **exploratory extension**, run on 2026-07-24 after the initial
release and **not part of the pre-registration**: the hypotheses here were formed
after we saw the v1 reversal, and we report them as post-hoc.

**Motivation.** The reversal of §4.1 was measured on one trader model
(`deepseek/deepseek-v4-flash`, "flash"). A natural worry is that it is an
artifact of a *cheap* model — perhaps a weak trader mishandles prices and
destroys information that a more capable trader would preserve. We test that by
re-running condition A's two-pass protocol on the same 40 questions × 3 seeds
with two more capable homogeneous tiers.

**Design.** Three homogeneous tiers, identical questions, seeds, personas, and
`b = 40`: **flash** (condition A), **pro** (`A_PRO`, `deepseek-v4-pro`), and
**luna** (`A_LUNA`, `gpt-5.6-luna`). Pro is the key control — it is the *same
model family* as flash, so it holds model knowledge roughly fixed while raising
capability, and any change it produces on the market side is trading skill, not
new knowledge. Luna is a stronger frontier model from a different vendor. Each
tier's pool gap, market gap, and deficit (market gap − mean-pool gap, positive ⇒
the market destroys information) are computed exactly as in §4. The extension
cost $3.06 ($0.45 for A_PRO, $2.61 for A_LUNA).

| Tier | Pool gap | Market gap | Deficit (market − pool) | Market Brier | Team ρ |
|---|---|---|---|---|---|
| flash | 0.218 | 0.269 | 0.051 [−0.001, 0.100] (p = 0.080) | 0.281 | 0.398 ± 0.018 |
| pro | 0.227 | 0.224 | −0.003 [−0.048, 0.041] (p = 0.377) | 0.215 | 0.378 ± 0.029 |
| luna | 0.191 | 0.192 | 0.002 [−0.046, 0.050] (p = 0.451) | 0.201 | 0.283 ± 0.005 |

(n = 120 per tier; deficit CIs are 95% bootstrap, p from Wilcoxon signed-rank.)

**Claim 1 — the deficit reaches parity, not superiority.** Flash's market runs a
deficit of 0.051 [−0.001, 0.100] (p = 0.080) against its own pool. At both
capable tiers the deficit vanishes — pro −0.003 [−0.048, 0.041] (p = 0.377) and
luna 0.002 [−0.046, 0.050] (p = 0.451), CIs straddling zero — so a more capable
trader **stops destroying information**. But it does not go past parity: no
tier's market beats its own pool, and the pro and luna market gaps (0.224, 0.192)
sit essentially on top of their pool gaps (0.227, 0.191). The capable-market
story is parity, not superiority.

**Claim 2 — the fix is trading skill, not knowledge.** Pro's independent pool is
no better than flash's: the pro − flash pool-gap difference is 0.010
[−0.011, 0.030] (Wilcoxon p = 0.289), indistinguishable from zero and if
anything nominally worse. Yet pro's *market* gap is lower than flash's by
Δ = −0.044 [−0.098, 0.008] (p = 0.127). Because pro shares flash's model family
its knowledge is held roughly fixed, and the improvement surfaces only once the
market runs — so the deficit fix is attributable to how the capable model
*trades*, not to what it *knows*.

**Claim 3 — capability matters independently of ρ.** One could instead explain
the fix through error correlation — lower ρ, less herding, smaller deficit. But
pro's team ρ (0.378 ± 0.029) is statistically indistinguishable from flash's
(0.398 ± 0.018) while their deficit outcomes are opposite, so capability moves
the market-vs-pool gap through a channel other than measured ρ. Luna does
confound capability with lower correlation (ρ = 0.283 ± 0.005, and its pool is
also the strongest — luna − flash pool gap −0.027 [−0.050, −0.004], p = 0.000),
but the same-family pro control isolates capability from ρ.

**Mechanism (supporting, descriptive).** The herding regression — §3's
sanctioned dynamics-only analysis, regressing each agent's stated in-market
belief on its own Pass-1 belief (b1) and the pre-trade price (b2) — offers a
consistent, if modest, mechanism. The pooled price-weight coefficient b2 is
0.045 [0.028, 0.060] for flash, 0.010 [−0.000, 0.021] for pro, and
0.023 [0.011, 0.036] for luna: flash's traders weight the running price
significantly more than either capable tier's (mean belief drift |y − x1| is
0.037 for flash vs. 0.012 for pro), while own-information weight b1 stays near 1
in every tier. This is **descriptive and the effect is small** — all three
coefficients are modest and the pro-vs-luna ordering is not distinguishable
(overlapping CIs) — so it is supporting evidence for the herding interpretation
of §5, not proof.

Fig 2-v2 (`analysis_out/v2/fig2_v2.png`) shows the phase map of §4.3 with pro and
luna added; Fig 3 (`analysis_out/v2/fig3.png`) shows the per-tier deficit and the
price-weight b2 by tier and round. The full tier and herding tables live in
`analysis_out/v2/tiers.md` and `herding.md`, regenerated with:

```
python -m pnyx.cli analyze-v2 --flash data/main/A --pro data/v2/A_PRO \
    --luna data/v2/A_LUNA --out analysis_out/v2
```

## 7. Limitations

We restate the binding caveats and add the design-scope ones.

1. **Recovery timing is structurally unmeasurable.** The adversary trades every
   round including the last, so there is no post-attack window; we report only
   the recovery fraction, never a recovery time (§4.4).
2. **The phase map is descriptive, not a fitted law.** The achieved team-ρ
   range is narrow (0.378–0.514) and n = 12, so the Spearman r_s = 0.280 is a
   description of direction, not a hypothesis test (§4.3).
3. **Thin markets.** All results are for six traders over three rounds — the
   regime where LMSR path-dependence and overcommitment are most likely to
   matter (§4.1, §5).
4. **Prompt and liquidity sensitivity.** `b` was chosen from a three-point
   pilot sweep and the prompts were frozen after a pilot, not exhaustively
   searched; different prompting could shift the market's behavior (§3).
5. **Synthetic rendering.** Signals are rendered into template-constrained prose
   and adversarially verified, but this may not transfer to real documents with
   their noise, redundancy, and stylistic tells (§2).
6. **Single environment family.** Everything here is a binary latent state with
   six conditionally independent signals under a uniform prior; other signal
   structures, priors, or outcome cardinalities are untested (§2).
7. **The capability axis is exploratory and thin.** The §6 extension was not
   pre-registered — its hypotheses were formed after the v1 reversal — and it
   adds only two tiers. Luna is the single non-deepseek frontier vendor tested,
   so cross-vendor capability rests on one model; and luna confounds capability
   with lower error correlation, a confound the same-family pro control mitigates
   but does not remove. All three tiers ran on the same 40-question environment
   (§2), so the parity finding inherits every environment caveat above.

Two further points. The static baselines had access to *exactly the same
information* as the market — that equality is the design, and it means the
comparison is fair rather than that the market was disadvantaged. And every
quantitative claim here is specific to this regime (6 agents, 3 rounds,
`b = 40`, the configured signal structure); we make no claim that the reversal
generalizes beyond it. Peer prediction — which could in principle distinguish
genuine private information from correlated agreement — is deliberately out of
scope for this version (arXiv:2601.20299) and left to a sequel.

## 8. Reproducibility

The pre-registration in SPEC.md fixes the hypotheses and the two-pass
elicitation protocol before any main run. The P4 event logs are released under
`data/main/`, and every figure, table, and statistic in this writeup
regenerates from them with a single command and no API calls:

```
python -m pnyx.cli analyze-main --data-root data/main \
    --configs pnyx/configs/main --out analysis_out
```

The run completes in a few seconds. All bootstrap statistics are deterministic
under a fixed bootstrap seed, so the CIs and p-values above are reproducible
exactly. Rerunning the experiments from scratch requires an OpenRouter key and
costs about $0.75 in total; the analysis of the released logs costs nothing.

## 9. Conclusion

In a controlled environment where the right answer is known exactly, LMSR
market trading among a small pool of LLM agents recovers *less* of the
available information than simply pooling those same agents' independent
beliefs — a reversal of the naive expectation, and the clearest single point on
our help/nothing/hurt map. The market helps in only one measured corner: a
low-correlation, quality-matched mixed team. Model diversity helps as prior
work found (a replication, not a new claim), adversarial capital moves prices
and degrades accuracy but is always taxed at settlement, and wealth persistence
hurts slightly. A post-hoc capability extension refines the reversal rather than
overturning it: cheap traders make LMSR destroy information, more capable
traders — including a same-family control that isolates trading skill from
knowledge — pull the market back to parity with the pool, and no tested tier's
market beats independent pooling. The value of the design is not any one of these numbers but the
separation that produced them: mechanism, model knowledge, and signal structure
pulled apart against an exact oracle, so that "the market helped" can be
checked rather than assumed. The interpretation of *why* the market hurts here
— overcommitment, herding on shared errors, thin-market path-dependence — is
offered as structure for follow-up, not as settled fact.

## 10. References

- Hanson, R. (2003). Combinatorial information market design. *Information
  Systems Frontiers*. (LMSR.)
- Storkey, A. (2011). Machine Learning Markets. *AISTATS*.
- Abernethy, J., Chen, Y., and Wortman Vaughan, J. (2013). Efficient market
  making via convex optimization, and a connection to online learning.
- Information Aggregation with AI Agents. arXiv:2604.20050. (Closest prior:
  LLM agents with dispersed private signals trading via LMSR; homogeneous vs.
  heterogeneous teams.)
- Preference Optimization Drives Monoculture in LLM Prediction Markets.
  arXiv:2606.26583. (Monoculture / DPO error correlation; cross-model
  diversity mitigation — the direction we replicate.)
- Decentralized Aggregation of LLM Predictions via Wagering Mechanisms.
  arXiv:2607.04389.
- Correlated Errors in Large Language Models. arXiv:2506.07962.
- The Oracle's Fingerprint. arXiv:2605.00844.
- Truthfulness Despite Weak Supervision. arXiv:2601.20299. (Peer prediction —
  why it is out of scope here and left to a sequel.)
