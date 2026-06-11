# NLAttack methodology

Conceptual background for the NLAttack benchmark: what a Natural Language
Autoencoder is, the two purposes the suite serves, how it grounds misuse
detection in the ATT&CK Navigator, and the validity discipline behind every
number. For the runnable evaluations see [EVALUATIONS.md](EVALUATIONS.md); for
results see [RESULTS.md](RESULTS.md).

## What is a Natural Language Autoencoder?

A **Natural Language Autoencoder (NLA)** is an interpretability method that
explains a model's internal state in plain language. It has two trained parts:

```
activation  --[ AV: activation verbalizer ]-->  natural-language text
text        --[ AR: activation reconstructor ]-->  activation'
```

The **AV** reads a hidden activation and writes a description of what it
represents. The **AR** reads that description and reconstructs the activation.
Training optimizes the round trip: a good description lets the AR reconstruct the
original activation closely (measured in activation space, e.g. cosine
similarity). The human-readable **bottleneck** is the AV's text. The promise of
NLAs over sparse autoencoders or linear probes is that the explanation is in
natural language, so a person can read it and even edit it to intervene.

NLAs are introduced in Anthropic's "Natural Language Autoencoders" work
([overview](https://www.anthropic.com/research/natural-language-autoencoders),
[technical writeup](https://transformer-circuits.pub/2026/nla/)). Interactive NLAs
for several open models are hosted on [Neuronpedia](https://www.neuronpedia.org/nla),
and an open-source training library is `kitft/natural_language_autoencoders`.


## Two interconnected purposes

NLAttack exists to evaluate an NLA on two fronts at once, and the families are
designed so that one body of measurement serves both.

1. **Fundamental capability.** Does the NLA work at all, on the basic level a weak
   or early-training NLA actually operates at? This is the floor question, asked
   four ways: did a concept survive the bottleneck? Can a probe still read it off
   the activation? Does the verbalizer actually say it? And does the reconstructor
   put it back? Families A, B, E, F, H, I, J, K, and L cover this.
2. **Usefulness as a safety monitor.** If you read the NLA's text to watch what a
   model is doing, can you actually catch misuse? NLAttack grounds this in the
   concrete techniques and risk axes from Anthropic's LLM ATT&CK Navigator (see the
   background section below). Families C, D, G, and M cover this: attack-technique
   survival, obfuscation see-through, dual-use intent laundering, kill-chain
   co-occurrence, agentic-orchestration markers, and deception decodability.

These are not two separate test suites bolted together. The misuse families are the
capability machinery pointed at attack content. The single number that scores
capability, the **bottleneck-minus-verbalizer gap**, is also the monitor's blind
spot: every concept the activation carries but the verbalization drops is a place an
attacker can route through undetected. Concretely, the misuse plans reuse the
capability primitives. P037 is the probe-grounded version of the attack-laundering
test, P038 is the probe-grounded version of the ARiES-axis test, and P040 turns the
whole thesis ("an NLA bottleneck as a safety monitor") into a falsifiable downstream
classifier experiment.


## What NLAttack measures, and why

An NLA can fail on two distinct sides, and NLAttack scores both:

1. **Bottleneck side (the activation).** Does the activation even carry the
   concept? Measured with linear **probes on the activation**, each with a
   label-permutation null control.
2. **Verbalizer side (the AV text).** Does the AV's description actually surface
   the concept that the bottleneck holds? Measured by matching concepts in the
   verbalization, and by discrimination, retrieval, and faithfulness axes.

The gap between the two (`probe accuracy − verbalizer accuracy`) isolates how much
the verbalizer loses, which is usually the part a practitioner is stuck on.

Two principles run through the suite:

- **Floor-first.** On a weak NLA the verbalizer can be degenerate and the AR
  near-random, so end-to-end metrics collapse to noise. The base measurement is a
  single reliable primitive: *did concept C (or a near neighbor) survive into the
  verbalization, yes or no?* Everything aggregates from one per-concept table.
- **Null controls on everything.** A probe at small n can hit high accuracy on
  pure noise. Every probe reports a label-permutation floor, and a result counts
  as signal only when it clears that floor by a margin. Matcher-dependent effects
  are flagged by an ensemble-agreement check.


## Background: the ATT&CK Navigator and the misuse families

[MITRE ATT&CK](https://attack.mitre.org/) is a public knowledge base of real
adversary tactics and techniques (reconnaissance, credential dumping, lateral
movement, and so on). Anthropic's red team adapted it to language models in the
**[LLM ATT&CK Navigator](https://red.anthropic.com/2026/attack-navigator/)**, which
maps how LLM misuse falls across those tactics and scores risk on three ARiES
axes: Threat, Vulnerability, and Impact. Two findings from that work shape the
misuse families directly. Risk is not uniform across tactics, so a monitor with
holes in the high-risk post-compromise tactics is worse than its average hole
rate suggests. And agentic orchestration, a model chaining steps autonomously,
is the single largest risk amplifier.

NLAttack treats those findings as the test specification for "is this NLA a usable
misuse monitor." `attack_concepts.py` encodes 18 techniques spanning 13 of the
ATT&CK tactics, each with its phase (pre/post-compromise), ARiES axis, and a benign
neighbor it can launder into. The misuse tests then ask whether the bottleneck preserves what a monitor
needs: technique coverage weighted by risk, see-through against an obfuscation
ladder, whether an attack term reconstructs as its benign neighbor (laundering),
whether agentic-orchestration markers survive, and whether a real downstream
classifier reading the bottleneck can still flag malicious inputs (P040). A hole
here is not an abstract capability gap. It is a specific technique an adversary can
push through the NL bottleneck without the monitor noticing.


## Methodology and honest limits

- **Concept matching is the weak link.** Prefer the embedding matcher and spot-
  check matched terms. The ensemble agreement / contested-rate check flags effects
  that only one matcher topology sees.
- **A low score is ambiguous** without controls. It can mean the NLA dropped the
  concept, or the verbalizer could not phrase it, or the matcher missed it. The
  bottleneck probe (ground truth) and the permutation null are how you tell these
  apart.
- **Read differentially.** Compare every number to the frequency-and-length
  matched control and the noise floor, not in isolation.
- **Hosted-API subset.** The probe and emergence axes need raw activations, which
  the hosted API does not expose, so they are full-access only. Over the API you can
  run the full API tier (retention, doc retrieval, discrimination, misuse tests).
  See [access tiers](#access-tiers-api-runnable-vs-full-access) for the exact split,
  which is also queryable in code via `nla_eval.access`.

The rationale behind these choices, including two external expert reviews, is in
[`DESIGN_REVIEW.md`](../DESIGN_REVIEW.md) and [`docs/`]().

