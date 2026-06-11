### P021 — False-belief (Sally-Anne) epistemic-state preservation
* **SIGNAL:** Both the relational probe and the AV matcher will hit the absolute floor (noise), as a rudimentary NLA's bottleneck and degenerate text cannot encode or preserve false-belief structures.
* **FAILURE MODE:** The linear probe will fail to train on a low-dimensional bottleneck or small dataset, while the AV matcher will output a constant default label on degenerate or empty text.
* **FLOOR VERSION:** Simplify the setup to probe merely for the physical presence or absence of a displaced object name (e.g., "basket" vs. "box") in a single-agent context.
* **MISSED FLAW:** If the AR is near-random (cosine ~0.1-0.4), the reconstructed activation space will lose alignment with the original space, making the linear probe results entirely uninterpretable regardless of bottleneck information.### P022 — Who-said-what attribution preservation
* **SIGNAL:** The relational binding signal will be lost to noise, though a retrieval-only verbalizer might sporadically output speaker names, causing the content baseline to look deceptively high while the binding accuracy saturates at random guess (50%).
* **FAILURE MODE:** When the AV emits repetitive, degenerate text, string-level attribution matchers fail catastrophically by defaulting to a single speaker or failing to match any name-claim pair.
* **FLOOR VERSION:** Reduce the task to a single speaker attributing a single claim ("A said X" vs. "X is true") to test basic source-attribution before moving to two-speaker relational binding.
* **MISSED FLAW:** In a low-dimensional or low-data regime, distinct proper names might collapse into a single generic "person" representation, causing spurious matching failures that are lexical rather than relational.### P023 — Reported-vs-asserted claim-status flattening (claim laundering)
* **SIGNAL:** Because status markers ("allegedly", "reported") are lexical, they may survive in some rudimentary verbalizers, but the probe will register pure noise due to the low capability of the NLA's representation.
* **FAILURE MODE:** An early-checkpoint AV emitting near-empty or highly degenerate text will drop the status markers entirely, causing a false positive for "claim laundering" (upgrading to asserted) when it is actually just standard vocabulary dropout.
* **FLOOR VERSION:** Test raw token retention of the lexical status words ("allegedly", "reported") in isolation before asserting a relational "laundering" effect.
* **MISSED FLAW:** Lexicon-based status matchers are highly sensitive to syntax; a weak verbalizer that outputs words in scrambled order will trigger the matcher's fallback rules, misclassifying status based on word position rather than semantic flattening.### P024 — Hedged-vs-certain modality preservation
* **SIGNAL:** On a rudimentary NLA, the modality signal will hit the floor, with the probe exhibiting chance performance and the AV text drifting to a default output due to vocabulary collapse.
* **FAILURE MODE:** Degenerate AV output (e.g., repeating the most frequent modal verb like "is") will look like systematic "hedged-to-certain" drift, when it is actually just high-entropy collapse to the model's unigram prior.
* **FLOOR VERSION:** Evaluate whether the NLA can distinguish between binary negation ("is" vs. "is not") before moving to fine-grained modal distinctions like "might" vs. "definitely".
* **MISSED FLAW:** If the AR round-trip cosine is 0.1-0.4, any high-frequency lexical item (like common certainty markers) will dominate the reconstruction space due to poor representation of low-frequency hedge words, creating a false illusion of systematic psychological drift.### P025 — First-vs-third-person knowledge-boundary preservation
* **SIGNAL:** Usable signal is limited to basic pronoun retention (first vs. third person), while the polarity cell ("doesn't know") will be entirely lost to noise.
* **FAILURE MODE:** With a tiny model or early checkpoint, negations ("not", "doesn't") are typically the first to drop, collapsing the polarity dimension entirely and making the 2x2 grid uninterpretable.
* **FLOOR VERSION:** Strip the knowledge/epistemic aspect and test a simple binary classification of grammatical person ("I" vs. "She") in short, non-nested sentences.
* **MISSED FLAW:** Pronoun replacement in degenerate text is highly subject to positional biases of the decoder, which the plan overlooks by assuming pronouns map reliably to the speaker.### P026 — Secret / private-info retention (does the bottleneck leak what shouldn't be revealed)
* **SIGNAL:** This plan provides a surprisingly usable signal even on rudimentary NLAs, as it reduces to simple concept survival (Family-A primitive) which operates above the floor even with weak bottlenecks.
* **FAILURE MODE:** If the AR is near-random (cosine ~0.1), the secret content won't reconstruct successfully, causing a floor effect where the leak rate is artificially zero despite bottleneck presence.
* **FLOOR VERSION:** Measure the presence of the secret token directly in the bottleneck activation using a simple logistic regression probe, bypassing the degenerate AV text entirely.
* **MISSED FLAW:** The plan assumes public and secret content are matched on frequency, but secret concepts (e.g., passwords or sensitive keys) have highly anomalous activation profiles that bypass standard calibration baselines.### P027 — Sandbagging-cue detectability (deliberate underperformance)
* **SIGNAL:** This plan will produce absolute zero signal (pure noise) in the rudimentary regime, as deliberate intent is a high-order relational concept far beyond a weak NLA.
* **FAILURE MODE:** Linear probes on a low-dimensional bottleneck will fail to converge, and the degenerate AV text will yield empty matches, leading to a trivial null result.
* **FLOOR VERSION:** Replace "intent to fail" with a simple sentiment or instruction-following check (e.g., negative vs. positive phrasing).
* **MISSED FLAW:** The plan conflates the lack of representation (the bottleneck never encoded it) with active stripping, failing to acknowledge that a tiny model lacks the capacity to represent inten
nt in the first place.

### P028 — Sycophancy / framing leakage into content
* **SIGNAL:** You will get a weak but usable signal on the probe side, as framing (sycophantic vs. neutral) often leaves a massive, low-frequency imprint on the activations of even small models.      
* **FAILURE MODE:** If the verbalizer is retrieval-only or degenerate, it will output the same static factual text regardless of framing, hiding the underlying activation-level contamination.
* **FLOOR VERSION:** Focus purely on the activation probe's ability to classify the prompt framing from the bottleneck representation, ignoring the degenerate AV output.
* **MISSED FLAW:** Framing differences often correlate with sentence length and token position, which can easily confound linear probes in low-dimensional bottlenecks.

### P029 — Intent-vs-action distinction preservation
* **SIGNAL:** The signal will saturate or collapse to noise, as a weak NLA will fail to distinguish grammatical aspect (planning vs. doing).
* **FAILURE MODE:** A degenerate AV will output only the root verb (e.g., "access"), which will look like a completed action, creating a false positive for "collapse to action."
* **FLOOR VERSION:** Test basic verb tense preservation (past vs. future) before attempting to distinguish nuanced planning from execution.
* **MISSED FLAW:** The plan assumes "completed" is the default collapse direction, but a weak decoder might collapse to the most frequent lexical token in its vocabulary, regardless of aspect.        

### P030 — Concealment / omission-marker preservation
* **SIGNAL:** The plan will yield pure noise for silent omission, though the lexical marker check might show a weak signal if the NLA has basic vocabulary retention.
* **FAILURE MODE:** Degenerate text will drop the complex negation and omission phrases, rendering the matcher-based metrics completely useless.
* **FLOOR VERSION:** Test basic text negation detection ("not X" vs. "X") as a proxy for the much harder concealment relations.
* **MISSED FLAW:** Probing for "absence" in low-dimensional spaces often picks up residual noise from context lengths, confusing sequence length with the semantic omission.

---

### Skip on rudimentary NLAs
* **P021** (Sally-Anne false-belief preservation)
* **P027** (Sandbagging-cue detectability)
* **P030** (Concealment / omission-marker preservation)

### Highest-value for the rudimentary regime
* **P026** (Secret / private-info retention)
* **P028** (Sycophancy / framing leakage)
* **P023** (Reported-vs-asserted claim-status flattening)
