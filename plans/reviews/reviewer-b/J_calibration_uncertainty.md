### P091 — Confidencedirection in the bottleneck
* **SIGNAL:** On a rudimentary NLA with a round-trip cosine of ~0.1-0.4 and low-dimensional bottlenecks, this probe will yield pure noise (AUC ≈ 0.50) because the representation lacks the geometric organization needed to structure self-recovery success.
* **FAILURE MODE:** The linear probe fails to find any direction because the activation space lacks coherent global geometry, leading to immediate overfitting on the tiny concept set.
* **FLOOR VERSION:** Replace the linear probe with a direct metric: calculate the raw cosine similarity or MSE between the input and reconstructed activation, using the reconstruction error itself as the confidence proxy.
* **MISSED FLAW:** The leave-one-concept-out CV assumes confidence is concept-independent, but in a weak NLA, the overall activation norm (energy) of the concept shortcut may confound the probe, predicting recovery success purely based on activation magnitude.### P092 — Abstention vs confabulation on underdetermined input
* **SIGNAL:** A degenerate AV that emits near-empty or repetitive text will yield a flat floor of 0% or a saturated 100% abstention rate due to template collapse, resulting in zero usable signal.
* **FAILURE MODE:** A retrieval-only or weak verbalizer lacks the conditional generation capacity to output abstention tokens in response to underdetermined cues, defaulting to static, repetitive completions.
* **FLOOR VERSION:** Measure the token-level entropy or vocabulary diversity of the AV output on underdetermined vs. determined inputs, treating higher-entropy output as a signal of underdetermination.
* **MISSED FLAW:** The AV might shortcut abstention based on surface lexical cues in the prompt prefix (e.g., matching the phrase "does not pin down") rather than representing semantic uncertainty in the bottleneck.### P093 — Confabulation index over information gaps
* **SIGNAL:** A rudimentary NLA will produce a flat slope of confabulation index vs. vagueness because the degenerate AV repeats the same restricted set of concepts regardless of input specificity.
* **FAILURE MODE:** The insertion detector fails at a round-trip cosine of ~0.1-0.4, as the low reconstruction fidelity causes almost all outputs to be classified as unrelated insertions, saturating the metric.
* **FLOOR VERSION:** Simplify the metric to a binary check: measure whether the Jaccard similarity of bags-of-words between the input and the AV output drops as input vagueness increases.
* **MISSED FLAW:** A weak or retrieval-only verbalizer may output generic, high-frequency concepts as a default completion strategy, making the confabulation index a measure of the AV's retrieval prior rather than input vagueness.### P094 — Ambiguity preservation vs forced disambiguation
* **SIGNAL:** The ambiguity preservation rate will flatline at 0% because the lossy bottleneck cannot hold multi-modal distributions, forcing a collapse to the verbalizer's prior or template.
* **FAILURE MODE:** A degenerate AV or a low-dimensional bottleneck is structurally incapable of representing or generating dual readings, causing the matcher ensemble to return false negatives due to grammar collapse.
* **FLOOR VERSION:** Evaluate whether the AR reconstruction of an ambiguous input can be decoded by a separate, strong probe into both target meanings, bypassing the degenerate AV entirely.
* **MISSED FLAW:** The plan assumes that choosing the higher-frequency sense indicates "forced disambiguation," but this collapse is indistinguishable from the AV's inherent lack of vocabulary depth.### P095 — AV-hedge calibration against AR faithfulness
* **SIGNAL:** On a rudimentary NLA, this plan will yield noise (Spearman ρ ≈ 0) because hedging tokens will be emitted randomly as degenerate output repetitions, entirely uncoupled from bottleneck fidelity.
* **FAILURE MODE:** A retrieval-only or weak verbalizer does not dynamically condition its output style (hedges) on the accuracy of the bottleneck reconstruction, breaking the correlation between surface styling and activation-space faithfulness.
* **FLOOR VERSION:** Measure the correlation between the reconstruction error (original vs. reconstructed activation cosine) and the output length or overall token perplexity, using perplexity as a proxy for surface uncertainty.
* **MISSED FLAW:** A weak AR might reconstruct a hedged phrase faithfully in activation space (high cosine similarity) simply because the hedge is a high-frequency token sequence, falsely indicating high calibration.### P096 — Uncertainty-direction stability across seeds
* **SIGNAL:** The pairwise cosine of confidence directions on a weak NLA will hit the shuffled-label floor (~0.0), showing no stable direction across seeds and confirming the absence of a structured confidence axis.
* **FAILURE MODE:** Because P091 fails to find any real signal in the rudimentary regime, P096 will measure only the random noise of the probe optimizer, making the stability metric uninformative.
* **FLOOR VERSION:** Measure the stability of the activation norms (L2 length) across different seeds or folds when processing inputs with different reconstruction success rates, rather than learning a full direction vector.
* **MISSED FLAW:** If the training set is very small, a linear probe can overfit to identical dataset splits across seeds, showing spurious "stability" (cosine > 0.5) that is actually an artifact of s
sample correlation rather than a true confidence direction.

### P097 — Refusal / safe-completion encoding in the bottleneck
* **SIGNAL:** The signal will saturate at chance (AUC ≈ 0.5) because a low-dimensional, weakly trained bottleneck fails to abstract the subtle, high-level meta-feature of "refusal-ness" away from topi
ical content.
* **FAILURE MODE:** The AV-render rate will be zero if the degenerate AV lacks the capability to construct compliant or refusal prose, while the low-fidelity AR will fail to reconstruct the refusal st
tance in activation space.
* **FLOOR VERSION:** Use a simple classification task on the activations of explicit compliance/refusal templates (without varying the topic) to test if the bottleneck can linearly separate direct tem
mplates.
* **MISSED FLAW:** A probe might easily achieve high AUC by latching onto stylistic artifacts of refusal responses (like sentence length or common prefix activations) rather than the abstract concept 
 of safety or refusal stance.

### P098 — Numeric / range uncertainty preservation
* **SIGNAL:** This plan will produce a floor signal (0% range-preservation) because the lossy bottleneck and degenerate AV will drop numeric qualifiers entirely or collapse all quantities to a few har
rdcoded numbers.
* **FAILURE MODE:** The regex and range-parser matcher ensemble fails when the AV outputs ungrammatical, fragmented, or repetitive text that does not form valid numeric ranges or point values.        
* **FLOOR VERSION:** Test only whether the NLA can preserve the order of magnitude (e.g., separating "tens" from "thousands") by comparing the cosine similarity of reconstructed activations of magnitu
ude-distinct inputs.
* **MISSED FLAW:** The model may collapse ranges to point estimates not because of a bottleneck limitation, but because the AR's training distribution contains overwhelmingly more point values than ra
ange expressions, bias-shifting the output.

### P099 — Known-unknown vs unknown-unknown distinction
* **SIGNAL:** On a weak NLA, this will yield floor results (no difference between condition b and c) as the low-dimensional bottleneck is unable to carry the meta-representation of a "redacted slot." 
* **FAILURE MODE:** The degenerate AV will confabulate or drop slots at the same rate across all conditions because it lacks the capacity to map the specific "redacted" activation pattern to an acknow
wledgement phrase.
* **FLOOR VERSION:** Measure the reconstruction error (reconstructed vs. original activation) for the redacted token itself; if the bottleneck registers the gap, the redacted region should have a dist
tinct error profile compared to omitted regions.
* **MISSED FLAW:** The explicit gap marker ("redacted") might introduce a strong local activation feature that the probe detects as a simple template shortcut, without the model actually understanding
g or representing the concept of a "known-unknown."

### P100 — Self-consistency across resamples as a confidence proxy
* **SIGNAL:** A degenerate, collapsed AV will yield a flat line (agreement pinned at 1.0) or noise (agreement ≈ 0.0), providing no correlation signal (ρ ≈ 0) with ground-truth faithfulness.
* **FAILURE MODE:** In the rudimentary regime, temperature sampling on a weak verbalizer either results in severe modal collapse (the model always outputs the same degenerate token) or high entropy no
oise unrelated to bottleneck quality.
* **FLOOR VERSION:** Measure the self-consistency of the reconstructed activations (AR output) over multiple stochastic passes of a noisy bottleneck, instead of relying on the AV text outputs.        
* **MISSED FLAW:** High self-consistency can occur on highly unfaithful reconstructions if the AR is strongly biased toward a small set of prior activations, making consistency a measure of AR bias ra
ather than recovery confidence.

---

### Skip on rudimentary NLAs
*   **P091 (Confidence direction):** Bottleneck lacks the metric structure or training depth to yield any linear separation of confidence above chance.
*   **P096 (Uncertainty-direction stability):** Without a valid underlying direction to measure, seed variance becomes a test of random optimizer state.
*   **P097 (Refusal / safe-completion):** High-level semantic stance tracking is washed out by lossy bottleneck projections and vocabulary collapse.
*   **P099 (Known vs unknown-unknown):** The meta-representation of explicit redaction collapses immediately into standard omission behaviors.

### Highest-value for the rudimentary regime
*   **P092 (Abstention vs confabulation):** Readily evaluable through output entropy shifts on paired inputs, bypassing complex activation probing.
*   **P093 (Confabulation index):** Captures how aggressively the lossy bottleneck/AV overrides vagueness with prior-driven tokens.
*   **P095 (AV-hedge calibration):** Directly tests the relationship between reconstruction loss and styling patterns without demanding complex multi-class probes.
*   **P100 (Self-consistency):** Provides a clean, activation-level look at decoder entropy, allowing a check for modal collapse vs. reconstruction failure.
