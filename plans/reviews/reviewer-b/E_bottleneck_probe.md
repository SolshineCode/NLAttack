### P041 — Probe-vs-AV gap per concept (the verbalizer+matcher loss)
* **SIGNAL:** Usable signal is highly degraded, as a degenerate AV will cause the AV matcher accuracy to hitthe floor, saturating the gap measurement near 1.0.
* **FAILURE MODE:** The degenerate text output from an early AV checkpoint triggers matcher failures that mask whether the representation was actually lost in the bottleneck.
* **FLOOR VERSION:** Simplify the AV matcher to look for simple token presence or substrings rather than semantic matching to bypass generator garbage.
* **MISSED FLAW:** Overlooks the confound that a large gap might be driven entirely by grammatical/generation failures rather than semantic retrieval loss.### P042 — Linear vs MLP probe: is concept encoding nonlinear at the bottleneck?
* **SIGNAL:** On a low-dimensional or tiny NLA, this will yield mostly noise due to severe overfitting in the MLP probe.
* **FAILURE MODE:** The MLP classifier will memorize small-sample label noise, artificially inflating MLP AUC and suggesting fake nonlinear encoding.
* **FLOOR VERSION:** Scale down the MLP to a tiny hidden size (e.g., 8 units) with aggressive dropout and weight decay.
* **MISSED FLAW:** An AUC delta can be caused by sub-optimal hyperparameter tuning of the linear probe rather than true nonlinearity in the bottleneck.### P043 — Layer sweep: which layer best encodes a given concept (needs re-extraction)
* **SIGNAL:** In rudimentary models, concepts are not stably formed, causing the layer-sweep AUC to be flat and noisy across all layers.
* **FAILURE MODE:** The extraction pipeline may fail to retrieve meaningful activations if the weak model's architecture differs or lacks deep layer separation.
* **FLOOR VERSION:** Restrict the sweep to three key points (input, middle, output) using simple, high-frequency lexical concepts.
* **MISSED FLAW:** The comparison is confounded by variance in activation norms across layers if standard scaling is not strictly applied per layer.### P044 — Probe-direction stability across seeds and subsamples
* **SIGNAL:** Provides a highly sensitive signal for detecting representational collapse, as weak NLAs will show extremely low stability.
* **FAILURE MODE:** If the concept AUC is near 0.5, the probe weights become random vectors, causing similarity metrics to collapse to zero.
* **FLOOR VERSION:** Compute cosine stability on a reduced-rank PCA projection of the weights rather than the full high-dimensional vector.
* **MISSED FLAW:** High stability can be an artifact of probe regularization or label imbalance rather than genuine concept alignment.### P045 — Concept geometry: angles between probe directions (superposition)
* **SIGNAL:** Useful for detecting forced superposition in low-dimensional bottlenecks, though noise can obscure semantic relationships.
* **FAILURE MODE:** Low-quality probe fits in weak models generate noisy weight vectors that compress all cosines toward the random null.
* **FLOOR VERSION:** Focus only on highly distinct antonym/contrast pairs to verify if they map to opposing axes.
* **MISSED FLAW:** High cosine similarity can be driven entirely by correlation in the underlying training text keyword labels rather than bottleneck superposition.### P046 — Verbalizer ablation: trained AV vs template/forced-decode AV
* **SIGNAL:** Excellent signal for isolating verbalizer deficits, showing exactly how much performance is lost to the generation process.
* **FAILURE MODE:** In a near-random AR (cosine 0.1–0.4), both the trained and forced AVs will fail, yielding a flat zero difference.
* **FLOOR VERSION:** Bypass text generation entirely by evaluating the direct dot-product of bottleneck activations with concept token embeddings.
* **MISSED FLAW:** The forced-decoder template itself may introduce structural bias that degrades performance on weak representations.### P047 — Probe accuracy gated by AR faithfulness (round-trip cosine threshold)
* **SIGNAL:** Hits a complete floor on rudimentary NLAs, as a near-random AR (cosine 0.1–0.4) results in almost all examples falling below the threshold.
* **FAILURE MODE:** Stratification fails due to insufficient sample sizes in high-fidelity bins.
* **FLOOR VERSION:** Use relative percentile-based stratification (top vs. bottom quintiles) instead of absolute cosine thresholds.
* **MISSED FLAW:** A low round-trip cosine indicates global reconstruction failure but does not necessarily imply the loss of the specific concept dimensions.### P048 — Multi-concept joint / structured probe
* **SIGNAL:** Primarily noise, as weak representations lack the capacity to encode joint multi-label boundaries.
* **FAILURE MODE:** The multi-output classifier overfits the scarce joint positive examples in low-dimensional spaces, failing to converge.
* **FLOOR VERSION:** Test joint presence using simple logical combinations of independent single-concept linear probes.
* **MISSED FLAW:** High joint AUC is often a reflection of label co-occurrence in the source corpus rather than structured representations in the bottleneck.### P049 — Probe transfer across corpora (train on corpus 1, test on corpus 2)
* **SIGNAL:** Shows floor performance across the board, as weak bottlenecks produce highly corpus-specific, non-generalizable directions.
* **FAILURE MODE:** Transfer AUCs collapse to 0.5 due to representation overfitting on the source corpus.
* **FLOOR VERSION:** Apply a linear alignment technique (like CCA) to the activations before testing transfer.
* **MISSED FLAW:** Variations in baseline label noise between corpora can be mistaken for representational transfer failure.

### P050 — Better concept labels: replace keyword labels with stage-2 LLM labels
* **SIGNAL:** Strong, positive signal because cleaner labels prevent the weak probe's signal from being completely drowned out by noise.
* **FAILURE MODE:** If the bottleneck is extremely degraded, it cannot capture the semantic nuances of LLM labels, leading to lower AUCs than crude keywords.
* **FLOOR VERSION:** Simplify the LLM labels to coarse, high-level topics rather than fine-grained semantic concepts.
* **MISSED FLAW:** LLM labels might rely on context that a tiny model's single-layer activations simply do not encode, setting an unachievable baseline.

---

### Skip on rudimentary NLAs
* **P042 (Linear vs MLP probe):** High risk of false-positive nonlinearity signal due to MLP overfitting on noisy, low-dimensional representations.
* **P047 (Probe accuracy gated by AR faithfulness):** Near-random ARs make absolute cosine thresholding partition all data into the noise floor.
* **P048 (Multi-concept joint probe):** Representation capacity is too low to model joint structures; results will be dominated by label co-occurrence artifacts.
* **P049 (Probe transfer across corpora):** Weak representations overfit local corpus statistics so heavily that cross-corpus transfer will hit a flat 0.5 AUC ceiling.

### Highest-value for the rudimentary regime
* **P044 (Probe-direction stability):** Directly measures whether the bottleneck is learning stable semantic directions or merely fitting noise, providing a clean indicator of representation quality. 
* **P046 (Verbalizer ablation):** Crucial for verifying if a degenerate AV's failures are due to token-generation issues or if the bottleneck itself is empty.
* **P050 (Better concept labels):** Maximizes the signal-to-noise ratio for weak probes by replacing noisy keyword heuristics with clean, semantic targets.
