### P031 — Full 14-tactic risk-weighted coverage heatmap
* **SIGNAL:** On a rudimentary NLA with degenerate or near-empty text, retention rates will collapse uniformly to near-zero, producing a flat, noise-dominated heatmap where risk-weighting is a mathematical artifact of the weights rather than true variation.
* **FAILURE MODE:** The lexical matchers will fail to match concepts in repetitive or degenerate text, driving the retention score to absolute zero across all 14 tactics and washing out the partial-correlation analysis.
* **FLOOR VERSION:** Simplify the metric to a binary detection test of whether *any* concept from *any* tactic survives in the bottleneck text (pooling all tactics), or measure raw vocabulary overlap instead of concept-level matching.
* **MISSED FLAW:** Frequency-length matching controls do not account for the fact that rudimentary AVs default to high-frequency background words, creating a confound where tactics with overlapping vocabularies to the model's degenerate output appear falsely "preserved."### P032 — Obfuscation see-through ratio across an obfuscation-family ladder
* **SIGNAL:** For a weak NLA, even L0 (plain) retention is close to the floor, meaning the see-through ratio (near-0 / near-0) will suffer from extreme division-by-zero noise and saturate at 0 immediately at L1.
* **FAILURE MODE:** Degenerate verbalizers cannot handle even simple spacing or leet perturbations, causing the "evasion threshold" to collapse immediately to L1 and rendering the slope and monotonicity calculations meaningless.
* **FLOOR VERSION:** Test only L0 vs L1 (spacing) or L2 (leet) using token-level edit distance of the bottleneck text rather than semantic concept matching.
* **MISSED FLAW:** A near-random AR might occasionally reconstruct random words that happen to match L3/L4/L5 by pure chance, yielding a false "see-through" signal that is actually just reconstruction noise.### P033 — Kill-chain multi-stage joint co-occurrence survival
* **SIGNAL:** On a rudimentary NLA, joint survival of $\ge 3$ stages will immediately hit a hard floor of 0%, turning the joint probability comparison into a flat zero line across all chain lengths.
* **FAILURE MODE:** Degenerate or near-empty bottleneck texts lack the capacity (due to low-dimensional bottlenecks or repetitive outputs) to contain multiple distinct concepts, causing the order preservation and independence model ratios to break down.
* **FLOOR VERSION:** Relax the joint co-occurrence requirement to a bag-of-words presence of any *one* stage's keyword, ignoring ordering and joint probability entirely.
* **MISSED FLAW:** The plan assumes that drop events are independent, but a rudimentary NLA's failure on the first stage due to length limits will deterministically cause all subsequent stages to drop (strict context-window exhaustion).### P034 — Dual-use disambiguation by a single contextual cue
* **SIGNAL:** On a rudimentary NLA, both the cue and the technique term will fail to reconstruct, resulting in 0% distinguishability and random downstream classifier performance (AUC ~0.5).
* **FAILURE MODE:** The intent classifier will train on noise or degenerate repeated text, leading to severe overfitting or random guessing because the bottleneck lacks the capacity to retain abstract qualifiers.
* **FLOOR VERSION:** Simplify the test to check if the NLA can distinguish between short, isolated single-word opposites (e.g., "safe" vs "attack") rather than complex contextual sentences.
* **MISSED FLAW:** The plan assumes downstream classifiers fail due to cue loss, but they may fail simply because the overall bottleneck text is too short or malformed for the classifier to parse.### P035 — Agentic-scaffolding / autonomous-orchestration marker survival
* **SIGNAL:** On rudimentary NLAs, abstract procedural markers (like "autonomously") will immediately drop to 0% retention, while concrete actions might show a weak, noisy signal, making the difference highly unstable.
* **FAILURE MODE:** A retrieval-only or degenerate AV will lack the semantic representation to verbalize abstract agentic concepts, leading to a complete floor effect for the entire agentic marker class.
* **FLOOR VERSION:** Swap abstract agentic markers for simple, concrete transition verbs (e.g., "then", "next", "repeat") to see if sequential execution logic survives at a basic lexical level.
* **MISSED FLAW:** High-frequency agentic markers like "loop" or "run" might show false positive survival if the degenerate AV naturally repeats common system prompts or tokens.### P036 — Severity monotonicity within a single technique family
* **SIGNAL:** On a weak NLA, the severity modifiers (e.g., "bulk", "ransom") will floor uniformly, meaning the Spearman correlation will collapse to near-zero or be undefined due to zero variance.
* **FAILURE MODE:** The collapse of the severity modifiers means the bottleneck output will be identical across all severity tiers, resulting in flat, indistinguishable reconstructions.
* **FLOOR VERSION:** Test binary extremes (e.g., lowest vs highest severity only) using direct activation cosine similarity instead of full rank correlation on text.
* **MISSED FLAW:** The NLA might exhibit spurious monotonicity simply because higher-severity inputs happen to contain more emotional or high-impact words that a weak AV is biased to output.### P037 — Benign-neighbor laundering of attack terms (probe-grounded)
* **SIGNAL:** A near-random AR (cosine 0.1-0.4) will produce activations with extremely poor signal-to-noise ratio, causing the linear probes to fail to classify either the attack or benign concept, r
rendering the probe-grounded breakdown useless.
* **FAILURE MODE:** The ensemble matcher will report 0% laundering simply because neither the attack nor the benign term is reconstructed in the degenerate bottleneck text.
* **FLOOR VERSION:** Focus the analysis purely on the activation space by measuring raw cosine similarity of the bottleneck activations to predefined benign/malicious prototype vectors.
* **MISSED FLAW:** Weak linear probes might report high "laundering" rates as a training artifact because the low-dimensional bottleneck conflates any security-related concepts into a single cluster. 

### P038 — ARiES axis separability via probes on activations
* **SIGNAL:** For a very early-checkpoint NLA, the probe performance (AUC) will hover near chance (0.5) for both linear and MLP models, leading to a flat signal that fails to distinguish any axes.    
* **FAILURE MODE:** The high-dimensional activation space of a poorly trained model will contain mostly noise, causing the MLP probe to overfit drastically compared to the linear probe, yielding a fal
lse non-linearity signal.
* **FLOOR VERSION:** Reduce the 3-way axis classification to a simple binary classification of "any threat concept" vs "benign control" on activations.
* **MISSED FLAW:** The plan assumes a high probe-AV gap indicates verbalizer compression, but on a rudimentary NLA, this gap is likely just a result of the probe exploiting superficial features of a p
poorly converged activation space.

### P039 — Cross-technique transfer of blind spots
* **SIGNAL:** If the NLA is rudimentary, the "blind spots" will encompass almost the entire technique space (retention near 0 everywhere), leading to high uniform correlation that yields a false posit
tive transfer signal.
* **FAILURE MODE:** The Mantel test will fail or output spurious significance because the retention-profile matrix will have near-zero variance, making the correlation calculation highly sensitive to 
 numerical noise.
* **FLOOR VERSION:** Group techniques into broad, coarse-grained categories (e.g., network vs host) and measure the overall average retention gap between these macro-groups.
* **MISSED FLAW:** The similarity matrix (based on name embeddings) may correlate with word length or token frequency, meaning the observed transfer is driven by lexical properties rather than semanti
ic similarity.

### P040 — Monitor-as-filter realism: downstream classifier on the bottleneck
* **SIGNAL:** A rudimentary NLA will produce a massive AUC gap between view A (raw) and view B (bottleneck), with view B performing near random (AUC ~0.5), while view C (activation probe) will also de
egrade heavily.
* **FAILURE MODE:** The downstream classifier on view B will fail to train or collapse to the majority class because the input feature matrix consists of empty or highly repetitive texts.
* **FLOOR VERSION:** Train the classifier on simple word count or vocabulary richness of the bottleneck text rather than tf-idf/embeddings of specific terms.
* **MISSED FLAW:** A classifier on degenerate text (view B) might show artificially high detection rates by learning to associate specific repetitive AV failure modes (e.g., repeating the word "the" o
or "error") with specific input classes.

---

### Skip on rudimentary NLAs
* **P033 (Kill-chain multi-stage joint co-occurrence survival):** Multi-stage joint survival will be 0% everywhere, resulting in flat, uninformative zeros across all chain lengths.
* **P036 (Severity monotonicity within a single technique family):** Monotonicity of modifiers is impossible to evaluate when modifiers do not survive at all, causing correlation scores to be undefine
ed or purely noise.
* **P038 (ARiES axis separability via probes on activations):** 3-way multi-class probes will completely fail to converge or find signal on the noisy activations of early checkpoints.
* **P039 (Cross-technique transfer of blind spots):** A Mantel test on a flat matrix of 0-retention will produce spurious correlations driven entirely by numerical noise.

### Highest-value for the rudimentary regime
* **P032 (Obfuscation see-through ratio):** When simplified to edit distance at L1/L2, it directly quantifies the degradation of the most basic lexical reconstructions.
* **P040 (Monitor-as-filter realism):** Demonstrates the absolute lower bound of downstream classification performance on degenerate text, establishing a solid baseline tax for bottleneck loss.       
* **P031 (Full 14-tactic risk-weighted coverage heatmap):** If simplified to macro-tactic or broad detection, it quickly maps the primary capability limits of the model.
