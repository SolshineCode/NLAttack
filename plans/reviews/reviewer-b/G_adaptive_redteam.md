### P061 — Query-budget evasion success curve
* **SIGNAL**: The success curve will likely floor at 0% or saturate immediately at 100% because a degenerate AV will either fail to verbalize anything coherent or completely fail to trigger the matcher, leaving no gradient for the search.
* **FAILURE MODE**: The optimization search space collapses because minor mutations on degenerate/repetitive text do not produce smooth changes in the monitor's binary output.
* **FLOOR VERSION**: Substitute the active optimization loop with a simple random-mutation baseline and lower the `intent_floor` substantially to match the low baseline reconstruction cosine of the weak AR.
* **MISSED FLAW**: The plan conflates the AV's vocabulary collapse (failing to verbalize anything) with a successful adversarial evasion.### P062 — Compositional blind spots (benign parts, malicious whole)
* **SIGNAL**: A rudimentary NLA will likely produce a flat floor of zero signal because a low-dimensional bottleneck cannot encode complex, multi-concept semantic compositions.
* **FAILURE MODE**: Complex triples will collapse into degenerate, repetitive loops or generic high-frequency tokens, preventing the individual component concepts from surviving.
* **FLOOR VERSION**: Simplify the triples to high-frequency, two-token combinations (e.g., "bad" + "file") and evaluate survival using activation-space cosine similarity rather than string matchers on the AV text.
* **MISSED FLAW**: False positives on the bottleneck probe can falsely indicate that the composite concept is present in the activation when it is actually just out-of-distribution noise.### P063 — Evasion transfer across NLAs / layers
* **SIGNAL**: Transfer rates will saturate at noise/chance levels because the representation spaces of weak NLAs are highly unstable and lack aligned semantic directions.
* **FAILURE MODE**: Adversarial inputs will overfit heavily to the specific tokenizer artifacts and degenerate outputs of NLA-A, failing completely on NLA-B.
* **FLOOR VERSION**: Measure transfer across layers of the *same* weak model using direct linear probes on activations rather than evaluating across different models with discrete text outputs.
* **MISSED FLAW**: The baseline of "chance transfer" is highly sensitive to the frequency of vocabulary tokens in the degenerate AV, which may inflate transfer rates.### P064 — Monitor-gaming feedback loop (optimize against the AV output)
* **SIGNAL**: The feedback loop will yield pure noise, as a degenerate AV producing repetitive or empty text offers no useful gradient signal for steering mutations.
* **FAILURE MODE**: The optimization policy will stagnate in infinite loops or trigger early termination rules due to vocabulary collapse.
* **FLOOR VERSION**: Optimize against the continuous probability distribution (logits) of the top-k AV tokens rather than hard string matchers or binary flags.
* **MISSED FLAW**: Optimizing directly against the AV on a small NLA will push the activations into out-of-distribution regions where the bottleneck probe behaves randomly, yielding false "retained intent" flags.### P065 — Steganographic channel capacity through the bottleneck
* **SIGNAL**: Usable signal will floor at zero capacity (BER ≈ 0.5) because a low-dimensional bottleneck or random AR cannot carry high-entropy steganographic payloads.
* **FAILURE MODE**: The encoder and decoder cannot be trained to converge due to the high noise and lack of semantic structure in the weak NLA's activation space.
* **FLOOR VERSION**: Reduce the payload to a single bit ($k=1$) and measure whether the presence of a specific high-frequency trigger word in the input is reconstructible.
* **MISSED FLAW**: Overlooks direct token leakage (e.g. bypass residual paths in the weak model) which mimics a covert channel but is actually just reconstruction failure.### P066 — Adaptive paraphrase across the retention boundary at constant probe-intent
* **SIGNAL**: The constant-intent region will be non-existent because minor paraphrase mutations will cause erratic, non-smooth jumps in both the noisy probe and the degenerate AV.
* **FAILURE MODE**: The search constraint $|\Delta \text{intent}| < \delta$ will filter out all generated candidates, leading to search failure.
* **FLOOR VERSION**: Relax the delta boundary ($\delta$) significantly and use a pre-computed dictionary of single-word synonyms instead of a live paraphrase generator.
* **MISSED FLAW**: A weak NLA's activation space lacks local smoothness, meaning two points with similar probe scores can represent entirely different semantics.### P067 — Adversarial concept ordering / dilution
* **SIGNAL**: High usable signal will exist, as weak AVs are highly sensitive to recency/position effects and vocabulary dilution, showing strong performance drop-offs.
* **FAILURE MODE**: A degenerate or retrieval-only AV will completely ignore dilution tokens beyond a very short context window, yielding flat or binary outputs.
* **FLOOR VERSION**: Restrict the context to 1-2 filler words and measure simple token-level reconstruction rates instead of full semantic concepts.
* **MISSED FLAW**: Dilution can push the weak model's activations into OOD regions, causing a drop in flagging due to model failure rather than selective evasion.### P068 — Prompt-injection that survives the bottleneck to reach a downstream reader
* **SIGNAL**: The signal will floor at zero because weak NLAs lack the capacity to reconstruct the complex instruction structures required for prompt injection.
* **FAILURE MODE**: The downstream parser will receive empty, repetitive, or nonsensical output from the degenerate AV, preventing any execution.
* **FLOOR VERSION**: Simplify the injection payload to a single system command (e.g., "delete") and use a simple keyword match on the AV output as the success metric.
* **MISSED FLAW**: Random generation from a degenerate AV could by chance output words matching the parser's criteria, leading to false positives.

### P069 — Encoder/decoder collusion under an AV-reading overseer
* **SIGNAL**: The collusion advantage will floor at zero because the random/weak AR cannot reliably transmit the encoded attack concept.
* **FAILURE MODE**: The joint encoder/decoder training will fail to converge on the highly noisy and discontinuous gradient surface of a weak NLA.
* **FLOOR VERSION**: Use a static, discrete codebook mapping single tokens directly to specific activation dimensions, bypassing the AV text generation entirely.
* **MISSED FLAW**: The decoder might overfit to the idiosyncratic, deterministic noise of a weak AR, giving a false impression of a generalizable covert channel.

### P070 — Robustness to UNSEEN obfuscation families (train on family 1, test on family 2)
* **SIGNAL**: The generalization gap will be extremely high (saturating at the floor), showing that weak NLAs are highly brittle and overfit to the training distribution.
* **FAILURE MODE**: The evaluation will fail to find successful evasions in the training phase, rendering the cross-family transfer step impossible to perform.
* **FLOOR VERSION**: Use highly similar perturbation families (e.g., character insertions vs. character deletions) rather than semantically distinct ones.
* **MISSED FLAW**: The generalization gap might be driven entirely by the brittleness of the bottleneck probe rather than the NLA's representation capabilities.

---

### Skip on rudimentary NLAs
* **P065 (Steganographic channel capacity)**: The capacity is too low and the bottleneck too noisy to train or evaluate multi-bit stego channels.
* **P068 (Prompt-injection survival)**: Rudimentary bottlenecks cannot reconstruct instruction structures, leading to flat zero survival rates.
* **P069 (Encoder/decoder collusion)**: Joint encoder-decoder optimization will not converge on noisy, degenerate representations.

### Highest-value for the rudimentary regime
* **P067 (Concept ordering / dilution)**: Readily surfaces basic sequence-length and position biases inherent in low-capacity or retrieval-only verbalizers.
* **P061 (Query-budget curves)**: With relaxed thresholds, this provides a clean baseline of how easily a weak monitor's discrete interface is broken by random mutations.
* **P070 (Robustness to unseen obfuscation)**: Directly tests the brittle generalization limits of small-model representations under simple perturbations.
