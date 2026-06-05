### P051 — Contested-rate as a confound thermometer
* **SIGNAL:** On a rudimentary NLA emitting degenerate or near-empty text, the contested rate will either saturate at 1.0 due to constant random disagreement, or floor at 0.0 if all matchers fail to match anything, yielding zero usable signal.
* **FAILURE MODE:** The variance of the test statistics across single backends collapses to zero because the degenerate AV text prevents any of the matchers from registering positive hits.
* **FLOOR VERSION:** Simplify the ensemble to a binary comparison between raw character overlap and absolute token silence rather than relying on complex semantic/lexical backends.
* **MISSED FLAW:** P051 assumes matcher disagreement is driven by boundary-case concept ambiguity, whereas on a weak NLA, disagreement is driven entirely by the random, repetitive vocabulary of the early-checkpoint generator.---### P052 — Embedding-model sensitivity of effects (swap 3 embedding models)
* **SIGNAL:** With a near-random AR (cosine 0.1–0.4), the Jaccard similarity of "substituted" rows will artificially saturate near 0 or 1 depending on whether the embeddings default to grouping garbage tokens together, masking any real NLA signal.
* **FAILURE MODE:** Swapping embedding models on gibberish or near-empty AV outputs merely measures the baseline semantic similarity of noise tokens rather than tracking actual concept substitution or laundering.
* **FLOOR VERSION:** Filter the AV output to only include high-frequency non-stopwords before running the embedding models to see if any non-random vocabulary signals persist.
* **MISSED FLAW:** P052 overlooks that different embedding models have wildly divergent tolerances for out-of-vocabulary or degenerate token sequences, which will dominate the Jaccard distance regardless of NLA behavior.---### P053 — AV decoding-temperature sensitivity (same activation, vary temperature)
* **SIGNAL:** A highly rudimentary AV will output degenerate text regardless of temperature, causing the flip-rate to saturate at noise-level variance (high-entropy gibberish) or floor at zero (constant repetitive outputs).
* **FAILURE MODE:** Temperature adjustments on a model that has not learned basic syntax will produce different permutations of nonsense, leading to high flip-rates that do not reflect true concept stochasticity.
* **FLOOR VERSION:** Measure the flip-rate of token-level character n-grams rather than concept-level matched terms to capture the raw entropy of the distribution.
* **MISSED FLAW:** P053 assumes the decoder operates in a regime where temperature scales semantic diversity, whereas for weak verbalizers, it merely gates chaotic token switching without changing underlying semantic representation.---### P054 — AV prompt-template sensitivity (reword the AV instruction)
* **SIGNAL:** For a tiny or retrieval-only verbalizer, prompt sensitivity will floor completely because the model lacks the capacity to follow instructions or contextualize prompts.
* **FAILURE MODE:** The model will ignore the security vs. neutral phrasing and output the same rigid retrieval results or empty templates, showing artificially flat invariance.
* **FLOOR VERSION:** Test raw prefix-matching sensitivity by prepending hard-coded tokens directly to the bottleneck activation before decoding.
* **MISSED FLAW:** P054 overlooks that early-checkpoint AVs are often "instruction-blind," meaning apparent robustness is actually a failure to represent the prompt instruction at all.---### P055 — Paraphrase invariance of verbalization
* **SIGNAL:** If the AR is near-random, different paraphrase activations will land in random regions of the bottleneck space, causing verbalization agreement to hit a noise floor.
* **FAILURE MODE:** The bottleneck's low dimensionality and lack of training mean it cannot map paraphrases to similar activations, making the joint NLA+AV agreement a measure of random layout rather than phrasing invariance.
* **FLOOR VERSION:** Use highly distinct, non-overlapping vocabulary pairs (e.g., "login" vs "sign-in") instead of full sentence paraphrases to test basic alignment.
* **MISSED FLAW:** P055 conflates the AV's inability to decode with the NLA's failure to map paraphrases to the same region; a bad AR ruins the round-trip independent of the AV's capability.---### P056 — Human-vs-auto matcher κ (inter-annotator reliability)
* **SIGNAL:** On a rudimentary NLA emitting repetitive or near-empty text, human annotators will immediately rate everything as "dropped" (yielding κ ~ 1.0 with a simple negative matcher), creating a 
 false sense of matcher validity.
* **FAILURE MODE:** The lack of meaningful semantic variety in degenerate text makes the annotation task trivial, masking the matcher's inability to handle nuanced concept boundaries.
* **FLOOR VERSION:** Have humans rate the coherence of the AV text first, and only compute κ on the subset of outputs containing at least one recognizable English noun.
* **MISSED FLAW:** P056 fails to account for "sparsity bias": when the base rate of positive matches is close to zero, Cohen's κ becomes highly sensitive to minor stochastic agreements.

---

### P057 — Matcher threshold sensitivity sweep
* **SIGNAL:** With a rudimentary NLA producing degenerate text, sweeping the threshold will show a steep binary drop-off from 100% dropped to 100% matched, offering no smooth sensitivity curve or stab
ble ranking.
* **FAILURE MODE:** The representation space of a weak NLA is not well-clustered, meaning embedding distances to target concepts will form a narrow, noisy distribution that collapses under threshold c
changes.
* **FLOOR VERSION:** Sweep thresholds on a normalized edit-distance or Jaccard token matcher instead of sentence embeddings to find the vocabulary floor.
* **MISSED FLAW:** P057 assumes there is a meaningful "decision boundary" to locate, but in a rudimentary regime, the cosine similarity space is isotropic noise.

---

### P058 — Lexical-vs-semantic matcher divergence map
* **SIGNAL:** A retrieval-only or degenerate verbalizer will produce repetitive keywords that favor the lexical matcher, while the semantic matcher will flag random connections, leading to extreme, no
oise-driven divergence.
* **FAILURE MODE:** Lexical matches will only succeed on exact degenerate token matches, while embedding matches will return garbage, meaning the off-diagonal cells reflect vocabulary overlap rather t
than semantic divergence.
* **FLOOR VERSION:** Restrict the lexical matcher to character substrings of length >3 and compare against a bag-of-words baseline.
* **MISSED FLAW:** P058 assumes semantic-only matches indicate "laundering" concepts, but for a weak NLA, they are typically artifacts of the embedding model trying to project garbage tokens into a de
ense space.

---

### P059 — Position-sampling sensitivity (which token positions are verbalized)
* **SIGNAL:** In a low-dimensional bottleneck or a model trained on very few examples, position information is likely lost or smeared, leading to flat, noisy retention across all sampling policies.   
* **FAILURE MODE:** The model cannot routing-link concept tokens to specific activation positions, making targeted sampling behave identically to random content sampling.
* **FLOOR VERSION:** Compare only the first token activation versus a mean-pooled representation of the entire short sequence.
* **MISSED FLAW:** P059 assumes the NLA has learned localized token representation, which is untrue for very early checkpoints where activations are dominated by position embeddings or global sequence
e bias.

---

### P060 — Verbalization-length vs recall (control for AV verbosity)
* **SIGNAL:** A weak AV that outputs repetitive gibberish will show a strong linear relationship between output length and false-positive matches, saturating the recall curve very quickly.
* **FAILURE MODE:** The negative control will match almost everything as length increases because the degenerate text generation lacks semantic constraints and eventually spits out high-frequency keyw
words by chance.
* **FLOOR VERSION:** Limit `max_new_tokens` to a very small range (e.g., 5 to 20 tokens) to prevent the degenerate generator from cycling through its entire vocabulary.
* **MISSED FLAW:** P060 overlooks that a weak AV's length is often a function of its inability to output an end-of-sequence (EOS) token, meaning length variations measure generation failure rather tha
an concept coverage.

---

### Skip on rudimentary NLAs
* **P054** (AV prompt-template sensitivity): Tiny/early models are prompt-blind; results will be flat noise.
* **P055** (Paraphrase invariance of verbalization): A near-random AR will scatter activations randomly, rendering paraphrase alignment impossible to measure.
* **P059** (Position-sampling sensitivity): Position embeddings and routing are unlearned in early checkpoints, making location comparison meaningless.

### Highest-value for the rudimentary regime
* **P060** (Verbalization-length vs recall): Crucial for quantifying how much "retention" in the weak model is just a symptom of long, repetitive vocabulary cycling.
* **P057** (Matcher threshold sensitivity sweep): Exposes whether the weak NLA's activation space has any clustering structure or is purely isotropic noise.
* **P051** (Contested-rate as a confound thermometer): Establishes the baseline noise floor of the ensemble matcher before any fine-tuning begins.
