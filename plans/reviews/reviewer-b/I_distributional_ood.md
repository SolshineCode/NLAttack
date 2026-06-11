### P081 — In-distribution vs OOD encodable-subspace probe-AUC gap
* **SIGNAL**: Linear probe AUCs on the original activation space will collapse toward a noisy baseline floor (0.5 AUC) under a near-random AR or low-dimensional bottleneck, making the in-dist vs OOD gap (ΔAUC) statistically undetectable.
* **FAILURE MODE**: The logistic-regression probe will overfit to high-dimensional noise or keyword-label noise when the NLA bottleneck is degenerate, causing the ΔAUC estimator to fluctuate randomly.
* **FLOOR VERSION**: Replace the logistic-regression probe training with a simple non-parametric distance-to-centroid comparison between concept-active and concept-inactive activations.
* **MISSED FLAW**: The ground-truth "keyword presence in source text" doesn't account for polysemy or negation, which heavily confounds probe AUC in lower-dimensional bottlenecks where these semantic nuances are compressed away first.### P082 — Dropout-map domain transfer
* **SIGNAL**: Under a weak NLA, the per-concept retention maps will be dominated by noise, causing Spearman rank correlations ($\rho$) to cluster near zero for all domain pairs and masking any real transfer signal.
* **FAILURE MODE**: When concept retention is near-random, Spearman $\rho$ becomes highly sensitive to small, noisy fluctuations in AUC rankings, leading to spurious clustering.
* **FLOOR VERSION**: Instead of correlating continuous ranks, binarize the retention metric (retained vs completely dropped) and evaluate domain overlap using Jaccard similarity.
* **MISSED FLAW**: Concept co-occurrence patterns in the evaluation corpora behave as a strong confound; if domains A and B share co-occurring keywords, their maps will look correlated purely due to data statistics.### P083 — Frequency-tier × domain interaction
* **SIGNAL**: Under a weak NLA, mid- and low-frequency bands will hit a floor (0.5 AUC) across all domains, flattening the frequency-retention slope and causing the interaction term to collapse to zero.
* **FAILURE MODE**: In low-data training regimes or for tiny models, the estimation of corpus-internal frequency bands is highly noisy, which destabilizes the interaction regression model.
* **FLOOR VERSION**: Collapse the three frequency tiers into a binary division (highly frequent vs rare) and run a simple t-test of the AUC difference between these two groups.
* **MISSED FLAW**: Concept abstractness (concrete nouns vs abstract verbs) strongly correlates with frequency and represents a major confound for how weak bottlenecks encode concepts.### P084 — Checkpoint-maturity effect
* **SIGNAL**: For a degenerate verbalizer emitting near-empty or repetitive text, AV-side recovered retention (`av_acc`) will remain at floor (near zero) across early checkpoints, failing to show the monotonic improvement curve.
* **FAILURE MODE**: The matcher ensemble will fail to run or yield entirely null outputs due to parser/tokenization crashes on degenerate, repetitive, or empty output strings.
* **FLOOR VERSION**: Evaluate AV maturity using character-level or token-level n-gram overlap (like BLEU or edit distance) with the source text, rather than relying on semantic concept matchers.
* **MISSED FLAW**: A narrowing gap can be caused by the matcher relaxing its thresholds for repetitive outputs rather than genuine verbalizer improvement.### P085 — Cross-model NLA transfer
* **SIGNAL**: If either NLA is rudimentary (e.g., retrieval-only or low-fidelity), the cross-model Spearman $\rho$ will hit a floor near zero, hiding any meaningful alignment or architecture-specific differences.
* **FAILURE MODE**: Querying Neuronpedia APIs or local pipelines for degenerate/empty outputs returns null/empty metadata fields, causing the correlation calculation to fail due to missing values.
* **FLOOR VERSION**: Compare models on a small, hand-curated list of 10-20 highly robust, concrete concepts (e.g., "numbers", "colors") instead of a broad, noisy shared vocabulary.
* **MISSED FLAW**: The prompt template used to query Neuronpedia serves as a massive confound, as different base models (Gemma vs Llama) react differently to the same query templates regardless of NLA quality.### P086 — Concept-prevalence-in-training vs probe-AUC
* **SIGNAL**: In a very rudimentary NLA trained on very few examples, the training prevalence of most concepts is zero or near-zero, leading to a massive floor effect where correlation cannot be reliably computed.
* **FAILURE MODE**: The regression model will experience severe collinearity between training prevalence and evaluation prevalence, particularly when the training set is extremely small or highly skewed.
* **FLOOR VERSION**: Group concepts into a binary category ("seen in training" vs "unseen/zero frequency") and measure the mean AUC difference between these two groups.
* **MISSED FLAW**: A concept's intrinsic linear separability in the base model's unbottlenecked activations is a major confound that directly inflates AUC regardless of NLA training prevalence.### P087 — Train-distribution leakage into verbalizations
* **SIGNAL**: For a degenerate AV emitting repetitive or near-empty text, the boilerplate ('diabetes') insertion rate might saturate at 100% or drop to 0%, providing zero resolution on domain-specific variance.
* **FAILURE MODE**: The concept matcher will falsely flag repetitive boilerplate fragments (e.g., repeating the same word infinitely) as multiple distinct concept insertions, inflating the metric.
* **FLOOR VERSION**: Measure boilerplate leakage using raw token frequencies or n-gram counts directly on the raw text outputs, bypassing the semantic concept matcher.
* **MISSED FLAW**: Base model fine-tuning bias is a major confound; the base model itself may naturally output the boilerplate text under low-temperature or high-repetition scenarios, independent of N
NLA leakage.

### P088 — OOD hallucination/insertion rate vs in-distribution
* **SIGNAL**: With a degenerate verbalizer or near-random AR, the insertion rate will saturate at a high, noisy baseline on both in-dist and OOD corpora, drowning out any distribution-shift signal.   
* **FAILURE MODE**: On OOD inputs, a rudimentary NLA's reconstructions may be so low-fidelity that the AV outputs degenerate gibberish, which breaks the matcher's ability to count insertions.
* **FLOOR VERSION**: Measure the ratio of output length to input length, or the token entropy of the outputs, as a proxy for confabulation/degeneration under OOD shift.
* **MISSED FLAW**: Input text complexity (e.g., perplexity under a reference model) is a major confound that drives AV hallucination independently of whether the input is in-distribution or OOD.      

### P089 — Layer × distribution interaction
* **SIGNAL**: For a tiny base model or weak NLA, representation quality collapses rapidly across layers, meaning linear probes will find no signal (floor AUC $\approx 0.5$) across all layers and distr
ributions.
* **FAILURE MODE**: The interaction model (layer $\times$ corpus_type) will lack statistical power and fail to converge because of the high variance and low magnitude of the AUC estimates.
* **FLOOR VERSION**: Evaluate the raw activation variance or cosine similarity between consecutive layers instead of training supervised linear probes.
* **MISSED FLAW**: Base model layer alignment is a confound; early layers of smaller models contain highly non-linear representations that linear probes fail to extract, mischaracterizing early-layer 
 concept survival.

### P090 — Encodable-subspace dimensionality by domain
* **SIGNAL**: If the AR is near-random or the bottleneck is extremely low-dimensional, the singular-value spectrum of the probe weights will collapse, making the effective rank estimate hit the floor 
 (flat spectrum/noise).
* **FAILURE MODE**: When probe AUCs are low, the resulting probe direction vectors represent random noise directions, causing the computed effective rank to look artificially high (approaching maximum
m rank).
* **FLOOR VERSION**: Instead of calculating the SVD of probe directions, calculate the mean pairwise cosine similarity of the raw activation vectors for a set of key concepts.
* **MISSED FLAW**: The number of samples (contexts) used to train each probe acts as a major confound, as low-data probes produce noisier weight vectors that artificially inflate the participation rat
tio.

---

### Skip on rudimentary NLAs
* **P082** (Dropout-map domain transfer): Noise in rank correlation masks actual domain differences under poor concept retention.
* **P085** (Cross-model NLA transfer): API-dependent queries and tokenization mismatches break when processing degenerate outputs.
* **P089** (Layer $\times$ distribution interaction): High-variance AUC estimates from tiny base models lead to non-converging regression models.
* **P090** (Encodable-subspace dimensionality): Noisy, weak probe directions artificially inflate the participation ratio and distort rank metrics.

### Highest-value for the rudimentary regime
* **P084** (Checkpoint-maturity effect): Directly tracks how a degenerate verbalizer evolves and separates interface loss from representation loss.
* **P087** (Train-distribution leakage): The most robust test for identifying training priors and repetitive boilerplate in degenerate text.
* **P081** (In-distribution vs OOD gap): Serves as the basic check of whether any bottleneck exists, easily simplified to centroid distances.
* **P086** (Concept-prevalence-in-training): Identifies whether failure modes are due to representation limits or simply lack of dataset exposure.
