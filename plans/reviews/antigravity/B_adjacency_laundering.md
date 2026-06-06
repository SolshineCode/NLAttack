### P011 — Minimal-pair separation
* **SIGNAL**: Noise/Floor. A degenerate or early-checkpoint AV will fail to generate negation/digit tokens for both inputs, leading to a saturated `pair_collapse_rate` of 1.0 for both target and control pairs, which drowns out the gap signal in noise.
* **FAILURE MODE**: Since the test relies on the generation of subtle distinguishing sub-token features, a weak NLA will output degenerate/unrelated text, preventing the Ensemble Matcher from identifying any distinguishing concept.
* **FLOOR VERSION**: Measure the linear separability of the distinguishing feature (e.g., negation) using a logistic regression probe directly on the bottleneck activations, bypassing the broken AV generation step.
* **MISSED FLAW**: A retrieval-only or highly biased AV will collapse outputs to a single template term, creating a high collapse rate that reflects output vocabulary restrictions rather than bottleneck information loss.### P012 — Hypernym drift
* **SIGNAL**: Floor. A weak NLA with a degenerate or retrieval-only verbalizer will systematically output high-frequency general terms like "thing" or "weapon" for all inputs, saturating the hypernym-drift ratio.
* **FAILURE MODE**: At low capabilities, the WordNet backend or taxonomy chain mapping will fail as the NLA outputs degenerate text that cannot be mapped to any valid WordNet synset.
* **FLOOR VERSION**: Restrict the target output space to a small, hand-curated set of 3–5 broad categories and measure category classification accuracy rather than hierarchical taxonomic drift.
* **MISSED FLAW**: The frequency permutation null assumes a smooth frequency distribution, which breaks if the early-checkpoint AV exhibits highly discontinuous token distributions or severe mode collapse.### P013 — Co-hyponym confusion
* **SIGNAL**: Floor. Rudimentary NLAs with round-trip cosines of 0.1–0.4 will fail to preserve child-specific features, causing sibling activations to collapse into the parent representation or random noise.
* **FAILURE MODE**: The lexical/overlap matcher will yield zero matches due to degenerate outputs, leaving only the embedding matcher, which produces noisy cross-matches at low cosine ranges.
* **FLOOR VERSION**: Test sibling sets with massive semantic distances (e.g., "car" vs "dog" under "nouns") rather than subtle distinctions like "Trojan" vs "ransomware".
* **MISSED FLAW**: Sibling concepts are inherently closer in embedding space than distractor concepts, creating an embedding-similarity confound that inflates the apparent cross-match rate even on a completely random AR.### P014 — Antonym / polarity collapse
* **SIGNAL**: Floor. A rudimentary NLA will lose the directional valence sign entirely, leading to a flat 100% collapse rate across all polarity pairs.
* **FAILURE MODE**: A degenerate AV that emits empty or repetitive text will lack explicit lexical polarity markers, causing the lexical sign check to fail and misclassify all trials.
* **FLOOR VERSION**: Train a linear probe on the bottleneck activations to perform binary sentiment/polarity classification, bypassing the AV's text generation entirely.
* **MISSED FLAW**: Lexical sign checks are brittle to spelling and morphological errors, which are common in weak NLAs and will be misidentified as polarity collapse.### P015 — Near-synonym winner-take-all
* **SIGNAL**: Usable Signal. Frequency-driven mode collapse is a primary characteristic of weak language models and degenerate AVs, making this test highly sensitive in the rudimentary regime.
* **FAILURE MODE**: If the NLA is extremely weak and outputs a single repetitive token, the self-retention rate will drop to zero, saturating the lift metric.
* **FLOOR VERSION**: Reduce synonym clusters to simple pairs (one high-frequency, one low-frequency) and evaluate the output distribution directly.
* **MISSED FLAW**: The matcher itself may have an inherent embedding-space bias toward high-frequency terms, conflating matcher similarity bias with NLA bottleneck collapse.### P016 — Metaphor vs. literal disambiguation
* **SIGNAL**: Floor. Rudimentary NLAs lack the context-tracking capacity to resolve figurative meaning, causing all homograph contexts to collapse to the dominant literal sense.
* **FAILURE MODE**: Early-checkpoint AVs will ignore the surrounding context words entirely and generate only the high-frequency literal sense of the homograph.
* **FLOOR VERSION**: Test simple homonyms (e.g., "bank" of a river vs money "bank") using highly simplified context templates.
* **MISSED FLAW**: Probes trained on the trigger token activation may leak context features from the model layers, yielding a false positive for NLA sense preservation.### P017 — Brand → category laundering
* **SIGNAL**: Usable Signal. Rare proper nouns (brands) are dropped much faster than common nouns in weak models, providing a clear brand-to-category laundering signal.
* **FAILURE MODE**: With a near-random AR, reconstruction will fail completely, causing the matcher to classify outputs as unrelated rather than category-laundered.
* **FLOOR VERSION**: Evaluate the NLA using extremely common global brands (e.g., "Google", "Apple") rather than niche cybersecurity attack tools.
* **MISSED FLAW**: Proper nouns have unique spellings that penalize lexical matchers, making brands appear to collapse faster than common nouns due to spelling distortions rather than semantic loss.

### P018 — Quantity rounding / magnitude flattening
* **SIGNAL**: Noise/Floor. Rudimentary NLAs with degenerate AVs rarely output valid numeric digits, leaving no numbers for the direct parser to evaluate.
* **FAILURE MODE**: The direct numeric parser will fail entirely when the AV outputs degenerate, non-numeric text, collapsing all results to the "dropped" category.
* **FLOOR VERSION**: Replace the direct numeric parser with a check for coarse magnitude words (e.g., "many", "few", "large", "small").
* **MISSED FLAW**: A near-random AR will map numerical representations to arbitrary values, making any reconstructed numbers look like random noise rather than systematic rounding to anchors.

### P019 — Temporal adjacency / order
* **SIGNAL**: Noise/Floor. Rudimentary NLAs fail to encode temporal order, causing sequence indicators to collapse immediately to co-occurrence or noise.
* **FAILURE MODE**: The lexical order scorer will fail to find any sequence indicators (e.g., "then", "after") in degenerate or early-checkpoint text.
* **FLOOR VERSION**: Simplify the input to a basic two-token sequence and measure if the NLA preserves their relative position index.
* **MISSED FLAW**: The Ensemble Matcher will match order-swapped event descriptions equally because it relies on bag-of-words/embedding similarities that are order-insensitive.

### P020 — Spatial / relational adjacency
* **SIGNAL**: Floor/Noise. Since relational structure is a frontier capability, a rudimentary NLA will fail completely, showing only floor effects and noise.
* **FAILURE MODE**: The lexical relation extractor cannot parse degenerate, ungrammatical, or early-checkpoint texts.
* **FLOOR VERSION**: Evaluate simple subject-verb-object strings (e.g., "cat eats fish") to check if the NLA can distinguish agent from patient.
* **MISSED FLAW**: Since the NLA's round-trip cosine is low, any relational alignment will be completely obscured by the shared token overlap between swapped inputs.

---

### Skip on rudimentary NLAs
* **P020 — Spatial / relational adjacency**: Exceeds the capacity ceiling of weak models; will yield pure noise.
* **P019 — Temporal adjacency / order**: Complex sequence tracking is absent in rudimentary setups; order markers will not generate.
* **P016 — Metaphor vs. literal disambiguation**: Context-based sense tracking collapses to literal frequency baselines.
* **P011 — Minimal-pair separation**: Collapses entirely on sub-token text generation.

### Highest-value for the rudimentary regime
* **P015 — Near-synonym winner-take-all**: Robustly measures the frequency-based mode collapse characteristic of weak models.
* **P017 — Brand → category laundering**: Captures the rapid drop-off of rare tokens (brands) relative to categories.
* **P012 — Hypernym drift (Floor Version)**: Direct categorisation mapping can successfully trace early abstraction hierarchies.
