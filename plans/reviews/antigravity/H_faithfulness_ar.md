### P071 — Faithfulness correlates with concept survival
- **SIGNAL**: On a rudimentary NLA with degenerate text and a near-random AR, the survival rates will hit a floor of zero across all quartiles, drowning out any correlation signal.
- **FAILURE MODE**: The logistic regression fails to converge because there is minimal variance in both the noisy `cos_sim` values and the binary survival labels.
- **FLOOR VERSION**: Simplify the predictor to a binary indicator of whether the AV emitted any non-trivial text, and correlate this with overall host activation magnitude recovery.
- **MISSED FLAW**: The plan assumes the matcher's survival classification is independent of `cos_sim`, but low-faithfulness text often contains repetitive patterns that trigger matcher false-positives, confounding the correlation.### P072 — Faithfulness-weighted retention vs raw retention
- **SIGNAL**: With a low-capability NLA, raw retention sits near the floor, and multiplying it by a noisy, low AR cosine (~0.1) compresses all values into a tiny range, yielding noise-dominated category rankings.
- **FAILURE MODE**: Small differences in raw retention values cause massive, spurious Spearman rank-correlation flips when weighted by noisy cosines.
- **FLOOR VERSION**: Apply a hard binary threshold to `cos_sim` (e.g., zeroing out all records below random chance) before calculating retention to filter out completely failed reconstructions.
- **MISSED FLAW**: The metric rewards repetitive, degenerate text that achieves high token overlap (inflating `cos_sim`) over diverse but paraphrased text that might be semantically richer.### P073 — AR concept drift toward the wrong neighbor (activation space)
- **SIGNAL**: A near-random AR will reconstruct activations that collapse to the global prior direction, causing both true and neighbor cosines to hover near zero-signal noise levels.
- **FAILURE MODE**: Reconstructions will align with the model's global prior vector rather than the text-conditional directions, rendering the relative drift calculation meaningless.
- **FLOOR VERSION**: Average the reconstructed activations across all dataset rows to create a single centroid vector, and evaluate if this global centroid is closer to the average benign representation than the true representation.
- **MISSED FLAW**: The plan assumes the host model's concept direction vectors are valid in the low-dimensional or poorly aligned representation space of a tiny NLA.### P074 — Round-trip stability: converge or drift?
- **SIGNAL**: A rudimentary AV checkpoint that emits degenerate or empty text will cause the round-trip loop to collapse to a static attractor in a single step, yielding a flat line from $k=1$ onwards.
- **FAILURE MODE**: The AV output becomes completely empty or repetitive at $k=1$, causing the AR to generate a constant bias vector and preventing the measurement of actual compounding drift.
- **FLOOR VERSION**: Replace the iterative loop with a single-step perturbation analysis, measuring the survival rate of the concept in the first-round text as a function of noise injected into $a_0$.
- **MISSED FLAW**: The plan mistakes vocab-collapse or token-bias attraction (surface text degeneracy) for semantic drift in the representation space.### P075 — Per-position faithfulness: concept tokens vs filler tokens
- **SIGNAL**: If a degenerate AV rarely emits the target concept token, the concept-bearing token set will be empty or near-empty, rendering the per-position comparison statistically underpowered.
- **FAILURE MODE**: The paired-difference test fails to run due to a lack of valid concept positions in the short, repetitive sentences produced by the rudimentary model.
- **FLOOR VERSION**: Compare the average reconstruction fidelity of entire sequences that contain a target keyword against sequences that contain only filler/structural words.
- **MISSED FLAW**: Frequent structural filler tokens, which are easier for a weak AR to reconstruct, dominate the average sequence cosine and artificially inflate the apparent faithfulness of failed concept reconstructions.### P076 — Reconstruction error by concept category
- **SIGNAL**: In a low-dimensional bottleneck or tiny model, the AR struggles universally across all categories, compressing the category-wise cosines into a narrow noise band where no ranking is possible.
- **FAILURE MODE**: Spearman rank correlation results will be dominated by random sampling error due to the lack of distinct, category-specific variance.
- **FLOOR VERSION**: Group concepts into two broad buckets (e.g., frequent/simple vs rare/complex) instead of fine-grained categories, and compare their average reconstruction cosines.
- **MISSED FLAW**: Apparent high faithfulness in some categories may merely reflect lexical overlap with the AR's training vocabulary distribution rather than true semantic preservation.### P077 — AR identity / cross-row leakage (does the AR ignore the text?)
- **SIGNAL**: This plan provides a strong, usable diagnostic signal even on the weakest NLA, as it directly checks whether the AR outputs change based on the input text.
- **FAILURE MODE**: If the AV is so degenerate that it outputs identical strings for all inputs, the cosine matrix will be uniform, making the diagonal and off-diagonal indistinguishable.
- **FLOOR VERSION**: Bypass the degenerate AV and feed the AR a set of hand-crafted, highly distinct synthetic text templates to check for basic conditioning capacity.
- **MISSED FLAW**: If the AR merely outputs activations whose magnitudes correlate with input sentence length, sequence length variance will create a false diagonal advantage.### P078 — Faithfulness calibration: cosine threshold vs true concept survival (probe)
- **SIGNAL**: In a tiny model or small-dataset regime, the bottleneck probes will be poorly trained and noisy, resulting in flat reliability curves and a random ROC-AUC.
- **FAILURE MODE**: The ground-truth labels from the probes are unreliable, making calibration metrics for the cosine threshold meaningless.
- **FLOOR VERSION**: Use a dictionary-lookup matching method on the input text as the ground truth presence label rather than relying on bottleneck probes.
- **MISSED FLAW**: A weak AR can output high-magnitude activations that trigger the target probe without actually reconstructing the specific coordinates of the concept.

### P079 — AR paraphrase delta (does paraphrasing the AV text change the activation?)
- **SIGNAL**: A degenerate AV output cannot be meaningfully paraphrased, and a weak AR will treat any surface changes as out-of-distribution noise, resulting in floor-level cosines.
- **FAILURE MODE**: The paraphrase step produces identical strings or fails, or the weak AR responds to minor lexical swaps with extreme, chaotic activation shifts.
- **FLOOR VERSION**: Replace natural paraphrasing with controlled character edits or single-word insertions to measure the AR's baseline sensitivity to input modifications.
- **MISSED FLAW**: A low paraphrase delta might indicate that the AR is completely ignoring the input text (extreme insensitivity) rather than showing robust semantic invariance.

### P080 — Activation-space neighbor analysis: nearer the true concept or a benign neighbor?
- **SIGNAL**: On a rudimentary NLA, low absolute cosines will cause rankings in the direction bank to be dominated by noise, resulting in random neighbor rankings.
- **FAILURE MODE**: If AR outputs collapse to a global mean, the ranking will be determined entirely by which concept direction vector is closest to that global mean.
- **FLOOR VERSION**: Limit the direction bank to three vectors (true concept, benign neighbor, and one random control) and report the fraction of times the true concept outranks the control.
- **MISSED FLAW**: In a low-dimensional or tiny model, the concept direction vectors are highly collinear, making the "nearest neighbor" selection a numerical artifact of tiny cosine differences.     

---

### Skip on rudimentary NLAs
- **P074 (Round-trip stability)**: Loops collapse immediately to degenerate/empty attractors.
- **P078 (Faithfulness calibration)**: Relies on bottleneck probes that are themselves poorly trained or noisy in rudimentary models.
- **P079 (AR paraphrase delta)**: Degenerate text cannot be paraphrased, and weak ARs are overly sensitive to any surface changes.

### Highest-value for the rudimentary regime
- **P077 (AR identity)**: Directly diagnoses whether the weak AR is conditioning on the text at all, which is the most fundamental failure mode.
- **P071 (Faithfulness correlates with concept survival)**: With a simplified floor version, it determines if bottleneck representation collapse maps to final output dropout.
- **P075 (Per-position faithfulness)**: Helps locate where the information loss occurs (concept vs structural tokens) when the model is capacity-constrained.
