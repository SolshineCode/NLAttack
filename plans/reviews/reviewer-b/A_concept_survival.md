### P001 — Retention-vs-frequency dropout law (continuous, probe-anchored)
* **SIGNAL:** On a rudimentary NLA with a degenerate AV and a low-fidelity AR (round-trip cosine ~0.1-0.4), the continuous frequency law will hit a floor effect where almost all rare and medium-frequency concepts show near-zero survival, flattening the continuous Zipf curve.
* **FAILURE MODE:** The logistic regression slope $\beta$ will fail to fit or yield chaotic, statistically insignificant slopes because the AV-matcher gap collapses to zero when the degenerate text output yields no matches.
* **FLOOR VERSION:** Replace the 300 continuous word-frequency values with a simple binary division of extremely high-frequency stopwords versus low-frequency content words, evaluating the bottleneck representation via direct cosine similarity rather than end-to-end matching.
* **MISSED FLAW:** The plan assumes the probe-side dropout is shallower than the AV-matcher dropout due to verbalizer failures; however, in a low-dimensional bottleneck trained on few examples, the representation itself may fail to encode the concept entirely, making both the probe and the matcher hit the floor simultaneously.### P002 — Category-selective dropout against freq+length-matched controls
* **SIGNAL:** This plan will produce high noise and a flat signal on a rudimentary NLA, as degenerate AV outputs (empty or repetitive text) cause both target categories and control groups to have near-zero retention rates.
* **FAILURE MODE:** The matching ensemble will fail to find matches across all categories, resulting in overlapping confidence intervals and a "blind-spot" table where every category appears equally and completely dropped.
* **FLOOR VERSION:** Simplify the categories to broad, structurally distinct classes (e.g., concrete high-frequency nouns vs. synthetically random character sequences) and relax the matcher to capture partial character overlaps or edit distances.
* **MISSED FLAW:** The plan matches controls on frequency and length, but overlooks the fact that a retrieval-only or weak verbalizer is highly sensitive to tokenization boundaries and subword patterns, confounding category-level deficits with subword fragmentation.### P003 — Blind-spot stability: variance of survival across many contexts
* **SIGNAL:** A near-random AR or degenerate AV will lead to flat, near-zero survival across all 15 contexts, saturating the within-concept variance at 0 (stable-dropped) for almost all concepts and rendering the clustering meaningless.
* **FAILURE MODE:** The low-variance "stable blind spot" cluster will be dominated by general model failure rather than concept-specific blind spots, making it impossible to identify routable holes.
* **FLOOR VERSION:** Reduce the context set to a few highly frequent, simple carrier sentences and evaluate stability using the variance of direct bottleneck activation cosine similarity rather than end-to-end survival.
* **MISSED FLAW:** The plan assumes that context stability indicates a structural property of the concept representation, but on a weak NLA, stable dropout is more likely a trivial consequence of low-capacity encoding that fails to register the concept under any condition.### P004 — BOS / position-0 verbalization noise (position confound on the floor)
* **SIGNAL:** An early-checkpoint AV that emits degenerate text will exhibit extreme positional bias (often only emitting at the very beginning or failing immediately after BOS), producing a high but artificial positional signal.
* **FAILURE MODE:** The end-to-end paired comparison will show massive, chaotic differences in retention based on position due to structural generation failures of the weak AV, which will completely overwhelm any true representation differences in the bottleneck.
* **FLOOR VERSION:** Measure the raw activation norm or token-attention weights at position 0 versus mid-sentence in the bottleneck directly, bypassing the verbalizer and matcher entirely.
* **MISSED FLAW:** The plan assumes the probe-side representation remains invariant across positions, but in low-capacity models, positional embeddings dominate the representation space, meaning a change in probe performance might reflect positional embedding shift rather than concept encoding.### P005 — Compression / length sensitivity: longer input → more dropout
* **SIGNAL:** For a rudimentary NLA with a low-dimensional bottleneck or small training set, introducing any amount of context will rapidly saturate the bottleneck, causing retention to immediately drop to floor levels for all inputs longer than a few tokens.
* **FAILURE MODE:** The length ladder will show a step-function drop at length 10 or 20 rather than a smooth monotonic decline, rendering slopes unmeasurable and CIs unstable.
* **FLOOR VERSION:** Shift the length ladder down to very short ranges (e.g., 2, 4, 6, 8, 12 tokens) and evaluate the target concept positioned exclusively at the end of the sequence to isolate basic memory decay.
* **MISSED FLAW:** The plan overlooks the fact that longer carrier sentences increase the chance of random matching false positives (hallucinations matching the target), which can artificially inflate retention scores on longer inputs in weak models.### P006 — Concept crowding: N concepts competing in one input
* **SIGNAL:** A weak NLA with a low-dimensional bottleneck will saturate immediately at N=2, resulting in chance-level retention and flat lines beyond N=1.
* **FAILURE MODE:** The estimated capacity k will collapse to 1 or 0, and the multi-concept inputs will produce chaotic, degenerate AV text that prevents the matcher ensemble from resolving which conc
cepts survived.
* **FLOOR VERSION:** Test only N ∈ {1, 2} with extremely distinct, highly frequent concepts (e.g., "the" vs "and" domain equivalents) to measure the absolute bare minimum capacity boundary.
* **MISSED FLAW:** When multiple concepts are present, a retrieval-only or weak verbalizer may output a single generic summary token that matches none of them, falsely indicating bottleneck eviction w
when the concepts might still be linearly probeable.

### P007 — Token-span vs survival: do multi-token concepts survive better?
* **SIGNAL:** A rudimentary NLA/AV that outputs degenerate or repetitive text will likely fail to reconstruct multi-token concepts entirely, as they require coherent sequential generation, leading to 
 a strong negative correlation between span and survival (saturation at the floor for longer spans).
* **FAILURE MODE:** Multi-token concepts will fragment during generation (e.g., "security guard" -> "guard guard" or "sec"), failing lexical matching completely and distorting the correlation analysis
s.
* **FLOOR VERSION:** Use a character-level edit distance or substring-level matcher rather than an exact token-span matcher, and test concepts that map to a single token in the vocabulary vs. multi-to
oken phrases.
* **MISSED FLAW:** The plan controls for frequency but ignores tokenization quality; in weak models, multi-token spans are highly sensitive to specific subword splits, confounding token span size with
h subword tokenization artifacts.

### P008 — Part-of-speech / modality retention: nouns vs verbs vs numbers
* **SIGNAL:** On a rudimentary NLA (especially with a near-random AR), numbers and abstract verbs will completely fail to retain, showing floor performance while concrete nouns may show a weak, noisy 
 signal.
* **FAILURE MODE:** The pairwise differences will collapse to zero or become dominated by the frequency of these POS classes in the tiny training corpus rather than their semantic modality.
* **FLOOR VERSION:** Restrict the test to concrete, high-frequency nouns vs. digits (using a regex-relaxed matcher that matches any digit or numerical token) to see if the model retains quantity at th
he coarsest level.
* **MISSED FLAW:** The plan assumes the carrier sentence ("The report mentions {}.") is class-neutral, but a weak AV is highly sensitive to grammatical context and may fail to generate grammatically a
awkward insertions, confounding POS survival with generation perplexity.

### P009 — Redundancy / repetition boosts survival
* **SIGNAL:** This plan is highly viable for a rudimentary NLA, as repeating a concept 3x provides a strong signal that can overcome high reconstruction noise (round-trip cosine ~0.1-0.4) and weak AV 
 thresholds.
* **FAILURE MODE:** If the AV is stuck in a degenerate repetition loop, the repetition of inputs might trigger self-attention loops that collapse the output into infinite repeats of a single unrelated
d token, breaking the monotonic relationship.
* **FLOOR VERSION:** Focus on exact repetitions (1x vs 3x) at very short total lengths and measure the absolute change in the bottleneck's cosine similarity to the target concept vector rather than re
elying on verbalization.
* **MISSED FLAW:** A weak matcher ensemble might over-count repeating outputs as multiple independent hits, inflating the apparent "survival" of repeated concepts when only a single corrupted fragment
t was generated.

### P010 — Salience-driven dropout: a dominant topic suppresses other concepts
* **SIGNAL:** In a low-capacity NLA, a highly salient dominant topic will completely overwrite the bottleneck representation, saturating the secondary concept's survival at absolute zero (floor).     
* **FAILURE MODE:** The paired delta will show total suppression at the first step of salience, preventing the measurement of relative salience gradients and making the probe AUC indistinguishable fro
om chance.
* **FLOOR VERSION:** Reduce the "dominance" of the salient topic to a single repeated token and use a direct activation probe to detect if the secondary concept's representation is shifted in space ra
ather than completely erased.
* **MISSED FLAW:** The plan matches total length and concept count, but a weak AV may exhibit strong semantic drift toward the dominant topic during decoding, meaning the "suppression" is a decoding-t
time hallucination rather than bottleneck-level eviction.

---

### Skip on rudimentary NLAs
* **P002 (Category-selective dropout)** — Fails to resolve semantic boundaries because the weak AV and matcher generate flat, zero-survival floor noise across all categories.
* **P003 (Blind-spot stability)** — Within-concept variance collapses to zero everywhere because the model baseline failure ensures almost all concepts drop stably across all contexts.
* **P006 (Concept crowding)** — Low-dimensional bottlenecks saturate immediately at $N=2$, causing capacity estimation to collapse to zero and generating degenerate outputs.
* **P007 (Token-span vs survival)** — Weak AVs fragment multi-token spans during generation, breaking exact matchers and showing a false negative correlation with survival.

### Highest-value for the rudimentary regime
* **P009 (Redundancy / repetition)** — Acts as a strong signal amplifier that can overcome high reconstruction noise and degenerate verbalizer thresholds.
* **P004 (BOS / position-0 noise)** — Crucial for isolating whether failures are due to the structural generation errors/biases of early-checkpoint AVs versus true bottleneck capacity.
* **P001 (Retention-vs-frequency)** — Provides the necessary baseline calibration of vocabulary limits, provided it is adapted from a continuous scale to a binary high-vs-low split.
