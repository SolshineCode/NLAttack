---

## Per-Plan Critiques for Rudimentary NLAs

### P081 — In-dist vs OOD encodable-subspace probe-AUC gap
**SIGNAL:** Strong. Probe-only (cached L23 + sklearn), bypasses AV/AR entirely. Encoder representational geometry is testable even on tiny/undertrained NLAs.  
**FAILURE MODE:** If encoder is so weak all probe AUCs collapse to ~0.5, ΔAUC becomes noise. Keyword ground truth misaligns with bottleneck features (syntactic/low-level patterns), making AUC a loose upper bound.  
**FLOOR VERSION:** Already probe-only. If n_concepts small or AUCs noisy, coarsen to binary (AUC > 0.6) and use Fisher's exact test on proportions; pool items; use Spearman rank correlation instead of mean ΔAUC.  
**MISSED FLAW:** MLP probe as "non-linear control" overfits badly at small n — needs explicit held-out test set, not just CV. Keyword labels are noisier for rare concepts, creating artifactual AUC-frequency correlation.

### P082 — Dropout-map domain transfer
**SIGNAL:** Moderate on probe side; poor on AV side. Probe-side works (same as P081). AV-side fails if verbalizer emits empty/degenerate/retrieval text — retention becomes all zeros, ρ undefined.  
**FAILURE MODE:** AV retention floor of zeros → matrix all zeros. Probe-side: if all AUCs ≈ 0.5, per-concept variance is noise → ρ reflects sampling noise. Shared-concept restriction (n_shared < 20) makes ρ unstable.  
**FLOOR VERSION:** Run probe-side ONLY. Coarsen retention to tertile bins (high/mid/low); use Kendall's W (concordance) across corpora instead of pairwise Spearman — more stable at low n. Compare to shuffled-activation control (permute concept labels per corpus) for noise floor.  
**MISSED FLAW:** "Shared concepts" assumes keyword-identical terms mean identical concepts across domains ("deception" in anthropic vs gemma4). Validity threat bites harder when probe AUCs are noisy — conflates concept instability with encoder instability.

### P083 — Frequency-tier × domain interaction
**SIGNAL:** Moderate. Probe-based, but needs sufficient concepts per freq_band per corpus. Rudimentary NLA compresses dynamic range (all AUCs 0.5–0.6), making slope estimation noisy.  
**FAILURE MODE:** Sparse low-band concepts (n < 10) → slope dominated by outliers. Corpus-internal freq ≠ NLA training freq (mis-specified predictor). Token length covariate multicollinear with frequency on token-level encoders.  
**FLOOR VERSION:** Collapse to 2 bands (median split); use robust/quantile regression or median difference + bootstrap CI. If n too small, permutation test on Δmedian. Report Cliff's delta (non-parametric) instead of slope.  
**MISSED FLAW:** Imposes monotonic freq→retention law. On rudimentary NLAs, relationship may be non-monotonic (stopwords ignored, mid-frequency best). meta.freq_band from eval corpus ≠ training frequency (acknowledged in P086 but not here).

### P084 — Checkpoint-maturity effect (AV training)
**SIGNAL:** Poor on rudimentary NLAs. Requires working AV across checkpoints. Early checkpoints emit empty/degenerate text → av_acc ≈ 0 for all concepts → gap = constant probe_acc. Retrieval-only AV measures cache growth, not encoder maturity.  
**FAILURE MODE:** Empty emissions → matcher 0 → av_acc 0 → flat gap. Degenerate text → matcher false positives. GPU cost high for low signal. Retrieval hits/misses create erratic jumps, not monotonic trend.  
**FLOOR VERSION:** If retrieval-only, measure cache hit rate vs checkpoint. If degenerate, filter outputs (length > threshold, repetition < threshold) and report av_acc on filtered + filter rate. Better: skip AV, use probe AUC trajectory across *encoder* checkpoints (if they exist) — isolates bottleneck maturity.  
**MISSED FLAW:** Assumes encoder frozen (probe AUC fixed). If joint AV+encoder training, probe AUC changes too → gap conflates encoder + verbalizer maturity. Degenerate emission rate only "reported," not corrected.

### P085 — Cross-model NLA transfer (Gemma vs Llama)
**SIGNAL:** Moderate probe side; poor AV side. Probe side compares two probe AUC vectors (works on small models). AV side needs Neuronpedia — custom/early NLAs won't be there. Description-recovered retention via matcher is weak proxy for retrieval/degenerate AV.  
**FAILURE MODE:** Neuronpedia lacks specific checkpoints. Both probe vectors near-chance → ρ ≈ 0 with huge CI. Matched concept set across tokenizers/model vocabularies is hard. Neuronpedia faithfulness scores are for *their* AR, not yours.  
**FLOOR VERSION:** Probe side only. Use PCA to match dimensionality before comparing probe AUCs. If n_concepts small, use rank correlation on binned AUCs. Skip AV side entirely unless both NLAs are on Neuronpedia.  
**MISSED FLAW:** Conflates model family (arch, size, pre-training) with NLA training. Low ρ could be any of these. To isolate NLA-specific blind spots, need SAME base model with DIFFERENT NLA training — not available. Different L23 dimensionality requires PCA matching before comparison.

### P086 — Concept-prevalence-in-training vs probe-AUC
**SIGNAL:** Strong. Probe-only, CPU. Even tiny encoder should show correlation if frequency shapes subspace.  
**FAILURE MODE:** Training mixture prevalence reconstruction noisy/biased. Very few training examples → prevalence binary (0/1) → kills rank correlation. Training/eval prevalence highly correlated (same domain) → partial correlation high variance. Small n_concepts with reliable counts → low power.  
**FLOOR VERSION:** If prevalence binary, use Mann-Whitney U (seen vs unseen AUC). Pool concepts into semantic clusters. Bayesian regression with informative prior on coefficient sign. Always compare to shuffled-control (permute prevalence labels).  
**MISSED FLAW:** Prevalence ignores context diversity ("deception in emails" ≠ "deception in abstracts" at same token count). Keyword label noise covaries with frequency (rarer → noisier) → artifactual AUC-prevalence correlation.

### P087 — Train-distribution leakage into verbalizations (boilerplate)
**SIGNAL:** Poor. Requires working AV generating non-degenerate text. Empty output → no insertions detectable. Retrieval-only → "insertions" are snippet content, not leakage. Degenerate repetition → matcher false positives.  
**FAILURE MODE:** High degenerate emission rate → most outputs filtered → low n. Retrieval-only: insertions track retriever behavior (input-dependent), not training distribution. Matcher on short/noisy text → high FPR.  
**FLOOR VERSION:** If retrieval-only, measure "retrieval bias": does retriever return training-frequent terms more than input predicts? If short text, use n-gram overlap with training corpus (surface leakage, no concept matcher). Compare to shuffled-activation control (random activations → noise floor).  
**MISSED FLAW:** "Input text" for keyword matching is source text, not what AV conditioned on (activation). AV may correctly decode a feature keyword matcher missed in source — looks like "insertion" but is faithful decoding. Bites harder when encoder weak (activation encodes less).

### P088 — OOD hallucination/insertion rate vs in-distribution
**SIGNAL:** Poor. Same AV-dependency as P087. Empty/degenerate/retrieval AV → insertion rate near 0, constant, or noise. Correlation with probe AUC requires meaningful AV variation.  
**FAILURE MODE:** Empty → 0 insertions → ratio NaN. Retrieval-only: rate depends on snippet length, not distribution shift. Degenerate: matcher FPR scales with output length; OOD may produce longer degenerate outputs → spurious correlation. Length-matching inputs ≠ output length.  
**FLOOR VERSION:** Skip if AV non-functional. If some text, use character-level perplexity or repetition rate (self-BLEU) as confabulation proxy — no concept matcher needed. Correlate with probe AUC. Better: measure AR faithfulness (cosine/MSE) directly on OOD vs in-dist activations — ground truth, no AV needed.  
**MISSED FLAW:** Assumes generative verbalizer with trained priors ("fills gap with priors"). False for retrieval-only or early-checkpoint NLAs (no priors learned, or just noise). Causal story invalid for rudimentary NLAs.

### P089 — Layer × distribution interaction
**SIGNAL:** Strong. Probe-only, uses cached multi-layer activations (L9/L21/L23). Tests encoder geometry, not verbalizer.  
**FAILURE MODE:** Layer dumps on 1B variant; main NLA may be different size → layer indices don't correspond (flagged but unsolved). All layers AUC ≈ 0.5 → no peak → interaction fits noise. Small n per layer×corpus cell → high variance in best-layer assignment.  
**FLOOR VERSION:** Collapse to 2 layers (early L9 vs late L23). Per concept, assign best layer (argmax AUC). McNemar's test: does fraction preferring L23 differ between in-dist/OOD? Bootstrap fraction difference. Report binary preference, not continuous AUC.  
**MISSED FLAW:** Linear probe AUC underestimates non-linearly encoded info. MLP helps but overfits at small n. 1B model's L23 ≠ 27B's L23 semantically (different depth/width scaling) — "trained read layer" comparison invalid across sizes.

### P090 — Encodable-subspace dimensionality by domain
**SIGNAL:** Moderate. Probe-only, operates on P081 weight vectors + SVD. Works if probes well-fit. On weak NLA, probe directions noisy (low AUC) → unstable effective rank.  
**FAILURE MODE:** Few probes pass AUC floor → subsample to tiny n (e.g., n=10) → SVD on 10 vectors in 4096-d meaningless (rank = n). Random weight vectors from chance probes inflate participation ratio. Concept-count matching discards in-dist power.  
**FLOOR VERSION:** Use ACTIVATION vectors directly: PCA participation ratio on all examples per corpus (n=1000s). Or report eigenvalue spectra (top 20) for in-dist vs OOD — visual comparison more robust than scalar at low n. Bayesian shrinkage estimator for effective rank.  
**MISSED FLAW:** Probe directions = probe's separating hyperplanes, not bottleneck's natural concept axes. Distributed code → probe directions arbitrary rotations within subspace → effective rank of probe directions ≠ subspace rank. Classic representational geometry confound.

---

## Summary Lists

### Skip on rudimentary NLAs
- **P084** (Checkpoint-maturity) — AV-side signal drowned by degenerate emissions; GPU cost unjustified
- **P087** (Training leakage / boilerplate) — requires coherent generative AV; retrieval/empty/degenerate outputs invalidate insertion detection
- **P088** (OOD hallucination rate) — same AV dependency as P087; causal story assumes trained generative priors
- **P085** (Cross-model AV transfer) — AV side needs Neuronpedia; probe side only valuable if both base models available locally with matched activations

### Highest-value for the rudimentary regime
1. **P081** (In-dist vs OOD probe-AUC gap) — pure probe, strong signal, directly tests core "encodable subspace" claim
2. **P086** (Training prevalence vs probe-AUC) — pure probe, operationalizes the training-dependence confound as testable correlation
3. **P089** (Layer × distribution) — pure probe, uses existing multi-layer dumps, tests whether fixed read layer is a domain-dependent design flaw
4. **P090** (Subspace dimensionality) — probe-only geometric test; floor version (activation PCA) works even when probe directions too noisy
5. **P082** (Probe-side dropout-map transfer) — probe-only, tests whether concept survival patterns are domain-stable; floor version (Kendall's W + shuffle control) robust at low n
session_id: 20260605_123948_b72441

