# Critique: Family E Plans on Rudimentary NLAs

---

### P041 — Probe-vs-AV gap per concept
**SIGNAL:** The probe side yields usable signal (bottleneck retention ceiling) even on a near-random AR; the AV side will floor at 0 for degenerate/empty verbalizations, so the gap collapses to `probe_acc` and you lose the verbalizer vs. matcher disentanglement.  
**FAILURE MODE:** Empty/repetitive AV output → matcher always 0 → `av_matcher_acc ≡ 0` → gap = probe_acc for every concept. With n=2–3 concepts, the mean gap has huge sampling variance and you cannot attribute loss to verbalizer vs. matcher.  
**FLOOR VERSION:** Drop the AV entirely for the weakest NLAs; report `probe_acc` alone as the bottleneck retention ceiling. If you need a non-zero verbalizer floor, swap the generative AV for a retrieval-kNN verbalizer (nearest cached snippet from a concept bank) which emits non-degenerate text even at checkpoint 0.  
**MISSED FLAW:** The probe is evaluated on *source-text* concept labels (keyword or LLM), but the bottleneck activations fed to the probe may already be AR reconstructions (round-trip cosine ~0.1–0.4). You're measuring retention in a corrupted bottleneck, not the true encoder output.

---

### P042 — Linear vs MLP probe
**SIGNAL:** On a rudimentary NLA, concepts may be nonlinearly encoded; an MLP *could* recover signal the linear probe misses. In practice, with 2–3 concepts and ~100–200 samples/concept, the MLP (256 hidden units) will severely overfit even with 5-fold CV and early stopping.  
**FAILURE MODE:** MLP memorizes label noise and fold-specific sampling artifacts; CV std dominates Δ. The reported "nonlinear encoding" is mostly overfit to keyword-label noise, which the MLP fits better than the linear probe.  
**FLOOR VERSION:** Use a heavily regularized tiny MLP (32 hidden units, L2=1.0, dropout=0.5) or skip MLP entirely and report linear probe AUC as a conservative lower bound. An RBF SVM is more sample-efficient for nonlinear boundaries at tiny n.  
**MISSED FLAW:** MLP can fit label noise *better* than linear, so MLP AUC may be *inflated* relative to true signal. A negative Δ (MLP < linear) from overfitting on noisy labels is misread as "encoding is linear" when it's just "MLP overfit worse."

---

### P043 — Layer sweep
**SIGNAL:** Requires GPU re-extraction at multiple layers. On a rudimentary NLA (e.g., a 2-layer tiny model or a single-checkpoint AV), the "bottleneck layer" may not exist or vary meaningfully. If the NLA is a standard multi-layer transformer, the probe side works but n=2–3 per layer gives high-variance AUC curves.  
**FAILURE MODE:** Re-extraction pipeline assumes the v8 extraction script and GPU access. If the model has <4 layers or the AR target layer is hardcoded, the sweep is ill-defined. Small n per layer makes peak-layer identification noise-dominated.  
**FLOOR VERSION:** Probe only the existing L23 dump plus one earlier layer (L9/L21 if 1B dumps exist and dims match). Skip GPU re-extraction; treat as a two-point comparison.  
**MISSED FLAW:** The plan assumes the NLA has a clear AV→text→AR pipeline with a designated bottleneck layer. A rudimentary NLA may be a single small model without this structure; "layer of the bottleneck" is then a category error.

---

### P044 — Probe-direction stability
**SIGNAL:** Correctly signals "no reliable encoding" for low-AUC concepts (cosine ~0). For any high-AUC concepts that exist, stability >0.8 would be meaningful evidence of a real direction.  
**FAILURE MODE:** Bootstrap subsamples at tiny n have very few positive examples; weight vectors are dominated by sampling noise. The correlation(stability, AUC) is largely driven by prevalence/label noise, not true encoding stability. Critically, `StandardScaler` is fit *per fold*, so coefficients across seeds/folds are on different scales — cosine of differently-scaled coefficients is meaningless unless you refit all seeds with a *single* scaler fit on full data.  
**FLOOR VERSION:** Report stability only for concepts with ≥20 positive examples. Use a parametric bootstrap (add Gaussian noise to activations, refit with fixed full-data scaler) instead of subsampling to increase effective sample size.  
**MISSED FLAW:** The per-fold StandardScaler makes cross-seed coefficient cosines incomparable. The stability metric as written measures scaler variance + sampling noise, not direction stability.

---

### P045 — Concept geometry (superposition)
**SIGNAL:** With n=2–3 concepts, you have 1–3 pairwise cosines — insufficient for clustering, null estimation, or any population claim. The random-direction null in high-d is tight (mean |cos| ~ 0), so any signal requires strong entanglement.  
**FAILURE MODE:** Keyword label co-occurrence (shared substrings) directly drives probe direction overlap. The "control" of reporting label correlation doesn't disentangle this because the probe directions are *learned from those same labels*.  
**FLOOR VERSION:** Skip entirely until n_concepts ≥ 10 (after P050). If forced, report only the single top-pair cosine as a sanity check with a large-grain caveat.  
**MISSED FLAW:** Direction cosine and label correlation are statistically coupled through the shared label matrix. The probe has no access to "true" concept presence independent of the keywords used to select concepts.

---

### P046 — Verbalizer ablation (trained AV vs forced/template AV)
**SIGNAL:** **Highest-value for rudimentary NLAs.** If the trained AV emits garbage, a forced decoder that reads the bottleneck (e.g., "Concept X: [probe_prob]") and emits a fixed phrasing directly tests whether the bottleneck retains the concept. The gap-narrowing Δ isolates verbalizer incompetence.  
**FAILURE MODE:** The forced decoder still needs a readout from the bottleneck. If the AR is near-random, the bottleneck activations may not align with the probe's readout direction. The ensemble matcher may also fail on template phrasing (expects natural language), artificially depressing forced-AV accuracy.  
**FLOOR VERSION:** Use the linear probe's per-example prediction as the forced verbalizer: emit "Concept X present" if probe_prob > 0.5. Requires no new decode code, works with degenerate AV, and the matcher can be swapped for a simple rule-based matcher on the template.  
**MISSED FLAW:** The matcher is held constant but validated only on natural AV output. Template outputs may fall outside the matcher's operating regime, making the forced AV look worse than it is. The matcher itself needs calibration on synthetic phrasings.

---

### P047 — Probe accuracy gated by AR faithfulness
**SIGNAL:** On a rudimentary NLA with AR cosine ~0.1–0.4, most examples fall in low-fidelity strata. If probe AUC is flat across strata, the bottleneck itself is the problem — useful signal.  
**FAILURE MODE:** High-fidelity stratum (top quartile) may have n < 20; probe AUC estimates are pure noise. Fidelity correlates with example difficulty/concept prevalence (easy examples reconstruct better), confounding the stratum Δ. Round-trip cosine measures *activation* reconstruction, not *concept* reconstruction — the AR can have high cosine while scrambling the concept direction.  
**FLOOR VERSION:** Use a binary split at median cosine (not quartiles). Pool concepts to get enough samples per stratum. Or skip and report overall probe_acc as the ceiling.  
**MISSED FLAW:** Faithfulness strata don't guarantee concept preservation. The AR could perfectly reconstruct the activation norm (high cosine) while rotating the concept direction to null. The probe on high-fidelity examples would still show low AUC, misleadingly attributed to "bottleneck loss."

---

### P048 — Multi-concept joint probe
**SIGNAL:** Requires ≥3–4 balanced concepts. On a rudimentary NLA with 2–3 concepts/corpus, this is infeasible per corpus. Merging chunks introduces distribution shift that the joint probe will fit as spurious structure.  
**FAILURE MODE:** `MultiOutputClassifier` with tiny n overfits catastrophically. Joint-pair cells (both present) may have 0–1 examples. Any "lift" is noise or label-co-occurrence leakage — the joint probe sees the *labels* of co-occurring concepts during training.  
**FLOOR VERSION:** Skip joint probe. Instead, add a single interaction term to logistic regression: `probe_A ~ activation + label_B` (using ground-truth label B, not predicted). Tests conditional encoding without multi-output overfit.  
**MISSED FLAW:** Label co-occurrence drives joint performance. The probe trains on (activation, label_B) → label_A; if label_A and label_B co-occur in the keyword labels, the probe learns the label correlation, not representational structure.

---

### P049 — Probe transfer across corpora
**SIGNAL:** On a rudimentary NLA, the bottleneck may not have stable directions at all. Transfer will fail (low off-diagonal), but this could be because the source probe is noise, not because directions are corpus-specific.  
**FAILURE MODE:** With 2–3 concepts/corpus, few concepts are balanced in *both* corpora. The transfer matrix is extremely sparse. `StandardScaler` from corpus A applied to corpus B assumes similar activation distributions; norm/scale mismatch causes artifactual transfer drop.  
**FLOOR VERSION:** Transfer only the single most prevalent shared concept (if any). Or skip transfer; probe each corpus independently and compare AUCs descriptively.  
**MISSED FLAW:** The scaler mismatch is a silent confound. If corpus B activations have different variance, the fixed scaler distorts features, and the drop reflects preprocessing artifact, not direction non-transferability.

---

### P050 — Better concept labels (LLM labels vs keywords)
**SIGNAL:** **Highest-value for rudimentary NLAs.** Keyword labels are especially destructive for weak NLAs because the verbalizer uses paraphrases/synonyms that substring matching misses. LLM labels recover semantic presence, increasing n_balanced_concepts from 2–3 to ≥10/corpus — this single change unblocks P044, P045, P048, P049.  
**FAILURE MODE:** Stage-2 LLM labels have their own noise (hallucination, inconsistency). Joining by `doc_id` assumes perfect alignment. Crucially, LLM labels measure concept presence in the *source text*, not in the NLA's verbalization. If the bottleneck drops a concept the source had, the probe shows high AUC (bottleneck has it) but AV matcher shows 0 — this *is* the P041 gap, but the LLM labels don't fix the fundamental mismatch: we're probing the bottleneck for source concepts, not verbalized concepts.  
**FLOOR VERSION:** Even 50–100 hand-verified labels for the top concepts would dramatically improve probe reliability over keyword labels.  
**MISSED FLAW:** The LLM labels are for the *source distribution*, not the *NLA's output distribution*. A concept present in source but dropped by the bottleneck will show as "probe_acc high, av_matcher_acc low" — correctly flagged as gap — but a concept *hallucinated* by the verbalizer (present in AV text, absent in source) is invisible to this probe because the label says 0. The ground truth is one-sided.

---

## Skip on rudimentary NLAs
- **P042** (MLP probe) — catastrophic overfit at tiny n; linear probe is the conservative floor
- **P043** (Layer sweep) — needs GPU re-extraction; assumes multi-layer bottleneck architecture
- **P045** (Concept geometry) — n_concepts too small; confounded by label overlap
- **P048** (Joint probe) — needs ≥3–4 balanced concepts; overfits; label leakage
- **P049** (Probe transfer) — too sparse; scaler mismatch artifacts dominate

## Highest-value for the rudimentary regime
1. **P050** (LLM labels) — fixes the root cause (label noise + small n) that cripples every other plan
2. **P046** (Verbalizer ablation) — directly tests "bottleneck has it but AV fails" with a probe-forced decoder
3. **P041** (Probe-vs-AV gap) — *probe side only*; AV side floors at 0, but `probe_acc` = bottleneck retention ceiling
4. **P047** (Fidelity-gated probe) — simplified to binary median split, pooled concepts
5. **P044** (Direction stability) — only for high-prevalence concepts (≥20 pos), with fixed full-data scaler
session_id: 20260605_122448_e01693

