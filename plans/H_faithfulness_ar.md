# Family H — Faithfulness / AR-fidelity

Activation-space faithfulness plans (P071–P080). This family operationalizes
independent review **P1 #5** (AR fidelity unmeasured): if the AR faithfully
reconstructs a *laundered* activation ("credential dumping" → "credential
stuffing" direction), the deception lives in the **AR**, not the NLA bottleneck —
so AR drift must be measured in **activation space** (the AR maps AV text back to
an activation; `cosine_similarity`/`mse` between the original and reconstructed
activation = round-trip faithfulness), not in AV text or via the matcher.

These plans are special among the families: they largely **are** the
activation-space ground truth that bypasses the AV-text + matcher confound (P0
#1, P0 #2). Where a plan uses Neuronpedia's per-position `cosine_similarity`/`mse`
or the local Gemma-4-E2B AR round-trip activation, the measurement does not pass
through the embedding/WordNet matcher at all — that is the point, and each plan
says so under **Controls**.

Faithfulness is **weak** on the local AR by design: the stage_ar_sft Gemma-4-E2B
AR has round-trip cos ≈ **0.438** — a genuinely lossy/weak reconstructor. Plans
are written to be informative on a weak AR, not to assume a good one.

---

### P071 — Faithfulness correlates with concept survival
- **Family:** H
- **Probes:** Whether activation-space round-trip faithfulness predicts whether a concept survives the bottleneck (is high `cos_sim`/low `mse` a precondition for survival, or orthogonal to it?).
- **Hypothesis:** Per-concept survival probability rises monotonically with the AR faithfulness (`cos_sim`) at that concept's token positions; concepts in the bottom faithfulness quartile survive at < 0.5× the rate of the top quartile.
- **Method:** Run the general + ATT&CK datasets through `NeuronpediaNLA.verbalize_concepts` (per-position `cos_sim`/`mse` populated on each `ConceptRow`). Bin rows into faithfulness quartiles by `cos_sim`; compute `retention_rate` per quartile. Fit logistic regression of `status∈{retained,substituted}` on `cos_sim` (and on `mse` separately); report slope, AUC, and quartile retention curve.
- **Metric:** Per-quartile retention; logistic slope of survival on `cos_sim` (with 95% CI); AUC of `cos_sim` as a survival classifier; same for `−mse`.
- **Feasibility:** high — Neuronpedia returns `cos_sim`/`mse` for free; the row fields already exist (`core.ConceptRow.cos_sim/mse`).
- **Controls:** This is itself an activation-space readout, so it sidesteps the matcher for the *predictor*; the survival *label* still comes from the matcher, so report the correlation separately within each matcher-ensemble agreement tier (`agreement==1.0` vs contested) to show it is not a matcher artifact. Frequency/length-match the quartiles (carry `meta.freq_band`, token length) so the curve is not just rarity.
- **Deception link:** A monitor that trusts an AV verbalization needs to know that low-faithfulness reconstructions are exactly where concepts silently vanish — the blind region a misuse actor would aim for.
- **Harness:** `core.mean_faithfulness`, `core.retention_rate`; new `t21_faithfulness_survival_corr(res)` doing the quartile/logistic group-by.
- **Null looks like:** Flat retention across faithfulness quartiles and logistic slope CI spanning 0 → faithfulness is orthogonal to survival, and `cos_sim`/`mse` carry no information about the dropout map.

### P072 — Faithfulness-weighted retention vs raw retention
- **Family:** H
- **Probes:** Whether down-weighting "retained" concepts by their AR faithfulness changes the qualitative conclusions of the survival tests (i.e., are some apparent survivals just unfaithful verbalizer artifacts?).
- **Hypothesis:** `faithfulness_weighted_retention` is materially lower than raw `retention_rate` (absolute gap > 0.05), and the gap is **category-dependent**: at least one category flips its rank order between raw and weighted retention.
- **Method:** For each test slice (category, freq band, ATT&CK tactic) compute raw `retention_rate` and `faithfulness_weighted_retention` (already in `core.py`). Report the per-slice delta and rank-correlation (Spearman) of category orderings under the two metrics. Bootstrap the per-category gap CI.
- **Metric:** raw − weighted retention per slice; Spearman ρ of category rank order (raw vs weighted); count of rank inversions; bootstrap CI on the largest gap.
- **Feasibility:** high — both metrics exist; pure aggregation over an existing run.
- **Controls:** Weighting *is* the activation-space correction the matcher cannot provide — it discounts survivals the AR could not faithfully reconstruct (P1 #5). Report rows missing `cos_sim` separately (they fall back to `default=1.0`); if most rows lack scores the result is uninformative — state coverage.
- **Deception link:** If weighting changes which categories look "safe," then end-to-end retention overstates monitor coverage for exactly the laundering-prone categories.
- **Harness:** `core.faithfulness_weighted_retention`, `core.retention_rate`; new `t22_weighted_vs_raw(res)` returning per-slice deltas + rank inversions.
- **Null looks like:** Weighted ≈ raw everywhere (gap < 0.05) and ρ ≈ 1 with zero inversions → faithfulness weighting does not change conclusions and the cheap raw metric suffices.

### P073 — AR concept drift toward the wrong neighbor (activation space)
- **Family:** H
- **Probes:** Whether the AR's reconstructed activation drifts toward a *benign neighbor* concept direction rather than the true concept direction — laundering located in the AR, in activation space, with no AV-text/matcher involvement.
- **Hypothesis:** For attack concepts with a `meta.benign_neighbor`, the AR-reconstructed activation has higher cosine to the **neighbor** concept direction than to the **true** concept direction in a non-trivial fraction (> 0.2) of rows — i.e., the AR itself bends "dumping" toward "stuffing."
- **Method:** Local `KitftNLA`/Gemma-4-E2B AR path. For each concept, build true-concept and benign-neighbor direction vectors (mean host-model activation at concept tokens minus a neutral-control mean, or a linear-probe weight vector). Reconstruct the activation from the AV text via the AR; compute cos(recon, true_dir) and cos(recon, neighbor_dir). Drift = neighbor cos − true cos.
- **Metric:** Mean drift (neighbor cos − true cos); fraction of rows with drift > 0; per-concept drift table; comparison of drift on attack concepts vs neutral controls.
- **Feasibility:** medium — needs the local AR + GPU and concept-direction vectors; the round-trip is the weak (cos ≈ 0.438) AR, so absolute cosines are small — interpret via the **difference** (neighbor vs true), not absolute magnitude.
- **Controls:** Purely activation-space; the matcher and AV-text are bypassed entirely (this is the ground-truth that adjudicates Family B's matcher-dependent laundering claims). Control with neutral concepts that have no adversarial neighbor — drift there should be ~0. Match neighbor/true directions for norm.
- **Deception link:** Directly localizes laundering: if drift is in the AR, the NLA bottleneck is innocent and the fix is the reconstructor, not the monitor's concept list.
- **Harness:** `adapters.KitftNLA.encode` + AR reconstruction; new `experiments/local_gemma_e2b/ar_drift.py` and `t23_ar_neighbor_drift`.
- **Null looks like:** Drift ≈ 0 (neighbor cos ≈ true cos) for attack and control alike → the AR does not preferentially bend concepts toward benign neighbors; observed laundering is matcher-side or bottleneck-side, not AR-side.

### P074 — Round-trip stability: converge or drift?
- **Family:** H
- **Probes:** Whether iterating the full loop (activation → AV text → AR activation → AV text → …) converges to a fixed point or drifts away from the original concept over rounds (compounding reconstruction error).
- **Hypothesis:** Cosine of the round-k activation to the original activation **decreases monotonically** with k for the first few rounds and the concept-survival rate (matcher on the round-k AV text) drops with k — the loop is contractive toward a generic attractor, not the original.
- **Method:** Local AR. Seed with host activation a0; for k=1..K: text_k = AV(a_{k-1}); a_k = AR(text_k). Track cos(a_k, a0), cos(a_k, a_{k-1}) (step size), `mse`, and concept survival on text_k. Repeat over the dataset; aggregate trajectories.
- **Metric:** cos(a_k, a0) vs k curve (with CI); step-size cos(a_k, a_{k-1}) → does it approach 1 (converged) or stay < 1 (drifting); survival(text_k) vs k; estimated fixed-point cos.
- **Feasibility:** medium — K× the AR cost on GPU; K small (e.g. 4) given the weak AR. Watch for degenerate empty AV text on early local checkpoints (note in `results/local_gemma_e2b/README.md`).
- **Controls:** Entirely activation-space for the stability claim. Compare against a shuffled-text control (feed AR an unrelated AV text) to calibrate what "drift to generic" looks like. The original a0 is the fixed reference — no matcher needed for the cos curve.
- **Deception link:** A monitor that re-verbalizes (or an attacker who iterates against the verbalizer, P0 #3 monitor-gaming) lives in this loop; if it drifts to a benign attractor, repeated processing launders intent.
- **Harness:** new `experiments/local_gemma_e2b/round_trip_stability.py`; `t24_round_trip_stability`.
- **Null looks like:** cos(a_k, a0) flat across k and step size ≈ 1 by k=2 → the loop converges immediately to (near) the original; no compounding drift, repeated verbalization is safe.

### P075 — Per-position faithfulness: concept tokens vs filler tokens
- **Family:** H
- **Probes:** Whether the AR reconstructs **concept-bearing** token positions less faithfully than **filler** positions (is the activation-space error concentrated exactly where the meaning is?).
- **Hypothesis:** Mean `cos_sim` at concept-token positions is **lower** than at filler positions (and `mse` higher) by a significant margin — the AR spends its fidelity budget on common/structural tokens and mangles the rare concept tokens.
- **Method:** From `verbalize_concepts`, partition per-position records into concept positions (those matched to a controlled concept via `_norm_tok` overlap) vs filler positions (the remainder of `__full__`). Compare `cos_sim`/`mse` distributions (paired within example). Use Neuronpedia per-position scores; cross-check on local AR.
- **Metric:** Mean/median `cos_sim` and `mse` for concept vs filler positions; paired-difference effect size (within-example); fraction of examples where concept positions are the least faithful.
- **Feasibility:** high — per-position `cos_sim`/`mse` already captured in `last_faithfulness` and `_pack`; pure analysis.
- **Controls:** Activation-space, per position — independent of whether the matcher judged survival. Match concept vs filler tokens on token frequency/length (filler includes frequent function words by construction; restrict filler to content words to avoid a frequency confound, then report both).
- **Deception link:** If faithfulness is systematically lowest at concept tokens, the AV bottleneck text is least trustworthy precisely where a monitor must read intent.
- **Harness:** `adapters.NeuronpediaNLA._pack`/`last_faithfulness`; new `t25_concept_vs_filler_faithfulness`.
- **Null looks like:** Concept and filler positions have indistinguishable `cos_sim`/`mse` distributions → AR fidelity is uniform across positions; concept tokens are not a fidelity blind spot.

### P076 — Reconstruction error by concept category
- **Family:** H
- **Probes:** Which concept **categories** the AR reconstructs worst in activation space (is the fidelity map category-shaped, and does it line up with the survival-dropout map?).
- **Hypothesis:** Mean `cos_sim` varies by `meta.category`/`meta.attack_tactic` by > 0.1, and the worst-faithfulness categories overlap with the worst-retention categories from Family A/D (the AR mangles the same categories the AV drops).
- **Method:** Group rows by category and ATT&CK tactic; compute `mean_faithfulness` (`cos_sim`) and mean `mse` per group. Cross-tabulate the faithfulness ranking against the retention ranking (Spearman). Bootstrap CIs per category; control-match for frequency/length.
- **Metric:** Per-category mean `cos_sim`/`mse` (ranked); Spearman ρ between faithfulness rank and retention rank across categories; identification of categories that are low-faithfulness but high-retention (suspect: survival via unfaithful reconstruction).
- **Feasibility:** high — group-by over existing `cos_sim`/`mse` fields.
- **Controls:** Activation-space metric, matcher-independent for the faithfulness side; the retention side uses the matcher, so the cross-tab itself tests whether the two agree. Report n per category and freq/length balance.
- **Deception link:** Categories that are simultaneously low-faithfulness and "high-retention" are laundering candidates — the AV/matcher say survived, the activation says the reconstruction is unreliable.
- **Harness:** `core.mean_faithfulness` with category key; new `t26_faithfulness_by_category`.
- **Null looks like:** Flat `cos_sim` across categories (range < 0.1) and ρ ≈ 0 with retention → AR fidelity is category-agnostic and unrelated to the dropout map.

### P077 — AR identity / cross-row leakage (does the AR ignore the text?)
- **Family:** H
- **Probes:** Whether the AR is text-conditional at all, or collapses toward a generic/global-prior activation regardless of input — a degenerate "reconstruct the average" mode that fakes faithfulness.
- **Hypothesis:** Cross-row identity holds: cos(recon_i, orig_i) is **not** meaningfully greater than cos(recon_i, orig_j) for random j≠i. If the diagonal advantage (own vs other) is small, the AR is largely ignoring the AV text and emitting a generic activation.
- **Method:** Local AR. Reconstruct activations for N inputs. Build the N×N cosine matrix between recon_i and orig_j. Compare the diagonal (own) distribution to the off-diagonal (cross) distribution. Also compute cos(recon_i, mean_orig) to test collapse toward the global mean.
- **Metric:** Mean diagonal cos − mean off-diagonal cos (the "identity margin"); fraction of rows where a non-self orig is a closer match than self (rank-1 identity error); cos(recon, global-mean activation).
- **Feasibility:** medium — N² cosines are cheap once activations are cached; needs the local AR + GPU to produce recon activations. On the weak AR (round-trip cos ≈ 0.438) the absolute diagonal is low, so the **margin over off-diagonal** is the load-bearing number.
- **Controls:** Pure activation-space; no matcher. The off-diagonal and global-mean baselines *are* the controls — they bound what "text-ignoring" looks like. Match input lengths so identity is not driven by length alone.
- **Deception link:** A text-ignoring AR makes round-trip faithfulness meaningless as evidence of concept preservation — every "faithful" number would be an artifact of the global prior, not of carrying the input's intent.
- **Harness:** new `experiments/local_gemma_e2b/ar_identity.py`; `t27_ar_cross_row_identity`.
- **Null looks like:** Identity margin ≈ 0 and rank-1 identity error high (own orig is rarely the nearest) → the AR is essentially input-agnostic; faithfulness scores are not measuring concept preservation. (A *healthy* AR shows a large positive margin — that would falsify the degeneracy hypothesis.)

### P078 — Faithfulness calibration: cosine threshold vs true concept survival (probe)
- **Family:** H
- **Probes:** Whether a `cos_sim` threshold is a **calibrated** indicator of true concept survival — i.e., does "faithful enough" (cos above τ) actually mean the concept is present in the bottleneck, as judged by a probe (ground truth), not the matcher?
- **Hypothesis:** There exists a `cos_sim` threshold τ for which P(concept present per probe | cos ≥ τ) is high (> 0.8) and well-separated from P(present | cos < τ); the reliability curve (predicted faithfulness vs probe-measured presence) is monotone but **over-confident** (ECE > 0.1) on the weak AR.
- **Method:** On the local NLA where bottleneck probes exist (`results/local_gemma_e2b`), pair each concept's `cos_sim` with the linear-probe presence label (ground truth at the bottleneck). Build a calibration/reliability curve and ROC of `cos_sim` predicting probe-presence; compute ECE and the best-F1 threshold τ.
- **Metric:** ROC-AUC of `cos_sim` vs probe presence; reliability curve + ECE; chosen τ with precision/recall; comparison to using the AV-text matcher verdict as the predictor instead of `cos_sim`.
- **Feasibility:** medium — needs the local probe labels (already produced for in-distribution concepts) aligned to AR `cos_sim`; CPU once activations + probes are cached.
- **Controls:** This *is* the ground-truth calibration the matcher cannot give (P0 #1): the label is the bottleneck probe, the predictor is the activation-space cosine — the AV text/matcher is excluded from both sides except as an explicit comparison predictor.
- **Deception link:** A monitor needs to know what cosine to trust; if `cos_sim` is uncalibrated/over-confident, a "faithful-looking" verbalization can still have lost the concept at the bottleneck.
- **Harness:** reuse `experiments/local_gemma_e2b/probe_bottleneck.py` outputs; new `t28_faithfulness_calibration` joining probe labels to `cos_sim`.
- **Null looks like:** ROC-AUC ≈ 0.5 and a flat/non-monotone reliability curve → `cos_sim` does not track true bottleneck presence at all; faithfulness cannot be thresholded into a survival decision. (Few probe concepts available — n is small; treat as method demo, per the local README caveats.)

### P079 — AR paraphrase delta (does paraphrasing the AV text change the activation?)
- **Family:** H
- **Probes:** Whether the AR is **paraphrase-invariant** in activation space — does feeding a meaning-preserving paraphrase of the AV text yield (nearly) the same reconstructed activation, or does surface wording move the activation?
- **Hypothesis:** Paraphrasing the AV text moves the reconstructed activation non-trivially (cos(recon_paraphrase, recon_original) < paraphrase-invariance threshold, e.g. < 0.9 on a healthy AR; far lower on the weak AR), and the movement is **larger for attack concepts** than neutral controls — surface form, not meaning, drives part of the reconstruction.
- **Method:** Local AR. For each AV verbalization, generate K meaning-preserving paraphrases (controlled-vocabulary swaps + a paraphrase model, validated to preserve concepts). Reconstruct activations for original and each paraphrase; compute pairwise cos and the cos to the *original host activation*. Compare paraphrase spread for attack vs control concepts.
- **Metric:** Mean cos(recon_para, recon_orig) (paraphrase-invariance score); variance of recon across paraphrases (spread); change in cos-to-true-host-activation under paraphrase; attack-vs-control difference in spread.
- **Feasibility:** medium — K× AR reconstructions on GPU + a paraphrase step; reuse the matcher only to *verify* paraphrases preserve the concept (gate), not to score the result.
- **Controls:** The faithfulness comparison is activation-space (recon vs recon, recon vs host activation) — matcher-free. Control = neutral concepts; if paraphrase spread is identical for attack and control, the effect is generic AR sensitivity, not concept-specific. Verify paraphrases are concept-preserving before inclusion.
- **Deception link:** If the AR is paraphrase-sensitive, an attacker can pick the AV-text surface form (or a monitor's re-phrasing can) that minimizes reconstructed-intent — a steganographic/monitor-gaming lever in activation space.
- **Harness:** new `experiments/local_gemma_e2b/ar_paraphrase.py`; `t29_ar_paraphrase_delta`; paraphrase gate via `matching.Matcher`.
- **Null looks like:** cos(recon_para, recon_orig) ≈ 1 with tiny spread, equal for attack and control → the AR is paraphrase-invariant; reconstructed activation depends on meaning, not wording.

### P080 — Activation-space neighbor analysis: nearer the true concept or a benign neighbor?
- **Family:** H
- **Probes:** For each reconstructed activation, the **nearest concept direction** among a bank of true-concept and benign-neighbor directions — the single decisive activation-space adjudication of whether the round trip preserved the concept or moved it to a neighbor.
- **Hypothesis:** For attack concepts, the AR-reconstructed activation's nearest neighbor in the direction bank is the **benign neighbor** more often than chance (and more often than for neutral controls), and the true-vs-neighbor cosine margin is small/negative — the activation itself sits closer to the laundered concept.
- **Method:** Local AR. Build a direction bank: true-concept and `meta.benign_neighbor` directions (probe-weight or mean-difference vectors), plus distractor directions. For each row, rank the bank by cos to the reconstructed activation; record top-1, the rank of the true direction, and (true cos − neighbor cos). Aggregate over attack vs control.
- **Metric:** Top-1 = benign-neighbor rate; mean rank of the true direction; mean (true cos − neighbor cos) margin; ROC of (margin) separating attack from control; per-concept confusion (which neighbor it lands on).
- **Feasibility:** medium — direction bank + AR recon activations on GPU; on the weak AR (cos ≈ 0.438) absolute cosines are low, so rely on **relative** ranking within the bank, not absolute thresholds.
- **Controls:** The definitive activation-space ground truth for laundering — fully matcher-independent; it directly adjudicates the matcher-dependent laundering claims of Family B (P0 #2) and the AR-drift of P073 with a hard top-1 decision. Norm-match all bank directions; include neutral controls and random directions to calibrate chance top-1.
- **Deception link:** This is the cleanest "where did the concept go" test: if the reconstructed activation's nearest concept is the benign neighbor, the round trip *is* the laundering channel, in the representation a downstream model would actually consume.
- **Harness:** shares the direction bank with P073; new `experiments/local_gemma_e2b/ar_neighbor_rank.py`; `t30_activation_neighbor_rank`.
- **Null looks like:** True direction is top-1 (or ranks above the benign neighbor) at the same rate for attack and control, margin ≥ 0 → the reconstructed activation stays nearest the true concept; no activation-space laundering, and Family B's neighbor effects are matcher artifacts.
