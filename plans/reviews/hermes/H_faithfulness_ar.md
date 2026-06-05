I'll critique each plan for the rudimentary NLA regime, then provide the summary lists.

---

### P071 — Faithfulness correlates with concept survival

**SIGNAL:** On a weak AR (cos ~0.438), the `cos_sim` predictor has restricted dynamic range and high noise; survival labels from the matcher are themselves unreliable when AV text is degenerate (empty/repetitive). The logistic slope will be dominated by sampling variance, and quartile binning with few rows per bin yields unstable retention estimates.

**FAILURE MODE:** If the AV emits near-empty text, `verbalize_concepts` produces few/no `ConceptRow`s — quartiles collapse or are empty. The matcher labels `retained/substituted` on degenerate text are near-random, making the survival label pure noise. Frequency/length matching across quartiles fails when n is tiny.

**FLOOR VERSION:** Skip quartiles; use a single threshold (e.g., `cos_sim > median`) for a 2×2 table (high/low faithfulness × retained/lost) with Fisher's exact test. Pool across all concepts regardless of category to boost n. Replace matcher survival with a probe-based ground truth where available (P078 style). Report the shuffled-activation control correlation to bound chance.

**MISSED FLAW:** The plan treats `cos_sim` as a per-concept property, but on a weak AR the per-position `cos_sim` is dominated by token-frequency artifacts (common tokens reconstruct better). The "concept" label from `_norm_tok` overlap may misalign with the actual concept-bearing positions when AV text is repetitive/empty, conflating token-frequency effects with concept survival.

---

### P072 — Faithfulness-weighted retention vs raw retention

**SIGNAL:** The weighting metric exists in code, but on a rudimentary NLA most rows lack `cos_sim` (fallback `default=1.0`), so the weighted metric silently reverts to raw retention. The rank-inversion count becomes a measure of missing-data pattern, not AR faithfulness.

**FAILURE MODE:** Early-checkpoint AVs produce low-coverage `cos_sim` — the plan's "state coverage" control will show most rows fall back to 1.0, making weighted ≈ raw by construction. Bootstrap CIs on per-category gaps are meaningless when n per category is < 5.

**FLOOR VERSION:** Report `cos_sim` coverage per category first; only compute weighted retention where coverage > 50%. For low-coverage categories, flag as "unmeasurable" rather than imputing 1.0. Use a continuous weight (the `cos_sim` itself) rather than a binary retained/weighted comparison.

**MISSED FLAW:** The plan assumes `faithfulness_weighted_retention` corrects the matcher, but the matcher's errors and the AR's errors are correlated (both worse on rare/attack concepts). Weighting by a noisy `cos_sim` can *increase* variance without reducing bias — the plan doesn't test whether weighted retention correlates better with probe ground truth.

---

### P073 — AR concept drift toward the wrong neighbor (activation space)

**SIGNAL:** This is one of the few plans that *can* work on a weak AR because it uses the **difference** (neighbor cos − true cos) rather than absolute cosines. The relative signal survives even when absolute cos ~0.1–0.4.

**FAILURE MODE:** Building "concept direction vectors" (mean-difference or probe weights) requires enough examples per concept. On a tiny dataset (few examples per attack concept), the direction vectors are noisy, and the drift metric inherits that noise. If the benign neighbor direction is estimated from even fewer examples, the drift sign flips randomly.

**FLOOR VERSION:** Use a *shared* neutral-control direction as the reference instead of per-concept true directions. Compute `cos(recon, neighbor_dir) - cos(recon, neutral_dir)` — this only needs one well-estimated neutral direction. Pool across all attack concepts for a single drift estimate with bootstrap CI. Skip per-concept tables.

**MISSED FLAW:** The plan assumes the "benign neighbor" direction is semantically distinct from the true direction. On a low-dimensional bottleneck, the true and neighbor directions may be highly collinear (cos > 0.8), making the drift metric ill-conditioned — small reconstruction noise flips the winner. The plan doesn't check direction collinearity.

---

### P074 — Round-trip stability: converge or drift?

**SIGNAL:** On a weak AR (cos ~0.438), the first round-trip already destroys most signal; iterating further yields a trajectory dominated by the AR's fixed point (likely a generic "average" activation). The curve will saturate at k=1 or 2, making monotonicity tests underpowered.

**FAILURE MODE:** Early-checkpoint AV emits empty/repetitive text → AR reconstructs a near-constant activation → `cos(a_k, a_0)` drops to noise floor immediately, `cos(a_k, a_{k-1}})` approaches 1 (convergence to the constant), but this "convergence" is degeneracy, not stability. The shuffled-text control will look identical.

**FLOOR VERSION:** Run only k=1,2,3. The key metric is not the curve but the *identity margin*: `cos(a_1, a_0) - cos(shuffled_recon, a_0)`. If this margin ≤ 0, the AR is already input-agnostic at round 1 — stop. Report the shuffled baseline explicitly as the null.

**MISSED FLAW:** The plan treats "convergence to original" vs "drift to generic" as the only two outcomes, but a weak AR can exhibit a third: immediate collapse to a *different* fixed point per input (input-dependent but not the original). The step-size `cos(a_k, a_{k-1}) → 1` would misleadingly suggest "converged" when actually each input goes to its own wrong attractor.

---

### P075 — Per-position faithfulness: concept tokens vs filler tokens

**SIGNAL:** Per-position `cos_sim`/`mse` from Neuronpedia is high-quality, but on a rudimentary NLA the AV may not emit the concept token at all (empty/repetitive text) — then there are *no* concept positions to measure, or they're misidentified via `_norm_tok` overlap on filler.

**FAILURE MODE:** If AV text is "the the the", `_norm_tok` finds no concept overlap → all positions labeled "filler" → concept-vs-filler comparison is empty. If AV text is empty, `last_faithfulness` is empty. The paired test has no pairs.

**FLOOR VERSION:** Fall back to a coarse binary: "concept token emitted in AV text?" (yes/no) vs mean `cos_sim` of *all* positions. Or use the local AR where you can force the concept token into the input (prompt the AV) and measure reconstruction at that forced position.

**MISSED FLAW:** The plan matches concept vs filler on frequency/length but ignores *position-in-sequence*. Early positions reconstruct better on most ARs. If concept tokens systematically appear later (e.g., at end of verbalization), the fidelity gap is positional, not semantic. The plan doesn't control for position index.

---

### P076 — Reconstruction error by concept category

**SIGNAL:** Grouping by category on a weak AR: per-category n is tiny (often 1–3 examples), `cos_sim` variance is huge, and Spearman ρ across categories is dominated by sampling noise. The cross-tab with retention ranking compounds two noisy rankings.

**FAILURE MODE:** Categories with n=1 get a `cos_sim` equal to that single example's noise. Bootstrap CIs are invalid with n<5. The "low-faithfulness but high-retention" cells are artifacts of the matcher labeling empty text as "retained" (default) while `cos_sim` is low.

**FLOOR VERSION:** Collapse to 2–3 super-categories (attack vs neutral vs control) to get n>10 per group. Report only mean `cos_sim` with bootstrap CI; skip rank correlations. Flag categories where n < 5 as "insufficient data" rather than plotting them.

**MISSED FLAW:** The plan assumes category labels are ground truth, but on a rudimentary NLA the *same* concept may be verbalized differently across examples (inconsistent AV), so the category grouping mixes different reconstruction difficulties. The plan doesn't measure within-category variance of `cos_sim`.

---

### P077 — AR identity / cross-row leakage (does the AR ignore the text?)

**SIGNAL:** **Highest-signal plan for weak ARs.** The diagonal-vs-off-diagonal margin is well-defined even at cos ~0.1–0.4. A text-ignoring AR shows margin ≈ 0; a minimally text-conditional AR shows margin > 0. N² cosines are cheap; no matcher, no AV text quality dependence.

**FAILURE MODE:** If all inputs produce nearly identical AV text (repetitive/degenerate), the AR *should* produce similar reconstructions — the off-diagonal cos will be high *because the inputs to the AR are similar*, not because the AR ignores text. The control "match input lengths" doesn't fix this; you need to match *AV text diversity*.

**FLOOR VERSION:** Compute the identity margin on the *AV text embeddings* first (cos(text_i, text_j) matrix). If AV text off-diagonal cos > 0.9, the AR identity test is confounded — report this and skip. Otherwise, the AR margin is interpretable.

**MISSED FLAW:** The plan uses the global mean activation as a collapse baseline, but a weak AR could collapse to *per-input* fixed points (different for each input, but not the original). The diagonal margin would still be positive (recon_i closer to orig_i than to orig_j), falsely suggesting text-conditionality. Need a "recon_i vs recon_j" off-diagonal to detect this.

---

### P078 — Faithfulness calibration: cosine threshold vs true concept survival (probe)

**SIGNAL:** **Conceptually the right approach** (probe ground truth), but on a rudimentary NLA: (a) probe labels only exist for in-distribution concepts (few), (b) the AR `cos_sim` range is narrow, (c) n is too small for reliability curves/ECE (need ~100+ points per bin).

**FAILURE MODE:** With n < 20 probe-labeled concepts, ROC-AUC has huge variance; ECE bins are empty or singletons; the "best-F1 threshold" overfits wildly. The comparison to matcher verdict is comparing two noisy predictors.

**FLOOR VERSION:** Treat as a *method demo only*. Report n, raw pairs (cos_sim, probe_present), and a single threshold at median `cos_sim` with a 2×2 table. Explicitly state: "not calibrated, n too small." Skip ECE, reliability curve, ROC.

**MISSED FLAW:** The probe measures *bottleneck* presence, but `cos_sim` measures *AR reconstruction* faithfulness. On a weak NLA, the bottleneck may retain the concept while the AR fails to reconstruct it (or vice versa). The calibration curve conflates AR quality with bottleneck fidelity — they're different failure modes.

---

### P079 — AR paraphrase delta (does paraphrasing the AV text change the activation?)

**SIGNAL:** On a weak AR, the paraphrase-invariance score `cos(recon_para, recon_orig)` will be low *even for neutral concepts* because the AR is noisy. The attack-vs-control difference in spread requires many paraphrases per concept (K≥5) and many concepts — infeasible on tiny data.

**FAILURE MODE:** Generating "meaning-preserving paraphrases" of degenerate AV text (empty, "the the the") is ill-posed. The matcher gate (verify paraphrase preserves concept) fails when the original text has no concept to preserve. K× AR cost on GPU is prohibitive for rudimentary NLAs where you'd want many concepts × many paraphrases.

**FLOOR VERSION:** Skip paraphrase generation. Instead: take the *same* AV text, add trivial perturbations (synonym swap, punctuation) that *should not* change meaning, and measure recon delta. Or: feed the AR the *concept name only* vs the full AV text, measure cos difference — this tests whether the AR uses context beyond the concept token.

**MISSED FLAW:** The plan assumes paraphrase sensitivity is a property of the AR, but on a weak AR the *AV text itself* may not contain the concept (degenerate verbalization). Paraphrasing empty text yields empty text; the AR reconstruction doesn't change because the input didn't contain the signal. The measured "invariance" is an artifact of the AV, not the AR.

---

### P080 — Activation-space neighbor analysis: nearest concept or benign neighbor?

**SIGNAL:** Like P073, this uses *relative* ranking within a direction bank, which survives low absolute cosines. The top-1 decision is discrete and robust to noise if the margin is large.

**FAILURE MODE:** Building the direction bank (true + neighbor + distractor directions) requires enough examples per direction. On a rudimentary NLA with few examples per concept, the direction vectors are noisy, and the nearest-neighbor decision becomes a noise coin-flip. Including "random directions" as controls helps calibrate chance, but with few directions total, the top-1 rate has high variance.

**FLOOR VERSION:** Use only 3 directions per concept: true, benign neighbor, and *one* shared neutral direction (same for all concepts). Pool across all attack concepts for a single top-1 rate with binomial CI. Skip per-concept confusion tables. Report the neutral-direction top-1 rate as the chance baseline.

**MISSED FLAW:** The plan assumes the direction bank spans the relevant concept space. On a low-dimensional bottleneck (e.g., 64-d), many concept directions are linearly dependent — the "nearest neighbor" is geometrically forced, not semantically meaningful. The plan doesn't measure the condition number of the direction bank or the angular separation between true and neighbor directions.

---

### Skip on rudimentary NLAs
- **P074** (Round-trip stability) — saturates at k=1; degeneracy mimics convergence
- **P079** (AR paraphrase delta) — infeasible paraphrase generation on degenerate AV text; K× GPU cost
- **P075** (Per-position faithfulness) — concept positions often absent/misidentified when AV is empty/repetitive
- **P078** (Faithfulness calibration) — n too small for calibration curves; probe labels only for in-distribution concepts

### Highest-value for the rudimentary regime
- **P077** (AR identity / cross-row leakage) — works at any AR fidelity; directly tests if AR is text-conditional at all; cheap N² cosines
- **P073** (AR neighbor drift) — relative drift metric survives low absolute cos; activation-space laundering localization
- **P080** (Activation-space neighbor rank) — discrete top-1 decision robust to noise; adjudicates laundering in representation space
- **P071** (Faithfulness→survival, floor version) — 2×2 table with pooled concepts + shuffled control gives a usable signal if coverage > 0
- **P076** (Reconstruction error by super-category) — collapse to attack/neutral/control gives n>10; reveals if AR mangles attack concepts worse
session_id: 20260605_123739_6f3a99

