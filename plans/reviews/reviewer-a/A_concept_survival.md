---

## Per-Plan Critique for Rudimentary NLAs

### P001 — Retention-vs-frequency dropout law
**SIGNAL:** Near-zero. Degenerate AV → matcher always 0 → end-to-end retention flat at 0 for all frequencies. Degenerate bottleneck → probe_auc flat at chance (0.5). Both logistic slopes β ≈ 0; the "AV-attributable fraction" (1 − β_probe/β_end2end) becomes 0/0 undefined.  
**FAILURE MODE:** Bootstrap CIs on β collapse to a point when variance is zero (pure floor), falsely implying precision. The continuous fit requires retention *variance* across frequencies — if everything drops, there's no curve to fit.  
**FLOOR VERSION:** Skip end-to-end entirely. Probe-only: train a regularized linear probe on bottleneck activations, measure probe_auc vs `wordfreq` using **Spearman rank correlation** (robust to floor/ceiling). Compare to **shuffled-activation control** (shuffle concept labels across activations) — if rank correlation persists, it's a probe artifact. If using a retrieval-kNN verbalizer, end-to-end may regain signal.  
**MISSED FLAW:** Assumes matcher ensemble achieves >2/3 agreement on *some* concepts. On rudimentary NLA, agreement may be 0 everywhere. The probe assumes linear readability; if the bottleneck encodes concepts non-linearly or the probe overfits (few examples per freq bin), probe_auc is noise masquerading as signal.

---

### P002 — Category-selective dropout
**SIGNAL:** Near-zero end-to-end (all categories 0 retention, matched controls also 0 → Δ = 0 with tight but meaningless CIs). Probe side: all categories at chance AUC ≈ 0.5, controls same → no detectable deficit.  
**FAILURE MODE:** Total collapse eliminates inter-category variance. Per-category probe AUCs have huge variance (≈20 concepts each) → CIs always overlap even if true differences exist. The ">2/3 matcher agreement" gate may fail for all categories on degenerate text, making the "deficit" unmeasurable.  
**FLOOR VERSION:** Probe-only, **pool across categories** for probe training (single multi-class probe predicting category), then inspect per-category coefficients/weights. Or: retrieval-kNN verbalizer + matcher, then test if matcher scores differ by category *after* controlling for frequency. Coarsen metric: "probe weight on category indicator" instead of per-category AUC.  
**MISSED FLAW:** The plan treats 20 concepts/category as sufficient for stable AUC estimates. With a weak signal, the effective n is far lower. Also, "technical/security terms" as a candidate category may systematically differ in tokenization (subword splits) from controls, confounding frequency matching.

---

### P003 — Blind-spot stability
**SIGNAL:** End-to-end: degenerate AV → survival = 0 for all concepts/contexts → within-concept std = 0 identically → **all concepts falsely flagged as "stable blind spots."** Probe side: probe_auc variance dominated by estimation noise (finite eval contexts), not true context sensitivity.  
**FAILURE MODE:** The "stable cluster" detection is statistically underpowered: with 15 contexts, the sampling variance of std(p̂) is ~0.1 even if true p=0.5. Probe confirmation assumes the probe is reliable — but on a weak NLA, the probe may memorize few training examples per context, creating spuriously high within-concept AUC variance.  
**FLOOR VERSION:** Probe-only with **≥50 contexts/concept**, strong probe regularization. Compare within-concept std to **shuffled-context null** (permute context labels within each concept). Use rank correlation: rank concepts by mean probe_auc vs std — true stable blind spots should be low-mean/low-std; test this correlation against permutation null.  
**MISSED FLAW:** The plan's threshold "std < 0.1" is arbitrary and not calibrated to the estimator's sampling variance. Also, "probe confirmation" uses the same probe that generated the variance estimate — circular if probe noise drives both.

---

### P004 — BOS / position-0 verbalization noise
**SIGNAL:** End-to-end: flat 0 at all positions (degenerate AV). **Probe side is the only viable signal** — BOS attention sink may make position-0 activations systematically different even in a weak bottleneck, so probe_auc could vary by position.  
**FAILURE MODE:** End-to-end gap decomposition requires both sides to have variance; if end-to-end is flat, gap = probe_effect − 0, but probe_effect may be noise (only ~30 concepts × 3 positions = 90 examples for probe with position interactions). The "3 paired examples per concept" is severely underpowered for probe interaction terms.  
**FLOOR VERSION:** **Probe-only**, skip end-to-end/matcher. Increase to ≥100 concepts, 5+ positions (BOS, early, mid, late, EOS). Train **one probe with position as a categorical feature** (not separate probes per position) to share strength. Control: shuffle position labels. If using retrieval-kNN verbalizer, end-to-end may work.  
**MISSED FLAW:** Input token position ≠ AV output token position. The matcher matches on AV *text*, which re-tokenizes and may not preserve input positions. The probe uses L23 activations (input-aligned), so end-to-end vs probe comparison has an **alignment confound**. The plan doesn't account for this.

---

### P005 — Compression / length sensitivity
**SIGNAL:** End-to-end: AV truncation (fixed output length) → retention drops with input length, but this is an **AV artifact**, not bottleneck compression. Probe side: if bottleneck is weak, probe_auc may drop with length (target activation diluted) OR stay flat at chance — either way, faithfulness (cos_sim) is ~0.1–0.4 flat noise.  
**FAILURE MODE:** The probe-vs-AV gap is supposed to separate compression from truncation, but if probe_auc is at chance for all lengths, gap = 0. Faithfulness weighting adds noise (cos_sim ≈ 0.2 ± 0.1) without signal. "Filler is neutral controls" — but filler text may contain substrings matching the target (false positives), causing retention to *increase* with length spuriously.  
**FLOOR VERSION:** Probe-only: train probe on **target-concept positions only** (not full sequence) at each length band. Test if probe weight on target position decays with length. Use 2 bins (short/long) with ≥100 examples each, Mann-Whitney on probe_auc. Skip faithfulness weighting. Retrieval-kNN verbalizer avoids truncation artifact.  
**MISSED FLAW:** "Length-matched" is ambiguous: fixed absolute target position vs fixed relative position? The bottleneck sees different absolute positions → different activation patterns. The probe uses L23 (position-aware), so this matters. Also, matcher false-positive rate likely increases with input length (more text to match against), confounding the retention metric.

---

### P006 — Concept crowding
**SIGNAL:** End-to-end: flat 0 at all N (degenerate AV). Probe side: probe_auc ≈ 0.5 at all N (weak bottleneck). "Capacity k" where retention crosses control floor is undefined (starts at floor).  
**FAILURE MODE:** Training separate probe per concept per N → severe overfitting (few examples per concept-N cell). "Concept identities randomized across slots" with ~25 concepts and N∈{1,2,4,8} yields few unique combinations → pseudoreplication. Length-matching via neutral filler adds filler activations to the bottleneck for low-N conditions, confounding "concept count" with "total activation density."  
**FLOOR VERSION:** Probe-only: **single multi-label probe** predicting all concepts simultaneously, measure per-concept AUC vs N. Or synthetic control: inject known orthogonal concept vectors into noise, test linear readout degradation with N (ground truth). Coarsen: 2 conditions (N=1 vs N=4) with ≥200 examples each, paired test. Retrieval-kNN verbalizer may restore end-to-end signal.  
**MISSED FLAW:** The "neutral filler" for length-matching still produces activations that occupy the bottleneck. Higher N = less filler = less distractor activation. The effect attributed to "concept crowding" could be "filler distraction." The probe sees the shared activation including filler, not isolated concept slots.

---

### P007 — Token-span vs survival
**SIGNAL:** End-to-end: flat 0 (degenerate AV). **Probe side has signal potential**: span-pooling (averaging multiple noisy activations) may boost probe SNR even on a weak bottleneck, so probe_auc could rise with span while end-to-end is flat.  
**FAILURE MODE:** "Partial correlation retention~span | (freq, char_band)" requires retention variance — undefined at floor. Probe comparison (single-position vs span-pooled) assumes all span positions carry equal signal; on a weak NLA, only the first token may encode the concept (like a [CLS] token), so pooling adds noise. Few concepts per span bucket (stratified + matched) → high probe variance.  
**FLOOR VERSION:** Probe-only, **skip end-to-end**. Use span-pooled probe as primary: average activations across concept's token span, train probe on pooled vector. Test Spearman correlation of probe_auc with span. Control: shuffle span labels. Increase n to ≥100 concepts spanning 1–5+ tokens, matched on freq.  
**MISSED FLAW:** Matcher ensemble agreement (>2/3) is **higher for multi-token concepts purely due to more n-gram matching opportunities**, not better survival. The plan treats this as a control but it's a confound. Also, span-pooling assumes uniform informativeness across span positions — untested and likely false for weak NLAs.

---

### P008 — POS/modality retention
**SIGNAL:** End-to-end: all classes 0 (degenerate AV). Probe side: single multi-class probe predicting POS from activation *could* work (pools data across 100 concepts). Numeric-presence probe (binary) vs exact-value probe (multi-class) is viable even on weak bottleneck.  
**FAILURE MODE:** "Noun−number Δ > 0.15" threshold unrealistic for weak NLA — differences will be smaller, CIs overlap. The carrier "The report mentions {}." forces all POS into identical syntactic slot, but numbers often appear in distinct contexts (quantifiers, measurements) that affect retention independently. Regex digit labeling labels *input*, not activation — probe could learn spurious activation-norm correlates.  
**FLOOR VERSION:** Probe-only: **single probe predicting POS class** (4-way) from activation, test accuracy > chance via permutation. For numbers: probe for "digit present" (binary) vs "exact value" (regression/classification), compare. Skip end-to-end unless retrieval-kNN. Use rank test (Mann-Whitney) on per-concept probe_auc by POS.  
**MISSED FLAW:** The "exact-vs-approximate" breakdown ("does '47' survive as 'around 50'?") requires the AV to output approximate numbers — impossible with degenerate AV. The probe tests bottleneck encoding, but the metric conflates bottleneck fidelity with AV generation capability. Also, POS classes differ systematically in tokenization (numbers often single-token; nouns often multi-token) — token-span confound not controlled.

---

### P009 — Redundancy / repetition boosts survival
**SIGNAL:** End-to-end: flat 0 (degenerate AV). **Probe side most promising**: pooling across multiple concept occurrences *should* boost probe SNR even on weak bottleneck (law of large numbers for noisy activations).  
**FAILURE MODE:** "Borderline concepts (≈50% baseline from P001)" — on rudimentary NLA, P001 baseline is 0%, not 50%. **No borderline concepts exist.** Length-padding with neutral filler means 1× condition has *more filler tokens* → more distractor activations in bottleneck. The probe sees full shared activation, so higher repetition = less filler = less distraction. This confounds "repetition benefit" with "filler reduction." Synonym-restatement assumes synonyms map to similar activations — false on weak NLA.  
**FLOOR VERSION:** Probe-only: for each concept, **pool ONLY its occurrence-token activations** (exclude filler), train probe on pooled vector. Test if probe_auc increases with occurrence count. 2 conditions (1× vs 3×) with ≥100 concepts. Control: shuffle repetition counts. Synthetic control: inject known vector K times into noise, test linear readout.  
**MISSED FLAW:** The length-matching via filler creates the filler-distraction confound. The probe must be trained on **concept-token activations only**, not the full sequence. Also, "synonym-restatement" assumes semantic equivalence in activation space — unvalidated and likely false for weak NLAs.

---

### P010 — Salience-driven dropout
**SIGNAL:** End-to-end: both conditions 0 (degenerate AV). Probe side: secondary concept probe_auc *might* be lower in dominant condition if bottleneck genuinely suppresses it — but the confound is severe (see below).  
**FAILURE MODE:** "Matched on concept count" — dominant topic contributes 1 concept but is **repeated/elaborated** (high coherence, low entropy); flat filler is diverse neutral concepts. The activation statistics (norm, PCs, sparsity) differ globally between conditions. The probe sees full activation and may respond to global differences, not specific secondary-concept suppression. Faithfulness weighting (cos_sim) is noise (~0.2) on near-random AR. "Insertion rate/hallucination" cross-ref t07 is meaningless with degenerate AV.  
**FLOOR VERSION:** Probe-only: train probe on **secondary-concept-token activations only**, compare weights/AUC flat vs dominant. **Synthetic control**: inject fixed "secondary vector" into activations with/without "dominant vector", test linear readout — isolates capacity from NLA-specific effects. 2 conditions, ≥100 secondary concepts, paired rank test. Skip end-to-end unless retrieval-kNN.  
**MISSED FLAW:** The "dominant topic = 1 controlled concept" matching is **invalid**. A repeated, elaborated dominant topic occupies far more semantic/activation space than 1 neutral concept. The probe cannot distinguish "salience competition" from "global activation distribution shift." Also, t07 hallucination check assumes generative AV — irrelevant for retrieval-kNN or degenerate AV.

---

---

## Skip on Rudimentary NLAs
*Expected pure-null / not worth running until NLA is stronger (generative AV + non-random bottleneck + probe linearity).*

- **P001** — Retention-vs-frequency law (continuous fit requires variance; both sides floor)
- **P002** — Category-selective dropout (all categories at floor; per-category n too small)
- **P003** — Blind-spot stability (degenerate AV → all concepts falsely "stable"; probe variance = noise)
- **P005** — Compression/length sensitivity (AV truncation confound; faithfulness noise; matcher false positives)
- **P006** — Concept crowding (filler-distraction confound; pseudoreplication; both sides floor)
- **P010** — Salience-driven dropout (invalid "matched concept count"; global activation confound)

---

## Highest-Value for the Rudimentary Regime
*Probe-only or retrieval-kNN variants that yield signal on weak/tiny NLAs. Run these first.*

1. **P004 (Probe-only: BOS/position effect)** — BOS attention sink is a structural property of the transformer, likely present even in tiny/early models. Probe with position feature is well-powered if n ↑.
2. **P007 (Probe-only: Token-span pooling benefit)** — Span-pooling is a pure SNR test; should work even on random bottleneck (averaging reduces noise). Synthetic control validates.
3. **P008 (Probe-only: POS/modality linear separability)** — Single multi-class probe pools data; numeric-presence vs exact-value is a clean bottleneck-capacity test.
4. **P009 (Probe-only: Repetition pooling benefit)** — Law of large numbers for noisy activations; synthetic control (K injections) gives ground truth. Must use concept-token-only pooling, not full-sequence.
5. **P001/P002/P005/P006 (Probe-only frequency/length/category/crowding slopes)** — All reduce to: "does probe_auc vary with X?" Use rank correlation + shuffled controls. Skip end-to-end entirely until AV is non-degenerate.
session_id: 20260605_115654_fa12d3

