---

### P011 — Minimal-pair separation
**Signal:** Floors at maximum collapse on rudimentary NLAs. Empty/degenerate AV text makes both members of every pair (including polarity-neutral controls) produce identical output → collapse rate ≈ 1.0 for all contrast types, gap ≈ 0. The metric saturates and loses discriminative power.  
**Failure mode:** AV emits empty string or repetitive tokens ("the the the") for both members → EnsembleMatcher sees identical text → collapse=1 everywhere. The distinguishing bit was never verbalized, so you cannot tell if the bottleneck dropped it or the AV lacked expressivity. Probe arbitration only helps if bottleneck activations exist and linear probes work — but with near-random AR (cosine 0.1–0.4) and few examples, probes overfit or sit at chance.  
**Floor version:** Skip generative AV entirely. Feed the same activations to (a) a retrieval-kNN verbalizer (nearest cached snippet) and (b) a linear probe for the distinguishing feature (negation, quantity, affix). If probe reads the bit but retrieval-kNN still collapses → AV capacity limit. If probe also fails → bottleneck never had it. Pool across many pairs per contrast type to beat sampling noise.  
**Missed flaw:** The plan assumes the AV produces *some* differentiable text. On degenerate AV, "collapse" conflates NLA bottleneck failure with AV expressivity floor. The polarity-neutral control ("blue car"/"red car") also collapses to empty, so gap ≈ 0 even if negation/quantity bits ARE preserved in the bottleneck — a false negative for NLA capability.

### P012 — Hypernym drift
**Signal:** Undefined or noise-dominated on rudimentary NLAs. Empty AV yields no substitutions to classify → substitution set empty → metric undefined. Even with retrieval-only AV, if the nearest cached snippet is generic ("weapon"), you cannot distinguish hypernym drift from retriever coverage gaps.  
**Failure mode:** No matched_term → no direction to measure. Low-capacity NLAs may only know hypernyms (training data skew) — not drift, just coverage gap. WordNet chains for technical terms ("Mimikatz", "Cobalt Strike") don't exist; the taxonomy backend fails silently, defaulting to lexical containment which is wrong for brand→category.  
**Floor version:** Go probe-only. Train two probes on the bottleneck: specific concept ("sniper rifle") and hypernym ("weapon"). If specific-probe AUC high + hypernym-probe AUC high → bottleneck has both. If specific-probe at baseline + hypernym-probe high → genuine hypernym drift at bottleneck. Use rank correlation (specific probe rank vs. hypernym probe rank) instead of absolute AUC to handle low-SNR probes. Skip AV entirely.  
**Missed flaw:** The plan treats "substitution" as observed in AV text, but on a weak NLA the substitution may happen in the AR (reconstruction), not the AV. The probe arbitration (two-probe) is the only valid signal, yet the plan still centers the AV-level direction histogram as the primary metric.

### P013 — Co-hyponym confusion
**Signal:** Saturates trivially on rudimentary NLAs. Empty/degenerate AV → all siblings produce identical text → cross-match rate = 1 for everything (saturation). Retrieval-kNN returns the same generic parent snippet for all siblings → false "confusion" that's just retriever coverage.  
**Failure mode:** Empty AV → cross-match = 1 trivially. Degenerate AV ("malware malware malware") → matcher sees identical text for all siblings. The frequency-winner correlation becomes meaningless when all outputs are identical. Multi-class probe over {A,B,…,P} on low-dim bottleneck with few examples → severe overfitting, chance-level AUC for all classes.  
**Floor version:** Probe-only pairwise discriminability. For each sibling pair (A,B), train a binary probe A-vs-B on the bottleneck. If AUC ≈ 0.5 for ALL pairs but parent-vs-other AUC high → bottleneck has parent but not children (genuine co-hyponym confusion at bottleneck). If A-vs-B AUC high but AV still collapses → AV/verbalizer limit. Report probe AUC matrix, not AV confusion matrix.  
**Missed flaw:** The "distractor-context" control assumes context-dependent disambiguation. On a tiny model with low-dim bottleneck, there's no capacity for context-dependence either way — both conditions floor. Technical sibling sets (ATT&CK techniques) have no clean WordNet siblings; curation is manual and brittle.

### P014 — Antonym / polarity collapse
**Signal:** Most robust of the family because it uses LEXICAL polarity markers (explicit word lists), not matcher cosine. Even short AV ("increase"/"decrease" vs "change") can be checked for sign. But if AV is empty or "the the the", sign-preservation = 0 for both members → collapse rate = 1 for all valence types including controls → gap ≈ 0 (false negative).  
**Failure mode:** Empty/degenerate AV → no polarity markers to detect → everything looks collapsed. Synonym controls also collapse → gap vanishes. Probe arbitration (sign probe) is the only salvage, but sign probe on low-dim bottleneck with few examples: high variance, overfitting risk.  
**Floor version:** Sign-probe-only. Train binary probe (+ vs −) on bottleneck for each antonym pair. Report mean AUC across pairs vs. shuffled-label null. Skip AV entirely. If sign-probe AUC > 0.6 consistently → bottleneck encodes polarity. If ≈ 0.5 → genuine polarity collapse at bottleneck.  
**Missed flaw:** The "topic-match" (both map to same dimension term) still uses EnsembleMatcher. On weak NLA, both antonyms may map to empty string → matcher says "same topic" (empty=empty) → topic-match=1 artifactually. Carrier sentences may not activate the concept strongly enough in a tiny model.

### P015 — Near-synonym winner-take-all
**Signal:** Floors at trivial winner-take-all on rudimentary NLAs. All cluster members → same empty/degenerate text → winner-take-all share = 1, pull-to-frequent lift undefined (permutation null also 1). The frequency correlation measures AV expressivity floor, not NLA bottleneck behavior.  
**Failure mode:** Empty AV → all self-retention = 0, all winner-share = 1. The "probe arbitration" (per-member discriminating probe) requires 30 clusters × ~4 members = 120 probes on low-dim bottleneck → multiple testing + overfitting dominates.  
**Floor version:** Probe-only discriminability matrix. For each cluster, train all-pairs probes on bottleneck. If within-cluster pairwise AUC ≈ 0.5 but cluster-vs-outside AUC high → bottleneck has cluster but not members (genuine winner-take-all at bottleneck). If pairwise AUC high → AV/verbalizer frequency prior. Report probe AUC heatmap per cluster.  
**Missed flaw:** Frequency bands assume general-domain wordfreq (or controls.estimate_freq_band on NLA training data, which may not exist). For technical jargon ("surreptitiously" vs "covertly" in cyber context), general wordfreq is wrong. Clusters curated for general English may not match the NLA's actual semantic neighborhoods.

### P016 — Metaphor vs. literal disambiguation
**Signal:** Expected null on rudimentary NLAs. Both senses → same empty/degenerate text → sense-collapse = 1 for all pairs including controls → gap ≈ 0. Even retrieval-kNN ignores context (same trigger word) → same snippet → false collapse.  
**Failure mode:** Empty AV → no sense markers. Retrieval-kNN returns same snippet for both contexts. Sense probe on bottleneck: the SAME trigger token activation for both senses → probe can't distinguish (it's the same activation!). The plan says "probe on trigger-token activation" but if the trigger token is identical, the activation IS identical (or nearly so in a tiny model) → probe AUC ≈ 0.5 by construction, not because sense is lost.  
**Floor version:** This fundamentally cannot work on weak NLAs without strong contextual modulation in the bottleneck. Only viable if bottleneck is context-sensitive (e.g., full transformer residual stream, not a compressed bottleneck). Otherwise skip.  
**Missed flaw (fatal):** The probe target is ill-posed. "Probe on the trigger-token activation" assumes the trigger token's activation DIFFERS by context. In a low-dim bottleneck or tiny model, it may not — the context window is too short or capacity too low to modulate the token representation. The plan would report "bottleneck doesn't encode sense" when actually the probe target is identical for both conditions.

### P017 — Brand → category laundering
**Signal:** Partially salvageable via freq-matched controls, but floors on very weak NLAs. Brands are rare → likely OOV or empty AV. Brand→category rate = 1, but matched common-noun controls (also rare) ALSO collapse to hypernyms at rate 1 → gap ≈ 0. The probe arbitration is the real signal.  
**Failure mode:** Empty AV → both rates = 1. Brand-probe vs category-probe: if brand concepts never appeared in training data, brand-probe sits at chance because the concept was never learned, not because of laundering. Multi-word brands ("Cobalt Strike") tokenize to pieces; bottleneck may not bind them as a unit.  
**Floor version:** Probe-only brand-vs-category. Curate brands that DO appear in training data (check tokenization). Train probes on bottleneck. Report brand-probe AUC minus category-probe AUC per brand. Positive = brand retained; negative = category-laundered. Skip AV entirely.  
**Missed flaw:** The plan reuses `attack_concepts.py` schema but many ATT&CK tools are multi-word — tokenizer splits them, bottleneck may not bind them. "Generic_category" annotation is manual and may not match the NLA's learned categories. Frequency matching on rare brands is noisy (few occurrences).

### P018 — Quantity rounding / magnitude flattening
**Signal:** STRONGEST for rudimentary NLAs because it BYPASSES the matcher entirely — numeric extractor parses numbers directly from AV text. Even short/degenerate AV may contain a number ("4096" → "4096" or "4000" or "large"). Empty AV → extraction returns None → "dropped" tier, still a valid data point.  
**Failure mode:** Empty AV → all quantities = "dropped", categorical controls = "retained" → retention gap maximal but uninformative (AV expressivity floor, not numeric-specific). Anchor-rounding bias test needs SOME numbers to survive. Probe arbitration (magnitude regression probe) is clean — works even if AV drops all numbers.  
**Floor version:** Two-track: (1) Numeric extractor on AV text (works down to very short text). (2) Magnitude regression probe on bottleneck (works even if AV empty). Compare: if probe R² high but AV extraction = dropped → AV verbalizer loss. If probe R² also low → bottleneck never had magnitude. Report both.  
**Missed flaw:** The numeric extractor assumes AV verbalizes quantities AS NUMBERS. A weak NLA may verbalize "4096 bytes" as "a large amount" (no number) → extractor returns None → scored "dropped" when magnitude survived qualitatively. The "direction" tier needs a lexical magnitude-word detector ("large", "small", "huge") which is not specified.

### P019 — Temporal adjacency / order
**Signal:** Weakest for rudimentary NLAs. Requires AV to encode order markers ("then", "before") AND preserve mention order. Weak NLA: events may survive but order markers dropped → AV texts identical → order-collapse = 1. Content-different controls also collapse → gap ≈ 0.  
**Failure mode:** Empty/degenerate AV → no order markers, no mention order → order-collapse = 1 for everything. "Precedence_marker_survival" lexical check finds nothing. Probe on bottleneck: if bottleneck is a compressed summary (not sequence-aware), order info is genuinely absent → probe at baseline, but this is an ARCHITECTURAL property, not laundering. The plan conflates architectural limitation with laundering.  
**Floor version:** Only run the order probe on the bottleneck (skip AV). If order-probe AUC > 0.6 → bottleneck encodes order. If ≈ 0.5 → architecture doesn't support order (report as architectural bound). Don't run the AV order-collapse metric on weak NLAs — it's pure noise.  
**Missed flaw:** The plan assumes "both events should be retained in both members" — but on a weak NLA, one or both events may be dropped entirely → order-collapse confounded with event-drop. Content-different controls don't isolate this: if both events drop in order-swapped pair but only one drops in content-different pair, "indistinguishable" rate differs for the wrong reason.

### P020 — Spatial / relational adjacency
**Signal:** Explicitly expected-null on rudimentary NLAs. The plan admits "run it as a CONTROL that bounds how much relational structure the bottleneck CAN carry" — it yields NO positive signal on weak NLAs.  
**Failure mode:** Empty AV → entities may survive (matcher confirms) but relation extractor finds nothing → role-collapse = 1. Content-different controls also role-collapse = 1 → gap ≈ 0. Relation probe on bottleneck: if baseline, the finding is "relation absent everywhere" — the expected result, not a discovery.  
**Floor version:** Run ONLY the relation probe on bottleneck as an architectural probe. Report: "bottleneck does/does not linearly encode relation R". This is a valid capability bound measurement, not a laundering test. Don't run the AV-level role-collapse metric.  
**Missed flaw:** The plan uses EnsembleMatcher to confirm both entities present. If entities are dropped (weak NLA), matcher says "not present" → relation-collapse metric becomes undefined (conditioned on empty set). The probe arbitration is the only well-defined part, and it's expected to be null.

---

### Skip on rudimentary NLAs
- **P016** (Metaphor vs. literal) — fatal probe confound; trigger-token activation identical by construction
- **P019** (Temporal adjacency) — requires relational markers in short text; architectural order-sensitivity conflated with laundering
- **P020** (Spatial/relational) — explicitly expected-null; only valid as architectural probe, not laundering test

### Highest-value for the rudimentary regime
- **P018** (Quantity/magnitude) — bypasses matcher with numeric extractor; two-track (AV + probe) works down to empty AV
- **P014** (Antonym/polarity via sign-probe) — lexical sign-check partially survives short AV; sign-probe-only floor version is clean
- **P011** (Minimal-pair via probe-only) — retrieval-kNN + probe discriminability isolates AV vs. bottleneck; pooling beats noise
- **P017** (Brand→category via probe-only) — freq-matched controls + probe diff (brand-probe − category-probe) is well-defined
- **P015** (Synonym clusters via probe discriminability matrix) — all-pairs probe AUC heatmap reveals bottleneck granularity without AV
session_id: 20260605_120325_2a6864

