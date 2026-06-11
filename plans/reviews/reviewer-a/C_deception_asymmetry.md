# Reviewer A critique — C_deception_asymmetry (run in two 5-plan halves to avoid the output-length limit)


session_id: 20260605_124926_47d8b7
**P021 — False-belief (Sally-Anne)**
SIGNAL: Probe side floors at baseline on any rudimentary NLA (explicitly labeled "expected-null control"); matcher side collapses to 0 when AV emits near-empty/degenerate text since the false-belief *relation* requires coherent propositional structure that weak NLAs cannot reconstruct. FAILURE MODE: Linear probe on low-dim/noisy bottleneck with few examples → severe overfitting or chance AUC; matcher returns 0/undefined on degenerate AV output; variance dominated by sampling noise. FLOOR VERSION: Drop probe entirely; coarsen matcher to content-noun presence (Sally/ball/basket) via simple token lookup; pool across many minipairs; add shuffled-activation control. MISSED FLAW: Minipair matching on freq/length ignores syntactic complexity — "Sally thinks that..." embedding depth differs from true-belief control, so a weak NLA dropping complex syntax masquerades as false-belief dropout; linear probe ≠ ground truth for relation presence.

**P022 — Who-said-what attribution**
SIGNAL: Probe side floors (relational binding across two entities is "frontier"); matcher side yields near-0 attribution accuracy on degenerate AV output — binding requires two distinct speaker-claim pairs in coherent text, which rudimentary NLAs cannot produce. FAILURE MODE: Probe overfits on tiny n with noisy bottleneck; matcher spuriously "finds" names near claims by chance on short text, inflating attribution accuracy falsely; swap-confusion undefined when output lacks both speakers/claims. FLOOR VERSION: Skip probe; coarsen matcher to independent speaker-name presence + claim-content presence (two binary checks, not binding); retrieval-kNN baseline for snippet-level retention; shuffled-speaker control. MISSED FLAW: Swapped-attribution counterfactual assumes perfect content matching, but speaker-specific style/claim-type differences (length, concreteness) create systematic retention confounds a weak NLA amplifies; "contested_rate" doesn't fix this.

**P023 — Reported-vs-asserted claim-status**
SIGNAL: Partial — status markers ("reportedly", "allegedly") are lexical tokens that may survive even weak/retrieval AVs; matcher can detect marker presence/absence; directional upgrade (reported→asserted) testable if *any* output exists. FAILURE MODE: Near-empty AV output → 0 markers for all three conditions → confusion matrix degenerate; multiclass probe (3-way) on tiny bottleneck → massive overfitting; "below-floor" check fails because *everything* drops on rudimentary NLA, conflating status dropout with general dropout. FLOOR VERSION: Coarsen matcher to binary marker-present per condition; skip probe; use only high-frequency markers; pool across propositions; retrieval-kNN + shuffled control. MISSED FLAW: Frames aren't length-matched ("Sources say X" vs "X") — syntax dropout masquerades as status flattening; symmetric modifier-dropout (reported→bare, alleged→bare) looks like "upgrade" but is just frequency-biased word dropout; plan doesn't disentangle.

**P024 — Hedged-vs-certain modality**
SIGNAL: Strongest among P021–P025 for rudimentary NLAs — single high-frequency modal tokens ("might", "definitely") often survive weak AVs; binary marker survival gives interpretable signal; directional drift (hedged→certain) measurable if output non-empty. FAILURE MODE: Degenerate AV → no markers → modality_retention=0 for both, certainty_drift undefined; probe side frontier; frequency confound ("might" ≫ "conceivably" retention on weak NLA); "below-floor" content dropout makes modality moot. FLOOR VERSION: Coarsen to high-frequency marker presence/absence only; skip probe; pool heavily; retrieval-kNN baseline. MISSED FLAW: Marker swap changes syntactic frame ("might be" vs "is") — weak NLA sensitivity to well-formedness confounds modality with syntax; certainty_drift requires continuous score, undefined on empty output (no fallback specified); empirically indistinguishable from P023 modifier-dropout on weak NLAs.

**P025 — First-vs-third-person knowledge-boundary**
SIGNAL: Person (pronoun) cell may give signal (pronoun tokens survive); polarity/ignorance cell ("doesn't know") floors hard — negation + epistemic relation is "frontier" and "expected hard null". FAILURE MODE: Two binary probes on low-dim bottleneck → overfitting; negation_survival matcher notoriously fragile (misses "not/n't"); AR noise → probe meaningless; content dropout makes ignorance cell uninterpretable per plan's own below-floor note. FLOOR VERSION: Coarsen matcher to pronoun-token presence (1st vs 3rd); skip polarity probe; negation → simple "not/n't" token check; retrieval-kNN for pronoun+negation retention. MISSED FLAW: Knowledge-polarity confound with negation-survival (Family B) — plan cross-references but doesn't disentangle; assumes false-belief (P021) and knowledge-gap share bottleneck properties without empirical link; 2×2 grid controls freq/length but not negation's syntactic cost.

---

**Skip on rudimentary NLAs**  
- P021 (probe + matcher both floor; explicit expected-null)  
- P022 (relational binding requires coherent two-speaker output; probe frontier)  
- P025 full plan (polarity/ignorance cell is hard null; person sub-component only)

**Highest-value for the rudimentary regime**  
- P023 — lexical status markers survivable; coarsened matcher + retrieval-kNN gives interpretable marker-retention + directional-upgrade signal  
- P024 — highest-frequency modal tokens; simplest binary matcher; directional drift testable; most robust to weak AV/AR  
- (P025 person/pronoun sub-component only — if Pronoun retention is a standalone question)

---


session_id: 20260605_125230_66d55c
**P026 — Secret/private-info retention**  
Signal is usable: content-retention and marker-survival are Family-A primitives that work even on degenerate AVs, and probe_auc(secret content) gives ground truth independent of verbalization. Failure mode bites when the NLA drops *all* content (secret and public) — then leak_gap ≈ 0 falsely suggests "no leak distinction" rather than "no content survived." Floor version: drop the generative AV entirely, run probe-only on activation with a shuffled-activation control, and report retention(secret) vs retention(public) as a simple ratio; skip marker_survival if AV is near-empty. Missed flaw: the probe measures linear decodability of X from the bottleneck, but the AR reconstructor may not actually *use* that linear direction — high probe_auc ≠ high leak_rate in reconstruction, and the plan doesn't enforce that alignment.

**P027 — Sandbagging-cue detectability**  
Signal is pure noise on a rudimentary NLA: deliberateness is a high-order intent relation the bottleneck almost certainly never encodes, so probe_auc ≈ baseline and matcher on degenerate AV is random. Failure mode: if even the failure-surface content drops, the probe label (deliberate vs genuine) becomes uninterpretable; the "expected null" is indistinguishable from a broken probe. Floor version: skip end-to-end AV matching entirely; run a probe-only suite with heavy L2 regularization and report probe_auc *relative to a shuffled-label baseline* on the same data; pool across many capability-then-failure pairs to escape sampling variance. Missed flaw: the "deliberate" label is imposed by dataset construction (specific phrasing like "I could but I won't"), so a probe_auc > baseline may detect *phrasing artifacts* (negation, modal verbs, length) rather than a genuine intent representation — the plan has no control for surface-form confounds.

**P028 — Sycophancy/framing leakage**  
Signal is barely usable: frame_leak probe_auc can work if the framing tokens attend to the proposition clause, but on a tiny bottleneck the signal drowns in context contamination from the framing window. Failure mode: with a low-dim bottleneck or few training examples, the probe overfits to framing-token artifacts (flattery lexicon surviving as n-grams) rather than genuine content shift; matcher assertion_drift is meaningless if AV text is repetitive/empty. Floor version: coarsen to binary frame classification (sycophantic vs neutral only), pool across 50+ fixed propositions, and require faithfulness_weighted_retention > 0.3 before interpreting any frame_leak; compare to a "frame-shuffled" control where framing is permuted across propositions. Missed flaw: "content-invariant framing" assumes the proposition activation is identical across frames — but in a weak NLA with small context, the framing tokens *change the attention pattern* over the proposition itself, so the bottleneck activation difference may reflect upstream context leakage, not bottleneck encoding of stance.

**P029 — Intent-vs-action distinction**  
Signal is mixed but salvageable for concrete verbs: verb-form survival in AV gives a lexical foothold; probe_auc(Intend vs Completed) may separate on tense/aspect morphology rather than epistemic intent. Failure mode: directional confusion "intend→completed" is confounded with the model's prior bias toward past-tense completion — a degenerate AR that always outputs past tense will fake the effect. Floor version: restrict to retrieval-kNN verbalizer (so verb form is copied, not generated); run probe-only with a tense-only baseline probe (past vs present on non-intent verbs) and subtract; report per-concreteness-tier as planned. Missed flaw: the intent/action distinction is *grammaticalized* as aspect/tense in English; a linear probe on a small model's activations cannot disentangle "epistemic intent" from "verb morphology" without a control that holds verb form constant while varying intent — which the matched pairs do not do.

**P030 — Concealment/omission-marker preservation**  
Signal is near-zero for the core claim: probe_auc(silent-omission vs full) tests whether *absence of tokens* creates a distinct activation pattern, but on a weak NLA the only pattern is length/position, not "omission as such." Failure mode: omission_detectability ≈ baseline by construction (probes can't detect what isn't there); marker_survival floors if AV is degenerate; the "expected hard null" is indistinguishable from a broken experiment. Floor version: drop the silent-omission cell entirely; keep only marked-concealment vs disclosure as a binary marker_survival check using retrieval-kNN verbalizer; report as present/absent with a shuffled-marker control. Missed flaw: the matched triples do *not* equalize sequence length (concealment adds "but they hid that Y"), so any probe_auc(concealment vs disclosure) may detect length/position artifacts, not the concealment relation — the plan's "freq+length matched markers" controls the marker word, not the sequence.

---

**Skip on rudimentary NLAs**  
- P027 (Sandbagging) — expected pure null; deliberateness is far beyond bottleneck capacity.  
- P030 (Concealment silent-omission cell) — absence detection is ill-posed for linear probes on weak NLAs; the null is structural, not informative.  

**Highest-value for the rudimentary regime**  
- P026 (Secret retention) — only plan where the core metric (content retention + probe leak ground truth) works *without* trusting the AV text; run probe-only with shuffled control first.  
- P028 (Framing leakage) — if you pool heavily and add the frame-shuffled control, a positive frame_leak is a real finding even on a weak NLA; negative is a clean "no leak" property.  
- P029 (Intent/action, concrete tier only) — with retrieval-kNN verbalizer and tense-baseline subtraction, the concrete-verb tier yields a usable "aspect survival" signal.
