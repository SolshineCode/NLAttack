# Reviewer A critique — F_matcher_verbalizer_confound (run in two 5-plan halves to avoid the output-length limit)


session_id: 20260605_125350_3db2a8
### P051 — Contested-rate as a confound thermometer
**Signal:** On a rudimentary NLA, the AV often emits near-empty or repetitive text; all matcher backends then agree on "dropped" (agreement=1, contested_rate≈0), so the thermometer flatlines and yields no variance to correlate with cross-backend instability. **Failure mode:** Degenerate AV output creates a *false null* — contested_rate≈0 everywhere, Spearman ρ≈0, and the plan incorrectly concludes "disagreement is noise" when actually the matcher had nothing to disagree *about*. **Floor version:** Restrict analysis to a retrieval-only verbalizer (nearest cached snippet) or human-written descriptions as AV input, which guarantees non-degenerate text so contested_rate varies meaningfully; alternatively, pool across many activations and filter to only rows where at least one backend returns a non-empty match. **Missed flaw:** The plan conflates "matcher topology disagreement" with "AV produced matchable text" — at low capability, contested_rate variation is driven by AV degeneracy, not matcher geometry, so the thermometer measures AV health, not matcher reliability.

### P052 — Embedding-model sensitivity of effects
**Signal:** If the generative AV emits near-empty text, all three embedding models produce near-zero cosine similarities → all concepts labeled "dropped" under every model → pairwise Jaccard=1 (trivially identical null sets), falsely suggesting "no embedding confound." **Failure mode:** The planned Jaccard comparison *requires* semantic content in the AV output to be sensitive to embedding geometry; a tiny/early-checkpoint AV provides none, so the confound is measured as absent when it's actually unmeasurable. **Floor version:** Run the embedding-model swap on a retrieval-only verbalizer or on human-authored descriptions (bypassing the generative AV entirely) to isolate pure matcher sensitivity; add a shuffled-activation control to establish the baseline Jaccard under pure noise. **Missed flaw:** The hypothesis assumes the AV text *has* semantic variation that embedding geometry could distort; at low capability the "confound" detected is actually just AV failure, not a property of the embedding models.

### P053 — AV decoding-temperature sensitivity
**Signal:** A retrieval-only verbalizer has no temperature parameter (N/A); a tiny generative AV at temp 0 often already emits degenerate/repetitive text, and raising temperature merely produces *more* garbage — flip-rate saturates at ~100% immediately with no monotonic signal. **Failure mode:** Mode-collapsed AV emits the same repetitive string at all temperatures → flip-rate≈0 → plan falsely concludes "verbalizer is deterministic" when it's actually broken; alternatively, temp 0 yields empty string so variance is undefined. **Floor version:** Use retrieval verbalizer as a zero-variance baseline, then compare generative AV against it; or measure AV output token entropy directly (bypassing the matcher) to quantify stochasticity without matcher noise. **Missed flaw:** The metric (verdict flip-rate) adds matcher instability on top of AV stochasticity; at low capability where matcher errors dominate, you're measuring matcher noise, not AV sampling variance.

### P054 — AV prompt-template sensitivity
**Signal:** A tiny/early AV often ignores the prompt entirely and emits the same degenerate text for all templates → retention range=0 → false null ("prompt-robust" when actually the AV is non-functional). Retrieval verbalizer ignores prompts by design (N/A). **Failure mode:** Security-framed prompt cannot inflate retention if the AV emits near-empty text; the plan measures matcher verdict change, not AV text change, so a brittle matcher (e.g., embedding threshold near decision boundary) will flip verdicts on trivial AV differences, overstating prompt sensitivity. **Floor version:** Test prompt sensitivity on human-written descriptions fed to the matcher (bypassing AV); for generative AV, measure prompt adherence via token overlap with prompt keywords in the AV output rather than downstream matcher verdicts. **Missed flaw:** The plan attributes verdict swings to prompt wording, but at low capability the matcher's own threshold brittleness is the dominant source of flips — the measured "prompt sensitivity" is largely matcher sensitivity.

### P055 — Paraphrase invariance of verbalization
**Signal:** A retrieval verbalizer returns the same cached snippet for all paraphrases → verdict agreement=100% trivially; a tiny generative AV emits the same degenerate text for all paraphrases → agreement=100% falsely suggesting "concept-driven invariance" when it's actually AV collapse. **Failure mode:** Low-dimensional bottleneck or few training examples can map all paraphrases to nearly identical activations → same verbalization → agreement=100% reflects bottleneck collapse, not true paraphrase robustness. **Floor version:** Bypass the AV entirely: use human-written paraphrase descriptions as matcher input to test paraphrase sensitivity of the matcher+NLA pipeline in isolation; or measure activation-space cosine similarity across paraphrases directly (before AV) to distinguish bottleneck collapse from verbalizer invariance. **Missed flaw:** The plan assumes activations *differ* across paraphrases; at low capability they often don't, so high agreement is a bottleneck artifact masquerading as NLA+AV robustness.


**Skip on rudimentary NLAs**
- P051 (thermometer flatlines on degenerate AV text)
- P052 (embedding sensitivity unmeasurable without semantic AV output)
- P053 (temperature sweep degenerates: mode collapse or saturation)
- P054 (prompt sensitivity invisible when AV ignores prompt)
- P055 (paraphrase invariance confounded with bottleneck collapse)

**Highest-value for the rudimentary regime**
- *None of P051–P055 as written* — all require a functional generative AV producing non-degenerate, prompt-responsive, paraphrase-sensitive text. For rudimentary NLAs, **replace the entire Family F with probe-only ablations**: (1) feed human-written descriptions / retrieval snippets directly to the matcher to characterize matcher behavior in isolation; (2) measure activation-space cosine across paraphrases / shuffled controls *before* the AV to quantify bottleneck collapse; (3) run matcher threshold/embedding sweeps on fixed human text to establish matcher noise floor. These give signal without a working AV.

---


session_id: 20260605_125614_3e637e
### P056 — Human-vs-auto matcher κ
**Signal:** Near-zero on rudimentary NLAs. The AV emits degenerate/near-empty text; human annotators cannot reliably judge "present/substituted/dropped" on garbage, so inter-human κ collapses and human-matcher κ measures annotator confusion, not matcher validity. Stratification by `contested_rate` fails when all rows fall in the "all dropped" bin.  
**Failure mode:** Empty AV output → matcher has no signal → unanimous "dropped" verdicts → κ undefined/unstable; small N (200) with floor-effect labels yields sampling-noise-dominated estimates. Domain knowledge gaps on ATT&CK terms further confound κ with concept difficulty.  
**Floor version:** Replace human annotation with a shuffled-activation control: match real vs permuted AV text; if matcher cannot distinguish, it is broken. Or run probe-only (skip AV, feed gold concept labels to matcher) to isolate matcher behavior from AV quality.  
**Missed flaw:** Assumes AV output is minimally interpretable. On rudimentary NLAs, verbalizer failure dominates — you measure human-matcher agreement on nonsense, not matcher validity. The "contested_rate" thermometer (P051) itself becomes uninformative when agreement is uniformly 0 or 1.

---

### P057 — Matcher threshold sensitivity sweep
**Signal:** **Usable** even on degenerate AV. Re-scoring fixed AV text across thresholds requires no NLA quality; the retention-vs-threshold curve and Kendall τ of category rankings are well-defined regardless of whether retention is high or near-zero.  
**Failure mode:** If AV is near-empty, all embedding scores ≈0 → retention ≈0 at every threshold → flat curve (correctly signals "matcher cannot operate on this AV output"). Category ranking τ becomes meaningless when all categories share floor retention, but the sweep itself still runs.  
**Floor version:** Run as-is on whatever AV text exists. If retention is uniformly low, report the score distribution and the threshold band where any signal emerges. Pool across many items for stable per-category estimates. Can also run on synthetic AV outputs (known concepts, controlled verbosity) to establish matcher baseline behavior.  
**Missed flaw:** Assumes the threshold sweeps a meaningful decision boundary. If the embedding space is collapsed (all AV outputs map to similar vectors because AV is broken), the sweep navigates a degenerate space. The "default threshold" may be completely miscalibrated for the actual score distribution — should report the raw score histogram first.

---

### P058 — Lexical-vs-semantic matcher divergence map
**Signal:** **Usable** on any AV text. Cross-tabulating lexical vs embedding verdicts works even if both say "dropped" everywhere (high diagonal agreement = neither matcher can operate). Divergence cells reveal matcher topology differences independent of NLA quality.  
**Failure mode:** Near-empty AV → both matchers output "dropped" → off-diagonals empty (correctly reflects insufficient AV signal). Retrieval-only AV returning cached snippets → lexical matches exact phrases, embedding matches semantically → real topology divergence appears. Low-dimensional bottleneck compresses embedding scores, reducing semantic matcher dynamic range.  
**Floor version:** Run as-is on fixed AV text. If off-diagonals are empty due to AV degeneracy, report "matcher topology not exercisable because AV output insufficient." Can run on synthetic AV outputs to establish expected divergence patterns. Use rank/ordinal comparison when cell counts are tiny.  
**Missed flaw:** Treats "semantic-only" matches as laundering candidates, but on rudimentary NLAs both matchers may be wrong *together* (AV fails to mention concept → both say dropped). The map assumes at least one matcher is sometimes right. Also conflates matcher behavior with NLA behavior: a semantic-only match could be AV hallucination, not NLA laundering.

---

### P059 — Position-sampling sensitivity
**Signal:** **Pure noise / floor** on rudimentary NLAs. Tiny/early-checkpoint NLAs have uninformative activations at *all* positions (near-random AR cosine 0.1–0.4). Concept tokens often unlocatable in degenerate AV output → `fallback_full` triggers >50% → targeted vs last-token comparison collapses. All position policies yield similar low retention; null result is uninterpretable.  
**Failure mode:** Early-checkpoint AV emits near-empty text → concept tokens not found → `fallback_full` dominates. Retrieval-only verbalizer ignores activation positions entirely → position policy has no effect (correctly, but uninformatively). Low-dimensional bottleneck makes all positions similar. Shuffled-activation control per position would show no structure.  
**Floor version:** Skip if `fallback_full` rate > 30%. Use probe-only: skip AV, train linear probes per token position on activations to test if concept info is locally encoded. Pool across many concepts to detect tiny position effects. Compare each position policy to its own shuffled-activation baseline.  
**Missed flaw:** Assumes concepts are encoded at specific token positions — a rudimentary NLA may not encode concepts locally at all (distributed or absent). "Concept token positions" assumes input text has identifiable concept tokens; for arbitrary activation sources, no token alignment exists. The harness's `fallback_full` silently conflates position sensitivity with sequence-level effects.

---

### P060 — Verbalization-length vs recall
**Signal:** **High-value** on rudimentary NLAs. The random-concept negative control works regardless of AV quality — it directly measures "longer text matches anything by chance." If retention rises with `max_new_tokens` but random-concept false positives rise equally, the apparent retention is a verbosity artifact.  
**Failure mode:** Early-checkpoint AV ignores `max_new_tokens` (emits same short/degenerate text) → no length variation → curve flat but uninformative. Retrieval-only verbalizer returns fixed-length snippets → no sweep possible. Near-random AR with floor retention → length coefficient ≈0 but this is a floor effect, not evidence against verbosity confound.  
**Floor version:** If AV doesn't vary length, bin naturally-occurring outputs by token count (even narrow range). Probe-only variant: skip AV, simulate verbosity by repeating/truncating gold concept descriptions and measuring match rate. Random-concept control always runnable. If all retention near-zero, report "length confound unmeasurable because signal absent."  
**Missed flaw:** Conflates two length effects: (1) more tokens → more chances to mention the *true* concept, (2) more tokens → more chances to mention *any* concept. Random-concept control isolates (2), but (1) remains confounded with AV's actual verbalization ability. On rudimentary NLAs, longer output may just repeat the same token — logistic regression assumes independent match probability per token, which fails for repetitive/degenerate output.

---

**Skip on rudimentary NLAs**  
- P056 — Human annotation on degenerate AV output is meaningless; κ unreliable; stratification collapses  
- P059 — Position sensitivity requires locatable concept tokens and position-varying activations; both fail on tiny/early NLAs  

**Highest-value for the rudimentary regime**  
- P057 — Threshold sweep works on any fixed AV text; reveals matcher fragility without needing NLA quality  
- P058 — Lexical vs semantic divergence works on any AV text; exposes matcher topology differences in isolation  
- P060 — Length-recall with random-concept control exposes verbosity confound; runnable even with broken AV
