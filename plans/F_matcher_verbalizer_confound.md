# Family F — Matcher & verbalizer confound

These 10 plans (P051–P060) operationalize DESIGN_REVIEW.md's P0 #2 (matcher
topology ≠ NLA topology) and its sibling — the AV verbalizer is a *second* lossy
model. Every observed retention/laundering number is `AV+NLA+AR end-to-end,
matcher-dependent`. This family **isolates and quantifies the two extra filters
(verbalizer, matcher) so retention can be corrected**, and tells you *which*
filter owns each effect: verbalizer vs matcher vs NLA.

The recurring design move: hold one stage fixed and perturb another.
- Vary the **matcher** while holding (activation, AV text) fixed → matcher's
  contribution (P051, P052, P057, P058, P059).
- Vary the **AV** (temperature, prompt, position sampling, length) while holding
  the activation fixed → verbalizer's contribution (P053, P054, P055, P060).
- Vary the **input phrasing** while holding the concept fixed → AV+NLA joint
  invariance, with the matcher as a constant (P055).
- Bring in **human ground truth** to anchor the matcher's own validity (P056).

Harness primitives already present: `EnsembleMatcher` / `EnsembleMatch.verdicts`
/ `agreement`, `core.contested_rate`, `Matcher(backend=...)`, the AV adapters'
`do_sample` / `max_new_tokens` / `AV_TEMPLATE` (`local_gemma_e2b.py`) and the
Neuronpedia `temperature` param.

---

### P051 — Contested-rate as a confound thermometer
- **Family:** F
- **Probes:** Whether the ensemble's *disagreement* rate predicts where end-to-end retention numbers are untrustworthy (matcher-dependent).
- **Hypothesis:** Effects (dropout-by-category, laundering) computed on high-contested-rate concept subsets are not reproducible across single matchers, while low-contested subsets reproduce; i.e. `contested_rate` is monotonically predictive of cross-matcher verdict instability.
- **Method:** Run a fixed dataset through one NLA with `EnsembleMatcher(backends=("lexical","fuzzy","overlap"), extra=("embedding","wordnet"))`. For each concept row record `agreement`. Bin rows by agreement (0, partial, 1.0). Recompute every Family-A/B test (t01–t04, t13) separately on contested (`0<agreement<1`) vs uncontested rows, and recompute each test under each single backend. Correlate per-bin `contested_rate` with the variance of the test statistic across the single backends.
- **Metric:** Spearman ρ between `contested_rate` and cross-backend std of retention; ratio of effect size (contested vs uncontested subset); fraction of "laundering" hits that are uncontested.
- **Feasibility:** high — `contested_rate` and `EnsembleMatch.verdicts` already exist; CPU-only on existing AV text.
- **Controls:** Isolates the **matcher** (AV text held fixed across all backends; only the matcher topology varies). Does NOT isolate verbalizer — that is P053+. Frequency/length is held constant by reusing the same rows across bins.
- **Deception link:** A laundering claim ("attack → benign neighbor") that only one matcher sees is a false misuse-detection alarm; the thermometer flags it before it's reported.
- **Harness:** `matching.EnsembleMatcher`, `core.contested_rate`, `tests.py` group-bys re-run per backend (new driver script).
- **Null looks like:** `contested_rate` uncorrelated with cross-backend variance (ρ≈0) → disagreement is noise, not a reliability signal, and the thermometer is useless; report it as such rather than as validation.

---

### P052 — Embedding-model sensitivity of effects (swap 3 embedding models)
- **Family:** F
- **Probes:** How much a "substitution/laundering" map is an artifact of the *specific* embedding model inside the matcher.
- **Hypothesis:** The set of concepts labeled `substituted` and the top laundering targets (t13) shift by >20% Jaccard when the embedding backbone is swapped (all-MiniLM-L6-v2 → all-mpnet-base-v2 → BAAI/bge-small-en), with technical/ATT&CK terms shifting most.
- **Method:** Hold (NLA, dataset, AV text) fixed. Instantiate three `Matcher(backend="embedding")` variants by parameterizing `_try_embedding` to load each model (small adapter to pass a model name). Re-run `core.run` three times (matcher-only swap), and compute t02 substitution_rate, t03 category dropout, t13 laundering set. Pairwise Jaccard of `substituted` row-IDs and of laundering target terms across the three models.
- **Metric:** Pairwise Jaccard of substituted-row sets; rank correlation of per-category retention; turnover of top-10 laundering targets; per-model substitution_rate spread.
- **Feasibility:** medium — needs the three sentence-transformers models installed (a few hundred MB) and a one-line model-name hook into `_try_embedding`; otherwise CPU.
- **Controls:** Isolates the **embedding-matcher** specifically (activation, AV text, lexical/fuzzy/overlap tiers all held fixed). Distinguishes "embedding-geometry artifact" from "NLA laundering": an effect stable across all three embeddings AND seen by lexical is NLA-side; one that swings with the embedding is matcher-side.
- **Deception link:** ATT&CK laundering verdicts (P0 example "credential dumping → password management") are exactly the embedding-geometry artifacts this isolates.
- **Harness:** `matching.Matcher._embedding_match` (parameterized), `tests.t02/t03/t13`.
- **Null looks like:** Jaccard ≈1, identical laundering targets across all three embeddings → the embedding choice is not load-bearing and embedding-model citation can be dropped from the methods; no confound here.

---

### P053 — AV decoding-temperature sensitivity (same activation, vary temperature)
- **Family:** F
- **Probes:** Verbalization variance attributable purely to AV stochastic decoding — the verbalizer's own contribution to "dropped".
- **Hypothesis:** For a fixed activation, raising AV temperature monotonically increases verbalization variance and *decreases* per-concept retention reproducibility; a non-trivial fraction of concepts flip retained↔dropped across temperature samples at the same activation.
- **Method:** Take N fixed activation vectors with known concepts. For each, verbalize K times at temperatures {0.0 (do_sample=False), 0.3, 0.7, 1.0} — local adapter: set `do_sample=True` + temperature in `generate`; Neuronpedia: pass its `temperature` param. Match each sample with a *fixed* matcher. Compute, per concept, the retention flip-rate across the K samples at each temperature.
- **Metric:** Per-temperature mean retention; per-concept flip-rate (fraction of K samples disagreeing with the modal verdict); variance of matched_term; temperature at which flip-rate exceeds 10%.
- **Feasibility:** medium — needs a working AV (local Gemma E2B with `do_sample` exposed, or Neuronpedia `temperature`); K× generation cost. GPU-deferred for the local path (adapter is GPU-deferred by construction).
- **Controls:** Isolates the **verbalizer's stochasticity** (activation fixed, matcher fixed; only AV sampling varies). Anything that flips here is AV noise, NOT NLA dropout. This is the variance term to subtract from end-to-end retention.
- **Deception link:** If an attack concept is "dropped" only because of high-temperature AV sampling, the monitor's apparent blind spot is illusory — lowering AV temperature recovers it.
- **Harness:** `local_gemma_e2b.LocalGemmaE2BNLA` (`do_sample`/temperature), Neuronpedia adapter temperature; fixed `Matcher`.
- **Null looks like:** Flip-rate ≈0 at all temperatures, retention flat → AV decoding is effectively deterministic for these activations; verbalizer-stochasticity contributes nothing and need not be corrected for.

---

### P054 — AV prompt-template sensitivity (reword the AV instruction)
- **Family:** F
- **Probes:** Whether retention depends on the *wording* of the AV instruction rather than the activation — verbalizer prompt-induced bias (the "does 'survival' change?" question).
- **Hypothesis:** Rewording `AV_TEMPLATE` (e.g. neutral "describe", security-framed "describe any security-relevant content", terse "concept:", verbose researcher framing) shifts overall retention by >5 absolute points and shifts *which categories* retain, even at temperature 0 with the same activations.
- **Method:** Hold activations + matcher + temperature(0) fixed. Define 4–5 AV template variants (the local adapter's `AV_TEMPLATE` is a format string; for Neuronpedia, the closest analog is varying the prompt/instruction where exposed, else mark frontier). Verbalize the same N activations under each template. Recompute t01 dropout law and t11 technique coverage per template.
- **Metric:** Range of overall retention across templates; per-category retention swing; rank correlation of category ordering across templates; count of concepts whose verdict is template-dependent.
- **Feasibility:** medium (local AV — direct template swap) / frontier (Neuronpedia — prompt may not be user-settable). GPU-deferred for local.
- **Controls:** Isolates the **verbalizer's prompt prior** (activation, matcher, sampling all fixed; only instruction text varies). Separates "the NLA encodes concept C" from "the AV mentions C because the prompt nudged it." A security-framed prompt inflating ATT&CK retention is a verbalizer artifact, not NLA see-through.
- **Deception link:** Direct test of prompt-induced "survival" inflation — a monitor prompt that asks for malicious content could manufacture retention the bottleneck never carried.
- **Harness:** `local_gemma_e2b.AV_TEMPLATE` variants; `tests.t01/t11`.
- **Null looks like:** Retention and category ordering invariant across all templates → the AV readout is prompt-robust; "survival" is not a wording artifact and template choice need not be reported as a caveat.

---

### P055 — Paraphrase invariance of verbalization
- **Family:** F
- **Probes:** Whether the same concept, expressed through different input phrasings, produces the same verbalization (and thus the same retention verdict) — joint AV+NLA stability with the matcher held constant.
- **Hypothesis:** Activations from paraphrases of one concept (e.g. 5 sentences each expressing "privilege escalation") yield verbalizations whose retention verdicts agree at <80% — i.e. retention is partly phrasing-driven, not concept-driven.
- **Method:** Build a paraphrase set: M concepts × P paraphrased input sentences each, with `meta.concept_id` linking paraphrases (extends `Example`/`pair_id` convention). Obtain the activation per paraphrase, verbalize (temp 0), match each against the shared concept with a fixed matcher. Per concept, compute verdict agreement across its P paraphrases (a within-concept-id analog of t09 blindspot stability, but over phrasings rather than contexts).
- **Metric:** Mean within-concept verdict agreement across paraphrases; per-concept verbalization similarity (matcher similarity among the P outputs); fraction of concepts with ≥1 paraphrase flipping the verdict.
- **Feasibility:** medium — needs activations for paraphrase sets (precomputed (text, vec) rows) + a working AV; matcher CPU-only.
- **Controls:** Holds **matcher and AV decoding fixed**; varies input phrasing → isolates the **AV+NLA joint sensitivity to surface form**. Pairs with P054 (which varies AV wording, input fixed): together they separate prompt-side from input-side phrasing effects.
- **Deception link:** An attacker who finds the phrasing of a malicious request that verbalizes benignly exploits exactly this non-invariance (paraphrase-driven blind spot).
- **Harness:** new paraphrase dataset builder + `meta.concept_id`; `tests.t09`-style within-group variance; fixed `Matcher`.
- **Null looks like:** Verdict agreement ≈1 across paraphrases → verbalization is concept-driven and phrasing-invariant; no paraphrase exploit surface and no confound from input wording.

---

### P056 — Human-vs-auto matcher κ (inter-annotator reliability)
- **Family:** F
- **Probes:** Whether the automatic matcher's retained/substituted/dropped labels agree with human judgment — the matcher's external validity (DESIGN_REVIEW P1 #6).
- **Hypothesis:** Cohen's/Fleiss' κ between the EnsembleMatcher verdict and human annotators on a stratified sample is in the "moderate" band (0.4–0.6), and is *lowest* on contested rows (`0<agreement<1`) and on technical/ATT&CK concepts — i.e. disagreement concentrates exactly where the thermometer (P051) flags it.
- **Method:** Sample ~200 (AV description, concept) pairs stratified by ensemble agreement bin and by category. Have ≥2 human annotators label each present/substituted/dropped with a short rubric. Compute pairwise human κ (annotator reliability) and human-vs-matcher κ, overall and per stratum. Cross-tabulate disagreement against `contested_rate`.
- **Metric:** Inter-human κ; human-vs-ensemble κ; per-stratum κ (by agreement bin, by category); confusion matrix of human vs matcher status.
- **Feasibility:** medium — automated side is trivial; cost is the human annotation pass (~200×2). One annotator + the author is a minimum viable, lower-confidence version.
- **Controls:** Anchors the **matcher** against ground-truth human judgment (the one thing all other F-plans assume but cannot self-verify). Does NOT touch verbalizer/NLA — humans read the *same* AV text the matcher reads, so this isolates matcher validity given the AV output, not AV quality.
- **Deception link:** If the matcher under-agrees with humans on attack terms, every misuse-detection number (Family D) inherits that unreliability; κ bounds how much to trust them.
- **Harness:** `EnsembleMatcher` verdicts + a small annotation CSV + κ computation (new script); `core.contested_rate` for stratification.
- **Null looks like:** κ ≥0.8 uniformly, including on contested/technical rows → the matcher is human-equivalent and `contested_rate` does NOT track human disagreement (which would *invalidate* the P051 thermometer); report the thermometer as unsupported.

---

### P057 — Matcher threshold sensitivity sweep
- **Family:** F
- **Probes:** How sensitive retention/substitution numbers are to the (arbitrary) embedding cosine `threshold` and ensemble vote `threshold`.
- **Hypothesis:** Overall retention varies by >15 absolute points across embedding thresholds in [0.45, 0.75], and the category *ranking* (which categories are blind spots) is stable only within a narrow threshold band — so any single-threshold claim is fragile.
- **Method:** Hold (NLA, AV text) fixed. Sweep `Matcher(backend="embedding", threshold=t)` for t in {0.45…0.75 step 0.05} and `EnsembleMatcher(threshold=v)` for v in {0.34, 0.5, 0.66}. Recompute t01/t02/t03 at each setting. Plot retention vs threshold; measure stability of category ordering (Kendall τ vs the default-threshold ordering).
- **Metric:** Retention-vs-threshold curve slope; absolute retention range over the band; Kendall τ of category ranking across thresholds; the threshold band where the top-blindspot category is stable.
- **Feasibility:** high — pure matcher re-scoring of existing AV text; CPU; thresholds are already constructor args.
- **Controls:** Isolates the **matcher's decision boundary** (activation, AV text, embedding model, ensemble membership all fixed; only the cutoff moves). Separates "real dropout" (robust across thresholds) from "threshold-tuned dropout" (only at one cutoff).
- **Deception link:** A misuse blind spot that appears only above threshold 0.7 is a reporting artifact; defenders shouldn't act on a threshold-tuned hole.
- **Harness:** `matching.Matcher(threshold=)`, `EnsembleMatcher(threshold=)`, `tests.t01/t02/t03`.
- **Null looks like:** Flat retention and stable category ranking (τ≈1) across the whole band → threshold choice is not load-bearing; the default cutoff is safe to report without a sensitivity caveat.

---

### P058 — Lexical-vs-semantic matcher divergence map
- **Family:** F
- **Probes:** *Where* lexical and semantic (embedding/wordnet) matchers systematically disagree — the regions where a "match" is purely surface or purely semantic.
- **Hypothesis:** Disagreement is non-random: lexical-only matches concentrate on morphological variants/substrings, semantic-only matches concentrate on the laundering set (attack→benign), and the semantic-only region overlaps heavily with the eventual P056 human-disagreement region.
- **Method:** Run both `Matcher(backend="lexical")` and `Matcher(backend="embedding")` (and wordnet if installed) on the same AV text. Cross-tabulate verdicts into a 2×2 (lexical present? × semantic present?). Characterize the two off-diagonal cells (lexical-only, semantic-only) by category, freq band, and matched_term type. Produce a "suspect map": concepts living in off-diagonal cells are flagged matcher-dependent.
- **Metric:** Size of each off-diagonal cell; category/freq composition of each cell; fraction of t13 laundering hits that are semantic-only; overlap of semantic-only set with high-`contested_rate` set.
- **Feasibility:** high — two existing backends on existing AV text; CPU.
- **Controls:** Isolates **lexical vs semantic matcher topology** directly (everything upstream fixed). The off-diagonals ARE the matcher-attributable verdicts; the on-diagonal "both agree" set is the matcher-robust core. Distinguishes matcher-side laundering from NLA-side.
- **Deception link:** Semantic-only laundering hits are the prime suspects for "the embedding model invented the laundering" — this map names them so Family D claims can be filtered.
- **Harness:** `Matcher(backend="lexical")` vs `Matcher(backend="embedding")`; `tests.t13`; `core.contested_rate`.
- **Null looks like:** Off-diagonal cells near-empty (lexical and semantic almost always agree) → matcher topology is not a confound for this dataset; laundering claims can be made without the ensemble caveat.

---

### P059 — Position-sampling sensitivity (which token positions are verbalized)
- **Family:** F
- **Probes:** Whether the retention verdict depends on *which* token positions' activations are fed to the AV — a sampling choice upstream of both AV and matcher.
- **Hypothesis:** Retention for a concept changes materially with the position-selection policy (concept-token positions vs last-token vs mean-pooled vs random content positions); concept-token-targeted verbalization yields higher and more stable retention than last-token, revealing that headline numbers are position-policy-dependent.
- **Method:** For multi-token inputs, build activation inputs under ≥3 position policies: (a) the concept's own token positions (current `verbalize_concepts` targeted path), (b) final-token activation, (c) mean over content positions, (d) random content positions (control). Verbalize each (temp 0, fixed AV, fixed matcher) and recompute t01/t11. Compare retention and within-concept verdict stability across policies. Use the `fallback_full` flag to exclude rows where the concept token couldn't be located.
- **Metric:** Retention per position policy; verdict agreement across policies per concept; gap between targeted and last-token retention; `fallback_full` rate per policy.
- **Feasibility:** medium — requires control over position selection when constructing activation inputs (precompute path), plus a working AV. The targeted path already exists in `core.run`.
- **Controls:** Isolates the **position-sampling stage** (AV decoding, matcher, NLA weights all fixed; only which activation positions enter the AV varies). Separates "concept is encoded somewhere" from "concept is encoded at the positions we happened to read." This is a confound *upstream* of the verbalizer, attributable to the harness's own sampling.
- **Deception link:** A concept encoded at non-final positions but read only at the last token would be a blind spot manufactured by the readout policy, not the NLA — an attacker could place malicious content where the monitor doesn't look.
- **Harness:** `core.run(concept_targeted=...)`, `ConceptRow.fallback_full`; new position-policy switch in the activation builder; `tests.t01/t11`.
- **Null looks like:** Retention identical across all position policies → the readout location doesn't matter; concept presence is position-robust and the targeted-vs-fulltext distinction in `core.run` is cosmetic.

---

### P060 — Verbalization-length vs recall (control for AV verbosity)
- **Family:** F
- **Probes:** Whether longer AV output trivially raises retention (more text → more chances to match), inflating recall independent of NLA quality.
- **Hypothesis:** Retention rises monotonically with AV `max_new_tokens` / output token count because a longer description mentions more concepts by chance; after controlling for output length, the residual retention is substantially lower and is the length-corrected NLA signal.
- **Method:** Hold (activations, matcher) fixed; sweep AV `max_new_tokens` in {40, 80, 120, 200} (local adapter param; Neuronpedia max-length where exposed). Also bin naturally-occurring outputs by token count. Regress per-concept retention on output length (logistic), report the length coefficient, and compute length-stratified / length-matched retention (reuse the freq+length control philosophy from DESIGN_REVIEW P1 #1). Add a *random-concept* negative control: how often a concept NOT in the input matches as a function of length (false-positive recall).
- **Metric:** Retention vs `max_new_tokens` curve; logistic length coefficient (and p); length-matched retention vs raw retention gap; false-positive match rate vs length on random concepts.
- **Feasibility:** medium — needs a working AV with adjustable max length; matcher CPU. The local adapter exposes `max_new_tokens` directly.
- **Controls:** Isolates the **verbalizer's verbosity** as a recall confound (activation, matcher fixed; only output length varies). The random-concept negative control separates "longer text mentions the real concept" from "longer text matches anything." This is the length analog of P053's temperature variance — both subtract AV-side inflation from end-to-end retention.
- **Deception link:** If retention is largely a verbosity artifact, a monitor's apparent coverage of attack concepts is inflated; forcing terse AV output would expose the true (lower) detection rate.
- **Harness:** `local_gemma_e2b.LocalGemmaE2BNLA(max_new_tokens=)`, `Matcher`; length-stratified retention helper (new); random-concept control rows.
- **Null looks like:** Retention flat across lengths AND length coefficient ≈0 with the random-concept false-positive rate also flat → verbosity is not inflating recall; raw retention needs no length correction.
