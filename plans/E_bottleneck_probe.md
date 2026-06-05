# Family E — Bottleneck-probe / ground-truth

Operationalizes DESIGN_REVIEW.md **P0 #1** ("no ground truth at the bottleneck").
For an activation→AV→text→AR→activation NLA, the bottleneck is the **activation
vector** (the AR's reconstruction target). A linear/MLP probe that reads concept C
off that vector measures *ground-truth* concept presence, independent of the
verbalizer and the matcher. The load-bearing quantity everywhere in this family:

    gap(C) = probe_acc(C) − av_matcher_acc(C)   =   verbalizer + matcher loss

High probe_acc + low av_matcher_acc ⇒ the NLA *kept* the concept; the AV/matcher
lost it (NOT an NLA dropout). Only probe_acc *also* low ⇒ a genuine bottleneck drop.

These plans build on `nla_eval/bottleneck_probe.py` (lazy sklearn, CPU) and the
local Gemma-4-E2B NLA: L23 activations in
`C:\Users\caleb\deception-nanochat-sae-research\experiments\v8_nla_local\data\stage0\*.parquet`
(cols `detokenized_text_truncated`, `activation_vector`, `activation_layer`,
`doc_id`); env `.venv-gemma4` (sklearn); AV checkpoint `av_v0_1_aux_readout`.
Prior result: in-dist AUC ~0.99, OOD ~0.70, probe−AV gap ~0.33 (n=2–3/corpus).
Standing caveats for the whole family: **(a) keyword labels are noisy** (substring
≠ semantic presence) so probe AUC is an *upper bound* on honestly-detectable
presence, and **(b) small n** (2–3 concepts/corpus under the 15–85% prevalence
band) — these are method demonstrations, not population estimates. Plan P050
directly attacks caveat (a)+(b) by swapping in the corpus's stage-2 LLM labels.
Because the probe IS the ground-truth control, "Controls" below notes the
*residual* threats (label noise, small n, leakage) rather than re-deriving the probe.

---

### P041 — Probe-vs-AV gap per concept (the verbalizer+matcher loss)
- **Family:** E
- **Probes:** Disentangles end-to-end "concept dropped" into NLA-bottleneck loss vs verbalizer+matcher loss, per concept.
- **Hypothesis:** For in-distribution concepts where the bottleneck clearly encodes C (probe_acc ≥ 0.90), the AV+matcher recovers far less, so gap = probe_acc − av_matcher_acc > 0.20. I.e. most apparent "dropout" on encodable concepts is downstream of the bottleneck.
- **Method:** On `gemma4_deception_chunk1.parquet`, run `run_probe_suite` (logistic, 5-fold CV) over auto-selected concepts → probe_acc(C). Independently run the AV (`verbalize_av.py`, ckpt `av_v0_1_aux_readout`) on the SAME rows, match the verbalization against C with the ensemble matcher → av_matcher_acc(C). `attach_av_accuracy` computes gap. Align by `doc_id`.
- **Metric:** Per-concept `probe_acc`, `av_matcher_acc`, `gap`; mean gap; fraction of concepts with gap > 0.20.
- **Feasibility:** medium — probe side HIGH (CPU now); AV side needs the 4GB GPU free (the `verbalize_av.py` half deferred in prior run). Reuses existing scripts.
- **Controls:** Probe is the ground truth. Residual threats: keyword label noise inflates probe_acc as an upper bound; small n (2–3 concepts). Use a fixed AV temperature and report matcher-ensemble agreement to avoid blaming the matcher alone.
- **Deception link:** A monitor reading only AV text will under-report present-but-unverbalized danger concepts; gap quantifies that systematic blind spot — the misuse-detection failure mode.
- **Harness:** `bottleneck_probe.run_probe_suite` + `attach_av_accuracy`; `experiments/local_gemma_e2b/{probe_bottleneck,verbalize_av}.py`. New test `t41_probe_av_gap`.
- **Null looks like:** gap ≈ 0 for all concepts (AV+matcher recover everything the probe sees) ⇒ no verbalizer/matcher confound; the AV-text retention numbers were valid as NLA properties.

### P042 — Linear vs MLP probe: is concept encoding nonlinear at the bottleneck?
- **Family:** E
- **Probes:** Whether concepts the linear probe misses are genuinely absent or merely nonlinearly encoded (linear-probe AUC under-reads presence).
- **Hypothesis:** For most concepts, MLP-probe AUC − linear-probe AUC < 0.05 (encoding is linearly accessible); a small minority show MLP ≫ linear (nonlinear/XOR-like encoding), and those are exactly the ones where naive linear ground-truth would be wrong.
- **Method:** Add an MLP variant to `probe_concept` (`sklearn.neural_network.MLPClassifier`, 1 hidden layer 256, same StratifiedKFold + StandardScaler). Run both on `gemma4_deception_chunk1` and `wikipedia_chunk1`. Compare AUCs per concept.
- **Metric:** Per-concept linear AUC, MLP AUC, Δ=MLP−linear; count of concepts with Δ ≥ 0.05; mean Δ.
- **Feasibility:** high — CPU, sklearn already present; pure probe-side addition.
- **Controls:** Same CV folds/seed for both probes (paired comparison). Guard against MLP overfit on small n by reporting CV std and using early-stopping; label noise still bounds both from above.
- **Deception link:** If danger concepts are nonlinearly encoded, linear monitors (and our own linear ground truth) under-detect them — a probe-class blind spot relevant to oversight.
- **Harness:** new `probe_concept(..., kind="mlp")` branch in `bottleneck_probe.py`; test `t42_linear_vs_mlp`.
- **Null looks like:** MLP ≈ linear everywhere (Δ near 0) ⇒ bottleneck encoding is linear; the existing linear probe is an adequate ground truth and DESIGN_REVIEW's linear-separability worry (P1 #3) is moot here.

### P043 — Layer sweep: which layer best encodes a given concept (needs re-extraction)
- **Family:** E
- **Probes:** Whether L23 (the NLA's chosen bottleneck) is where a concept is most linearly available, or whether the AR target sits off the concept's peak.
- **Hypothesis:** Per-concept probe AUC is non-monotone in layer and peaks at a concept-dependent layer; for some concepts the peak is several layers from L23, so the bottleneck layer choice itself contributes to "dropout."
- **Method:** Re-extract activations at a layer set (e.g. L9, L21, L23 — `stage0_gemma3_1b_L9/L21` already exist for the 1B; extract matching layers for E2B on the same `doc_id`s) with the v8 extraction script. Probe each concept at each layer with `probe_concept`. Plot AUC vs layer per concept.
- **Metric:** Per-concept AUC-by-layer curve; argmax layer; AUC(peak) − AUC(L23); fraction of concepts whose peak ≠ L23.
- **Feasibility:** medium — probing is CPU/HIGH, but re-extraction at new layers needs the GPU + the v8 extraction pipeline. L9/L21 dumps already exist for gemma3_1b, lowering cost.
- **Controls:** Same doc_ids and labels across layers (paired). Account for differing activation norms with the StandardScaler already in the probe; small n + label noise caveats persist.
- **Deception link:** If danger concepts peak away from the AR target layer, an NLA-as-monitor is structurally weakest exactly where it most needs sensitivity.
- **Harness:** re-use v8 extraction; `bottleneck_probe.probe_concept` per layer; test `t43_layer_sweep`.
- **Null looks like:** flat AUC-vs-layer or peak always at L23 ⇒ concept presence is layer-robust and the bottleneck layer is not costing detectability.

### P044 — Probe-direction stability across seeds and subsamples
- **Family:** E
- **Probes:** Whether the learned concept direction is a stable property of the bottleneck or an artifact of a particular sample/seed (reliability of the ground truth itself).
- **Hypothesis:** For high-AUC concepts, probe weight vectors from different CV seeds / 70% subsamples have mean pairwise cosine > 0.8 (stable direction); low-AUC concepts have unstable directions (cosine near 0), i.e. instability tracks genuine non-encoding.
- **Method:** Refit the logistic probe under K seeds (and K bootstrap subsamples) on `gemma4_deception_chunk1`; collect `clf.coef_`; compute all pairwise cosine similarities of the (scaler-normalized) weight vectors per concept. Relate mean stability cosine to probe AUC.
- **Metric:** Per-concept mean pairwise weight-cosine; correlation(stability, AUC); fraction of high-AUC concepts with stability > 0.8.
- **Feasibility:** high — CPU, just refit the existing probe many times.
- **Controls:** Use the same StandardScaler fold logic so coefficients are comparable; bootstrap CIs on stability. Label noise lowers both AUC and stability — report so a null isn't mistaken for noise.
- **Deception link:** An unstable ground-truth direction means "the bottleneck encodes C" is sample-dependent — a caution against over-trusting any single probe when auditing.
- **Harness:** new `probe_direction_stability()` in `bottleneck_probe.py`; test `t44_direction_stability`.
- **Null looks like:** all concepts (incl. high-AUC) show low/unstable cosine ⇒ probe directions are not reproducible; ground-truth claims need many-seed averaging before use.

### P045 — Concept geometry: angles between probe directions (superposition)
- **Family:** E
- **Probes:** Whether distinct concepts occupy distinct bottleneck directions or share/oppose directions (superposition / entanglement), which would make per-concept survival claims non-independent.
- **Hypothesis:** Semantically related concepts have probe-direction cosines well above the random baseline (entangled), and at least one antonym/contrast pair has negative cosine (shared axis) — so "C survived" partly implies "C′ survived."
- **Method:** Fit linear probes for all auto-selected concepts on a single corpus (`gemma4_deception_chunk1`). Build the matrix of pairwise weight-vector cosines. Compare to a null from random Gaussian directions in the same dimension. Cluster the cosine matrix.
- **Metric:** Pairwise direction-cosine matrix; mean |cosine| vs random-null mean; count of pairs with |cosine| > null 99th percentile; number of clusters.
- **Feasibility:** high — CPU; reuses fitted probes from P044.
- **Controls:** Random-direction null (high-dim cosines concentrate near 0, so the null is tight). Confounded by shared keyword co-occurrence in labels — report concept-label correlation alongside direction cosine to separate label overlap from representational overlap.
- **Deception link:** Superposed danger/benign directions are the mechanistic substrate for laundering (DESIGN_REVIEW P0 #2) — geometric evidence for or against the matcher-topology worry.
- **Harness:** new `concept_geometry()` in `bottleneck_probe.py`; test `t45_concept_geometry`.
- **Null looks like:** all pairwise cosines within the random-null band ⇒ concepts are near-orthogonal at the bottleneck; survival claims are independent and superposition is not driving laundering here.

### P046 — Verbalizer ablation: trained AV vs template/forced-decode AV
- **Family:** E
- **Probes:** Isolates the *verbalizer's* contribution to the gap by replacing the trained AV with a constrained decoder, holding bottleneck + matcher fixed.
- **Hypothesis:** A forced-vocabulary / templated decoder ("The concept is: [X]" over a fixed concept list, scored from the bottleneck) recovers concepts at higher matcher-accuracy than the trained `av_v0_1_aux_readout` on concepts the probe says are present ⇒ a large share of the P041 gap is verbalizer (not matcher) loss.
- **Method:** Build a template/forced-decode AV: for each concept, score its presence from the bottleneck (or constrain the AR/AV decode to a closed concept vocabulary) and emit a fixed phrasing; run the SAME ensemble matcher. Compare av_matcher_acc(trained AV) vs av_matcher_acc(forced AV) on shared `doc_id`s, both against probe_acc.
- **Metric:** Per-concept av_matcher_acc(trained), av_matcher_acc(forced), and the gap-narrowing Δ = forced − trained; share of P041 gap attributable to verbalizer.
- **Feasibility:** medium — forced/templated decode needs the local AV/AR weights and modest new decode code; the early checkpoint emits degenerate text on some inputs (known v8 issue), which this ablation specifically routes around.
- **Controls:** Bottleneck and matcher held identical across AV variants (only the verbalizer changes). Probe ground truth anchors both. Small n + label noise caveats.
- **Deception link:** Directly tests DESIGN_REVIEW's "swap AV for a constrained decoder" remedy — establishes whether the verbalizer (not the NLA) is the security-relevant bottleneck.
- **Harness:** new `experiments/local_gemma_e2b/verbalize_av_forced.py`; reuse matcher + `attach_av_accuracy`; test `t46_verbalizer_ablation`.
- **Null looks like:** forced AV ≈ trained AV (Δ ≈ 0) ⇒ the verbalizer is not the bottleneck; the remaining gap is matcher loss or genuine, and AV swapping won't help.

### P047 — Probe accuracy gated by AR faithfulness (round-trip cosine threshold)
- **Family:** E
- **Probes:** Whether the bottleneck's ground-truth concept signal is preserved through the AR reconstruction, by conditioning probe accuracy on per-example AR fidelity.
- **Hypothesis:** Restricting the probe's test set to high-fidelity examples (round-trip cosine ≥ τ) raises probe AUC for concepts whose loss is reconstruction-driven; concepts whose AUC is flat across fidelity strata are lost (or absent) at the *bottleneck*, not in the AR.
- **Method:** Compute per-example AR faithfulness (round-trip cosine between original and AR-reconstructed activation; or the `cosine_similarity` field where available). Stratify rows by fidelity (e.g. quartiles, and a τ sweep). Run `probe_concept` within each stratum on the same concept labels.
- **Metric:** Probe AUC as a function of fidelity quartile / τ per concept; Δ AUC(top vs bottom quartile); concepts that are fidelity-sensitive vs fidelity-flat.
- **Feasibility:** medium — probing is CPU/HIGH; needs AR reconstructions to get per-example round-trip cosine (GPU or precomputed faithfulness column).
- **Controls:** Faithfulness strata are matched on concept prevalence (re-balance per stratum) to avoid base-rate artifacts. Implements DESIGN_REVIEW P1 #5 (faithfulness weighting) as a probe-side control; label noise unchanged.
- **Deception link:** Separates "AR hallucinated away the danger concept" from "bottleneck never had it" — locating where in the pipeline a misuse signal dies.
- **Harness:** add fidelity-stratified split to `run_probe_suite`; pull cosine from AR run; test `t47_fidelity_gated_probe`.
- **Null looks like:** probe AUC identical across fidelity strata for every concept ⇒ AR fidelity doesn't modulate ground-truth recoverability; faithfulness weighting buys nothing for these concepts.

### P048 — Multi-concept joint / structured probe
- **Family:** E
- **Probes:** Whether jointly modeling concepts (multi-label / structured probe) recovers presence better than independent per-concept probes — i.e. whether the bottleneck encodes concept *combinations*.
- **Hypothesis:** A joint multi-output probe (or a probe conditioned on a co-occurring concept) yields higher AUC on co-occurring concept pairs than independent probes, evidencing that the bottleneck encodes concept structure, not just isolated features.
- **Method:** On a corpus with several balanced concepts (`gemma4_deception_chunk1` or a merged multi-chunk set), fit (a) independent logistic probes and (b) a joint `MultiOutputClassifier` / a structured probe predicting one concept given activation + indicator of a partner concept. Compare per-concept and joint-pair AUC under the same CV.
- **Metric:** Per-concept AUC independent vs joint; joint-pair (both-present) detection AUC; Δ from independence; mutual-information-style lift.
- **Feasibility:** high — CPU, sklearn `MultiOutputClassifier`; needs ≥3–4 balanced concepts (merge chunks if a single corpus is too thin).
- **Controls:** Same folds/seed and StandardScaler across both setups (paired). Watch label-co-occurrence leakage — report concept-label correlation so joint lift isn't just shared keywords. Small-n caveat amplified for pair cells.
- **Deception link:** Compositional attacks (DESIGN_REVIEW P0 #3) rely on individually-benign concepts that are jointly malicious; a joint probe is the ground-truth test of whether the bottleneck retains the *combination*.
- **Harness:** new `joint_probe()` in `bottleneck_probe.py`; test `t48_joint_probe`.
- **Null looks like:** joint AUC ≈ independent AUC and no pair lift ⇒ the bottleneck encodes concepts independently; no structured/compositional signal to exploit or rely on.

### P049 — Probe transfer across corpora (train on corpus 1, test on corpus 2)
- **Family:** E
- **Probes:** Whether a concept direction learned on one corpus generalizes to another — i.e. whether the bottleneck's concept encoding is corpus-invariant or distribution-bound (the in-dist 0.99 vs OOD 0.70 result, made transfer-explicit).
- **Hypothesis:** Probe-direction transfer accuracy (train corpus A, test corpus B) drops well below within-corpus CV accuracy for cross-domain pairs (e.g. deception→fineweb), and the drop is larger than for similar-domain pairs (arxiv→wikipedia), showing concept directions are partly corpus-specific.
- **Method:** For concepts present (balanced) in two corpora, fit the probe on corpus A's activations+labels, freeze, evaluate on corpus B (align dims; reuse A's StandardScaler). Pairs from `{gemma4_deception, fineweb_edu, arxiv, wikipedia, cnn_dailymail}_chunk1`. Compare to within-corpus 5-fold CV (the diagonal).
- **Metric:** Transfer-accuracy matrix (train×test) per concept; off-diagonal drop vs diagonal; same-domain vs cross-domain transfer gap.
- **Feasibility:** high — CPU; all needed stage0 parquets already exist.
- **Controls:** Restrict to concepts balanced in BOTH corpora (prevalence matched); apply train-corpus scaler to test (no test leakage). Diagonal (within-corpus CV) is the control ceiling. Label-noise differences across corpora can masquerade as transfer loss — note it.
- **Deception link:** A monitor trained/validated on one distribution may silently lose ground-truth sensitivity on deployment traffic — the distribution-shift oversight failure.
- **Harness:** new `probe_transfer()` in `bottleneck_probe.py`; test `t49_probe_transfer`.
- **Null looks like:** off-diagonal ≈ diagonal everywhere ⇒ concept directions are corpus-invariant; the in-dist/OOD AUC gap was about label balance, not transferable encoding.

### P050 — Better concept labels: replace keyword labels with stage-2 LLM labels
- **Family:** E
- **Probes:** Whether swapping crude keyword labels for the corpus's existing stage-2 LLM labels lifts usable concept count (n) and changes probe AUC — directly attacking the family's label-noise + small-n caveats.
- **Hypothesis:** LLM-labeled concept presence (semantic, paraphrase-aware) admits many more balanced concepts than the 15–85% keyword auto-selector (n rises from 2–3 to ≥10/corpus), and per-concept probe AUC shifts (rises where keywords missed paraphrase; falls where keywords caught mere mentions), so keyword-based AUC was a biased estimate of the true ground truth.
- **Method:** Join stage-2 LLM labels (e.g. `stage0_fineweb_anthropic*` / `stage2_v0_*` label parquets) to stage0 activations by `doc_id`. Build a `label_by_llm()` alongside `label_by_keyword`. Re-run `run_probe_suite` with LLM labels; compare concept count, prevalence, and per-concept AUC head-to-head with keyword labels on the same rows.
- **Metric:** n_balanced_concepts (keyword vs LLM); per-concept prevalence and AUC under each labeling; mean |Δ AUC|; sign of Δ by concept type (paraphrase-heavy vs literal).
- **Feasibility:** high — CPU; the anthropic/stage2 label parquets already exist; only a join + new labeler function needed.
- **Controls:** Compare on the EXACT same activation rows (paired). LLM labels have their own noise/bias — treat as a *better* ground truth, not perfect; spot-check a sample by hand for label-source agreement (Cohen's κ). Now the small-n caveat is the thing being fixed, not just noted.
- **Deception link:** Honest ground truth requires semantic, not lexical, concept presence; this is the prerequisite for trustworthy danger-concept survival numbers across the whole family.
- **Harness:** new `label_by_llm()` + join helper in `bottleneck_probe.py`; test `t50_llm_label_probe`.
- **Null looks like:** LLM labels yield the same n and near-identical per-concept AUC as keywords ⇒ keyword labeling was adequate; the family's label-noise caveat was over-stated for these corpora.
