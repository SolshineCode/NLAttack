# Family I — Distributional / OOD / training-dependence

How concept survival depends on the NLA's training distribution, the eval
corpus's domain, concept frequency-in-training, training maturity (checkpoint),
the layer read, and the underlying model. The unifying claim hardened by the
local-Gemma result (`results/local_gemma_e2b/README.md`): the bottleneck is **not
a uniform concept filter** — it linearly encodes in-distribution concepts at
probe AUC ≈0.99 but generic OOD concepts at ≈0.70. "Concept C dropped" is
therefore confounded by whether C lives in the bottleneck's *encodable subspace*,
which is set by training distribution, not by C alone. Every plan here either
(a) formalizes that gap, or (b) tests whether a dropout/survival law measured in
one distribution generalizes to another. Probe-side plans are CPU-only (cached
L23 `activation_vector` + sklearn) and run while the GPU is busy. Corpora live in
the companion NLA-training repo at `<NLA_DATA_ROOT>/experiments/v8_nla_local/data/stage0/`
(read-only): `gemma4_deception_chunk1`, `arxiv_chunk1`, `cnn_dailymail_chunk1`,
`fineweb_edu_chunk1`, `wikipedia_chunk1`, `mmlu_moral_chunk1`, `cai_harmless_chunk1`,
`pku_saferlhf_chunk1`, `anthropic_discrim_explicit_chunk1`,
`anthropic_global_opinions_chunk1`, `anthropic_persuasion_chunk1` — same model,
same L23 read, different domain. Layer variants: `stage0_gemma3_1b_L9/L21`.
Cross-model: `stage0_crossmodel` + Neuronpedia (`gemma-3-27b/kitft-l41`,
`llama3.3-70b-it/kitft-l53`).

### P081 — In-distribution vs OOD encodable-subspace probe-AUC gap
- **Family:** I
- **Probes:** Whether concept survival is gated by an encodable subspace whose shape is set by the NLA's training distribution, not by the concept per se.
- **Hypothesis:** For matched concepts (same freq band, same token length), mean linear-probe AUC on the L23 `activation_vector` is materially higher for concepts drawn from the NLA's training/in-distribution corpus (`gemma4_deception`) than for concepts drawn from OOD corpora (`fineweb_edu`, `wikipedia`, `arxiv`): ΔAUC ≥ 0.15, and the gap survives freq+length matching.
- **Method:** Reuse `experiments/local_gemma_e2b/probe_bottleneck.py`. For each of ≥8 stage0 corpora, auto-select concepts in a widened 10–90% prevalence band, train logistic-regression probes (5-fold CV) on the cached L23 vectors, ground truth = keyword presence in source text. Bin corpora into in-dist (deception/business) vs OOD (web/encyclopedic/scientific). Report per-concept AUC, then mean AUC per corpus and the in-dist−OOD contrast. Also fit an MLP probe to bound the linear-separability assumption (DESIGN_REVIEW P1 #3).
- **Metric:** Per-corpus mean probe AUC; in-dist−OOD ΔAUC with bootstrap CI; partial correlation of AUC with corpus-type controlling for freq+length.
- **Feasibility:** high — CPU, cached activations + sklearn; extends an already-running script to more corpora.
- **Controls:** Probe (not AV/matcher) is ground truth, so verbalizer+matcher loss is excluded. Freq+length matched concept pools across corpora (DESIGN_REVIEW item D). MLP probe as a non-linear control. Report n_pos per concept; treat AUC as an upper bound under noisy keyword labels.
- **Deception link:** A monitor's blind spots are domain-shaped: an attacker phrases misuse in an OOD register (scientific/encyclopedic) where the bottleneck barely encodes the relevant concepts, evading detection without obfuscating intent.
- **Harness:** `experiments/local_gemma_e2b/probe_bottleneck.py` (extend corpus loop + matched-pool selector); new `nla_eval` reporter for the in-dist/OOD contrast.
- **Null looks like:** ΔAUC ≈ 0 after freq+length matching — probe AUC is concept-driven, not distribution-driven; the encodable-subspace story collapses and survival is a property of C alone.

### P082 — Dropout-map domain transfer (does domain A's map predict domain B?)
- **Family:** I
- **Probes:** Whether the per-concept survival/dropout map is a stable NLA property or is re-shaped by the eval domain.
- **Hypothesis:** The per-concept dropout map measured on domain A transfers poorly to domain B: rank correlation (Spearman) of per-concept retention between any two distinct-domain corpora is weak (ρ < 0.4) for cross-family pairs (e.g. `arxiv`↔`cai_harmless`) and strong (ρ > 0.7) only for same-family pairs (`anthropic_*` among themselves).
- **Method:** Restrict to concepts that occur in ≥2 corpora (shared vocabulary across `gemma4_deception`, `cnn_dailymail`, `wikipedia`, `arxiv`, `fineweb_edu`, `cai_harmless`, `anthropic_*`). For each corpus compute per-concept probe AUC (P081 pipeline). Build a concept×corpus AUC matrix; compute pairwise Spearman ρ across the shared-concept rows; cluster corpora by agreement. Repeat with AV-side retention (when AV checkpoint available) and report whether the transfer pattern matches the probe-side pattern.
- **Metric:** Pairwise Spearman ρ of per-concept AUC across corpus pairs; mean within-family vs cross-family ρ; number of "domain-flipping" concepts (high AUC in A, low in B).
- **Feasibility:** high — probe side CPU; AV side medium (needs a working AV checkpoint, see P084).
- **Controls:** Only shared concepts compared (no selection-driven inflation). Freq+length matched within each corpus. Probe ground truth removes matcher topology confound (DESIGN_REVIEW P0 #2). Report n shared concepts per pair.
- **Deception link:** If dropout maps don't transfer, a monitor validated on benchmark corpus A has uncharacterized holes on deployment domain B — a direct false-assurance failure for NLA-as-monitor.
- **Harness:** New `nla_eval/transfer.py` (concept×corpus AUC matrix + Spearman); reuses P081 probes.
- **Null looks like:** Uniformly high ρ (>0.7) for all corpus pairs including cross-family — the dropout map is a portable NLA property and domain doesn't reshape it.

### P083 — Frequency-tier × domain interaction (is the dropout law domain-dependent?)
- **Family:** I
- **Probes:** Whether the frequency dropout law (`t01_dropout_law`: retention falls with rarity) has the same slope across domains, or whether domain modulates it.
- **Hypothesis:** The freq→retention slope is steeper in OOD corpora than in-distribution: in `gemma4_deception` even low-band concepts retain moderately (shallow slope), but in `fineweb_edu`/`arxiv` low-band concepts collapse (steep slope) — a significant freq_band×corpus_type interaction term.
- **Method:** Assign `meta.freq_band` (high/mid/low) per concept per corpus from corpus-internal counts. Run `t01_dropout_law` (probe-AUC variant) per corpus. Fit a regression: AUC ~ freq_band * corpus_type (+ token_length covariate). Test the interaction coefficient. Visualize as one retention-vs-freq curve per corpus overlaid.
- **Metric:** Per-corpus freq→AUC slope; interaction coefficient (freq_band×corpus_type) with p-value/CI; spread of slopes across corpora.
- **Feasibility:** high — CPU; freq bands computed from the same parquets.
- **Controls:** Token length as explicit covariate (DESIGN_REVIEW P1 #1 — separates rarity from length). Frequency-matched controls (`meta.is_control`) per band per corpus. Probe ground truth. Report band sizes (small low-band n is the main caveat).
- **Deception link:** If rare high-risk terms drop *fastest specifically in the domains where they're rare*, the danger-term blind spot (`t14_risk_weighted_dropout`) is domain-amplified — worst exactly where the attacker operates off-distribution.
- **Harness:** `nla_eval/tests.py::t01_dropout_law` + new interaction-model reporter in `nla_eval/transfer.py`.
- **Null looks like:** Parallel freq→AUC curves across corpora (interaction ≈ 0) — one universal dropout law, domain only shifts the intercept (overall difficulty), not the slope.

### P084 — Checkpoint-maturity effect (does survival improve with AV training?)
- **Family:** I
- **Probes:** Whether apparent concept dropout shrinks as the AV verbalizer trains — separating immature-verbalizer loss from genuine bottleneck dropout over training.
- **Hypothesis:** Holding the probe (bottleneck) fixed, AV-side recovered retention rises monotonically with checkpoint step (early `block_to000100` → later checkpoints), narrowing the probe−AV gap (currently ≈0.33 at step 100) without the probe AUC changing. I.e. most of the early gap is training maturity of the verbalizer, not bottleneck dropout.
- **Method:** Run `verbalize_av.py` at ≥3 AV checkpoints (`av_v0_1_aux_readout/block_to000100`, plus later `stage3_v0_1*` checkpoints). For a fixed concept set on `gemma4_deception`, hold the linear probe AUC constant (same L23 vectors) and measure `av_acc` (ensemble-matcher recovery) per checkpoint. Track `probe_acc − av_acc` (the verbalizer+matcher loss) vs step.
- **Metric:** av_acc vs checkpoint step; probe−av gap vs step (slope, monotonicity); degenerate/empty-emission rate vs step.
- **Feasibility:** medium — AV side needs GPU + multiple checkpoints; the early checkpoint emits empty/degenerate text on some inputs (known v8 maturity issue), so later checkpoints are required for a clean comparison.
- **Controls:** Probe AUC held fixed across checkpoints isolates verbalizer maturity from bottleneck change. Matcher ensemble for av_acc (DESIGN_REVIEW item C). Same concept set + freq+length matched across runs. Report empty-emission rate as a confound on early steps.
- **Deception link:** Calibrates whether "NLA dropped C" claims from early/immature NLAs are real bottleneck holes or just under-trained verbalizers — prevents over-claiming monitor failures on prototype NLAs.
- **Harness:** `experiments/local_gemma_e2b/verbalize_av.py` (checkpoint-loop arg); reuse probe outputs; new gap-vs-step reporter.
- **Null looks like:** Flat probe−av gap across checkpoints — the loss is not training-maturity; the bottleneck genuinely lacks the concept and more verbalizer training won't recover it.

### P085 — Cross-model NLA transfer (Gemma vs Llama NLA on the same concepts)
- **Family:** I
- **Probes:** Whether the encodable subspace / dropout map is model-specific or a shared property of the concept across model families.
- **Hypothesis:** Per-concept survival correlates only weakly across models: on a shared concept set, Spearman ρ between the Gemma NLA (`gemma-3-27b/kitft-l41`) and the Llama NLA (`llama3.3-70b-it/kitft-l53`) is < 0.5, and each model has concepts the other encodes well that it drops (model-specific blind spots).
- **Method:** Build a shared concept set with parallel inputs. For the local probe side, use `stage0_crossmodel` (gemma vs llama activations) → per-model probe AUC. For the AV side, query Neuronpedia for both NLAs on matched feature/text inputs and compare `description`-recovered retention (matcher ensemble) plus the returned `cosine_similarity`/`mse` faithfulness scores. Correlate per-concept survival across the two models.
- **Metric:** Cross-model Spearman ρ of per-concept survival (probe and AV separately); count of model-specific blind spots (high in one, low in other); agreement between probe-side and AV-side cross-model patterns.
- **Feasibility:** medium — local probe side needs `stage0_crossmodel` activations; AV side is Neuronpedia API (no raw activations, description+faithfulness only). Two model families = two pipelines.
- **Controls:** Matched concept set + matched inputs across models. Matcher ensemble on AV descriptions (DESIGN_REVIEW P0 #2). Faithfulness (`cosine_similarity`/`mse`) reported to separate AR-fidelity from bottleneck (DESIGN_REVIEW P1 #5). Probe ground truth where local activations exist.
- **Deception link:** If blind spots are model-specific, swapping the monitor's base model silently relocates the holes — an attacker who mapped one model's gaps must re-probe, but a defender who upgrades models inherits *new* uncharacterized gaps.
- **Harness:** New `nla_eval/crossmodel.py`; reuse `probe_bottleneck.py` for the local side; Neuronpedia client for the API side.
- **Null looks like:** High cross-model ρ (>0.7) — survival is a concept property shared across model families; no model-specific blind spots, dropout maps port across models.

### P086 — Concept-prevalence-in-training vs probe-AUC
- **Family:** I
- **Probes:** Whether how well the bottleneck encodes a concept is predicted by how often that concept appeared in the NLA's *training* distribution.
- **Hypothesis:** Per-concept probe AUC increases monotonically with the concept's prevalence in the NLA training corpus (the deception/business stage0 distribution): a positive, significant rank correlation (ρ > 0.5) between training-prevalence and AUC, even after controlling for eval-corpus prevalence and token length.
- **Method:** Estimate per-concept training prevalence from the NLA's training stage0 distribution (`gemma4_deception` + the `stage0_*` mixture used to train the local AV). Measure per-concept probe AUC on a held-out eval split. Regress AUC ~ log(training_prevalence) + log(eval_prevalence) + token_length. Report the partial correlation of AUC with training prevalence.
- **Metric:** Spearman ρ (training_prevalence, AUC); partial correlation controlling for eval prevalence + length; regression coefficient on log(training_prevalence) with CI.
- **Feasibility:** high — CPU; prevalence counts from parquets, AUC from P081 probes. Main work is reconstructing the training mixture's per-concept counts.
- **Controls:** Eval-corpus prevalence and token length as covariates (separates "trained on it" from "common in the test set" and from length, DESIGN_REVIEW P1 #1). Probe ground truth. Honest caveat: training mixture must be the *actual* one the AV saw, not assumed.
- **Deception link:** Directly operationalizes the training-dependence confound — "NLA dropped C" may simply mean "C was rare in NLA training," reframing dropout claims as data-coverage claims and pointing at a fix (train-set augmentation) rather than an inherent limit.
- **Harness:** New `nla_eval/train_prevalence.py` (count concepts in training mixture); reuse P081 AUCs.
- **Null looks like:** No relationship (ρ ≈ 0) between training prevalence and AUC after controls — encoding quality is not driven by training frequency; the encodable subspace isn't a frequency artifact.

### P087 — Train-distribution leakage into verbalizations (the "diabetes" boilerplate)
- **Family:** I
- **Probes:** Whether the AV verbalizer inserts training-corpus-specific boilerplate/concepts into outputs regardless of input — a distribution-driven insertion bias distinct from input-conditioned hallucination.
- **Hypothesis:** A small set of training-frequent concepts (the "diabetes"/business-boilerplate class) appear in AV verbalizations at a rate far above their input prevalence, and this excess is *constant across eval domains* (it tracks the training distribution, not the input). Excess-insertion rate for these terms > 3× their input rate on OOD corpora.
- **Method:** On AV verbalizations (`av_verbalizations.jsonl` + new runs across `fineweb_edu`, `arxiv`, `wikipedia`, `cai_harmless`), tally inserted concepts (in output, not in input) via `t07_co_occurrence_hallucination` and the insertions list. Identify the top training-frequent inserted terms. Compare each term's insertion rate across eval domains; test whether it's flat (training-driven) vs input-correlated (content-driven).
- **Metric:** Per-term excess-insertion rate (output rate − input rate); cross-domain variance of that rate (low variance = training-driven leakage); list of top leakage terms with rates.
- **Feasibility:** medium — needs AV runs across corpora (GPU) + the insertions pipeline; a later/cleaner checkpoint preferred (early one emits degenerate text).
- **Controls:** Input prevalence per term as the baseline (insertion = output minus input). Cross-domain constancy test separates training-leakage from input-driven hallucination (which would vary with domain). Matcher ensemble for "inserted" verdicts (DESIGN_REVIEW P0 #2). Report degenerate-emission rate.
- **Deception link:** Boilerplate insertion is a laundering surface in reverse — a monitor reading AV text sees fabricated benign training-context ("diabetes", "earnings") that can mask or dilute the real flagged content, and it does so independent of what was actually present.
- **Harness:** `nla_eval/tests.py::t07_co_occurrence_hallucination` + new leakage reporter cross-tabbing insertion rate × domain.
- **Null looks like:** Inserted-term rates track input prevalence and vary with domain — insertions are input-conditioned hallucination, not training-distribution leakage; no constant boilerplate signature.

### P088 — OOD hallucination/insertion rate vs in-distribution
- **Family:** I
- **Probes:** Whether the AV fabricates more (insertions per example) on OOD inputs than on in-distribution inputs — distribution-shift-driven confabulation.
- **Hypothesis:** Insertions-per-example (fabricated concepts in output absent from input) is significantly higher on OOD corpora (`fineweb_edu`, `arxiv`, `wikipedia`) than on in-distribution (`gemma4_deception`): OOD insertion rate ≥ 1.5× in-dist, mirroring the probe-AUC collapse (when the bottleneck encodes the input poorly, the verbalizer fills the gap with priors).
- **Method:** Run AV verbalization on matched-size samples from in-dist and OOD corpora. Apply `t07_co_occurrence_hallucination` per corpus. Correlate per-corpus insertion rate with that corpus's mean probe AUC from P081 (prediction: insertion rate ↑ as encodability ↓). Length-match inputs across corpora to remove output-length confounds.
- **Metric:** Insertions-per-example per corpus; OOD/in-dist insertion ratio with CI; correlation (insertion rate, mean probe AUC) across corpora.
- **Feasibility:** medium — AV runs across corpora (GPU); probe AUCs reused from P081.
- **Controls:** Input length matched across corpora (longer inputs ≠ more insertions). Matcher ensemble for insertion detection. Probe AUC linkage uses ground truth, not AV self-report. Degenerate-emission rate reported and excluded.
- **Deception link:** A monitor that confabulates more on off-distribution traffic produces more false context exactly where it's least reliable — an attacker operating OOD both evades detection (low encodability) and induces noise (high insertion) that obscures audit.
- **Harness:** `nla_eval/tests.py::t07_co_occurrence_hallucination` per corpus + correlation reporter against P081 AUCs.
- **Null looks like:** Flat insertion rate across in-dist and OOD (ratio ≈ 1, no AUC correlation) — confabulation is input-content-driven, not distribution-shift-driven.

### P089 — Layer × distribution interaction
- **Family:** I
- **Probes:** Whether the layer that best encodes a concept depends on the eval distribution — i.e. the optimal read layer is domain-dependent, not fixed.
- **Hypothesis:** The per-concept probe AUC layer-profile (across L9/L21/L23) differs by domain: in-distribution concepts peak at the trained read layer (L23), but OOD concepts peak earlier (e.g. L9/L21) or never reach high AUC — a significant layer×corpus_type interaction. The single fixed read layer is therefore optimal only for the training distribution.
- **Method:** Use the layer-variant activation dumps (`stage0_gemma3_1b_L9`, `_L21`, and the L23 stage0 set). For a shared concept set, train probes at each layer per corpus (P081 pipeline). Build a concept×layer×corpus AUC tensor. Fit AUC ~ layer * corpus_type. Report the best layer per (concept, domain) and whether it shifts with domain.
- **Metric:** Per-layer mean AUC per corpus; layer×corpus_type interaction coefficient; fraction of concepts whose best layer differs between in-dist and OOD.
- **Feasibility:** high — CPU, cached multi-layer activation dumps already exist (`stage0_gemma3_1b_L9/L21`).
- **Controls:** Same concept set across layers/corpora. Freq+length matched pools. Probe (linear + MLP) ground truth. Note layer dumps are on the 1B model variant — keep model fixed across the layer comparison; flag if model differs from the L23 set.
- **Deception link:** If the best read layer is domain-dependent, a monitor reading a single fixed layer has domain-shaped blind spots invisible to single-layer evaluation — the attacker's domain choice moves the concept out of the read layer's reach.
- **Harness:** New `nla_eval/layer_sweep.py` (multi-layer probe loop, interaction model); reuses `probe_bottleneck.py` core.
- **Null looks like:** Same best layer (L23) across all domains, parallel layer profiles (interaction ≈ 0) — the optimal read layer is concept/architecture-driven, distribution doesn't move it.

### P090 — Encodable-subspace dimensionality by domain (effective rank of concept directions)
- **Family:** I
- **Probes:** Whether the bottleneck devotes a richer (higher-dimensional) concept subspace to in-distribution domains than to OOD domains — the geometric form of the encodable-subspace claim.
- **Hypothesis:** The effective rank of the set of learned per-concept probe directions is higher for in-distribution corpora than for OOD corpora: the in-dist concept-direction matrix spans more effective dimensions (participation ratio / 90%-variance dimension) than the OOD one, even with concept count matched. The bottleneck "spends" representational capacity on the training domain.
- **Method:** For each corpus, collect the per-concept linear probe weight vectors (directions in L23 space) from P081. Stack into a matrix per corpus; compute effective rank via participation ratio of the singular-value spectrum and the count of components for 90% variance. Match the number of concepts per corpus (subsample to the minimum). Compare in-dist vs OOD effective rank; also report mean pairwise cosine between concept directions (lower = more spread = higher effective dimensionality).
- **Metric:** Effective rank / participation ratio per corpus (concept-count matched); in-dist−OOD difference with bootstrap CI; mean inter-direction cosine per corpus.
- **Feasibility:** high — CPU; operates on probe weight vectors already produced by P081, plus a small SVD.
- **Controls:** Concept count matched across corpora (rank scales with #directions). Use only well-fit probes (AUC > floor) so directions are meaningful, or report rank as a function of the AUC inclusion threshold. Directions normalized before stacking. Honest caveat: small concept counts make effective-rank estimates noisy — report CIs and the n used.
- **Deception link:** A low-dimensional OOD concept subspace means many distinct OOD attack concepts collapse onto few directions (poor separability) — the geometric reason laundering and category-collapse get worse off-distribution.
- **Harness:** New `nla_eval/subspace_rank.py` (stack probe directions, SVD, participation ratio); consumes P081 probe weights.
- **Null looks like:** Equal effective rank for in-dist and OOD at matched concept count — the bottleneck allocates dimensionality uniformly; encodability differences are not a subspace-capacity phenomenon.
