# NLA Evaluation Plans — Master Index (P001–P112)

112 plans across 12 families. Schema + themes: see `README.md`. Feasibility: high / medium / frontier.

**Feasibility mix:** high=40 · medium=56 · frontier=15 · mixed=1 (P037)

> Small/weak NLA? see `RUDIMENTARY_TIERS.md`. Emergence detection: Family K + `nla_eval/emergence.py`.
> Literature basis: `../docs/LITERATURE.md`; integration + new plans (Family L): `../docs/LITERATURE_INTEGRATION.md`.


## Family A — Concept-survival & dropout-map  (`A_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P001 | Retention-vs-frequency dropout law (continuous, probe-anchored) | medium |
| P002 | Category-selective dropout against freq+length-matched controls | high |
| P003 | Blind-spot stability: variance of survival across many contexts | high |
| P004 | BOS / position-0 verbalization noise (position confound on the floor) | medium |
| P005 | Compression / length sensitivity: longer input → more dropout | medium |
| P006 | Concept crowding: N concepts competing in one input | medium |
| P007 | Token-span vs survival: do multi-token concepts survive better? | medium |
| P008 | Part-of-speech / modality retention: nouns vs verbs vs numbers | high |
| P009 | Redundancy / repetition boosts survival | medium |
| P010 | Salience-driven dropout: a dominant topic suppresses other concepts | medium |

## Family B — Content-adjacency & laundering  (`B_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P011 | Minimal-pair separation (does the bottleneck keep the bit that distinguishes "X" from "not-X" / "30 mg" from "300 mg") | high |
| P012 | Hypernym drift (specific → general; "sniper rifle" → "weapon") | high |
| P013 | Co-hyponym confusion (two siblings under one parent blend together) | medium |
| P014 | Antonym / polarity collapse (opposite meanings map to the same region) | high |
| P015 | Near-synonym winner-take-all (cluster collapses to its most frequent member) | high |
| P016 | Metaphor vs. literal disambiguation (figurative sense laundered into literal, or vice-versa) | medium |
| P017 | Brand → category laundering (named product collapses to its generic class) | high |
| P018 | Quantity rounding / magnitude flattening (numeric precision collapses) | high |
| P019 | Temporal adjacency / order (before vs after, sequence preserved or flattened) | medium |
| P020 | Spatial / relational adjacency (relation between entities collapses to mere co-mention) | frontier |

## Family C — Deception & knowledge-asymmetry  (`C_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P021 | False-belief (Sally-Anne) epistemic-state preservation | frontier |
| P022 | Who-said-what attribution preservation | frontier |
| P023 | Reported-vs-asserted claim-status flattening (claim laundering) | medium |
| P024 | Hedged-vs-certain modality preservation | medium |
| P025 | First-vs-third-person knowledge-boundary preservation | frontier |
| P026 | Secret / private-info retention (does the bottleneck leak what shouldn't be revealed) | high |
| P027 | Sandbagging-cue detectability (deliberate underperformance) | frontier |
| P028 | Sycophancy / framing leakage into content | medium |
| P029 | Intent-vs-action distinction preservation | medium |
| P030 | Concealment / omission-marker preservation | frontier |

## Family D — ATT&CK / misuse detection  (`D_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P031 | Full 14-tactic risk-weighted coverage heatmap | high |
| P032 | Obfuscation see-through ratio across an obfuscation-family ladder | high |
| P033 | Kill-chain multi-stage joint co-occurrence survival | medium |
| P034 | Dual-use disambiguation by a single contextual cue | medium |
| P035 | Agentic-scaffolding / autonomous-orchestration marker survival | medium |
| P036 | Severity monotonicity within a single technique family | medium |
| P037 | Benign-neighbor laundering of attack terms (probe-grounded) | mixed (high+frontier) |
| P038 | ARiES axis separability via probes on activations | frontier |
| P039 | Cross-technique transfer of blind spots | high |
| P040 | Monitor-as-filter realism: downstream classifier on the bottleneck | medium |

## Family E — Bottleneck-probe / ground-truth  (`E_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P041 | Probe-vs-AV gap per concept (the verbalizer+matcher loss) | medium |
| P042 | Linear vs MLP probe: is concept encoding nonlinear at the bottleneck? | high |
| P043 | Layer sweep: which layer best encodes a given concept (needs re-extraction) | medium |
| P044 | Probe-direction stability across seeds and subsamples | high |
| P045 | Concept geometry: angles between probe directions (superposition) | high |
| P046 | Verbalizer ablation: trained AV vs template/forced-decode AV | medium |
| P047 | Probe accuracy gated by AR faithfulness (round-trip cosine threshold) | medium |
| P048 | Multi-concept joint / structured probe | high |
| P049 | Probe transfer across corpora (train on corpus 1, test on corpus 2) | high |
| P050 | Better concept labels: replace keyword labels with stage-2 LLM labels | high |

## Family F — Matcher & verbalizer confound  (`F_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P051 | Contested-rate as a confound thermometer | high |
| P052 | Embedding-model sensitivity of effects (swap 3 embedding models) | medium |
| P053 | AV decoding-temperature sensitivity (same activation, vary temperature) | medium |
| P054 | AV prompt-template sensitivity (reword the AV instruction) | medium |
| P055 | Paraphrase invariance of verbalization | medium |
| P056 | Human-vs-auto matcher κ (inter-annotator reliability) | medium |
| P057 | Matcher threshold sensitivity sweep | high |
| P058 | Lexical-vs-semantic matcher divergence map | high |
| P059 | Position-sampling sensitivity (which token positions are verbalized) | medium |
| P060 | Verbalization-length vs recall (control for AV verbosity) | medium |

## Family G — Adaptive red-team / adversarial  (`G_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P061 | Query-budget evasion success curve | medium |
| P062 | Compositional blind spots (benign parts, malicious whole) | medium |
| P063 | Evasion transfer across NLAs / layers | frontier |
| P064 | Monitor-gaming feedback loop (optimize against the AV output) | medium |
| P065 | Steganographic channel capacity through the bottleneck | frontier |
| P066 | Adaptive paraphrase across the retention boundary at constant probe-intent | medium |
| P067 | Adversarial concept ordering / dilution | medium |
| P068 | Prompt-injection that survives the bottleneck to reach a downstream reader | medium |
| P069 | Encoder/decoder collusion under an AV-reading overseer | frontier |
| P070 | Robustness to UNSEEN obfuscation families (train on family 1, test on family 2) | medium |

## Family H — Faithfulness / AR-fidelity  (`H_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P071 | Faithfulness correlates with concept survival | high |
| P072 | Faithfulness-weighted retention vs raw retention | high |
| P073 | AR concept drift toward the wrong neighbor (activation space) | medium |
| P074 | Round-trip stability: converge or drift? | medium |
| P075 | Per-position faithfulness: concept tokens vs filler tokens | high |
| P076 | Reconstruction error by concept category | high |
| P077 | AR identity / cross-row leakage (does the AR ignore the text?) | medium |
| P078 | Faithfulness calibration: cosine threshold vs true concept survival (probe) | medium |
| P079 | AR paraphrase delta (does paraphrasing the AV text change the activation?) | medium |
| P080 | Activation-space neighbor analysis: nearer the true concept or a benign neighbor? | medium |

## Family I — Distributional / OOD / training-dependence  (`I_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P081 | In-distribution vs OOD encodable-subspace probe-AUC gap | high |
| P082 | Dropout-map domain transfer (does domain A's map predict domain B?) | high |
| P083 | Frequency-tier × domain interaction (is the dropout law domain-dependent?) | high |
| P084 | Checkpoint-maturity effect (does survival improve with AV training?) | medium |
| P085 | Cross-model NLA transfer (Gemma vs Llama NLA on the same concepts) | medium |
| P086 | Concept-prevalence-in-training vs probe-AUC | high |
| P087 | Train-distribution leakage into verbalizations (the "diabetes" boilerplate) | medium |
| P088 | OOD hallucination/insertion rate vs in-distribution | medium |
| P089 | Layer × distribution interaction | high |
| P090 | Encodable-subspace dimensionality by domain (effective rank of concept directions) | high |

## Family J — Calibration / uncertainty / self-knowledge  (`J_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P091 | Confidence direction in the bottleneck | frontier |
| P092 | Abstention vs confabulation on underdetermined input | medium |
| P093 | Confabulation index over information gaps | medium |
| P094 | Ambiguity preservation vs forced disambiguation | medium |
| P095 | AV-hedge calibration against AR faithfulness | medium |
| P096 | Uncertainty-direction stability across seeds | frontier |
| P097 | Refusal / safe-completion encoding in the bottleneck | frontier |
| P098 | Numeric / range uncertainty preservation | medium |
| P099 | Known-unknown vs unknown-unknown distinction | frontier |
| P100 | Self-consistency across resamples as a confidence proxy | medium |

## Family K — Emergence & additive capability index  (`K_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P101 | Sufficiency: incremental decodability over trivial input features | high |
| P102 | Selectivity: confound-subtracted decodability | high |
| P103 | Graded / ordinal encoding (continuum, not just binary) | high |
| P104 | Abstraction transfer (cross-context, not memorized) | high |
| P105 | Effective-rank sweet spot (structured, not collapsed or noise) | high |
| P106 | Additive Emergence Index (the composite) | high |

## Family L — Literature-informed evaluations  (`L_*.md`)

| ID | Title | Feasibility |
|----|-------|-------------|
| P107 | Simulatability axis (ALMANACS) | medium |
| P108 | Steerability / read-write check (ActAdd / RepE) | high |
| P109 | Control-task selectivity as primary decodability statistic (Hewitt & Liang) | high |
| P110 | Explanation-scoring matcher vs keyword matching (auto-interp scoring) | medium |
| P111 | Developmental / longitudinal emergence across checkpoints (metric mirage guard) | frontier |
| P112 | Simulatability-minus-decodability gap (bottleneck utility beyond linear decode) | medium |
