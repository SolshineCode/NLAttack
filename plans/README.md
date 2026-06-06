# NLA Evaluation Plans (P001–P100)

A catalog of 100 evaluation plans for Natural Language Autoencoders, built on the
NLAttack thesis (a weak NLA is a **lossy concept filter**; the floor primitive is
*did concept C survive the bottleneck?*) and hardened by the Hermes design review
(`../DESIGN_REVIEW.md`): the AV verbalizer + matcher are extra lossy filters, so
every plan states how it handles that confound.

Each plan is falsifiable, weak-NLA-aware, and maps to the harness (`nla_eval/`).

## Plan schema

```
### P0XX — <title>
- **Family:** <A–J>
- **Probes:** <NLA property / failure mode being tested>
- **Hypothesis:** <falsifiable prediction>
- **Method:** <dataset + NLA call pattern + what is measured>
- **Metric:** <the number(s) reported>
- **Feasibility:** high | medium | frontier — <why, on genuinely weak NLAs>
- **Controls:** <Hermes confounds addressed: probe ground-truth / matcher
  ensemble / freq+length / faithfulness weighting / below-floor calibration>
- **Deception link:** <knowledge-asymmetry / misuse-detection relevance>
- **Harness:** <existing module or test, or new code needed>
- **Null looks like:** <what a null / negative result is — so silence ≠ success>
```

## Families (10 × 10 = 100)

| Fam | Range | Theme |
|---|---|---|
| A | P001–P010 | **Concept-survival & dropout-map** — retention curves, category-selective dropout, blind-spot stability, position/compression effects (the floor) |
| B | P011–P020 | **Content-adjacency & laundering** — neighbor substitution, hypernym drift, minimal-pair separation, polarity/antonym collapse |
| C | P021–P030 | **Deception & knowledge-asymmetry** — epistemic-state preservation, attribution/claim-status laundering, false-belief, sandbagging cues |
| D | P031–P040 | **ATT&CK / misuse detection** — tactic coverage, obfuscation see-through, kill-chain co-occurrence, dual-use, agentic-scaffolding, severity ordering |
| E | P041–P050 | **Bottleneck-probe / ground-truth** — linear/MLP probes on activations, probe-vs-AV gap, layer sweep, probe-direction stability, verbalizer ablation |
| F | P051–P060 | **Matcher & verbalizer confound** — ensemble agreement, decoding/temperature & template sensitivity, paraphrase invariance, human-vs-auto κ |
| G | P061–P070 | **Adaptive red-team / adversarial** — query-budget evasion, compositional blind spots, evasion transfer, monitor-gaming, steganographic capacity |
| H | P071–P080 | **Faithfulness / AR-fidelity** — cosine/MSE vs survival, error-weighted retention, AR concept drift in activation space, round-trip stability |
| I | P081–P090 | **Distributional / OOD / training-dependence** — encodable-subspace in-vs-OOD, domain transfer, frequency-tier interactions, checkpoint maturity, cross-model transfer |
| J | P091–P100 | **Calibration / uncertainty / self-knowledge** — does the bottleneck encode its own uncertainty, abstention handling, hallucinated specificity over gaps, ambiguity preservation |

## Feasibility legend
- **high** — runnable today on a weak NLA with the current harness (often CPU/probe-side).
- **medium** — needs modest new code or a working AV/server.
- **frontier** — expected-null or stretch on weak NLAs; run as a control, don't lean on it.

Family files: `A_concept_survival.md` … `J_calibration.md`. Master list: `INDEX.md`.
