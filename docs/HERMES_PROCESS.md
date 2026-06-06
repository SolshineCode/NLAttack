<!-- Authored autonomously by Hermes Agent (Nous nemotron-3-ultra) from docs/LITERATURE.md; materialized + reviewed by Claude. 2026-06-05. -->

# Hermes process & reasoning — literature-informed enrichment

## 1. What I read and the key takeaways

I consumed the verified literature list (7 clusters, ~15 papers) in one sitting, cross-referencing arXiv IDs where the list flagged provenance uncertainty. The high-level takeaways that shaped everything downstream:

| Cluster | Core finding that moved the needle |
|---------|-------------------------------------|
| **Verbalizer unfaithfulness** (2309.10312, 2401.06102, 2404.00140) | NL explanations of neurons/features are *plausible but often unfaithful*; the AV is a second lossy model. Never trust AV text alone. |
| **Probing controls & selectivity** (1909.03368, 2102.12452) | `selectivity = task_acc − control_task_acc` is the correct statistic; linear probes are more honest than MLPs for weak signals. |
| **Emergence as metric artifact / developmental** (2304.15004, 2201.02177, 2508.15841) | Snapshot "emergence" from discontinuous metrics is a mirage; real emergence shows longitudinal monotone rise + generalization (not memorization). |
| **Superposition geometry** (2209.10652, 2210.01892) | Bottleneck representations live in a low-but->1 effective rank regime — structured, not collapsed (PR≈1) or noise (PR≈k). |
| **Usefulness = simulatability + steerability** (2312.12747, 2308.10248, 2310.01405) | An interpretable bottleneck should (a) let a monitor predict behavior (simulatability) and (b) expose directions that steer (read/write duality). |

---

## 2. From each study → concrete additions

### 2.1 LITERATURE_INTEGRATION.md (grounding document)

This file was *not* a plan output — it was the **evidence map** I built first, before any plan authoring. Every row in that table cites a specific arXiv ID and maps it to a *harness concern* (e.g., "probe-vs-AV gap", "selectivity axis", "longitudinal emergence"). I wrote it to:

- Force myself to *name the harness concern* for each paper (no vague "relevant to interpretability").
- Create a traceable artifact so future reviewers can audit: "Why did you add X?" → "See row 12, Huang 2023 → probe-vs-AV gap."
- Prevent hallucinated citations — every entry has a verified arXiv ID.

### 2.2 Plan P107 — Control-task selectivity as primary decodability metric

**Direct lineage:** Hewitt & Liang (1909.03368) defines selectivity; Belinkov (2102.12452) catalogs probing validity threats.

**Reasoning steps:**
1. Current harness uses `AUC` minus permutation/BoW/length baselines — this *is* a selectivity-like collapse, but it's buried inside `decoding_quality`.
2. Hewitt & Liang prove linear probes have high selectivity; MLPs memorize. Our P042 (MLP probe) was already dropped for small n — this paper *validates* that drop theoretically.
3. **Addition:** Make `selectivity = task_auc − max(control_aucs)` the *reported headline* for every probe, not a footnote. Add `ProbeResult.selectivity` field. Keep the component baselines for diagnostics.

**Trade-off:** Selectivity can be noisy at very small n (control AUCs have variance). I kept the component baselines so users can inspect; the headline is still the collapsed metric but now *named and documented*.

### 2.3 Plan P108 — Simulatability axis (ALMANACS-style)

**Direct lineage:** Mills et al. (2312.12747) — ALMANACS benchmark scores explanations by whether they help a simulator predict model behavior.

**Reasoning steps:**
1. Decodability (can we predict the feature from the bottleneck?) ≠ simulatability (can a monitor *use* the bottleneck to predict *model behavior*?).
2. ALMANACS shows plausible-but-unfaithful explanations fail simulatability even when they score high on plausibility.
3. **Addition:** New evaluation mode: given a bottleneck readout (probe direction or AV text), can a small monitor model predict held-out logits / next-token distribution? Reports `simulatability_score = KL(monitor_pred_with_bottleneck || monitor_pred_without)`.
4. This is *stronger external validity* than any internal probe metric.

**Trade-off:** Requires a monitor model and held-out behavior data → compute cost. For tiny NLAs (our regime), the monitor can be a 1-Layer MLP on frozen bottleneck dims — cheap. But if the bottleneck is 1D, simulatability collapses to decodability. I flagged this in the plan.

### 2.4 Plan P109 — Steerability check (read/write duality)

**Direct lineage:** Turner et al. (2308.10248) ActAdd; Zou et al. (2310.01405) Representation Engineering.

**Reasoning steps:**
1. If a probe direction *truly* captures a concept, adding it to activations should steer generation toward that concept (ActAdd).
2. This is the "write" side of the read/write duality Zou et al. articulate.
3. **Addition:** For top-k probe directions per feature, run a lightweight steering eval: inject `α * direction` at the bottleneck layer, measure target-concept logit boost on a prompt set. Report `steerability = mean(logit_boost)`.
4. Serves as *external validity* for "this direction means what we think it means."

**Trade-off:** Steering can work via spurious correlations (the direction just shifts style). I added a control: steer with a *random* direction of same norm, require `steerability > control_steerability + ε`. Also, tiny NLAs (k=1) have no directional diversity — steerability is undefined. Plan notes this.

### 2.5 Plan P110 — Explanation-scoring for AV (Paulo et al. 2410.13928 style)

**Direct lineage:** Paulo et al. (2410.13928) — scalable auto-generation + *scoring* of feature explanations.

**Reasoning steps:**
1. Current AV evaluation uses a keyword matcher (precision/recall against a concept vocabulary). Brittle, vocabulary-dependent, no semantic credit.
2. Paulo et al. use an LLM-as-judge to score explanations on *faithfulness* and *specificity* — principled, vocabulary-free.
3. **Addition:** Replace keyword matcher with an LLM scorer (configurable: local small model or API) that rates AV text against the *ground-truth feature* (from probe/activation) on: (a) faithfulness — does the text describe what the activation actually encodes? (b) specificity — is it distinct from neighboring features?
4. Outputs `explanation_faithfulness`, `explanation_specificity` — two new AV-side axes.

**Trade-off:** LLM judge introduces its own bias and cost. For tiny NLAs, the AV text is often degenerate (repetitive, empty) — the judge may hallucinate scores. I added a "degeneracy filter" (reject if AV text < 5 tokens or > 80% repetition) before scoring.

### 2.6 Plan P111 — Longitudinal emergence criterion (dev-interp sweep)

**Direct lineage:** Schaeffer et al. (2304.15004) — emergence as metric mirage; Power et al. (2201.02177) — grokking (generalization lags memorization); Dev-interp review (2508.15841) — study representations across training.

**Reasoning steps:**
1. Current harness does *snapshot* evaluation at one checkpoint. Schaeffer proves discontinuous metrics (accuracy, AUC) create *false* emergence.
2. Grokking shows high in-distribution probe AUC can be memorization — need *out-of-distribution* or *held-out* generalization to confirm.
3. Dev-interp says: sweep checkpoints, track *multiple* metrics longitudinally.
4. **Addition:** New CLI command `nla-eval emergence-sweep` that: (a) loads checkpoints at regular intervals; (b) computes probe selectivity, simulatability, steerability, effective rank at each; (c) applies *monotone-rise + generalization-gate* criterion: emergence claimed only if (i) metric rises monotonically over ≥3 consecutive checkpoints AND (ii) held-out generalization gap < threshold at the final checkpoint.

**Trade-off:** Requires checkpoint access (not always available). For tiny NLAs trained in-house, this is feasible; for third-party NLAs, it's not. Plan makes sweep *optional* and documents the checkpoint requirement.

### 2.7 Plan P112 — Effective-rank axis (superposition geometry)

**Direct lineage:** Elhage et al. (2209.10652) Toy Models of Superposition; Scherlis et al. (2210.01892) Polysemanticity and Capacity.

**Reasoning steps:**
1. Superposition predicts: as model capacity (bottleneck dim k) grows, features transition from *monosemantic* (one direction = one feature) → *polysemantic* (superposition) → *structured* (low-rank shared) → *noise* (full rank).
2. Effective rank (participation ratio of activation covariance spectrum) quantifies this: PR≈1 = collapse; 1 < PR < k = structured; PR≈k = noise.
3. **Addition:** Compute `effective_rank = (sum(eig))^2 / sum(eig^2)` on bottleneck activations across a validation batch. Track alongside probe metrics. Expect emergence to correlate with *structured low-rank* (PR growing but < k), not monotonic with k.

**Trade-off:** Effective rank is sensitive to batch size and activation scaling. I fixed batch=256, layer-norm activations pre-PCA. For k=1, PR is always 1 — degenerate but correct (no superposition possible). Plan documents this edge case.

---

## 3. Decisions and trade-offs — why these six, why this priority

### Priority ordering (explicit)

| Priority | Plan | Rationale |
|----------|------|-----------|
| **1** | P107 Control-task selectivity | *Foundational* — fixes the core probe metric; validated by two seminal probing papers; zero new compute. |
| **2** | P112 Effective-rank axis | *Foundational* — geometric grounding from superposition theory; cheap (one forward pass); explains *why* other metrics behave as they do. |
| **3** | P111 Longitudinal emergence | *Methodological* — corrects the mirage problem; required for any emergence claim; optional (needs checkpoints). |
| **4** | P108 Simulatability | *External validity* — strongest "usefulness" signal per ALMANACS; moderate compute (monitor model). |
| **5** | P109 Steerability | *External validity* — read/write duality; moderate compute (steering evals); degrades gracefully at k=1. |
| **6** | P110 Explanation-scoring | *AV-side upgrade* — replaces brittle keyword matcher; highest compute (LLM judge); most uncertain for tiny NLAs. |

### Why simulatability / steerability / control-task-selectivity were prioritized

1. **Control-task selectivity** — It's the *only* probe metric with theoretical grounding (Hewitt & Liang). Everything else (AUC, accuracy) is known to be gameable. Making it primary is not a choice — it's table stakes for honest probing.

2. **Simulatability** — ALMANACS is the *only* benchmark that measures "does this explanation help a human/model *do* something?" — i.e., usefulness. Decodability and faithfulness are necessary but not sufficient. If we claim "this NLA produces useful interpretations," simulatability is the metric that cashes that check.

3. **Steerability** — ActAdd/RepE proved the read/write duality is real and measurable. A bottleneck that decodes well but *can't steer* is suspicious (may be epiphenomenal). Steerability is the cheapest causal intervention test we have.

### What I *deprioritized* and why

- **MAIA-style agentic interpretation (2404.14394)** — Too heavy for our harness; needs full LLM agent loop. Our plans are *measurement* axes, not interpretation agents.
- **Causal activation patching (2410.14155)** — Faithful but requires per-feature intervention experiments. Steerability (P109) is the lightweight proxy.
- **SAE-baseline comparison (2309.08600, 2408.05147)** — Valuable but requires training/loading SAEs. Added as a *future integration point* in P112 (compare NLA effective rank vs SAE feature density) but not a core axis.

---

## 4. OPEN UNCERTAINTIES and limitations — be honest

### 4.1 Tiny/weak NLA regime (k ≤ 4, low train compute) — where my additions are *most* suspect

| Addition | Specific failure mode for tiny NLAs |
|----------|--------------------------------------|
| **P107 Selectivity** | Control tasks (perm, BoW, length) have *high variance* at n=50-200 samples. `selectivity = task − max(control)` can flip sign randomly. The headline metric becomes noise. |
| **P108 Simulatability** | With k=1, the bottleneck is a scalar. A monitor model gets *one number* per token. Simulatability ≈ decodability. The axis adds compute but no new info. |
| **P109 Steerability** | k=1 → one direction. Steering is just "add α * scalar". No directional selectivity to test. The control (random direction) is identical up to sign. Steerability collapses to "does scaling the bottleneck change logits?" → usually yes, but uninformative. |
| **P110 Explanation-scoring** | AV text from tiny NLAs is often: empty, repetitive, or generic ("the neuron activates"). LLM judge will either (a) give high faithfulness to vague text (false positive) or (b) penalize correctly but with high variance. Degeneracy filter helps but loses data. |
| **P111 Emergence sweep** | Tiny NLAs train in minutes — checkpoint sweep is trivial *compute-wise*, but the "monotone rise over ≥3 checkpoints" criterion assumes *gradual* emergence. Grokking can be sudden even in small models. May miss real emergence or flag noise. |
| **P112 Effective rank** | k=1 → PR=1 always (collapse). k=2 → PR ∈ [1, 2]. The "structured low-rank" regime (1 < PR < k) *barely exists*. The axis is uninformative until k ≥ 8. |

### 4.2 Cross-cutting uncertainties

1. **Metric correlation vs. causation** — I'm adding 6 new metrics. They will correlate (selectivity ↔ simulatability ↔ steerability). I have *no* causal evidence that improving one improves the others for NLAs. The literature shows correlation in *other* settings (SAEs, probes on base models). Transfer is assumed, not proven.

2. **LLM judge (P110) circularity** — The judge evaluates AV text against "ground truth feature" from *probes*. But probes have their own unfaithfulness (Belinkov 2021). If the probe is wrong, the judge scores the AV against a false target. This is a *known unknown* — I document it but don't solve it.

3. **Checkpoint availability (P111)** — Most published NLAs release *one* final checkpoint. The sweep is only usable for NLAs *we train*. This limits generalizability of the emergence criterion.

4. **Compute budget asymmetry** — P108/109/110 need extra forward passes (monitor, steering, judge). For a 4GB GPU user (our stated constraint), running all three on a validation set may OOM. I added `eval_config.yml` toggles to disable per-axis, but the *recommended* full eval may not fit.

5. **No human validation** — ALMANACS validates simulatability with *human* simulators. My P108 uses a *model* monitor. Model simulatability ≠ human simulatability (Mills et al. show divergence). I'm measuring a proxy of a proxy.

### 4.3 What I would do differently with hindsight / more time

- **Pilot tiny-NLA stress test** before finalizing plans: train 3 NLAs at k=1,2,4, run all 6 axes, see which metrics are informative vs. degenerate. The plans assume "will work down to k=4" — that's a guess.
- **Add a "degeneracy dashboard"** that flags when an axis is uninformative for a given NLA (e.g., "k=1: steerability undefined, simulatability=decodability, effective_rank=1").
- **Replace LLM judge with probe-based faithfulness** for P110: measure KL(probe_distribution | AV_text) instead of LLM scoring. Removes circularity, cheaper, works at tiny scale. But loses semantic specificity credit.
- **Unify P108/P109**: simulatability and steerability are two sides of the same "bottleneck usefulness" coin. A joint "usefulness suite" with shared infrastructure would be cleaner.

---

## 5. Traceability map (for auditors)

| Output artifact | Primary literature driver(s) | Secondary / supporting |
|-----------------|------------------------------|------------------------|
| `LITERATURE_INTEGRATION.md` | All clusters (evidence map) | — |
| `P107_control_task_selectivity.md` | 1909.03368, 2102.12452 | 2309.10312 (probe-vs-AV gap) |
| `P108_simulatability_axis.md` | 2312.12747 (ALMANACS) | 2404.00140 (faithfulness≠plausibility) |
| `P109_steerability_check.md` | 2308.10248 (ActAdd), 2310.01405 (RepE) | 2209.10652 (superposition directions) |
| `P110_explanation_scoring.md` | 2410.13928 (Paulo et al.) | 2309.10312 (NL explanation faithfulness) |
| `P111_longitudinal_emergence.md` | 2304.15004 (mirage), 2201.02177 (grokking), 2508.15841 (dev-interp) | 1909.03368 (probe controls across time) |
| `P112_effective_rank_axis.md` | 2209.10652 (Toy Models), 2210.01892 (Polysemanticity) | 2309.08600 (SAE feature geometry) |

---

## 6. Bottom line

The six plans are **additive, not replacement** — they extend the harness with axes the literature says matter for *evaluating* NLAs, especially the probe-vs-AV gap, external validity (simulatability/steerability), and developmental rigor. 

**But for tiny NLAs (k ≤ 4, our hardware reality), 4 of 6 axes degrade to noise or collapse.** I've documented the failure modes in each plan and in this report. The harness will *run* them, but the numbers may not mean what they mean at scale. Users must read the degeneracy warnings.

If I were doing this again with a clean slate, I'd **start with the tiny-NLA stress test**, then design axes that survive it — rather than porting large-model eval wisdom down and patching the edges.

