# Family L — Literature-informed evaluations

<!-- Plans P107-P112 authored autonomously by Hermes Agent (Nous nemotron-3-ultra) from docs/LITERATURE.md; materialized + reviewed by Claude. 2026-06-05. -->

# Family L — Literature-informed evaluations

### P107 — Simulatability axis (ALMANACS)
- **Family:** L
- **Probes:** A frozen "monitor" LM (small, e.g., 1.5B-7B) receives NLA bottleneck readout + context, predicts held-out model behavior on downstream task (next-token, classification, or reasoning step).
- **Hypothesis:** If bottleneck captures decision-relevant information, a cheap monitor should predict behavior significantly above context-only baseline.
- **Method:** Sample prompts from GSM8K, MMLU, and a synthetic logic dataset. For each: run base model → NLA bottleneck → monitor predicts answer token / label. Compare to: (a) context-only monitor, (b) full-activation monitor (upper bound). Use linear probe on bottleneck as monitor for minimal capacity.
- **Metric:** ΔAccuracy = Acc(bottleneck+context) − Acc(context-only). Normalize by full-activation ceiling. Report AUC across prompts.
- **Feasibility:** medium — needs a second model for monitor; can use API or small local model. Weak NLAs may have ΔAccuracy ≈ 0.
- **Controls:** Permute bottleneck dims (kills structure) → ΔAccuracy should collapse. Random bottleneck from same marginal distribution → same. Task-mismatched prompts (e.g., monitor trained on GSM8K, eval on MMLU) → transfer test.
- **Deception link:** Deceptive models may hide intent in activations not captured by bottleneck; simulatability gap flags this.
- **Harness:** New module `simulatability.py` wrapping token-generation + monitor prediction loop; reuse `activations.py` for extraction.
- **Null looks like:** ΔAccuracy ≤ 0.02 (within noise of context-only) on all datasets; permuted and random bottlenecks match original.

### P108 — Steerability / read-write check (ActAdd / RepE)
- **Family:** L
- **Probes:** Extract probe directions (linear classifier weights) for target concepts (truthful/deceptive, harmful/benign, factual/hallucinated) from bottleneck. Add scaled direction to bottleneck during generation (ActAdd) or to residual stream (RepE), measure behavior shift.
- **Hypothesis:** Directions that decode from bottleneck also causally steer model behavior; magnitude of steering correlates with probe AUC.
- **Method:** Train linear probes on bottleneck for concept labels on contrastive pairs (e.g., TruthfulQA, HarmBench, Sycophancy). For each probe: inject α·w into bottleneck (α ∈ {−2,−1,−0.5,0,0.5,1,2}) during generation on held-out prompts. Measure target behavior rate (refusal, truthfulness, etc.).
- **Metric:** Steering slope = d(behavior_rate)/dα at α=0. Correlation ρ(probe_AUC, steering_slope) across concepts.
- **Feasibility:** high — local forward passes with hook injection; no extra model. Weak NLAs: probe AUC may be low, steering slope may be flat (negative control).
- **Controls:** (1) Inject random unit-norm direction (null). (2) Inject probe direction trained on permuted labels. (3) Inject at wrong layer. (4) BoW/length-matched control prompts.
- **Deception link:** If deceptive concept direction steers toward/away from deception, bottleneck captures causal mechanism.
- **Harness:** Extend `activations.py` with `steer_generation(direction, alpha, layer)`; reuse probe training from `probes.py`.
- **Null looks like:** Steering slope ≈ 0 for all α; ρ(AUC, slope) ≈ 0; random/permuted directions match true directions.

### P109 — Control-task selectivity as primary decodability statistic (Hewitt & Liang)
- **Family:** L
- **Probes:** Linear probes on bottleneck for target tasks. For each task probe, train matched control probe on same bottleneck with randomized labels (permutation) or on construct-irrelevant but distribution-matched control task (e.g., part-of-speech for sentiment).
- **Hypothesis:** Selectivity = task_AUC − control_AUC is a stricter, less gamed measure of decodability than raw AUC; it filters probing memorization and dataset artifacts.
- **Method:** For each axis dataset (truthfulness, sycophancy, refusal, etc.): train linear probe on bottleneck → task_AUC. Train 10 permuted-label controls → mean perm_AUC. Train 3 construct-irrelevant controls (length, BoW, POS) → mean irr_AUC. Selectivity = task_AUC − max(perm_AUC, irr_AUC).
- **Metric:** Primary: Selectivity. Secondary: raw task_AUC, control_AUCs. Report distribution across seeds.
- **Feasibility:** high — reuses existing probe training; permutation controls are free. Works at any NLA strength.
- **Controls:** (1) Per-dimension permutation (preserves marginals, kills structure). (2) BoW control probe on token counts. (3) Length control. (4) Cross-task generalization: probe trained on Task A, tested on Task B control labels.
- **Deception link:** High selectivity on deception probes = bottleneck isolates deceptive signal, not just correlates.
- **Harness:** Modify `probes.py` to return `ProbeResult.selectivity` as primary field; add control-task registry.
- **Null looks like:** Selectivity ≤ 0.02 (within permutation noise); task_AUC ≈ control_AUC; permuted and BoW controls match or exceed task.

### P110 — Explanation-scoring matcher vs keyword matching (auto-interp scoring)
- **Family:** L
- **Probes:** Replace keyword-based AV matcher with explanation-scoring: feed AV-output concept description + held-out activation examples to a judge LM (or learned scorer), score faithfulness. Compare to keyword matcher and human-eval subset.
- **Hypothesis:** Explanation-scoring (2410.13928 style) correlates with human faithfulness judgments better than keyword overlap, and exposes AV hallucinations keyword matcher misses.
- **Method:** For each NLA unit/direction: generate AV description on train set. On held-out activations: (a) keyword matcher score, (b) LLM judge scores "Does description match this activation?" (1-5), (c) simulatability (P107) as ground truth. Compare Spearman ρ(keyword, judge), ρ(judge, simulatability), ρ(keyword, simulatability).
- **Metric:** ρ(judge, simulatability) > ρ(keyword, simulatability) + 0.15. Judge AUC on faithfulness classification (human-labeled subset).
- **Feasibility:** medium — requires LLM judge calls (API or local). Can batch. Weak NLAs: AV descriptions may be generic; judge should penalize.
- **Controls:** (1) Shuffle AV descriptions across units → judge score should drop. (2) Template descriptions ("this unit activates on X") → judge should detect vagueness. (3) Human eval on 50 units for validation.
- **Deception link:** Keyword matcher may match "deception" tokens in AV text even when activation doesn't encode deception; judge + simulatability catch this.
- **Harness:** New `explanation_scoring.py` with judge prompt templates; reuse `activations.py` for held-out examples.
- **Null looks like:** ρ(judge, simulatability) ≤ ρ(keyword, simulatability); shuffled descriptions score same as true; judge AUC ≈ 0.5 on human subset.

### P111 — Developmental / longitudinal emergence across checkpoints (metric mirage guard)
- **Family:** L
- **Probes:** Sweep NLA training checkpoints (or base model checkpoints if NLA trained on top). Compute full eval suite (selectivity, simulatability, steerability, effective rank) at each. Track monotonicity and emergence shape.
- **Hypothesis:** Genuine emergence shows smooth, monotonic rise across checkpoints on *multiple* axes (not one discontinuous jump); metric mirages show sharp threshold on discontinuous metrics (accuracy) but smooth on continuous (AUC, selectivity).
- **Method:** Use Gemma-2-2B checkpoints (public via Gemma Scope) or NLA checkpoints saved during training. Evaluate at 8-10 evenly spaced steps. Metrics: selectivity (P109), simulatability ΔAcc (P107), steering slope (P108), effective rank (P105). Fit emergence curve: step function vs logistic vs linear.
- **Metric:** Monotonicity score = fraction of adjacent pairs where metric increases. Emergence sharpness = max derivative of fitted logistic / linear slope. Require: (a) monotonicity ≥ 0.7 on ≥3 axes, (b) sharpness < 5× linear slope on continuous metrics.
- **Feasibility:** frontier — requires many checkpoints; may not exist for custom NLA. Can use base model checkpoints + retrain NLA on each (expensive). Alternative: train NLA once, evaluate on base model checkpoints (activation distribution shift).
- **Controls:** (1) Permuted-label probes at each checkpoint → should stay flat. (2) Random bottleneck probe → flat. (3) Discontinuous metric (accuracy@k) vs continuous (AUC) comparison on same data.
- **Deception link:** Deceptive capabilities may emerge late/non-monotonically; longitudinal tracking catches delayed emergence.
- **Harness:** New `longitudinal.py` orchestrating checkpoint sweep; reuses all axis modules. Checkpoint loader for Gemma/HF.
- **Null looks like:** All metrics flat (monotonicity ≈ 0.5) or only accuracy@k jumps while AUC/selectivity/simulatability stay flat (mirage signature).

### P112 — Simulatability-minus-decodability gap (bottleneck utility beyond linear decode)
- **Family:** L
- **Probes:** Compute both simulatability (P107) and selectivity (P109) for each concept/direction. Gap = simulatability − selectivity (normalized). Positive gap = bottleneck has non-linear/structured info usable by monitor but not linear probe.
- **Hypothesis:** Useful bottlenecks show positive gap for high-level concepts (reasoning, deception, intent); random/weak bottlenecks show gap ≈ 0 or negative (monitor overfits).
- **Method:** For each concept in axis suite: train linear probe → selectivity. Train monitor (linear or 1-layer MLP on bottleneck) → simulatability ΔAcc. Compute gap. Correlate gap with concept abstraction level (operationalized by human ratings or depth in causal graph).
- **Metric:** Mean gap across concepts. Gap > 0.05 for ≥50% of high-abstraction concepts. Correlation ρ(gap, abstraction_level).
- **Feasibility:** medium — runs P107 + P109; adds monitor training. Weak NLAs: both low, gap noisy.
- **Controls:** (1) Permuted bottleneck → gap ≈ 0. (2) Full activations → large positive gap (upper bound). (3) Random projection of bottleneck → gap ≤ 0.
- **Deception link:** Deception may be high-abstraction; positive gap means bottleneck captures it non-linearly (harder to detect with probes alone).
- **Harness:** Composite script `gap_analysis.py` calling P107 and P109 internals; aggregates per-concept.
- **Null looks like:** Mean gap ≤ 0.01; ρ(gap, abstraction) ≈ 0; permuted and random projections match true bottleneck.

