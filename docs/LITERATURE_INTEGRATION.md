
# Literature Integration: Concrete Harness Implications

---

## DO NOW (cheap, valid on weak/tiny NLAs)

### 1. Hewitt & Liang 1909.03368 — *Designing and Interpreting Probes with Control Tasks*
**Harness change:** Make **selectivity = task_AUC − control_AUC** the *primary* decodability metric on every probe; demote raw AUC to secondary display. Implement the control task as the existing label-permutation null (already present) and ensure `ProbeResult.signal` stores `selectivity` explicitly. This validates dropping MLP probes (P042) at small *n* — linear probes have higher selectivity per the paper.

### 2. Huang et al. 2309.10312 — *Rigorously Assessing Natural Language Explanations of Neurons*
**Harness change:** Add a **probe−AV gap report** to every concept evaluation: `gap = bottleneck_probe_selectivity − AV_verbalizer_accuracy`. Log this as `ConceptResult.probe_av_gap`. The paper shows NL explanations are often plausible but unfaithful; this gap quantifies exactly that failure mode for the AV.

### 3. Schaeffer et al. 2304.15004 — *Are Emergent Abilities of LLMs a Mirage?*
**Harness change:** Harden the **longitudinal emergence criterion**: require monotone non-decreasing `decoding_quality` across ≥3 consecutive checkpoints *and* a minimum effect size (Cohen's *d* ≥ 0.5 vs. first checkpoint). Replace any single-checkpoint "emerged" flags with this longitudinal gate. The paper proves snapshot/discontinuous metrics produce false emergence.

### 4. Cunningham et al. 2309.08600 — *Sparse Autoencoders Find Highly Interpretable Features in LMs*
**Harness change:** Add **SAE baseline probes** to the harness: train a linear probe on the SAE latent activations (Gemma Scope SAEs for the target model) for each concept and report `sae_selectivity` alongside `nla_selectivity`. This gives a fixed, strong baseline to calibrate whether the NLA bottleneck is actually extracting features or just passing noise.

### 5. Elhage et al. 2209.10652 — *Toy Models of Superposition*
**Harness change:** Implement the **effective-rank axis (P105)** as: `participation_ratio = (sum(s))^2 / sum(s^2)` on bottleneck activation singular values across a concept batch. Gate the "structured" emergence tier on `1.5 < PR < 0.8 * bottleneck_dim`. This directly operationalizes the paper's phase transitions: collapse (PR≈1), structured (low-but->1), noise (PR≈k).

### 6. Belinkov 2102.12452 — *Probing Classifiers: Promises, Shortcomings, and Advances*
**Harness change:** Extend the permutation null to **three control baselines per probe**: (a) label permutation, (b) bag-of-words shuffle, (c) length-matched random. Store all three in `ProbeResult.controls` and define `decoding_quality = AUC − max(controls)`. The paper catalogs validity threats; this multi-control collapse is the textbook mitigation.

---

## LATER (needs more compute, stronger NLAs, or downstream integration)

### 7. Mills et al. 2312.12747 — *ALMANACS: A Simulatability Benchmark for LM Explainability*
**Harness change:** Add a **simulatability axis**: train a tiny monitor (1–2 layer MLP) on bottleneck readouts to predict *held-out model behavior* (next-token distribution KL, or specific circuit output). Report `simulatability_score = 1 − KL(monitor_pred || true_behavior)`. This is stronger than decodability — it tests whether the bottleneck *causally supports* behavior prediction.

### 8. Turner et al. 2308.10248 — *Activation Addition: Steering LMs Without Optimization (ActAdd)*
**Harness change:** Add a **steerability check**: for each concept with high probe selectivity, extract the probe direction, scale by ±2σ, add to residual stream at the bottleneck layer, and measure behavioral change on a targeted eval (e.g., sentiment flip rate). Report `steerability_effect_size`. Read/write duality = external validity for "interpretable capability."

### 9. Paulo et al. 2410.13928 — *Automatically Interpreting Millions of Features in LLMs*
**Harness change:** Replace the keyword matcher for AV evaluation with an **explanation-scoring pipeline**: (a) generate candidate explanations from AV text via LLM, (b) score each by simulating with the explanation and measuring token-level F1 vs. true activations, (c) report `explanation_faithfulness = max(scored_explanations)`. This is a principled alternative to brittle keyword matching.

### 10. Zou et al. 2310.01405 — *Representation Engineering: A Top-Down Approach to AI Transparency*
**Harness change:** Unify probe directions and steering vectors: store `ConceptResult.concept_direction` (normalized probe weight) and use it for *both* the linear probe readout *and* the ActAdd steering intervention. Report alignment `cosine(probe_dir, steer_dir)` as a coherence metric. The read/write duality should be literal in the code.

### 11. 2404.00140 — *Does Faithfulness Conflict with Plausibility?*
**Harness change:** Split every concept report into two panels: **Faithfulness** (bottleneck probe selectivity, simulatability, steerability) and **Plausibility** (AV text quality, human rating, explanation F1). Make the final verdict require *both* above threshold. The paper shows they diverge; the harness must surface this explicitly.

### 12. 2410.14155 — *Towards Faithful NL Explanations via Activation Patching*
**Harness change:** Add an **activation-patching faithfulness check**: for top-k concepts, patch the bottleneck activation with the concept direction ablated, measure output logit change on concept-relevant prompts, and correlate with probe selectivity. Report `patch_correlation`. Causal grounding beyond observational probes.

---

## Summary Table

| Priority | Paper | Harness Change | Component Touched |
|----------|-------|----------------|-------------------|
| DO NOW | 1909.03368 | Selectivity as primary metric; log `probe_av_gap` | Bottleneck probes, AV gap |
| DO NOW | 2309.10312 | Probe−AV gap report per concept | ConceptResult |
| DO NOW | 2304.15004 | Longitudinal monotone + effect-size gate | Emergence verdict |
| DO NOW | 2309.08600 | SAE baseline probes (Gemma Scope) | Probe runner |
| DO NOW | 2209.10652 | Effective-rank (participation ratio) axis + tier gate | Emergence tiers |
| DO NOW | 2102.12452 | Three-control max collapse for decoding_quality | ProbeResult, decoding_quality |
| LATER | 2312.12747 | Simulatability axis (monitor → behavior) | New axis |
| LATER | 2308.10248 | Steerability check (ActAdd on probe dirs) | New axis |
| LATER | 2410.13928 | Explanation-scoring pipeline for AV | AV evaluation |
| LATER | 2310.01405 | Unified concept_direction for probe + steer | ConceptResult |
| LATER | 2404.00140 | Faithfulness vs Plausibility split panels | Reporting |
| LATER | 2410.14155 | Activation-patching faithfulness correlation | Probe validation |
