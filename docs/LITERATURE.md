# Literature informing NLA evaluation (verified arXiv list + synthesis)

Compiled via an automated arXiv search, then cross-checked against direct arXiv
searches; only IDs confirmed in those searches are kept. Each entry
notes relevance to *evaluating NLAs, especially weak/tiny ones*. NLA =
`activation → AV verbalizer → text → AR reconstructor → activation`.

> Provenance/uncertainty: arXiv IDs below were each seen in a verified search.
> Author lists are abbreviated ("et al."); verify exact author order on arXiv
> before formal citation. A few well-known papers are included from domain
> knowledge and flagged.

## 1. Verbalizing / decoding activations to text (the NLA mechanism)
- **Patchscopes: A Unifying Framework for Inspecting Hidden Representations of LMs** — Ghandeharioun et al. — arXiv:2401.06102 (2024). Decodes a hidden state by patching it into a fresh prompt that elicits a natural-language readout — the conceptual ancestor of an AV. *Relevance:* the AV is a learned, lossy Patchscope; their faithfulness caveats motivate our probe-vs-AV gap.
- **Rigorously Assessing Natural Language Explanations of Neurons** — Huang et al. — arXiv:2309.10312 (2023). Shows NL explanations of units are often unfaithful even when plausible. *Relevance:* direct warning that AV text ≠ ground truth → our bottleneck-probe controls.
- **A Multimodal Automated Interpretability Agent (MAIA)** — Shaham et al. — arXiv:2404.14394 (2024). Agentic experiment-driven interpretation. *Relevance:* template for automated, but needs the verification scaffolding we add.
- **Automatically Interpreting Millions of Features in LLMs** — Paulo et al. — arXiv:2410.13928 (2024). Scalable auto-generation + scoring of feature explanations. *Relevance:* the explanation-scoring methodology informs our matcher/faithfulness axes.
- **Linear Explanations for Individual Neurons** — arXiv:2405.06855 (2024). *Relevance:* simple linear readouts as a floor verbalizer (cf. our forced-probe verbalizer P046).

## 2. Sparse autoencoders / dictionary learning (the bottleneck's cousin)
- **Sparse Autoencoders Find Highly Interpretable Features in LMs** — Cunningham et al. — arXiv:2309.08600 (2023). *Relevance:* SAEs vs NLAs both extract concepts from activations; their eval pitfalls transfer.
- **Gemma Scope: Open SAEs Everywhere All At Once on Gemma 2** — Lieberum et al. — arXiv:2408.05147 (2024). *Relevance:* the SAE suite for the exact model family our local NLA targets; a probe/feature baseline.
- **Toy Models of Superposition** — Elhage et al. — arXiv:2209.10652 (2022). Phase transitions between mono/polysemantic regimes by sparsity. *Relevance:* grounds our effective-rank axis (P105) and the "structured vs collapsed vs noise" emergence reading.
- **Polysemanticity and Capacity in Neural Networks** — Scherlis et al. — arXiv:2210.01892 (2022). *Relevance:* capacity framing for what a low-dim bottleneck can encode at all.

## 3. Probing methodology — controls, selectivity, pitfalls (our null backbone)
- **Designing and Interpreting Probes with Control Tasks** — Hewitt & Liang — arXiv:1909.03368 (2019). Defines **selectivity = task acc − control-task acc**; linear probes have high selectivity. *Relevance:* the exact logic of our selectivity axis (P102) and a strong argument for linear (not MLP) probes on weak NLAs (validates dropping P042 at small n).
- **Probing Classifiers: Promises, Shortcomings, and Advances** — Belinkov — arXiv:2102.12452 (2021). Survey of probing validity threats. *Relevance:* motivates permutation/control baselines on every probe (our `ProbeResult.signal`).

## 4. Evaluating interpretability — faithfulness vs plausibility, simulatability
- **ALMANACS: A Simulatability Benchmark for LM Explainability** — Mills et al. — arXiv:2312.12747 (2023). Explanations scored by whether they help predict model behavior. *Relevance:* the "monitor reads the bottleneck" framing = simulatability; a target metric for a *useful* NLA.
- **Does Faithfulness Conflict with Plausibility?** — arXiv:2404.00140 (2024). *Relevance:* our faithfulness-vs-plausibility separation (AV plausibility ≠ bottleneck faithfulness).
- **Towards Faithful NL Explanations via Activation Patching** — arXiv:2410.14155 (2024). *Relevance:* causal grounding of explanations; complements our activation-space faithful-rank axis (P080).
- **Mechanistic Interpretability for AI Safety — A Review** — Bereska & Gavves — arXiv:2404.14082 (2024). *Relevance:* situates NLAs among interp methods and their evaluation gaps.

## 5. Emergence / developmental dynamics (snapshot vs longitudinal)
- **Are Emergent Abilities of LLMs a Mirage?** — Schaeffer et al. — arXiv:2304.15004 (2023). Apparent emergence can be an artifact of discontinuous metrics. *Relevance:* the core reason we (a) dropped the masking weighted-mean for tiers and (b) require a *longitudinal* monotone-rise criterion, not a snapshot, to claim "emergence."
- **Grokking: Generalization Beyond Overfitting** — Power et al. — arXiv:2201.02177 (2022). Delayed generalization after overfit. *Relevance:* why stability + abstraction (generalization) gate emergence — a high in-dist probe can be memorization pre-grok.
- **A Review of Developmental Interpretability in LLMs** — arXiv:2508.15841 (2025). Studying representations *across training*. *Relevance:* the checkpoint-sweep emergence curve is a developmental-interpretability instrument.

## 6. Superposition / effective dimensionality
- (Toy Models 2209.10652 and Polysemanticity 2210.01892, above) — *Relevance:* directly underpin the effective-rank axis (P105): emergence shows shared low-but->1 rank, not collapse (PR≈1) or noise (PR≈k).

## 7. Steering / representation reading (downstream usefulness)
- **Activation Addition: Steering LMs Without Optimization (ActAdd)** — Turner et al. — arXiv:2308.10248 (2023). Contrastive steering vectors. *Relevance:* a *usable* bottleneck should expose directions that also steer — an external validity check on "interpretable capability."
- **Representation Engineering: A Top-Down Approach to AI Transparency** — Zou et al. — arXiv:2310.01405 (2023). Reading/controlling concept directions. *Relevance:* the read/write duality; our probe directions are the "read" side.

---

## Synthesis — what the literature implies for this harness

1. **The verbalizer is a second lossy model, and NL explanations are often
   unfaithful** (2309.10312, 2401.06102, 2404.00140). → Vindicates the central
   move: never trust AV text alone; measure the **bottleneck directly** and report
   the probe−AV gap.
2. **Probing needs controls; selectivity = task − control; linear > MLP for
   honesty** (1909.03368, 2102.12452). → Our permutation null, the
   `decoding_quality = AUC − max(perm, BoW, length)` collapse, and the choice of
   linear probes (dropping MLP P042 at small n) are textbook-correct.
3. **Emergence claims from snapshots/discontinuous metrics can be mirages**
   (2304.15004); generalization can lag (grokking, 2201.02177); study it
   *developmentally* (2508.15841). → Hence tiers (not a smooth masking mean), the
   abstraction gate, and the **longitudinal** emergence criterion.
4. **Superposition predicts the geometry** (2209.10652, 2210.01892). → The
   effective-rank axis is the right shape; expect structured low-rank, not
   collapse or noise.
5. **Usefulness = simulatability + steerability** (2312.12747, 2308.10248,
   2310.01405). → Future axes: does a monitor reading the bottleneck *predict
   behavior* (simulatability), and do probe directions *steer*? These are the
   strongest external-validity targets for "useful interpretability capability."

## Concrete gaps this suggests adding (candidates for additive enrichment)
- A **simulatability axis** (ALMANACS-style): can a cheap model predict held-out
  behavior from the bottleneck readout? (stronger than decodability).
- A **steerability check**: do probe/AV directions move model behavior (read/write
  duality) — external validity for "interpretable".
- **Explanation-scoring** (2410.13928 style) as a principled alternative to the
  keyword matcher for the AV side.
- Adopt **control-task selectivity** (1909.03368) as the primary, not secondary,
  decodability statistic.
