# NLAttack roadmap — from two independent reviews (2026-06-08)

Synthesis of a targeted review (verbalizer-axes critique) + an open-ended,
web-grounded review (read the Anthropic/transformer-circuits/Neuronpedia NLA
sources). Records what shipped now and what's gated on resources.

## Shipped now (this branch — CPU, additive)
- **`minimal_pair_discrimination` rebuilt** → AUC/d-prime over the matcher's
  *continuous* pos-vs-neg scores (chance 0.5), label-permutation p-value, and a
  `matcher_ceiling_auc` covariate (separates AV signal from matcher quality).
  *The single highest-value add per both sessions.* Ships with **`build_attack_pairs`**
  so the default demo is the **ATT&CK laundering-discrimination** test
  (attack term vs its benign neighbor — can the monitor tell "credential dumping"
  from "password management"?).
- **`doc_retrieval`** → reports char-vs-semantic gap (lexical parroting detector),
  copy-doc upper reference; semantic embedder when supplied.
- **`prior_deviation`** (the team's contrastive method), **`mode_collapse`**
  (diversity / collapse flag), **`calibration_entropy`** (self-consistency).
- **`nla_eval/confabulation.py`** — `factual_grounding` (confabulation rate on
  specific entity/number claims unsupported by source), `thematic_fidelity`,
  `consistency`. Addresses the #1 stated NLA limitation; for the misuse monitor a
  confabulating verbalizer is untrustworthy.
- Axis side-tagging (`AXIS_SIDE`), `allow_low_coverage` override, docs.

## Gated — needs the AR+model loop (GPU), a live AV/server, or humans

### P0 — Causal-fidelity family (the north-star; open review's #1)
NLAttack is currently correlational; the *defining* NLA claim (Anthropic) is
**causal**: edit explanation → reconstruct (AR) → steering vector → behavioral Δ.
Spec: for steering concepts {refusal, language, style, fact-recall}: AV(act) →
edit/flip concept → AR(edited) − AR(orig) = steer; apply to model; correlate
verbalized-concept strength with steer magnitude. **A zero-Δ-but-concept-present
hit is a *caught confabulation*** — links the two top gaps. Needs the full
target-model + AR pipeline (GPU).

### P1 — Steganography / covert channel (safety-critical for the misuse monitor)
- human-readability vs AR-decodability gap (needs human ratings + AR);
- paraphrase robustness: perturb the verbalization → AR reconstruction drop
  (semantic content is robust, stego codes are brittle) — needs AR;
- MI(verbalization tokens; reconstruction) vs human-estimated semantic content.

### P1 — Downstream human utility (the gold standard)
Replicate the Anthropic **auditing game** (misaligned model + hidden motive) +
debugging time-to-discovery + hypothesis-generation quality (LLM-as-judge). Needs
humans/infra.

### P2 — Standard interp-eval practices to adopt
cross-layer generalization (train probe layer N → test N+1); OOD robustness
(held-out domains / adversarial); compute-efficiency frontier (tokens-per-activation
vs fidelity); SAE / attribution baselines alongside the probe-vs-AV gap.

### P2 — Reporting reframe (open review #4/#7; partially done)
Keep the conjunctive tier as a *conservative gate* (it already avoids
averaging-masking) but treat the activation-side composite as a **secondary
diagnostic / Pareto frontier**, and frame the longitudinal signal as a
**"capability trajectory / learning dynamics"** (track *which* capabilities emerge
in what order), not a single "emergence" number — construct-validity caveat.
Also: replace `allow_low_coverage`'s low-confidence aggregate with **per-concept
reporting + cluster-robust SE** below the floor.

## Note on scope
The activation-side capability/emergence axes are ~flat across AV-LoRA checkpoints
**by design** (they read the frozen base activation). To track AV training /
conditioning, use the verbalizer-side axes above. See `AV_CONDITIONING_AXES.md`.
