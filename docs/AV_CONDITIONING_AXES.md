# Activation-side vs verbalizer-side axes — and how to track AV training

Added in response to collaborator feedback (2026-06-08): the emergence axes
characterize the **frozen base activation**, not the **verbalizer's reading** of
it — so they can't track AV-LoRA training, which is the actual lever for
conditioning / prior-deviation / contrastive work.

## The distinction (know which axis answers which question)

**Activation-side** (`nla_eval/emergence.py`; see `AXIS_SIDE`) — read the frozen
base-model activation. They tell you *what the activation carries*. They are
**~constant across AV-LoRA checkpoints** (the base activations don't change as you
train the AV), so **do NOT use them to track AV training**:
`decoding_quality`, `selectivity`/`sufficiency` (folded in), `stability`,
`effective_rank`, `dose_response`, `abstraction`, `graded_encoding`, and the
activation-space `content_adjacency` (minimal-pair *distance*).

**Verbalizer-side** (`nla_eval/verbalizer_axes.py`) — **run the AV**, so they
**move as the AV is trained**. Use these to track AV conditioning:
- **`minimal_pair_discrimination(pairs, verbalize_fn)`** — the verbalizer-facing
  minimal-pair axis the feedback asked for. `pairs` = `(concept, act_pos,
  act_neg)` with `act_neg` a HARD NEGATIVE. Correct = AV verbalizes `act_pos` with
  the concept and `act_neg` without it. Clean null = **0.0** (an AV that ignores
  the activation emits the same text for both). Verified to rise 0.0 → 1.0 as a
  simulated AV goes from unconditioned to conditioned.
- **`doc_retrieval(items, verbalize_fn)`** — the held-out doc-retrieval signal the
  team is already using, now first-class: does the AV's verbalization retrieve its
  own source doc among distractors? Chance = 1/N; rises with conditioning.

> So NLAttack is no longer "structurally unable" to provide an AV-conditioning
> signal — these two axes provide exactly that. Sweep them across AV-LoRA
> checkpoints; the activation-side emergence index will be ~flat over the same
> sweep, by construction (that flatness is now expected and documented, not a bug).

## Usage (needs the AV → local GPU or a hosted AV)

```python
from nla_eval.verbalizer_axes import minimal_pair_discrimination, doc_retrieval
from nla_eval.local_gemma_e2b import LocalGemmaE2BNLA      # or a Neuronpedia AV wrapper

nla = LocalGemmaE2BNLA(av_checkpoint="...").load()
vfn = nla.verbalize_activation

# (concept, activation-with-concept, hard-negative-activation) from your minimal pairs
pairs = [("credential dumping", act_pos, act_neg), ...]
print(minimal_pair_discrimination(pairs, vfn))           # tracks AV training

items = [(activation_i, source_doc_text_i), ...]
print(doc_retrieval(items, vfn))                          # held-out doc retrieval
```

Run the same calls at each AV-LoRA checkpoint to plot the conditioning curve.

## Low concept coverage (the 8-concept floor that blocked the general-domain run)

The capability tier hard-blocks at `< min_concepts` (default 8) because every
activation-side axis is statistically unreliable below that. To run anyway (e.g.
a general-domain set that only yields ~7 keyword concepts), pass
`allow_low_coverage=True`:

```python
emergence.run_capability(X, texts, concepts, allow_low_coverage=True)
# -> still computes tiers/profile, but stamps low_confidence=True + a LOW-CONFIDENCE
#    reason; trust the bootstrap CI width, which will be wide.
```

The real fix for coverage is semantic (not keyword) labels (plan P050) to lift the
concept count past the floor.

## Update 2026-06-08 — hardened after two Hermes reviews

- **`minimal_pair_discrimination` is now AUC-based** (chance = 0.5), with a
  permutation p-value and a `matcher_ceiling_auc` covariate, instead of the
  earlier `hit ∧ ¬hit` (which conflated AV conditioning with matcher recall and
  used a wrong null). Verified to rise 0.50 → 1.00 as a simulated AV conditions.
- **ATT&CK demo shipped:** `verbalizer_axes.build_attack_pairs(activation_of)`
  builds attack-term-vs-benign-neighbor pairs from `attack_concepts.py`, so the
  default discrimination demo *is* the misuse laundering test (can the monitor
  tell "credential dumping" from "password management"?).
- **New verbalizer-side axes:** `doc_retrieval` now reports the char-vs-semantic
  gap (lexical-parroting detector) + copy-doc reference; `prior_deviation`
  (contrastive: does the activation add signal beyond the prior?); `mode_collapse`
  (diversity / collapse flag); `calibration_entropy` (self-consistency across
  temperature samples).
- **`nla_eval/confabulation.py`** — confabulation rate (specific entity/number
  claims unsupported by the source), thematic fidelity, cross-sample consistency.
  The #1 stated NLA limitation and a misuse-monitor trust blocker.
- Bigger gated items (causal-fidelity north-star, steganography, the auditing
  game, cross-layer/OOD/efficiency baselines) are in `ROADMAP.md`.
