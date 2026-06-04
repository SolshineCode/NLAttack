# Bottleneck-probe results — local Gemma-4-E2B NLA

Implements Hermes review **P0 #1** (ground truth at the bottleneck) against the
local NLA at `deception-nanochat-sae-research/experiments/v8_nla_local`
(read-only). The bottleneck = the L23 `activation_vector` the AV verbalizes;
ground-truth concept presence = keyword in the source text. A linear probe
(logistic regression, 5-fold CV) reads concepts off the bottleneck.

## Result (probe side — CPU, run 2026-06-04)

**In-distribution (deception/business corpus, `gemma4_deception_chunk1`, n=1200):**

| concept | probe AUC | bal. acc | baseline | n_pos |
|---|---|---|---|---|
| report | 0.996 | 0.951 | 0.829 | 205 |
| team   | 0.979 | 0.934 | 0.627 | 448 |

mean AUC ≈ **0.988** → the bottleneck encodes these concepts almost perfectly.

**Out-of-distribution (diverse web, `fineweb_edu_chunk1`, n=800):**

| concept | probe AUC | bal. acc | baseline | n_pos |
|---|---|---|---|---|
| war    | 0.700 | 0.618 | 0.714 | 229 |
| people | 0.691 | 0.614 | 0.756 | 195 |

mean AUC ≈ **0.695** → barely above chance; balanced accuracy ≈ baseline.

## What this means (and the load-bearing point for NLAttack)

1. **The bottleneck is not a uniform concept filter.** It linearly encodes
   in-distribution (business) concepts at ~0.99 AUC but generic OOD web concepts
   at ~0.70. "Concept survival" is confounded by whether the concept is even in
   the bottleneck's encodable subspace — which depends on the NLA's training
   distribution, not just the concept.
2. **This is exactly Hermes #1 made concrete.** Where probe AUC is high (~0.99),
   any later "drop" in the AV verbalization is *verbalizer/matcher loss, not NLA
   dropout*. The probe gives the ground truth the AV-text matcher cannot.
3. The probe side is **CPU-only** (cached activations + sklearn), so it runs even
   while the GPU is busy with other work.

## Honest caveats

- **n = 2 concepts per corpus.** The keyword-prevalence auto-selector (15–85%
  band) admits few concepts on these narrow/short-text corpora. This is a
  *method demonstration*, not a population estimate. Fixes: lower the band, add
  candidates, use multiple chunks, or — better — replace keyword labels with the
  corpus's existing stage-2 LLM labels.
- **Keyword labeling is crude** (substring presence ≠ semantic presence; misses
  paraphrase, catches mentions-of-absence). Probe AUC is an *upper bound* on
  honestly-detectable presence under noisy labels.
- **The AV-side number is not in yet.** `probe_acc − av_matcher_acc = gap` (the
  verbalizer+matcher loss) needs the GPU AV run (`verbalize_av.py`), deferred
  because the 4GB card was occupied by another session's `exp2_rft.py` at build
  time. Until then, only the ground-truth (probe) half exists.

## Reproduce

```bash
cd C:/Users/caleb/deception-nanochat-sae-research
# probe side (CPU, any time):
KMP_DUPLICATE_LIB_OK=TRUE .venv-gemma4/Scripts/python.exe \
  C:/Users/caleb/nla-eval-harness/experiments/local_gemma_e2b/probe_bottleneck.py
KMP_DUPLICATE_LIB_OK=TRUE .venv-gemma4/Scripts/python.exe \
  C:/Users/caleb/nla-eval-harness/experiments/local_gemma_e2b/probe_bottleneck.py \
  --parquet experiments/v8_nla_local/data/stage0/fineweb_edu_chunk1.parquet \
  --generic --out-suffix _fineweb
# AV side (GPU, when free): experiments/local_gemma_e2b/verbalize_av.py
```
