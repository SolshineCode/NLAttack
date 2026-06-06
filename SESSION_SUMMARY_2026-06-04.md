# NLAttack — deep-work / GPU-grant session 2026-06-04

Autonomous session: fold the Hermes design review into NLAttack. All work
committed **locally only — NOT pushed** (held per Caleb's instruction).

## What shipped (7/7 tasks)

| # | Item | Hermes ref | Status |
|---|---|---|---|
| 1 | `DESIGN_REVIEW.md` — critique + prioritized issues | — | ✅ |
| 2 | Faithfulness weighting (cos_sim/mse → confidence) | P1 #5 | ✅ |
| 3 | `EnsembleMatcher` + agreement / `contested_rate` | P0 #2 | ✅ |
| 4 | `controls.py` freq+length matched controls | P1 #1 | ✅ |
| 5 | Bottleneck probes on local Gemma-4-E2B NLA | P0 #1 | ✅ (probe + AV gap) |
| 6 | `redteam.py` adaptive evasion + compositional blindspot | P0 #3 | ✅ scaffold |
| 7 | Verify + commit locally | — | ✅ |

## Headline empirical result (the deepest fix, P0 #1)

On the local weak NLA (`v8_nla_local`, read-only), the L23 bottleneck encodes
concepts linearly:
- In-distribution (business corpus): probe AUC **0.996 / 0.979** (mean 0.988).
- OOD (fineweb web text): probe AUC **~0.70** — the bottleneck is *not* a uniform
  filter; survival is confounded by the encodable subspace.

Full probe-vs-AV gap (n=100): probe_acc ≈ 0.92 vs AV+matcher acc ≈ 0.59 →
**gap ≈ 0.33 = verbalizer+matcher loss, not NLA dropout.** This converts NLAttack's
AV-text observations into bottleneck claims — exactly what the review demanded.

## Key reframe baked into the repo

All dropout/laundering numbers are now framed as **AV+NLA+AR end-to-end,
matcher-dependent** — not NLA properties — unless backed by a bottleneck probe.

## New modules

`bottleneck_probe.py`, `local_gemma_e2b.py` (AV adapter), `controls.py`,
`redteam.py`; `EnsembleMatcher` + faithfulness/contested metrics in
`matching.py`/`core.py`. Experiments under `experiments/local_gemma_e2b/`,
results under `results/local_gemma_e2b/`.

## Environment notes

- GPU: GTX 1650 Ti 4GB. A concurrent session's `exp2_rft.py` held the card mid-
  session; bottleneck probe (CPU) ran alongside it, AV side ran after it freed.
- Local NLA inference uses the deception repo's `.venv-gemma4` (transformers/peft/
  bnb/sklearn). The deception repo was treated **read-only**.
- Neuronpedia NLA server (RunPod) was down (502) for the latter half — the
  concept-targeting upgrade is verified by the earlier live run + mock; re-run
  `test_neuronpedia_live.py` when it recovers.

## Open / next (not done)

- **Push** — everything is local; awaiting Caleb's go.
- AV-side gap on a **later/better AV checkpoint** + OOD corpus (this run used an
  early 100-step checkpoint that emits degenerate text on some inputs).
- More probe concepts (replace crude keyword labels with the corpus's stage-2 LLM
  labels; current n=2–3 per corpus).
- Run `redteam.adaptive_evade` against the real local pipeline
  (`make_local_nla_pipeline`) once a stronger AV checkpoint is chosen.
- Re-run Neuronpedia live test when the server is back.
