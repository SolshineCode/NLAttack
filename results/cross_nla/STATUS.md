# Cross-NLA run status

**Comparison target:** Gemma-3-27B (`kitft-l41`) vs Llama-3.3-70B (`kitft-l53`),
attributed by NLA name (see README "Result attribution" convention).

| NLA | status |
|---|---|
| `kitft/Llama-3.3-70B-NLA-av @ L53` (llama3.3-70b-it) | ✅ evaluated 2026-06-10 (re-run) |
| `kitft/nla-gemma3-27b-av @ L41` (gemma-3-27b-it) | ✅ evaluated 2026-06-10 (recovered) |

The Gemma-3 NLA inference server recovered on 2026-06-10 (it had returned 502 since
~2026-06-05). It now serves normal requests but still emits transient
"NLA server error: 500" responses on some calls; the adapter's retry set was
extended to include 500 so a single transient fault no longer aborts a run. Both
NLAs were evaluated on the identical 20-example dataset and matcher, so retention,
see-through, and laundering are directly comparable; results are in
`cross_nla_summary.json` and the per-NLA JSON/CSV files.

Still open: Gemma-3 held-out doc retrieval (the protocol-anchored metric comparable
to the local Gemma-4 NLA) has not been run yet. The probe/emergence axes need
activations the hosted API does not expose, so they remain full-access only.
