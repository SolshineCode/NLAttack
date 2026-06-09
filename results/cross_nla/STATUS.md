# Cross-NLA run status

**Comparison target:** Gemma-3-27B (`kitft-l41`) vs Llama-3.3-70B (`kitft-l53`),
attributed by NLA name (see README "Result attribution" convention).

| NLA | status |
|---|---|
| `kitft/Llama-3.3-70B-NLA-av @ L53` (llama3.3-70b-it) | ✅ evaluated 2026-06-08 |
| `kitft/nla-gemma3-27b-av @ L41` (gemma-3-27b-it) | ⏳ pending — its NLA inference server returns 502 (since ~2026-06-05) |

Diagnosis: the call format is correct (identical request returns 200 for Llama);
`/api/nla/sources` + the webapp are up; only the Gemma-3 NLA inference instance is
down — likely related to recent NLA-server infra commits (hqq-on-Blackwell /
sglang). A persistent retry runs the full side-by-side automatically when Gemma-3
recovers. This is the API-runnable subset (retention + AV faithfulness); the
probe/emergence dashboard needs activations the hosted API doesn't expose.
