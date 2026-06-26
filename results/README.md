# Results — generation provenance map

Committed result artifacts, labeled by benchmark generation. Per the freeze policy
([`../docs/VERSIONING.md`](../docs/VERSIONING.md)), v1 artifacts are never
overwritten; v2 adds alongside them. Findings and caveats are written up in
[`../docs/RESULTS.md`](../docs/RESULTS.md).

## v1 (frozen)

The published v1 leaderboard and emergence results.

| Path | Contents |
|---|---|
| `cross_nla/` | Cross-NLA leaderboard: Llama-3.3-70B, Gemma-3-27B, and the local Gemma-4-E2B NLA on the shared dataset (per-NLA JSON + rows, summary, and the benchmark graphs). |
| `emergence/` | EmergenceIndex runs (base vs RL checkpoints, pooled) behind the published 0.601 composite. |
| `deception/` | Family-M deception probe on real honest-vs-deceptive behavioral-split activations (nanochat-d32, L12). |
| `local_gemma_e2b/` | Full-access white-box artifacts for the local Gemma-4-E2B NLA: bottleneck probe, probe-vs-AV gap, AV verbalizations, and the domain-aware-contrastive verbalizer-axis eval (an honest negative on the generation axes). |

## v2 (current)

Additive; v1 artifacts above are unchanged.

| Path | Contents |
|---|---|
| `ctf/` | Family-N CTF Red/Blue scoreboard. `ctf_red_blue_mock.json` is the offline, reproducible **control-tier** demo against `MockNLA` (mechanics + the verbalizer blind spot, **not** a monitor certification — see P128). A primary-tier run needs a bottleneck-probe oracle over the model's own behavior. |

> Multi-domain v2 content results are not yet produced; see
> [`../docs/ROADMAP_v2.md`](../docs/ROADMAP_v2.md). The tagged v2.0.0 does not claim
> domain-coverage results it has not run.
