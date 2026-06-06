# Rudimentary-NLA triage of the 100 plans

Derived from the two independent reviews (`reviews/SUMMARY.md`). Tiers reflect
where Hermes and Antigravity **converged** on what to run vs shelve when the NLA
is small/weak (degenerate AV, near-random AR, low-dim bottleneck, tiny n). Use
this to sequence work on a rudimentary NLA; the full per-plan detail and floor
versions are in `reviews/`.

## Tier 1 — RUN FIRST (both reviewers' highest-value)

Start here: answer the existential questions, then the probe-grounded basics. Each
maps to ready harness tooling (mostly the new `nla_eval/rudimentary.py`,
all CPU, no working AV required).

| Plan | What it answers | Tooling |
|---|---|---|
| **P077** | Is the AR/AV even conditioned on the input (vs a generic constant)? | `rudimentary.identity_leakage_vectors` / `_text` |
| **P081** | Does a usable bottleneck exist at all (signal above the noise floor)? | `rudimentary.bottleneck_exists` (+ permutation control) |
| **P046** | Is "drop" the bottleneck or just a broken verbalizer? | `rudimentary.fit_forced_verbalizer` |
| **P050** | Unblock probes — replace keyword labels with LLM/semantic labels | `bottleneck_probe.run_probe_suite(aliases=...)` + better labels |
| **P004** | BOS/position-0 structural effect (present even in tiny models) | probe-only position feature |
| **P023** | Reported-vs-asserted status markers (coarsened, partially survive) | `EnsembleMatcher` + retrieval/coarse matcher |
| **P092** | Abstention vs confabulation on underdetermined input | entropy/length proxy + matcher ensemble |
| **P095** | Are AV hedges coupled to AR faithfulness, or decorative? | `faithfulness_weighted_retention` + hedge bins |
| **P067** | Position/compression invariance under reordering (probe-only grid) | probe over fixed reorder grid |

Cross-cutting requirement for ALL Tier-1 probe work: report the **permutation
control** (`ProbeResult.signal`) — never an absolute AUC alone.

## Detecting EMERGENCE (the most nuanced use) — Family K + the dashboard

For watching capability *lift off the floor* in a super-weak/tiny NLA, run the
additive **Emergence Index** (`nla_eval/emergence.py`,
`experiments/emergence_dashboard.py`) — it composes the null-gated axes:
decodability (P081) · sufficiency (P101) · selectivity (P102) · content-adjacency
(P011) · faithful-rank (P080) · stability (P044) · dose-response (P086) · graded
(P103) · abstraction (P104) · effective-rank (P105) → composite (P106). Every axis
carries its own null margin and the index is gated on decodability clearing the
permutation floor, so a high score can't come from overfit/degeneracy. Sweep
checkpoints (`--sweep`) for the emergence curve; the most sensitive trio at the
floor is **stability (P044), faithful-rank (P080), dose-response (P086)**.

## Tier 3 — SKIP until the NLA is stronger (both reviewers' "skip")

Expected pure-null on a rudimentary NLA — needs relational binding, multi-stage
structure, metacognition, or a functioning generative AV:

`P002` category dropout · `P016` metaphor/literal · `P019` temporal order ·
`P020` spatial/relational · `P021` false-belief · `P027` sandbagging ·
`P042` MLP probe (overfits at tiny n) · `P055` paraphrase invariance ·
`P065` steganographic capacity · `P074` round-trip stability ·
`P079` AR paraphrase delta · `P096` uncertainty-direction stability ·
`P097` refusal encoding.

## Tier 2 — everything else: run the FLOOR VERSION

The remaining ~78 plans are runnable on a rudimentary NLA **only via the
floor/probe-only adaptation** each reviewer specified (bypass the degenerate AV;
coarsen the metric to binary/rank/pooled; relax the matcher to
character/edit-distance/numeric; add a shuffled-activation control). See the
per-plan "FLOOR VERSION" notes in `reviews/hermes/` and `reviews/antigravity/`.

## Reviewer disagreements (decide per your NLA) — from `reviews/SUMMARY.md`
- **P089** layer×distribution: Hermes run / Antigravity skip → run only with
  several layers dumped and enough n.
- **Family F** wholesale: on a truly degenerate AV, follow Hermes — replace
  P051–P055 with probe-only matcher-noise-floor ablations on fixed human text.
- **Family D** picks: prefer the probe-grounded P034/P037/P040 over coverage/
  obfuscation heatmaps (matcher hallucination on generic text).
