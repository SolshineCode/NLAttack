# NLAttack versioning policy

## Guiding principle

**Once a version is tagged and results are published, its benchmark content is
frozen.** Changing domain coverage, NLA sources, concept sets, or scoring thresholds
after publication invalidates prior comparisons — which is worse than limited
coverage. New capabilities land in the next release; the prior release stays exactly
as reported.

## Generations

NLAttack ships as numbered benchmark generations. A generation is a self-contained,
citable release: its catalog, datasets, thresholds, and published results do not
change after it is tagged.

| Generation | Tag | Status | Headline |
|---|---|---|---|
| **v1** | `v1.0.0` (released as `0.1.0`, 2026-06) | **FROZEN** | 118-plan catalog (families A–M); EmergenceIndex 0.601 ("established") on `gemma-3-27b-it / kitft-l41`; the cross-NLA leaderboard |
| **v2** | `v2.0.0` (this release) | current | 128-plan catalog (families A–N); adds the **CTF Red/Blue** adversarial family (Family N); v1 results carried forward unchanged |

> Earlier development drafts referred to these as `v0.1` and `v0.2`. They are the
> same content; v2 adopts plain `v1`/`v2` generation labels. References to a
> *checkpoint's* own version (e.g. "Gemma-4-E2B NLA v0.1", "the v0.1 AV") name the
> NLA artifact, not the NLAttack generation, and are unchanged.

## Version semantics

| Bump | When | Examples |
|---|---|---|
| **patch** (vN.0.x) | Harness bug that materially affects result validity; no content change | Fix BOS-noise fallback; fix NaN JSON output; fix PYTHONHASHSEED nondeterminism |
| **minor** (vN.x.0) | Additive within a generation: new eval scripts, new adapters, new result artifacts on existing content | Add a doc-retrieval runner; add a hosted NLA result |
| **major** (vN.0.0) | A new generation: a new evaluation family or paradigm, expanded domain coverage, or any change that restructures the catalog | v2.0.0 — adds the Red/Blue CTF family |

**Patch rule.** A harness patch is allowed when a code bug (not a design choice)
makes published numbers unreliable. Patches rerun only the affected metric on the
same benchmark content and document the delta. If the delta is < 0.01 on the
published composite, the original number is noted as "confirmed within 0.01 under
patch" rather than retracted.

**Additive rule.** New benchmark content always lands in a new generation. The prior
generation's card/README gets a one-line pointer to the new one; its results do not
change.

## Frozen content per generation

### v1 (frozen)
- **Published result:** EmergenceIndex 0.601 ("established") on `gemma-3-27b-it /
  kitft-l41`, plus the cross-NLA leaderboard (Llama-3.3-70B, Gemma-3-27B,
  Gemma-4-E2B), evaluated against the original concept/document set. Do not change
  this content; see [RESULTS.md](RESULTS.md).
- **Known harness issues (carried forward, not yet patched):**
  - `emergence.py` — BoW buckets use `hash()` without `PYTHONHASHSEED`; the Tier-1
    verdict is nondeterministic across processes (does not affect the v1 numbers,
    which were produced in a single session with no process restart).
  - `adapters.py` — a schema-drift HTTP-200 silently falls back to BOS noise instead
    of failing (low risk for the v1 eval; the Neuronpedia API was stable during it).
  - `emergence_dashboard.py` — `json.dumps(..., allow_nan=True)` writes bare `NaN`
    tokens (invalid JSON for strict parsers).
  - `adapters.py` — verbalized positions are a contiguous prefix 1..16, not
    evenly-spaced as the docstring claims.
  These are tracked as **v2-development harness fixes** in
  [ROADMAP_v2.md](ROADMAP_v2.md). A v1 patch will be issued only if evidence emerges
  that any of them affected the published composite.

### v2 (current)
- **Additive content:** the CTF Red/Blue family (Family N, P119–P128;
  `nla_eval/ctf.py`; [CTF_RED_BLUE.md](CTF_RED_BLUE.md)) and release/versioning
  maturation. v1 capability and safety results are carried forward verbatim.
- **In development (not part of the tagged v2.0.0 results):** the multi-domain
  corpus expansion and the harness-correctness fixes above. These are the v2.x
  roadmap; see [ROADMAP_v2.md](ROADMAP_v2.md). The tagged release does not claim
  results it has not produced.

## Result directories

v1 result artifacts live in the existing `results/` subdirectories (`cross_nla/`,
`emergence/`, `deception/`, `local_gemma_e2b/`); v2 additions live in `results/ctf/`
and alongside them. Generation provenance for every artifact is mapped in
[`../results/README.md`](../results/README.md). Prior-generation artifacts are never
overwritten.

## Rollout checklist for a new generation

- [ ] Branch kept off `master` until feature-complete
- [ ] Prior generation's known-issue harness fixes applied (or explicitly deferred,
      documented, and shown not to affect prior published numbers)
- [ ] New content added without altering any prior-generation artifact
- [ ] Version constant bumped in `nla_eval/__init__.py`
- [ ] `CITATION.cff` `version` field updated
- [ ] `CHANGELOG.md` entry added
- [ ] Prior generation's README/card note: "See vN for …"
- [ ] External review completed (two lenses) via a release-review packet under
      `docs/reviews/` — e.g. [`reviews/v2.0.0-release-review.md`](reviews/v2.0.0-release-review.md)
- [ ] Git tag `vN.0.0` pushed after external review (not before)
