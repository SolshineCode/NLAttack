# NLAttack versioning policy

## Guiding principle

Once a version is tagged and results are published, its benchmark content is frozen.
Changing domain coverage, NLA sources, concept sets, or scoring thresholds after
publication invalidates prior comparisons — which is worse than limited coverage.
New capabilities land in the next minor version; the prior version stays exactly
as reported.

---

## Version semantics

| Bump | When | Examples |
|---|---|---|
| **patch** (0.1.x) | Harness bug that materially affects result validity; no content change | Fix BOS-noise fallback; fix NaN JSON output; fix PYTHONHASHSEED nondeterminism |
| **minor** (0.x.0) | Additive: new domains, new NLA sources, new axes, new eval scripts | v0.2.0 — expanded domain coverage; harness fixes |
| **major** (x.0.0) | Breaking: scoring schema change, tier-threshold change, concept-set restructure that breaks backward comparability | Reserved |

**Patch rule:** a harness patch is allowed when a code bug (not a design choice)
causes published numbers to be unreliable. Patches rerun only the affected metric
on the same benchmark content and document the delta. If the delta is < 0.01 on
the published composite, the original number is noted as "confirmed within 0.01
under patch" rather than retracted.

**Minor rule:** new content always uses a new version number. The v0.1 HF card/
README gets a one-line note pointing to v0.2 for expanded coverage; it does not
change its results.

---

## Current versions

### v0.1 (released 2026-06)
- **Status: FROZEN.** Published EmergenceIndex 0.601 ("established") on
  gemma-3-27b-it / kitft-l41, evaluated against the original concept/document
  set. Do not change this benchmark content.
- **Known harness issues (not yet patched):**
  - `emergence.py:63` — BoW buckets use `hash()` without `PYTHONHASHSEED`; Tier-1
    verdict is nondeterministic across processes (does not affect v0.1 numbers if
    the original eval ran in a single session with no process restart).
  - `adapters.py:193-196` — schema-drift HTTP-200 silently falls back to BOS noise
    instead of failing; low risk for v0.1 eval (Neuronpedia API was stable during
    that run).
  - `emergence_dashboard.py:117-120` — `json.dumps` with `allow_nan=True`; both
    committed result files contain bare `NaN` tokens (invalid JSON for strict parsers).
  - `adapters.py:197-199` — positions are a contiguous prefix 1..16, not evenly-
    spaced as the docstring claims.
  These are carried forward as **fixes in v0.2.0**. A v0.1 patch will be issued
  only if evidence emerges that any of them affected the published composite.

### v0.2.0 (in development — LOCAL ONLY, not pushed)
See `docs/ROADMAP_v0.2.md` for scope.

---

## Rollout checklist for a new minor version

- [ ] Branch `v{N}-dev` branched from master, kept local until feature-complete
- [ ] All harness fixes from the prior version's known-issues list applied
- [ ] New content (domains, NLA sources, concept sets) added under a new config
      key so both versions can run from the same codebase
- [ ] `results/` subdirectory named `v{N}/` — never overwrite prior version results
- [ ] Version constant bumped in `nla_eval/__init__.py`
- [ ] HF card updated: new version section added; prior version section unchanged
- [ ] CITATION.cff `version` field updated
- [ ] Git tag `v{N}.{M}.{P}` pushed after external review (not before)
- [ ] Prior version README note: "See v0.2 for expanded domain coverage."
