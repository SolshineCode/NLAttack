# NLAttack v0.2 roadmap (LOCAL DRAFT — not yet pushed)

v0.2.0 is the first minor version after v0.1. Primary goals: expanded domain
coverage, harness correctness fixes, and cleaner multi-version result management.
Content scope is additive — v0.1 benchmark content is untouched.

---

## 1. Harness fixes (from ultrareview 2026-06-12, confirmed ×2-of-3 Opus lenses)

These fix code bugs, not benchmark design. All required before any new results
under the v0.2 label.

| # | File | Issue | Fix |
|---|---|---|---|
| H1 | `emergence.py:63` | `hash(w) % dim` uses PYTHONHASHSEED — Tier-1 nondeterministic across processes | Replace with `mmh3.hash(w) % dim` or `int(hashlib.md5(w.encode()).hexdigest(),16) % dim` |
| H2 | `adapters.py:193-196` | Schema-drift 200 → BOS noise / empty string; no JSON parse guard | Fail loudly on missing `prompt_length`/`results`; validate before use |
| H3 | `emergence_dashboard.py:117` | `json.dumps` with `allow_nan=True` writes invalid JSON | `allow_nan=False`; replace NaN with `null` |
| H4 | `adapters.py:197-199` | Positions are contiguous prefix 1..16; docstring says "evenly-spaced" | Fix docstring (if prefix is intentional) OR implement true evenly-spaced sampling |
| H5 | `verbalizer_axes.py:249-272` | Agreement margin trivially ≥ 0; 0.5 null is vacuous | Normalize against permutation null or switch to Cohen's kappa |
| H6 | `emergence.py:115-120` | BoW/length null defaults to 0.5 on exception; weakens Tier-1 gate | Propagate exception or exclude null-failed concepts |
| H7 | `adapters.py:173-179` | 429 backoff ~9s; `Retry-After` ignored; hourly quota never clears | Honor `Retry-After`; checkpoint partial rows |
| H8 | `adapters.py:193-200` | 2 HTTP requests per `reconstruct()`; callers budget as 1 | Add request counter + pacing; fix budget comments |
| H9 | `real_nla_example.py:23` | `layer` param doesn't exist in constructor → `TypeError` on default entry | Remove `layer=` kwarg |
| H10 | `matching.py:94-98` | Lexical match is unanchored substring; "art" matches "departure" | Use word-boundary regex or strip-and-split |
| H11 | `controls.py:94-99` | Controls scored only on freq_band; token-length match ignored | Add token-length scoring term per docstring spec |
| H12 | `core.py:132-135` | Insertion detection never checks `ex.text`; flags source-text words as hallucinations | Include source text words in the reference set |

---

## 2. Expanded domain coverage

v0.1 evaluates on the original concept/document set (primarily deception domain).
v0.2 adds multi-domain coverage using the same NLAs, so the eval speaks to
generalization rather than domain specialization.

### 2a. Target domains for v0.2
Drawn from the ARM A balanced corpus (academically-sourced, cluster-level holdout):

| Domain | Source | Status |
|---|---|---|
| legal | Free Law Project (CourtListener) | available |
| math | AMPS / Hendrycks-MATH | available |
| medicine | MedNLI / PubMedQA | available |
| reviews | Amazon ESCI / Yelp (academic) | available |
| science | S2ORC / AllenAI | available |
| arxiv | arXiv abstracts | available |
| wiki | Wikipedia (Wikimedia dumps) | available |
| news | CC-News / RealNews | available |
| persuasion | Persuasion for Good | available |
| global_opinions | GlobalOpinionQA | available |
| deception | (v0.1 original) | existing |

Domains intentionally excluded from v0.2 expansion (data provenance issues or
insufficient academic sourcing): fineweb raw, news_rl, pku_safety, mmlu_moral.
These remain under review for v0.3.

### 2b. Concept/document set construction
- Per-domain concept set: 8 concepts minimum (the v0.1 reliability floor)
- Document pool: 10–20 documents per domain, held out from any AV training data
- Concept–document pairing: same methodology as v0.1 (canonical NLA concept list
  for the tested NLA source, paired with source documents from each new domain)
- Concept survival threshold: unchanged from v0.1 (preserves backward comparability
  of the threshold, not the concept set)

### 2c. Result directory structure
```
results/
  v0.1/          ← v0.1 results untouched here
    emergence_gemma4_deception_chunk1.json
    ...
  v0.2/          ← all new results under this prefix
    emergence_v0.2_gemma4_multidomain.json
    ...
```

---

## 3. Version constant + config gating

```python
# nla_eval/__init__.py
__version__ = "0.2.0.dev"
```

New domain sets live under a `BENCHMARK_VERSION` config key so a single codebase
can run either version:
```python
# nla_eval/datasets/registry.py  (new file)
VERSIONS = {
    "v0.1": {"domains": ["deception"], "min_concepts": 8},
    "v0.2": {"domains": [...], "min_concepts": 8},
}
```

---

## 4. NLA sources (unchanged from v0.1 for now)

- `gemma-3-27b-it / kitft-l41` (Neuronpedia)
- `llama3.3-70b-it / kitft-l53` (Neuronpedia)

Adding new NLA sources (e.g., Gemma-4-E2B fine-tuned AVs from the deception
research program) is gated on: (a) the AV checkpoint being stable/published, and
(b) external review of the conditioning results. Not in v0.2 scope unless ARM
A/B/C/D produce a publishable conditioning result first.

---

## 5. Out of scope for v0.2

- Causal-fidelity (AR loop) — still P0 on the roadmap, still GPU-gated
- Human utility evals — still P1, still infra-gated
- Changing tier thresholds or the EmergenceIndex formula — would break v0.1
  backward comparability; reserved for v1.0 if justified

---

## 6. Rollout gate

Per `VERSIONING.md`: branch `v0.2-dev` stays **local only** until:
- [ ] All H1–H12 harness fixes implemented and tested
- [ ] New domain concept/document sets constructed and reviewed
- [ ] At least one full eval run on a known NLA source produces a v0.2 result
- [ ] External review requested (Gemini + manual)
- [ ] Version constant set to `0.2.0` (drop `.dev`)
- [ ] Git tag `v0.2.0` pushed to remote
