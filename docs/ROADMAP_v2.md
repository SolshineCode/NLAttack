# NLAttack v2 development roadmap

**Shipped in v2.0.0:** the CTF Red/Blue family (Family N) and release/versioning
maturation — see [CHANGELOG.md](../CHANGELOG.md). This file tracks the remaining v2
engineering: expanded domain coverage, harness-correctness fixes, and multi-version
result management. All of it is **additive** — v1 benchmark content is untouched —
and none of it is claimed in the tagged v2.0.0 results until it is run and reviewed.

---

## 1. Harness fixes (from ultrareview 2026-06-12, confirmed ×2-of-3 Opus lenses)

These fix code bugs, not benchmark design. Required before any new *content* results
(domain expansion below) are published under v2.

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

v1 evaluates on the original concept/document set (primarily deception domain).
v2 adds multi-domain coverage using the same NLAs, so the eval speaks to
generalization rather than domain specialization.

### 2a. Target domains
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
| deception | (v1 original) | existing |

Domains intentionally excluded from this expansion (data provenance issues or
insufficient academic sourcing): fineweb raw, news_rl, pku_safety, mmlu_moral.
These remain under review for a later generation.

### 2b. Concept/document set construction
- Per-domain concept set: 8 concepts minimum (the v1 reliability floor)
- Document pool: 10–20 documents per domain, held out from any AV training data
- Concept–document pairing: same methodology as v1 (canonical NLA concept list
  for the tested NLA source, paired with source documents from each new domain)
- Concept survival threshold: unchanged from v1 (preserves backward comparability
  of the threshold, not the concept set)

### 2c. Result directory convention
v1 artifacts stay in their existing `results/` subdirectories (never overwritten);
new multi-domain results are added alongside them with a `v2`/multidomain marker in
the filename, and every artifact's generation is recorded in `results/README.md`
(provenance map). Example new file: `results/local_gemma_e2b/v2_gemma4_multidomain.json`.

---

## 3. Version constant + config gating

```python
# nla_eval/__init__.py
__version__ = "2.0.0"
```

New domain sets live under a `BENCHMARK_VERSION` config key so a single codebase
can run either generation:
```python
# nla_eval/datasets/registry.py  (new file)
VERSIONS = {
    "v1": {"domains": ["deception"], "min_concepts": 8},
    "v2": {"domains": [...], "min_concepts": 8},
}
```

---

## 4. NLA sources (unchanged from v1 for now)

- `gemma-3-27b-it / kitft-l41` (Neuronpedia)
- `llama3.3-70b-it / kitft-l53` (Neuronpedia)

Adding new NLA sources (e.g., Gemma-4-E2B fine-tuned AVs from the deception
research program) is gated on: (a) the AV checkpoint being stable/published, and
(b) external review of the conditioning results. Not in v2.0.0 scope unless ARM
A/B/C/D produce a publishable conditioning result first.

---

## 4b. Family N — CTF Red/Blue (new, additive)

A game-theoretic packaging of Families G + M into a Red-vs-Blue Capture-the-Flag,
so an NLA-monitor can be reported as an adversarial scoreboard (clean-capture rate,
catch rate, stealth gap, evasion cost). Additive: it touches no v1 content.
**Shipped in v2.0.0.**

Status:
- [x] Harness module `nla_eval/ctf.py` (Flag/catalog, BlueMonitor, capture oracles,
      naive + adaptive Red policies, `play_ctf`, `CTFScore`) — reuses
      `redteam.adaptive_evade` and `deception.DECEPTION_ANCHORS`.
- [x] Plan catalog `plans/N_ctf_red_blue.md` (P119–P128) + design `docs/CTF_RED_BLUE.md`.
- [x] Control-tier offline demo `experiments/ctf_red_blue_demo.py` (MockNLA, CPU) +
      committed artifact `results/ctf/ctf_red_blue_mock.json`.
- [ ] **Primary-tier run** — swap the survival-proxy oracle for a real
      bottleneck-probe oracle (`ctf.probe_capture_oracle`) over the local NLA's own
      honest-vs-deceptive behavior (gated on work item E: local Gemma-4-E2B + GPU).
- [ ] Escalation ladder (P123) and category catalog expansion (P124) for power.
- [ ] Cross-monitor transfer runner (P127) once ≥2 probe+AV pipelines exist.

Gate: like Family M, only a primary-tier (probe oracle over model-own behavior)
result may be reported as a headline; control-tier runs are mechanics demos (P128).

## 5. Out of scope for v2

- Causal-fidelity (AR loop) — still P0 on the roadmap, still GPU-gated
- Human utility evals — still P1, still infra-gated
- Changing tier thresholds or the EmergenceIndex formula — would break v1
  backward comparability; reserved for a future generation if justified

---

## 6. Rollout gate for the v2 multi-domain content

The CTF family (Family N) and release maturation already shipped in **v2.0.0**. The
**multi-domain content** above stays out of any published v2 result table until:
- [ ] All H1–H12 harness fixes implemented and tested
- [ ] New domain concept/document sets constructed and reviewed
- [ ] At least one full eval run on a known NLA source produces a multi-domain result
- [ ] External review requested (Gemini + manual)
- [ ] Result artifacts added alongside v1 (never overwriting) + provenance recorded
      in `results/README.md`
- [ ] `CHANGELOG.md` updated and a `v2.x` tag pushed after review
