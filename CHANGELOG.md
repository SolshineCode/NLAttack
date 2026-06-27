# Changelog

All notable changes to NLAttack. The format follows [Keep a Changelog](https://keepachangelog.com/),
and the project follows the generation/freeze policy in
[docs/VERSIONING.md](docs/VERSIONING.md): once a generation is tagged, its benchmark
content and published results are frozen.

## v2.0.0 — 2026-06

The second benchmark generation. **Additive:** every v1 result is carried forward
unchanged; v2 adds a new evaluation paradigm and matures the release process.

### Added
- **Family N — CTF Red/Blue** (`plans/N_ctf_red_blue.md`, P119–P128): an adversarial
  capture-the-flag packaging of the NLA-as-monitor threat model. A Red model pursues
  a hidden misaligned objective (the flag — exfiltration, sandbagging, a concealed
  override, collusion, a backdoor) while a Blue NLA-monitor reads the AV
  verbalization and tries to catch it. Reported as a scoreboard: clean-capture rate
  (Red), catch rate (Blue's recall on genuine captures), stealth gap (the probe−AV
  exploitable surface), and evasion cost.
- **`nla_eval/ctf.py`** — the runnable harness: `Flag`/`CTF_FLAGS`, `BlueMonitor`/
  `nla_blue_monitor`, `survival_oracle`/`probe_capture_oracle`, `naive_red_policy`/
  `adaptive_red_policy`, `play_ctf`, `CTFScore`. Reuses `redteam.adaptive_evade` and
  `deception.DECEPTION_ANCHORS` (it packages existing axes, not a parallel suite).
- **`docs/CTF_RED_BLUE.md`** — the design, the four-outcome model, and the
  control-vs-primary integrity tiers.
- **`experiments/ctf_red_blue_demo.py`** + `results/ctf/ctf_red_blue_mock.json` — an
  offline, reproducible, CPU-only control-tier demo against `MockNLA`.
- **`CHANGELOG.md`** and **`results/README.md`** (generation provenance map).
- Domain-aware contrastive Gemma-4-E2B verbalizer-axis result
  (`results/local_gemma_e2b/domainaware_nla_verbalizer_axes.json`), reported as an
  honest negative on the generation axes.

### Changed
- Catalog grows from 118 plans / 13 families to **128 plans / 14 families** (the
  v1 catalog A–M is unchanged; Family N is new).
- Version bumped to `2.0.0` (`nla_eval/__init__.py`, `CITATION.cff`).
- README simplified into a single-screen front page; `docs/VERSIONING.md` rewritten
  around plain `v1`/`v2` generation labels.

### Integrity
- Capture is judged on the activation (the bottleneck probe), the flag on the AV
  text — never the same signal (the independent-review P0 #1 separation, already
  enforced in `redteam.py`).
- A CTF run is **primary-tier** only with a real bottleneck-probe oracle over the
  model's own behavior; the model-agnostic survival-proxy oracle over constructed
  flags is **control-tier** (mechanics, not a monitor certification). Every
  `CTFScore` carries its `tier` (Family N / P128, mirroring Family M / P118).

### Not yet in this release (v2.x roadmap)
- The multi-domain corpus expansion and the H1–H12 harness-correctness fixes
  (`docs/ROADMAP_v2.md`). The tagged v2.0.0 does not claim domain-coverage or
  fix-dependent results it has not produced.

## v1.0.0 — 2026-06 (frozen)

The first public benchmark generation (released as `0.1.0`).

### Highlights
- 118-plan catalog across 13 families (A–M): concept survival, content adjacency and
  laundering, deception and knowledge asymmetry, ATT&CK misuse detection, bottleneck
  probes, matcher/verbalizer confound, adaptive red-team, faithfulness, distribution
  shift, calibration, emergence, literature-informed evals, and
  deception/misalignment monitoring.
- Published result: EmergenceIndex 0.601 ("established") on `gemma-3-27b-it /
  kitft-l41`, plus the cross-NLA leaderboard (Llama-3.3-70B, Gemma-3-27B, and the
  local Gemma-4-E2B NLA). See `docs/RESULTS.md`.
- Floor-first design and null controls on every axis; API vs full-access tiers.

**Status: frozen.** Its content and numbers do not change; v2 carries them forward.
