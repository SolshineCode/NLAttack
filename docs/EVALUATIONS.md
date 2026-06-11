# NLAttack evaluation suite

The full catalog of evaluation plans, the implemented harness modules, the
access tiers that decide what runs on a hosted vs a local NLA, and the deception
monitoring family. For concepts see [METHODOLOGY.md](METHODOLOGY.md); for results
see [RESULTS.md](RESULTS.md).

## The evaluation suite

The suite has two layers: a **plan catalog** (the design space) and an
**implemented harness** (runnable code).

### Plan catalog: 118 plans, 13 families

Each plan is a falsifiable evaluation with a fixed schema (hypothesis, method,
metric, feasibility, controls, and a "null looks like" line). The full table is in
[`plans/INDEX.md`](../plans/INDEX.md); the schema is in [`plans/README.md`](../plans/README.md).

| Family | Plans | Theme |
|---|---|---|
| A | P001–P010 | Concept survival and the dropout map (the floor primitive) |
| B | P011–P020 | Content adjacency and laundering (neighbor substitution, hypernym drift) |
| C | P021–P030 | Deception and knowledge asymmetry (epistemic-state preservation) |
| D | P031–P040 | ATT&CK / misuse detection (does the bottleneck preserve attack content) |
| E | P041–P050 | Bottleneck probes / ground truth (probe-vs-verbalizer gap) |
| F | P051–P060 | Matcher and verbalizer confound (isolating measurement error) |
| G | P061–P070 | Adaptive red team (compositional blind spots, query-budget evasion) |
| H | P071–P080 | Faithfulness / AR fidelity (activation-space reconstruction) |
| I | P081–P090 | Distributional / OOD / training dependence |
| J | P091–P100 | Calibration / uncertainty / self-knowledge |
| K | P101–P106 | Emergence and the additive capability index |
| L | P107–P112 | Literature-informed evaluations (simulatability, steerability, etc.) |
| M | P113–P118 | Deception / misalignment monitoring |

For evaluating weak or tiny NLAs, [`plans/RUDIMENTARY_TIERS.md`](../plans/RUDIMENTARY_TIERS.md)
sorts the plans into run-first, floor-version, and skip-until-stronger tiers.

### Implemented harness (`nla_eval/`)

| Module | What it does |
|---|---|
| `core.py` | the per-concept retention table and `run()` |
| `adapters.py` | the `NLA` contract plus `MockNLA`, `NeuronpediaNLA`, `KitftNLA`, `CallableNLA` |
| `matching.py` | `Matcher` and `EnsembleMatcher` (lexical/fuzzy/overlap, plus embedding/wordnet when installed) with a continuous `soft_score` |
| `tests.py` | 20 coded retention tests (general adjacency + ATT&CK misuse) as group-bys |
| `bottleneck_probe.py` | linear probes on the activation with a permutation null |
| `emergence.py` | hierarchical-tier capability verdict and the longitudinal emergence criterion |
| `verbalizer_axes.py` | AV-conditioning axes that move with AV training: minimal-pair discrimination (AUC), held-out doc retrieval, prior deviation, mode collapse, calibration |
| `deception.py` | deception / misalignment monitoring (probe, discrimination, cross-scenario transfer) |
| `confabulation.py` | factual grounding, thematic fidelity, consistency |
| `rudimentary.py` | floor checks (does a bottleneck exist, and is the AR/AV input-conditioned) |
| `controls.py` | frequency-and-length-matched controls |
| `redteam.py` | adaptive evasion and compositional blind-spot search |
| `attack_concepts.py` | ATT&CK technique dictionary for the misuse family |
| `local_gemma_e2b.py` | adapter for a local Gemma-E2B NLA (full activation access) |
| `access.py` | the access-tier map (API-runnable vs full-access) and leaderboard-metric list |

Worked examples are in `experiments/` (`cross_nla_eval.py`, `plot_cross_nla.py`,
`emergence_dashboard.py`, `deception_probe_demo.py`, and others).


## Access tiers: API-runnable vs full-access

For NLAttack to work as a public benchmark, the suite is split by the access a
given NLA requires, the way SWE-bench or a closed-book exam separate what every
system can attempt from what only some can. The split is encoded in code at
[`nla_eval/access.py`](../nla_eval/access.py), so you can ask it directly
(`access.requires_full_access("bottleneck_probe")`).

**Tier 1, API-runnable (the universal leaderboard).** Everything computable from
what a hosted endpoint returns: the AV's verbalization text, plus a scalar
reconstruction faithfulness (cosine / mse) if the endpoint provides one. This runs
on **any** NLA, hosted or local, so every NLA gets a row. Local NLAs are scored
here too. They simply also unlock Tier 2. This tier covers concept retention,
content adjacency and laundering, the ATT&CK misuse tests, held-out doc retrieval,
minimal-pair discrimination, mode collapse, calibration, and the verbalizer side of
deception monitoring.

**Tier 2, full-access (the white-box extension).** Anything that needs the raw
activation vector, the model internals, or training checkpoints, which a hosted API
does not expose. This is where the linear bottleneck probes, the probe-minus-
verbalizer gap, activation-space faithfulness, the emergence and longitudinal
capability index, and the probe-grounded misuse and deception plans live. It runs
only on NLAs you have the weights for.

| Family | Theme | Access tier |
|---|---|---|
| A | concept survival / dropout map | API |
| B | content adjacency / laundering | API |
| C | deception and knowledge asymmetry | mixed (discrimination API, epistemic probe full) |
| D | ATT&CK / misuse detection | mixed (survival, see-through, laundering API; P037/P038/P040 full) |
| E | bottleneck probes / ground truth | full-access |
| F | matcher and verbalizer confound | API |
| G | adaptive red team (text-level) | API |
| H | faithfulness / AR fidelity | mixed (scalar cos/mse API if exposed, activation-space analysis full) |
| I | distributional / OOD / training dependence | full-access |
| J | calibration / uncertainty | API |
| K | emergence and additive capability index | full-access |
| L | literature-informed axes | mixed (simulatability API; steerability full) |
| M | deception / misalignment monitoring | mixed (discrimination API; probe and transfer full) |

A second, independent axis matters for the leaderboard: **comparability**. Of the
API-tier metrics, only the **protocol-anchored** ones (a fixed null regardless of
model or dataset) are honestly comparable across different NLAs: held-out doc
retrieval (chance set by the distractor count), minimal-pair and deception
discrimination AUC (chance 0.5), obfuscation see-through, and laundering rate. These
are `access.LEADERBOARD_METRICS`. Activation-space scalars like raw cosine are not
on a common scale across base models, so they are reported per-NLA, not ranked.

This access split lines up with an older distinction the suite still uses for a
different reason. **Activation-side** axes (decodability, selectivity, stability,
effective rank) read the frozen base activation, so they are roughly constant across
AV checkpoints and do not track AV training. They are also the full-access axes.
**Verbalizer-side** axes (in `verbalizer_axes.py`) run the AV, move as it trains,
and are the API-tier axes. To track AV conditioning across training checkpoints, use
the verbalizer-side axes. See
[`docs/AV_CONDITIONING_AXES.md`](AV_CONDITIONING_AXES.md).


## Deception / misalignment monitoring (Family M)

The deception family asks: can this NLA be used to *catch* deceptive or misaligned
behavior? It adapts the methodology of prior SAE-based deception-detection work,
where honest and deceptive activations are collected under an **identical prompt**
(the behavior is the model's own choice, not an instruction) and labeled by a
multi-judge consortium. NLAttack wires that substrate into its existing axes:

- `deception_probe`: is honest-vs-deceptive linearly decodable from the
  bottleneck? (with the label-permutation null as the specificity control)
- `deception_discrimination`: does the AV text distinguish deceptive from honest
  on the same prompt? (the NLA-as-monitor metric)
- `cross_scenario_transfer`: does detection generalize across scenario domains?

**Integrity gate (non-negotiable):** labels must come from the model's own
behavior under an identical prompt. Instructed deception or role-assignment turns
the task into prompt classification, so any such run is a control, not a primary
result. This is enforced in the family's documentation and result metadata.

