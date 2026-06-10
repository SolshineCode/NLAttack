# NLAttack

**An evaluation suite for Natural Language Autoencoders (NLAs).**

NLAttack measures how faithfully a Natural Language Autoencoder turns a model's
internal activations into human-readable text, and how well that text can be used
to monitor model behavior, including deceptive or misaligned behavior. It ships a
Python harness, a catalog of 118 evaluation plans across 13 families, ready-made
adapters for hosted and local NLAs, and result artifacts.

The design goal is to work even on weak, small, or early-training NLAs (where most
metrics saturate or collapse) and to keep every claim honest with explicit null
controls. This README is self-contained: it defines the terms, explains the
methodology, and tells you how to run, extend, and cite the suite.

- Repository: https://github.com/SolshineCode/NLAttack
- Status: research software, actively developed. APIs may change.
- License: Apache-2.0 (see [License](#license)).

---

## What is a Natural Language Autoencoder?

A **Natural Language Autoencoder (NLA)** is an interpretability method that
explains a model's internal state in plain language. It has two trained parts:

```
activation  --[ AV: activation verbalizer ]-->  natural-language text
text        --[ AR: activation reconstructor ]-->  activation'
```

The **AV** reads a hidden activation and writes a description of what it
represents. The **AR** reads that description and reconstructs the activation.
Training optimizes the round trip: a good description lets the AR reconstruct the
original activation closely (measured in activation space, e.g. cosine
similarity). The human-readable **bottleneck** is the AV's text. The promise of
NLAs over sparse autoencoders or linear probes is that the explanation is in
natural language, so a person can read it and even edit it to intervene.

NLAs are introduced in Anthropic's "Natural Language Autoencoders" work
(https://www.anthropic.com/research/natural-language-autoencoders,
https://transformer-circuits.pub/2026/nla/). Interactive NLAs for several open
models are hosted on Neuronpedia (https://www.neuronpedia.org/nla), and an
open-source training library is `kitft/natural_language_autoencoders`.

## What NLAttack measures, and why

An NLA can fail on two distinct sides, and NLAttack scores both:

1. **Bottleneck side (the activation).** Does the activation even carry the
   concept? Measured with linear **probes on the activation**, each with a
   label-permutation null control.
2. **Verbalizer side (the AV text).** Does the AV's description actually surface
   the concept that the bottleneck holds? Measured by matching concepts in the
   verbalization, and by discrimination, retrieval, and faithfulness axes.

The gap between the two (`probe accuracy − verbalizer accuracy`) isolates how much
the verbalizer loses, which is usually the part a practitioner is stuck on.

Two principles run through the suite:

- **Floor-first.** On a weak NLA the verbalizer can be degenerate and the AR
  near-random, so end-to-end metrics collapse to noise. The base measurement is a
  single reliable primitive: *did concept C (or a near neighbor) survive into the
  verbalization, yes or no?* Everything aggregates from one per-concept table.
- **Null controls on everything.** A probe at small n can hit high accuracy on
  pure noise. Every probe reports a label-permutation floor, and a result counts
  as signal only when it clears that floor by a margin. Matcher-dependent effects
  are flagged by an ensemble-agreement check.

## Installation

```bash
git clone https://github.com/SolshineCode/NLAttack.git
cd NLAttack
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The core harness has **no required dependencies** (the lexical matcher and the
mock NLA run on the standard library). The extras in `requirements.txt` unlock
stronger components:

- `sentence-transformers`, `nltk` for the strongest concept-matching backends and
  semantic retrieval,
- `scikit-learn`, `numpy`, `scipy` for the probe / emergence / deception axes,
- `torch`, `transformers`, `safetensors` only for running a **local** NLA.

## Quickstart

Run the offline smoke test (a deliberately lossy mock NLA, no network or GPU):

```bash
python run_example.py
```

Plug in any NLA by implementing one method:

```python
from nla_eval import NLA

class MyNLA(NLA):
    name = "my-nla"
    def reconstruct(self, text: str) -> str:
        return my_verbalizer(text)          # text -> activation -> AV description
    # optional: encode(text) -> activation vector  (used by the probe axes)
```

Evaluate a hosted NLA on Neuronpedia (no API key, rate limited to 120 req/hr/IP):

```python
from nla_eval import NeuronpediaNLA, EnsembleMatcher, Example, run

# A dataset is a list of Example(id, text, concepts) — `concepts` are the
# controlled concepts present in `text` that we check survived the bottleneck.
my_dataset = [
    Example(id="ex1", text="The committee published its annual budget report",
            concepts=["budget", "report"]),
    Example(id="ex2", text="A storm warning was issued for the coastal region",
            concepts=["storm", "warning"]),
]

nla = NeuronpediaNLA(model_id="llama3.3-70b-it", nla_source_id="kitft-l53")
result = run(nla, my_dataset, matcher=EnsembleMatcher())
```

Discover the hosted NLAs with `GET https://www.neuronpedia.org/api/nla/sources`.
At time of writing two are hosted: `gemma-3-27b-it`/`kitft-l41` and
`llama3.3-70b-it`/`kitft-l53`. The hosted inference backend can be intermittent
(the Gemma-3 endpoint was returning 502 at time of writing); the suite retries
gateway errors and records per-NLA failures rather than crashing.

## The evaluation suite

The suite has two layers: a **plan catalog** (the design space) and an
**implemented harness** (runnable code).

### Plan catalog: 118 plans, 13 families

Each plan is a falsifiable evaluation with a fixed schema (hypothesis, method,
metric, feasibility, controls, and a "null looks like" line). The full table is in
[`plans/INDEX.md`](plans/INDEX.md); the schema is in [`plans/README.md`](plans/README.md).

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

For evaluating weak or tiny NLAs, [`plans/RUDIMENTARY_TIERS.md`](plans/RUDIMENTARY_TIERS.md)
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

Worked examples are in `experiments/` (`cross_nla_eval.py`, `plot_cross_nla.py`,
`emergence_dashboard.py`, `deception_probe_demo.py`, and others).

## Activation-side vs verbalizer-side axes

A practical distinction the suite makes explicit. **Activation-side** axes
(decodability, selectivity, stability, effective rank) read the frozen base
activation, so they are roughly constant across AV checkpoints and do **not**
track AV training. **Verbalizer-side** axes (in `verbalizer_axes.py`) run the AV,
so they move as it is trained. To track AV conditioning across training
checkpoints, use the verbalizer-side axes. See
[`docs/AV_CONDITIONING_AXES.md`](docs/AV_CONDITIONING_AXES.md).

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

## Reproducible findings to date

These come with the suite (`results/`) and are stated with their caveats.

- **Cross-NLA, hosted (no GPU).** On a fixed dataset, the **Llama-3.3-70B** NLA
  (`kitft-l53`) reaches concept retention 0.95, zero attack-to-benign laundering,
  and mean AV faithfulness (cosine) 0.888. Held-out doc retrieval, matched to the
  Gemma-4 protocol below (same docs, 12 distractors so chance is 0.077, MiniLM
  embedder): char 0.396, semantic 0.503.
- **Local Gemma-4-E2B NLA.** The bottleneck encodes in-distribution concepts at
  about 0.99 probe AUC and out-of-distribution concepts at about 0.70, so the
  bottleneck is not a uniform filter. The verbalizer is the weak side: held-out
  doc retrieval is about 0.135 (chance 0.077, not significant), placing it at the
  conditioning wall. Same metric, the Llama NLA is roughly 3.7x higher.
- **Deception probe on real behavioral-split data.** On honest-vs-deceptive
  activations from prior work (nanochat-d32, layer 12, n=1327), the deception
  probe reaches AUROC 0.932 / balanced accuracy 0.866 versus a shuffled-label
  floor of 0.49, reproducing that line of work's roughly 0.87 baseline through
  NLAttack's own permutation-controlled probe.

Numbers from the hosted API can change under a fixed source id (the backend is
redeployed), which is exactly why results are attributed and dated.

## Result attribution: name results by the NLA, not the base model

A base model can host many different NLAs (different AV/AR checkpoints, layers,
and training runs). `gemma-3-27b-it` is a *model*; `kitft/nla-gemma3-27b-av` at
layer 41 is an *NLA*, and two NLAs on the same base model can score very
differently. **Every result in this suite is attributed to the NLA.**

Canonical NLA id (record it with every result):

```
<av_checkpoint>__<base_model>__L<layer>__<source_id>
# hosted example:  kitft/nla-gemma3-27b-av__gemma-3-27b-it__L41__kitft-l41
# local example:   av_v0_1__google/gemma-4-E2B__L23
```

Short label for plots and tables: `<av_basename>@L<layer>`, e.g.
`nla-gemma3-27b-av@L41`. When you publish a number, record the NLA id, the base
model, the layer, the AV and AR checkpoints, the eval suite version (a commit
hash), the dataset, the matcher backend, and the date. `nla_name()` in
`experiments/cross_nla_eval.py` builds these fields automatically.

## Methodology and honest limits

- **Concept matching is the weak link.** Prefer the embedding matcher and spot-
  check matched terms. The ensemble agreement / contested-rate check flags effects
  that only one matcher topology sees.
- **A low score is ambiguous** without controls. It can mean the NLA dropped the
  concept, or the verbalizer could not phrase it, or the matcher missed it. The
  bottleneck probe (ground truth) and the permutation null are how you tell these
  apart.
- **Read differentially.** Compare every number to the frequency-and-length
  matched control and the noise floor, not in isolation.
- **Hosted-API subset.** The probe and emergence axes need raw activations, which
  the hosted API does not expose. Over the API you can run the verbalizer-side and
  retention evals, and run the probe axes against a local NLA.

The rationale behind these choices, including two external expert reviews, is in
[`DESIGN_REVIEW.md`](DESIGN_REVIEW.md) and [`docs/`](docs/).

## Repository layout

```
nla_eval/         the harness (importable package)
plans/            118 evaluation plans (A–M), INDEX.md, schema, rudimentary tiers
experiments/      runnable scripts (cross-NLA eval, plots, emergence, deception)
results/          committed result artifacts (cross_nla/, emergence/, deception/, ...)
docs/             literature review, AV-conditioning axes, design notes
DESIGN_REVIEW.md  validity threats and the rationale for the controls
requirements.txt  optional extras (core needs none)
```

## How to cite

If you use NLAttack in research, please cite the software. A `CITATION.cff` is
included so GitHub shows a "Cite this repository" button.

Recommended citation:

> DeLeeuw, C. (2026). *NLAttack: An evaluation suite for Natural Language
> Autoencoders* [Computer software]. https://github.com/SolshineCode/NLAttack

BibTeX:

```bibtex
@software{deleeuw_nlattack_2026,
  author  = {DeLeeuw, Caleb},
  title   = {{NLAttack}: An Evaluation Suite for Natural Language Autoencoders},
  year    = {2026},
  url      = {https://github.com/SolshineCode/NLAttack},
  note    = {Version 0.1.0}
}
```

**Citing a result produced with NLAttack.** A score is meaningful only with the
NLA it was measured on. Report the canonical NLA id, the suite commit hash, the
dataset, the matcher backend, and the date alongside the number. For example:
"retention 0.95 on `kitft/Llama-3.3-70B-NLA-av__llama3.3-70b-it__L53__kitft-l53`,
NLAttack `@<commit>`, cross_nla dataset, ensemble matcher, 2026-06-08." This lets
others reproduce and compare against the exact NLA and suite version.

## Acknowledgements and references

- Natural Language Autoencoders: Anthropic
  (https://www.anthropic.com/research/natural-language-autoencoders;
  https://transformer-circuits.pub/2026/nla/).
- Hosted NLAs and the inference API: Neuronpedia (https://www.neuronpedia.org/nla).
- NLA training library: `kitft/natural_language_autoencoders`.
- The misuse family is grounded in Anthropic's LLM ATT&CK Navigator
  (https://red.anthropic.com/2026/attack-navigator/) and the MITRE ATT&CK
  framework.
- The methodology draws on the probing-classifier literature (control tasks and
  selectivity), the SAE interpretability literature, and work on measuring
  emergence. The full reading list with arXiv ids is in
  [`docs/LITERATURE.md`](docs/LITERATURE.md).

## License

NLAttack is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for
the full terms and [`NOTICE`](NOTICE) for the attribution notice. You may use,
modify, and redistribute it, including commercially, provided you retain the
copyright, attribution, and license notices and state any changes you made
(Apache-2.0 Section 4). The license also includes an explicit patent grant.

## Contact

Caleb DeLeeuw (`SolshineCode`), caleb.deleeuw@gmail.com.
