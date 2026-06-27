<p align="center">
  <img src="docs/figures/nlattack-hero.png" alt="NLAttack overview: what a Natural Language Autoencoder is (activation, AV verbalizer, plain-language text bottleneck, AR reconstructor), the two questions it answers (capability and safety), the two access tiers, the headline finding, and how to get started." width="560">
</p>

# NLAttack

**An evaluation suite for Natural Language Autoencoders (NLAs).**

![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![version](https://img.shields.io/badge/release-v2.0.0-brightgreen.svg)
![status](https://img.shields.io/badge/status-research%20software-orange.svg)

A **Natural Language Autoencoder** explains a model's internal state in plain
language: an activation verbalizer (AV) turns a hidden activation into text, and an
activation reconstructor (AR) rebuilds the activation from that text.

```
activation --[ AV ]--> natural-language text --[ AR ]--> activation'
```

NLAttack asks two questions about that human-readable bottleneck — **does the NLA
work at all** (capability) and **can you catch misuse by reading it** (safety) — and
answers them with explicit null controls, on NLAs as weak as a 4 GB-GPU checkpoint.

## Quickstart

```bash
git clone https://github.com/SolshineCode/NLAttack.git && cd NLAttack
pip install -r requirements.txt        # core harness needs no dependencies
python run_example.py                  # offline smoke test (mock NLA, no GPU)
```

Score any hosted NLA in three lines:

```python
from nla_eval import NeuronpediaNLA, EnsembleMatcher, Example, run

nla = NeuronpediaNLA(model_id="llama3.3-70b-it", nla_source_id="kitft-l53")
result = run(nla, [Example(id="ex1", text="A storm warning was issued for the coast",
                           concepts=["storm", "warning"])], matcher=EnsembleMatcher())
```

To evaluate your own local NLA, implement the one-method `NLA` adapter
([docs/EVALUATIONS.md](docs/EVALUATIONS.md)).

## Results (v1, frozen)

The v1 benchmark results are published and frozen; v2 carries them forward
unchanged. Full tables and caveats in [docs/RESULTS.md](docs/RESULTS.md).

![NLAttack benchmark overview across the evaluated NLAs](results/cross_nla/benchmark_overview.png)

| NLA | Access | Concept retention | Doc retrieval (semantic) | Bottleneck probe (in / OOD) |
|---|---|---|---|---|
| `Llama-3.3-70B-NLA-av@L53` | hosted | 0.95 | 0.503 | not available over API |
| `nla-gemma3-27b-av@L41` | hosted | 0.90 | not yet run | not available over API |
| `Gemma-4-E2B-NLA@L23` | local | 0.00 (out-of-domain) | 0.135 (in-domain) | 0.988 / 0.695 |

The hosted NLAs are verbalizer-strong (a reader of their AV text recovers most
concepts). The local Gemma-4-E2B NLA is the mirror image — its bottleneck probes
near-perfectly in-distribution but its verbalizer is weak and domain-specific.
Separating those two failure modes is the point of the suite.

## What's inside

- **A 128-plan catalog across 14 families (A–N).** Every plan is a falsifiable test
  with a hypothesis, a method, a metric, and a null-control baseline. Index:
  [plans/INDEX.md](plans/INDEX.md).
- **Two access tiers.** An API tier scores any hosted, text-only NLA (the universal
  leaderboard); a full-access tier adds white-box probe/emergence axes that need raw
  activations. Query it in code via `nla_eval.access`.
- **Deception / misalignment monitoring** (Family M): can the NLA's text be read to
  catch a model's own deceptive behavior?
- **CTF Red/Blue** (Family N, **new in v2**): a Red model captures a hidden
  misaligned-objective flag while a Blue NLA-monitor reads the AV text to catch it,
  reported as an adversarial scoreboard (clean-capture rate vs. catch rate). Offline
  demo: `python experiments/ctf_red_blue_demo.py` · design:
  [docs/CTF_RED_BLUE.md](docs/CTF_RED_BLUE.md).
- **Two principles throughout:** floor-first (one reliable per-concept primitive, so
  weak NLAs still yield signal) and null controls on everything (a result counts only
  when it clears a permutation floor).

## Documentation

| Document | Contents |
|---|---|
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | What NLAs are, the two purposes, and the validity limits |
| [docs/EVALUATIONS.md](docs/EVALUATIONS.md) | The 128-plan catalog, the harness modules, and the access tiers |
| [docs/RESULTS.md](docs/RESULTS.md) | Reproducible findings, the leaderboard, and the attribution convention |
| [docs/CTF_RED_BLUE.md](docs/CTF_RED_BLUE.md) | The v2 Red/Blue capture-the-flag family (Family N) |
| [CHANGELOG.md](CHANGELOG.md) · [docs/VERSIONING.md](docs/VERSIONING.md) | Release history and the freeze-on-release policy |
| [DESIGN_REVIEW.md](DESIGN_REVIEW.md) · [docs/LITERATURE.md](docs/LITERATURE.md) | Validity threats; the reading list with arXiv ids |

## How to cite

A `CITATION.cff` is included, so GitHub shows a "Cite this repository" button.
A score is meaningful only with the NLA it was measured on — cite the canonical NLA
id, the suite version, the dataset, and the date (see [docs/RESULTS.md](docs/RESULTS.md)).

```bibtex
@software{deleeuw_nlattack_2026,
  author = {DeLeeuw, Caleb},
  title  = {{NLAttack}: An Evaluation Suite for Natural Language Autoencoders},
  year   = {2026},
  url    = {https://github.com/SolshineCode/NLAttack},
  note   = {Version 2.0.0}
}
```

## Acknowledgements

NLAs are introduced in Anthropic's Natural Language Autoencoders work
([overview](https://www.anthropic.com/research/natural-language-autoencoders),
[writeup](https://transformer-circuits.pub/2026/nla/)); interactive NLAs are hosted
on [Neuronpedia](https://www.neuronpedia.org/nla). The misuse family is grounded in
Anthropic's [LLM ATT&CK Navigator](https://red.anthropic.com/2026/attack-navigator/)
and [MITRE ATT&CK](https://attack.mitre.org/).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Contact: Caleb DeLeeuw
(`SolshineCode`), caleb.deleeuw@gmail.com.
