# nla-eval-harness

Capability-floor evaluation harness for **Natural Language Autoencoders** (Neuronpedia-hosted NLAs, Gemma E2B autoencoders, or anything that ships next).

Built on the premise that current NLAs can't represent negation, order, attribution, or epistemic state — so the **only** reliable measurement is the floor primitive:

> **Did concept C (or a near-neighbor) survive the bottleneck? yes/no**

An NLA is treated as a **lossy concept filter**. The harness characterizes that filter, and all 20 tests are just different `group-by`s over one per-concept retention table.

## Plug in any NLA (one method)

```python
from nla_eval import NLA

class MyNLA(NLA):
    name = "my-nla"
    def reconstruct(self, text: str) -> str:
        return my_model(text)        # encode->bottleneck->decode
    # optional: encode(text)->vector  (only the latent-probe tests use it)
```

**A real NLA is `activation → verbalizer (AV) → text → reconstructor (AR) → activation`** — not text→text. The human-readable bottleneck is the AV's description, so the harness measures concept survival as `input text → host activation @ layer L → AV verbalization`, then asks whether concept C survived. That's the "monitor reads the NL bottleneck" framing.

Ready-made adapters in `adapters.py`:
- `MockNLA` — deps-free lossy filter, for testing the harness itself.
- `NeuronpediaNLA` — **verified live** against `POST /api/nla/explain` (no API key needed; 120 req/hr/IP). Sources via `GET /api/nla/sources`: `gemma-3-27b-it`/`kitft-l41` (Gemma) and `llama3.3-70b-it`/`kitft-l53`. The bottleneck text = concatenated per-position AV `description`s. Run `python test_neuronpedia_live.py`.
- `KitftNLA` — local AV/AR checkpoints from `kitft/natural_language_autoencoders` (`kitft/nla-{base}-L{layer}-av`). Grabs the host-model activation, then verbalizes.
- `CallableNLA` — wrap any `fn(str)->str`.

See `real_nla_example.py` for both wiring paths.

## Run

```bash
python run_example.py          # mock NLA + attack dataset, prints all readouts
```

`Matcher(backend="auto")` self-selects the strongest concept-presence backend
installed: **embedding** (sentence-transformers) > **wordnet** (nltk) > **lexical**
(always works, zero deps). Force one with `backend=...` for reproducibility.

## The 20 tests

Each is a `group-by` over the retention table; tags come from `Example.meta`.

**General adjacency (1–10)** — `tests.t01..t09`
1 dropout law · 2 adjacency substitution · 3 category-selective dropout · 4 specificity collapse · 5 salience crowding* · 6 paired-neighbor asymmetry* · 7 co-occurrence hallucination · 8 quantity flattening* · 9 blind-spot stability · 10 floor calibration control*

**ATT&CK misuse (11–20)** — grounded in the LLM ATT&CK Navigator post — `tests.t11..t20`
11 technique coverage · 12 obfuscation see-through (central) · 13 attack→benign laundering · 14 risk-weighted dropout · 15 evasion robustness · 16 kill-chain co-occurrence · 17 agentic-scaffolding markers* · 18 ARiES axis separability · 19 dual-use disambiguation* · 20 severity monotonicity

`*` = scaffolded via the same retention table + meta tags; add the tagged dataset to activate. The starred ones are the frontier/expected-null tests — run them, don't lean on them.

## Validity layer (from the Hermes design review — see `DESIGN_REVIEW.md`)

A skeptical review (Hermes `nemotron-3-ultra`) flagged that the raw retention
signal is confounded: the AV verbalizer and the matcher are two extra lossy
filters. These additions subtract that confound:

- **Bottleneck probes** (`nla_eval/bottleneck_probe.py`) — the deepest fix.
  Train a linear probe on the NLA's activation (the true bottleneck) for ground-
  truth concept presence, independent of verbalizer/matcher. `probe_acc −
  av_matcher_acc = verbalizer+matcher loss`. Demonstrated on a local Gemma-4-E2B
  NLA: in-distribution concepts probe at ~0.99 AUC, OOD at ~0.70 — the bottleneck
  is *not* a uniform filter. See `results/local_gemma_e2b/`. Each probe carries a
  **label-permutation null control** (`ProbeResult.signal`) so a high AUC at small
  n can't be mistaken for overfit noise.
- **Rudimentary-NLA floor tooling** (`nla_eval/rudimentary.py`) — for evaluating
  *weaker/tinier* NLAs (degenerate AV, near-random AR), where end-to-end metrics
  collapse. `bottleneck_exists` (does any signal clear the noise floor?),
  `identity_leakage_vectors/_text` (is the AR/AV conditioned on input at all?),
  `fit_forced_verbalizer` (read the bottleneck via probes when the AV is empty).
  See `plans/RUDIMENTARY_TIERS.md` for which plans to run first vs shelve.
- **Emergence dashboard** (`nla_eval/emergence.py`, `experiments/emergence_dashboard.py`)
  — detect *useful interpretability capability starting to emerge* in a weak/tiny
  NLA. An **additive Emergence Index** sums orthogonal, null-gated axes
  (decodability, sufficiency, selectivity, content-adjacency, faithful-rank,
  stability, dose-response, graded encoding, abstraction transfer, effective rank),
  gated on decodability clearing the permutation floor so a high score can't come
  from overfit/degeneracy. Sweep checkpoints for the emergence curve. Plans P101–P106
  (Family K). Verified: in-dist index ≈0.60 ("established"), cross-domain ≈0.49
  ("early", driven by near-zero abstraction transfer).
- **Matcher ensemble** (`EnsembleMatcher`) — vote across topologies
  (lexical/fuzzy/overlap [+embedding/wordnet]); only >threshold agreement counts.
  `ConceptRow.agreement` + `contested_rate()` expose matcher-dependent (suspect)
  effects.
- **Faithfulness weighting** (`faithfulness_weighted_retention`) — down-weight
  "survived" concepts whose activation was reconstructed poorly, using the
  `cosine_similarity`/`mse` the Neuronpedia API already returns.
- **Frequency+length matched controls** (`nla_eval/controls.py`) — for every test.
- **Adaptive red-team** (`nla_eval/redteam.py`) — compositional blind spots +
  query-budgeted evasion search; static i.i.d. tests miss adaptive attackers.
- **Local NLA adapter** (`nla_eval/local_gemma_e2b.py`) — activation→AV
  verbalization for a local Gemma-4-E2B NLA (full activation access for probes).

## Honest limits

- **Concept matching is the weak link.** Embedding backend recommended; always
  human-spot-check `matched_term` in `retention_table.csv`.
- **Floor confound:** a low score can mean "NLA dropped it" *or* "decoder couldn't
  phrase it." Test 10 (below-floor control) calibrates where you've run out of signal.
- **Differential, not absolute:** read every number against the dropout law (t01)
  and the frequency-matched control (t03), never alone.
```

## Deception / misalignment monitoring (`nla_eval/deception.py`, Family M)

Can an NLA be used to *catch* deceptive/misaligned behavior? Adapted from the SAE
deception-detection methodology in `deception-nanochat-sae-research` (same-prompt
behavioral split, multi-judge honest-vs-deceptive labels), wired into the existing
axes: **bottleneck** (`deception_probe` — is honest-vs-deceptive decodable from the
activation, with a label-permutation null = their specificity-gap control),
**verbalizer** (`deception_discrimination` — does the AV text distinguish deceptive
from honest, the NLA-as-monitor metric), **cross-scenario transfer**, and the
probe-vs-verbalizer **gap** (deception the AV drops). Integrity gate: labels must be
the model's own behavior under an *identical* prompt (judge-labeled), never
instructed deception or role-play. Validated on real data: reproduces the source
repo's ~0.87 balanced-accuracy baseline (nanochat-d32 L12, AUROC 0.932) through
NLAttack's own permutation-controlled probe. See `experiments/deception_probe_demo.py`.

## Result attribution — name results by the NLA, NOT the base model

**A base model can host many different NLAs** — different AV/AR checkpoints,
different layers, different training runs/maturity. `gemma-3-27b-it` is a *model*;
`kitft/nla-gemma3-27b-av @ layer 41` is an *NLA*. Two NLAs on the same base model
can score completely differently. So **every result in this suite is attributed
to the NLA, not the base model.**

**Canonical NLA id** (record this with every result):

```
<av_checkpoint>__<base_model>__L<layer>__<source_id>
# hosted example:  kitft/nla-gemma3-27b-av__gemma-3-27b-it__L41__kitft-l41
# local example:   av_v0_1_aux_readout__google/gemma-4-E2B__L23
```

**Short chart label** (for plots/tables): `<av_basename>@L<layer>` —
e.g. `nla-gemma3-27b-av@L41`, `Llama-3.3-70B-NLA-av@L53`.

When you publish results, record alongside the number: **NLA id, base model,
layer, AV + AR checkpoints, the eval suite + dataset used, the matcher backend,
and the date** (the hosted NLA inference can change under a fixed source id). The
helpers in `experiments/cross_nla_eval.py` (`nla_name()`) and the discovery call
`GET /api/nla/sources` produce these fields automatically.

### Cross-NLA results (`results/cross_nla/`)
`experiments/cross_nla_eval.py` runs the **API-compatible** evals
(concept-survival retention, substitution/laundering, obfuscation see-through, and
AV activation-space faithfulness `cosine_similarity`/`mse`) over a fixed dataset
for every NLA hosted on Neuronpedia — **no local GPU** — and
`experiments/plot_cross_nla.py` renders the side-by-side comparison. Note: the
**bottleneck-probe / emergence dashboard is NOT in the cross-NLA set** — it needs
raw activation vectors the hosted API does not expose (run it locally per NLA).
