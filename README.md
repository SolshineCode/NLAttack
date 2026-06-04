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

## Honest limits

- **Concept matching is the weak link.** Embedding backend recommended; always
  human-spot-check `matched_term` in `retention_table.csv`.
- **Floor confound:** a low score can mean "NLA dropped it" *or* "decoder couldn't
  phrase it." Test 10 (below-floor control) calibrates where you've run out of signal.
- **Differential, not absolute:** read every number against the dropout law (t01)
  and the frequency-matched control (t03), never alone.
```
