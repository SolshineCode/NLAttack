# Contributing to NLAttack

Thanks for helping evaluate Natural Language Autoencoders. There are two main ways
to contribute: **submit a result** for an NLA, or **improve the harness/catalog**.

## Submit a result (add your NLA to the leaderboard)

NLAttack is built so any NLA can be scored with one adapter method.

1. **Implement the adapter.** Provide an `NLA` with a `reconstruct(text) -> str`
   method (the AV verbalization of the activation). Hosted, text-only NLAs use the
   universal API tier; local NLAs with raw activations unlock the full-access tier.
   See [`docs/EVALUATIONS.md`](docs/EVALUATIONS.md).
2. **Run the suite** and write the result JSON under `results/` (a new file — never
   overwrite another NLA's artifact). Worked runners are in `experiments/`.
3. **Attribute it.** Name results by the **NLA**, not the base model, and record the
   canonical NLA id, suite version (`nla_eval.__version__`), dataset, matcher
   backend, and date — see the attribution convention in
   [`docs/RESULTS.md`](docs/RESULTS.md). `nla_name()` in
   `experiments/cross_nla_eval.py` fills these in.
4. **Open a PR** that adds the result file, a row in `docs/RESULTS.md`, and an entry
   in [`results/README.md`](results/README.md) (generation provenance). Report a
   number with its **null control** — a result counts only when it clears the
   permutation/chance floor.

## Improve the harness or the plan catalog

- **New evaluation plan:** follow the schema in [`plans/README.md`](plans/README.md)
  (hypothesis, method, metric, feasibility, controls, "null looks like"). Add it to
  the right family file and to [`plans/INDEX.md`](plans/INDEX.md).
- **New coded axis:** add the module under `nla_eval/`, export it from
  `nla_eval/__init__.py`, and add a unit test under `tests/`.

## Development

```bash
pip install numpy scikit-learn pytest   # probe/emergence/deception axes + tests
python run_example.py                    # offline smoke test
python experiments/ctf_red_blue_demo.py  # CTF Red/Blue demo
pytest -q                                # unit tests
```

CI runs the smoke test, the CTF demo, and `pytest` on Python 3.9 / 3.11 / 3.12; keep
them green. Two house rules:

- **Null controls everywhere.** Every reported number clears an explicit
  permutation/chance floor; negatives are reported honestly, not hidden.
- **Freeze-on-release.** Published results are frozen per generation
  ([`docs/VERSIONING.md`](docs/VERSIONING.md)); new content lands in a new
  generation and never edits a prior one's artifacts.

By contributing you agree your contributions are licensed under Apache-2.0
([`LICENSE`](LICENSE)).
