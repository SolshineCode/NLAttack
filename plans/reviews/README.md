# Dual-reviewer critiques of the 100 plans (rudimentary-NLA focus)

Two independent agents critiqued all 100 plans (P001–P100), specifically for
evaluating **smaller / more rudimentary NLAs** than the ~2B weak NLA the plans
were written for (tiny models, near-empty/degenerate AV, retrieval-only
verbalizers, near-random AR with round-trip cosine ~0.1–0.4, low-dim bottlenecks,
tiny training sets).

For each plan each reviewer gave: **signal** (usable or floors/saturates/noise),
**failure mode** at low capability, a **floor version** (how to simplify so it
still yields signal), and a **missed flaw**; each ended with per-family
"skip on rudimentary NLAs" and "highest-value" lists.

## Reviewers
- **`reviewer-a/`** — the first independent reviewer.
- **`reviewer-b/`** — the second independent reviewer.

## Files
`reviewer-a/<family>.md` and `reviewer-b/<family>.md` — one file per family A–J,
each covering its 10 plan IDs. **`SUMMARY.md`** — the consolidated synthesis
(convergent guidance, per-family verdicts from both, and disagreements).

Each reviewer was given the same per-family critique instructions.
