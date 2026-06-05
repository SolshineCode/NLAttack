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
- **`hermes/`** — Hermes Agent, `nvidia/nemotron-3-ultra:free` (Nous Portal),
  direct CLI. (C & F were re-run in 5-plan halves after the first pass hit the
  output-length limit / a timeout.)
- **`antigravity/`** — Antigravity CLI (`agy`), Gemini-3.5-Flash, via the ConPTY
  helper (`gemini-collab/scripts/agy_pty.py`) with a no-tools / answer-directly
  preamble (agy is agentic and otherwise tries to explore the workspace).

## Files
`hermes/<family>.md` and `antigravity/<family>.md` — one file per family A–J,
each covering its 10 plan IDs. **`SUMMARY.md`** — the consolidated synthesis
(convergent guidance, per-family verdicts from both, and disagreements).

The instruction prompts are `~/nla_review_instr.txt` (Hermes) and
`~/nla_review_instr_agy.txt` (agy, with the no-tools preamble).
