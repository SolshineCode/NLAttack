# NLAttack — Design Review & Validity Threats

Peer review of the 20-test harness + the "weak lossy concept filter" hypothesis.
Source: skeptical review by Hermes Agent (`nemotron-3-ultra:free`, Nous Portal,
session `20260604_161851_f72fb6`, 2026-06-04), synthesized with the harness
author's notes. This file is the standing issue list — work items reference it.

The headline: **the suite is well-built for *describing AV-description phenomena*;
it becomes a valid *NLA characterization* only once the verbalizer/matcher
confound is quantified and subtracted.** Everything below is about closing that gap.

---

## P0 — No ground truth at the bottleneck (the deepest problem)

We treat the AV description as a transparent readout of the bottleneck. It is not:
the verbalizer (AV) is a **second lossy model**. So three distinct failures all
collapse into one observed "dropped":

| What actually happened | We record |
|---|---|
| NLA bottleneck lost the concept | "dropped" |
| Bottleneck kept it, verbalizer couldn't lexicalize it | "dropped" |
| Bottleneck kept it, verbalizer emitted a benign neighbor | "laundered" |

We have **no independent access to the bottleneck representation**, so "concept
survival" currently measures *end-to-end AV+NLA+AR*, not the NLA's dropout map.
The "filter" framing only holds if verbalizer error is negligible or calibrated —
and Test 10 does not establish that.

**Fix (requires activation access):**
1. Train **linear probes on the AR's *input* activations** (the true bottleneck)
   for each test concept. Probe accuracy = ground-truth concept presence.
2. Gap between probe accuracy and AV-description-matcher accuracy = verbalizer+matcher noise.
3. **Verbalizer ablation:** swap AV for a constrained/forced-vocabulary decoder
   ("The concept is: [X]") and re-measure. If retention jumps, AV was the bottleneck.

**Feasibility:** the Neuronpedia API exposes *no* raw activations (only the
`description` + `cosine_similarity`/`mse` faithfulness scores). The probe path
needs a **local** NLA with AV/AR weights (the `KitftNLA` path). Published kitft
checkpoints (gemma3-12b/27b, llama-70b) exceed a 4GB GPU — this is a
bigger-GPU / smaller-NLA item. Tracked in work item E.

---

## P0 — Matcher topology ≠ NLA topology (the laundering confound)

Tests 2, 3, 6, 7, 13, 16 detect *specific neighbor concepts* in the AV text via
our matcher (embedding cosine / WordNet / lexical). But the matcher has its own
similarity geometry: "credential dumping → password management" may be an
**embedding artifact of our matcher**, not NLA laundering. Swap the matcher and
the "laundering" patterns shift — and we have no ground truth for which topology
is real.

**Fix (cheap, do now):**
- Run a **matcher ensemble** (≥3 embedding models + lexical [+ human on a sample]).
  Only claim a retained/substituted/laundered effect when **>2/3 agree**.
- Report per-matcher verdicts and an agreement score alongside every effect.
- Long-term: replace the matcher with probe ground-truth (P0 above) for these tests.

Tracked in work item C.

---

## P0 — Static eval misses adaptive exploitation

We measure which concepts drop under i.i.d. inputs. A real attacker against an
NLA-as-monitor is **adaptive**:
- **Probes** the monitor to map blind spots (adaptive queries).
- **Compositional attacks:** "run script" + "elevate privileges" + "access
  database" each survive *benignly*; the *combination* is the attack. None
  individually flags. (Test 16 checks co-occurrence survival, not adversarial
  *composition* of individually-benign concepts.)
- **Monitor-gaming:** optimizes against the *verbalizer output*, not the raw
  bottleneck — a feedback loop our static tests don't model.
- **Novel obfuscation:** real evasion ≠ leetspeak/spacing (Test 15's fixed set).

**Fix:** an **adaptive red-team loop** — give a red-team model query access to the
full pipeline (input → activation → AV description → retention decision) and have
it optimize inputs that maximize [malicious intent retained in activation space]
while minimizing [AV-description flags]. Report success rate vs query budget.
Add an explicit **compositional-blindspot** test (individually-benign concepts
that jointly imply an attack). Tracked in work item F.

---

## P1 — Secondary flags

| # | Issue | Why it matters | Fix | Item |
|---|---|---|---|---|
| 1 | **Frequency/length confound** in tests 1,3,4,14 | rare concepts drop more — risk-sensitivity or just rarity? | match on freq **and** token length for *every* test; report partial correlations | D |
| 2 | **Test 10 doesn't disentangle** the 3 failure modes | only sets an easy-concept baseline | needs probes (P0) + matcher calibration curves; downgrade Test 10's claim | A/E |
| 3 | **Test 18 assumes linear separability** of ARiES axes | Threat/Vuln/Impact may not be linearly separable in embedding space | use probe accuracy (linear **and** MLP), not cosine geometry | later |
| 4 | **Test 19 needs context** | "pentest" vs "attack" depends on surrounding concepts, not the word | test in full-sentence controlled-framing contexts | later |
| 5 | **AR fidelity unmeasured** | if AR faithfully reconstructs "dumping → stuffing", deception is in **AR**, not the NLA bottleneck | capture per-position `cosine_similarity`/`mse` already returned by Neuronpedia; weight retention by it | B |
| 6 | **No matcher inter-annotator reliability** | cosine thresholds are arbitrary; WordNet paths sparse for technical terms | human-annotate ~200 AV descriptions × 5 concepts; report Cohen's κ | later |

---

## Minimum controls before claiming ANY dropout/laundering effect is an NLA property

1. Bottleneck probe suite (linear probes on AR-input activations) — ground truth. *(E)*
2. Verbalizer swap (constrained decoder) — re-run 2/6/7/13/16; vanished effects were AV artifacts. *(E)*
3. Matcher ensemble — report across ≥3 matchers + lexical; claim only >2/3 agreement. *(C)*
4. Frequency/length-matched controls for **every** test. *(D)*
5. Adaptive red-team eval — even a 500-query budget reveals exploitable blind spots. *(F)*

---

## Work-item index (this session)

- **A** DESIGN_REVIEW.md (this file) + downgrade overclaims in README/test docstrings.
- **B** Faithfulness weighting from `cosine_similarity`/`mse` — *cheap, doable now (API already returns it).*
- **C** Matcher ensemble + agreement score — *cheap, doable now.*
- **D** Frequency+length matched control helper across all tests — *doable now.*
- **E** Local-NLA bottleneck probes (the P0 fix) — *needs local Gemma E2B NLA + GPU; gated on availability.*
- **F** Adaptive red-team loop + compositional-blindspot test — *scaffold now, full run later.*

Items B, C, D, F (scaffold) land this session. E is gated on a local NLA that
fits the 4GB GPU. Until E lands, **all dropout/laundering numbers are reported as
"AV+NLA+AR end-to-end, matcher-dependent," not as NLA properties** — stated
explicitly wherever results are emitted.
