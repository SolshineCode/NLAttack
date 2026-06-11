# Consolidated review — evaluating rudimentary NLAs

Synthesis of two independent reviewers' critiques of all 100 plans, focused on **smaller/weaker NLAs** than the plans
assume. Full per-plan detail in `reviewer-a/` and `reviewer-b/`.

## Convergent guidance (both reviewers, repeatedly)

1. **Go probe-only / activation-space.** On a rudimentary NLA the AV is degenerate
   (empty/repetitive) and the AR near-random, so any *end-to-end* metric collapses:
   AV emits nothing → matcher always 0 → retention/laundering/gap metrics become
   degenerate or undefined. Both reviewers' floor versions almost always reduce to:
   **bypass the verbalizer and matcher; measure the bottleneck activation directly**
   (linear probe, cosine, centroid distance, shuffled-activation control).
2. **Floor effects, not findings.** When everything floors, "0% survival
   everywhere" looks like a uniform blind spot but is just capacity failure.
   Controls must distinguish *representation absent* from *interface (AV) broken* —
   the recurring confound both flag.
3. **Coarsen metrics.** Continuous fits (frequency slopes, monotonic ladders)
   need variance that weak NLAs don't provide → replace with binary / rank /
   pooled measures; relax matchers to character-level / edit-distance / numeric
   extractors; pool across many items.
4. **First question is existential:** *does a usable bottleneck exist at all, and
   is the AR even conditioned on the input?* Several "highest-value" picks are
   really sanity checks for that (P077, P081).

## Run FIRST on a rudimentary NLA — both reviewers' highest-value overlap

| Plan | Why (both agree) |
|---|---|
| **P077** AR identity / cross-row leakage | the most fundamental check — is the AR text-conditional at all? works at any fidelity, cheap N² cosines |
| **P081** in-dist vs OOD probe-AUC gap | pure probe; tests whether *any* encodable bottleneck exists; simplifiable to centroid distance |
| **P046** verbalizer ablation (forced/template decoder) | isolates "AV is broken" from "bottleneck is empty" — the central rudimentary-NLA confound |
| **P050** better concept labels (LLM not keyword) | unblocks every probe (lifts n from 2–3 to ≥10/corpus); keyword matching is fatal when AV paraphrases |
| **P004** BOS/position-0 (probe-only) | BOS attention sink is structural, present even in tiny models |
| **P023** reported-vs-asserted (coarsened) | lexical status markers partly survive; retrieval-kNN + coarse matcher gives signal |
| **P092 / P095** abstention vs confabulation; hedge-vs-faithfulness | evaluable via simple entropy/length/repetition proxies, no deep probing |
| **P067** ordering/dilution (probe-only grid) | tests position/compression invariance in activation space; no adaptive search needed |

(Near-misses each reviewer ranked highly: reviewer A also P100, P073, P080, P089, P086;
reviewer B also P009, P044, P057, P060, P084, P032.)

## SKIP until the NLA is stronger — both reviewers' "skip" overlap

`P002` (category dropout — all at floor), `P016` (metaphor/literal — probe
confound), `P019` (temporal order), `P020` (spatial/relational — explicit null),
`P021` (false-belief), `P027` (sandbagging — deliberateness beyond a bottleneck),
`P042` (MLP probe — overfits at tiny n), `P055` (paraphrase invariance — needs a
working AV), `P065` (steganographic capacity), `P074` (round-trip stability —
degeneracy mimics convergence), `P079` (AR paraphrase delta), `P096` (uncertainty-
direction stability), `P097` (refusal encoding). Theme: anything requiring
relational binding, multi-stage structure, metacognition, or a functioning
generative AV is expected-null on a rudimentary NLA.

## Where the reviewers DISAGREE (decide per your NLA)

- **P089 (layer × distribution):** reviewer A ranks it **highest-value** (pure probe
  on existing multi-layer dumps); reviewer B says **skip** (high-variance AUC on tiny
  models won't converge). → Run it *only* if you have several layers dumped and
  enough n; otherwise defer.
- **Family F (matcher/verbalizer confound) wholesale:** reviewer A says **none of
  P051–P055 as written** survive a degenerate AV and would **replace the whole
  family** with probe-only matcher-noise-floor ablations on fixed human text; reviewer B
  keeps **P051/P057/P060** as usable baselines. → On a truly rudimentary AV,
  follow reviewer A (characterize the matcher on human text, not on AV output).
- **Family D top picks:** reviewer B skips P033/P036 and elevates P032/P040/P031; reviewer A
  skips P031/P032 (matcher hallucination on generic text) and elevates P034/P037
  (minimal-pair + probe ground truth). → The probe-grounded ones (P034, P037,
  P040) are the safer bets; pure coverage/obfuscation heatmaps are noise-prone.

## One-line takeaway
For a rudimentary NLA, **stop measuring the AV text**: run the probe-only /
activation-space floor versions, start with the "does a bottleneck even exist /
is the AR conditional" checks (P077, P081, P046, P050), and shelve every
relational / metacognitive / generative-AV-dependent plan until the NLA matures.
