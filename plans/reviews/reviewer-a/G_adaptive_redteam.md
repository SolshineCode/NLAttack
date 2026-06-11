---

# Per-Plan Critique (P061–P070) for Rudimentary NLAs

### P061 — Query-budget evasion success curve
**SIGNAL:** Floors to noise. Requires both `probe_predict` (ground truth for intent) and `verbalize` (overseer view). On rudimentary NLA: probe is untrained → `intent_retained` is random; verbalize is empty/degenerate → `EnsembleMatcher` always returns 0. The adaptive loop receives a constant (flagged=0) or noise (random intent) reward signal. Budget curve saturates immediately at chance level — measures nothing.
**FAILURE MODE:** `default_mutations` on empty AV text produces identical empty output → no gradient for adaptation. "Success" = random probe noise passing floor + matcher always 0. Success-rate@500 ≈ probe-noise-floor, indistinguishable from null.
**FLOOR VERSION:** Probe-only mode — train a linear probe on frozen activations (feasible today) as ground truth; skip AV entirely. Compare evasion search against a keyword matcher on *input text* (not AV) to establish a baseline. Report success vs shuffled-activation control.
**MISSED FLAW:** Assumes `probe_predict` is a reliable ground truth. On rudimentary NLA with near-random AR (cosine 0.1–0.4), the "AR-input activation" may bear little relation to the original concept. The probe measures reconstruction artifacts, not input intent. Also assumes mutation space has meaningful variation — false when AV is constant.

### P062 — Compositional blind spots
**SIGNAL:** Partial but confounded. `compositional_blindspot` runs components through NLA and checks if attack composite is flagged. Retrieval verbalizer: "exploited" triggers on cache misses (composite not in cache), not true compositionality failure. Degenerate verbalizer: all components → empty → matcher says absent → exploited-rate=0 (floor). Probe cross-check needs working probe.
**FAILURE MODE:** Template formatting breaks on empty/repetitive AV output. "Benign components survive" measured by matcher on AV — if AV is retrieval, survival = cache hit; if degenerate, survival = 0. Probe-vs-AV gap = noise vs noise.
**FLOOR VERSION:** Probe-only — test if bottleneck probe activates for composite when components are present (linear probe on frozen activations). For retrieval verbalizer, add cache-aware baseline: explicitly log cache hits/misses per component vs composite. Coarsen metric to present/absent per component; pool across many triples.
**MISSED FLAW:** Conflates *genuine compression loss* (bottleneck doesn't compose) with *adversarial blind spot* (attacker exploits monitor gap). Null "probe shows composite never enters activation" could mean the NLA simply doesn't compose concepts — not an exploitable blind spot. The triples' "genuinely implied" label assumes compositionality the NLA may not have.

### P063 — Evasion transfer across NLAs / layers
**SIGNAL:** Pure noise. Needs ≥2 working probe+AV pipelines (two NLAs or two layer-probes). Rudimentary regime has neither. Cross-layer within same tiny model: all layers have near-random AR → probes on all layers untrained/noisy. "Transfer" = correlation of noise.
**FAILURE MODE:** "Succeeded on A" uses A's probe as ground truth — if A's probe is random, evasion set is random. Replay on B with B's probe: random inputs × random probe = chance. Random-benign baseline also chance → transfer ≈ chance always, indistinguishable from null.
**FLOOR VERSION:** Single-NLA layer ablation only — train linear probes on *same labeled data* for layer L1 vs L2 (controlled comparison). Metric: rank correlation of evasion success across layers, not absolute transfer rate. Or skip entirely until NLA stronger.
**MISSED FLAW:** Assumes "evasions found against NLA-A" are real. On rudimentary NLA, they're inputs that fooled a broken probe/matcher. Transfer of nonsense = nonsense. Cross-layer "transfer" may just reflect all layers being equally bad, not shared blind spots.

### P064 — Monitor-gaming feedback loop
**SIGNAL:** Degenerate. Feedback `mutate` reads AV text/flag to steer. Empty/repetitive AV → no signal to read (constant text, flag=0 or random). Mutate has no gradient → reduces to random walk = blind baseline. Probe-intent trajectory needs working probe.
**FAILURE MODE:** AV empty → no tokens to "remove/soften". Matcher always 0 → flag-suppression trivial (already 0). Probe untrained → "intent held constant" unverifiable. Feedback vs blind comparison confounded: both fail for same reason (no signal), not because feedback provides no advantage.
**FLOOR VERSION:** Synthetic feedback loop — use a known-working matcher (e.g., keyword-based on *input text*, not AV) as overseer proxy. Or test on retrieval verbalizer where cache hits/misses provide discrete feedback. Metric: feedback vs blind on simplified task where feedback signal exists.
**MISSED FLAW:** Assumes AV verbalization contains usable signal for attacker to optimize against. On rudimentary NLA, AV output may be constant — there's no "feedback" to exploit. Also: if probe is ground truth for intent but probe is untrained, the "intent held constant" check is meaningless.

### P065 — Steganographic channel capacity
**SIGNAL:** Zero. Needs encoder/decoder training through AV loop. Degenerate AV: constant output → no surface choices → capacity=0 (true null for wrong reason). Retrieval AV: encoder picks from fixed cache → capacity=log₂(cache_size), but this is *cache exploitation*, not bottleneck stego. Near-random AR: decoder reads noise → BER≈0.5.
**FAILURE MODE:** "Surface choices via default_mutations" assumes generative AV with token-level control — false for retrieval/degenerate. Decoder recovery from AV text: empty/repetitive → no bits recoverable. Probe-side ablation needs working probe (E). Overseer flag on empty text → flag-rate=0 trivially.
**FLOOR VERSION:** Cache-channel capacity for retrieval verbalizer: measure distinct cache entries encoder can selectively retrieve (real stego channel for this NLA type). Probe-only: skip AV, measure mutual information between encoder-chosen inputs and bottleneck activations (linear probe).
**MISSED FLAW:** Conflates "bottleneck stego" with "AV surface stego". On rudimentary NLAs, AV may pass surface verbatim (retrieval) or destroy it (degenerate). Measured capacity reflects AV behavior, not bottleneck properties. Probe-side ablation is the right control but needs working probe.

### P066 — Adaptive paraphrase across retention boundary
**SIGNAL:** Partial, probe-dependent. Paraphrase mutation + intent-band filter on probe. If probe works (trained linear probe on frozen activations): CAN get signal — paraphrase inputs, filter by probe-intent constancy, check matcher flip. If probe untrained: filter is noise. Degenerate verbalize: matcher always 0 → no flag-flips (floor). Retrieval: paraphrase may not change cache key.
**FAILURE MODE:** Paraphrase MutateFn on retrieval → same cache hit → same output → no flag flip. Probe untrained → |Δprobe|<δ filter retains random subset. δ threshold: noisy probe → Δprobe dominated by noise → keeps everything (δ large) or nothing (δ small). "Constant-intent flag-flip region" size = probe noise variance.
**FLOOR VERSION:** Use trained linear probe (frozen backbone) as intent ground truth — feasible on rudimentary NLA. Skip AV: measure if bottleneck activation changes under paraphrase (probe on AR-input activation). Coarsen to present/absent flag flip. Pool across many paraphrase pairs per concept.
**MISSED FLAW:** Assumes paraphrase mutations produce meaningfully different AV outputs. On retrieval, paraphrases map to same cache entry; on degenerate, all map to empty. "Flag flip" may not occur because AV is insensitive, not because matcher is robust. Also: probe measures "intent" on AR-input activation — but near-random AR means AR-input activation may not correspond to input concept at all.

### P067 — Adversarial concept ordering / dilution
**SIGNAL:** Usable (probe-dependent). Fixed grid sweep: position × dilution variants, measure flag-rate (AV+matcher) and probe-intent. Doesn't need adaptive search — measurable even with noisy components. If probe works: probe-intent per variant gives signal (does bottleneck preserve concept regardless of position/fillers?).
**FAILURE MODE:** Probe untrained → probe-intent slope noise, can't verify "stays flat". Verbalize degenerate → flag-rate=0 flat → null for wrong reason (AV broken). Fillers from benign_neighbors: on tiny NLA, "benign" concepts may not exist in bottleneck. Position effect confounded with dilution (last + 8 fillers = different context length).
**FLOOR VERSION:** Probe-only version — skip AV/matcher entirely, measure probe-intent across position/dilution grid. Directly tests Family A position/compression effects in activation space. Metric: probe-intent variance across grid (low = position-invariant). Coarsen to rank correlation vs position.
**MISSED FLAW:** Assumes "hold concept token fixed" — but low-dim bottleneck + few examples may drop concept entirely when diluted. Probe-intent drop = genuine compression loss, not matcher/verbalizer gating. Null "flag-rate flat" could mean: (a) matcher position-invariant, (b) AV too broken to vary, (c) concept never survives bottleneck. Not distinguished.

### P068 — Prompt-injection surviving bottleneck
**SIGNAL:** Low. Needs: (a) AV for reconstruction, (b) downstream parser/probe for "recoverable", (c) EnsembleMatcher for injection flag. Degenerate AV → reconstruction empty → survive-rate=0 (floor). Retrieval AV → reconstruction = cached snippet → "recoverable" only if cache has injection-like text. Downstream parser on garbage → noise. Matcher on empty → unflagged=1 always.
**FAILURE MODE:** "Laundered into benign phrasing" assumes generative AV — retrieval/degenerate can't. adaptive_evade optimizes against broken pipeline → random walk. Benign-instruction control: same issues. "Recoverable" by parser on AV output — if AV garbage, parser noise dominates.
**FLOOR VERSION:** Probe-only: does bottleneck activation contain injection-like pattern? (linear probe for "instruction" concept). Skip AV/reconstruction: test if input injection survives to AR-input activation (bottleneck probe). Synthetic injection: known token sequences, measure probe activation.
**MISSED FLAW:** Assumes injection survives bottleneck AND re-emerges in AV text. Rudimentary bottleneck may not preserve sequential/instructional structure at all. "Survive-but-unflagged" gap conflates: (a) bottleneck drops injection, (b) AV fails to reconstruct, (c) matcher fails to flag. Also: downstream reader modeled by parser on *text* — real threat is downstream agent acting on *reconstructed activation*, not text.

### P069 — Encoder/decoder collusion
**SIGNAL:** Zero. Most demanding plan: trained encoder/decoder, shared codebook, AV loop, probe verification. Degenerate AV: no surface variation → no channel. Retrieval AV: encoder selects cache keys → collusion = cache exploitation. Overseer on empty text → flag-rate=0 → advantage = decoder recovery (0). Non-colluding baseline also 0.
**FAILURE MODE:** "Benign-surface choices" assumes generative AV with controllable output — false. Retrieval: codebook = cache subset. Degenerate: no surface variation. Probe-side ablation needs working probe. adaptive_evade hardening needs functional overseer.
**FLOOR VERSION:** Cache-collusion for retrieval verbalizer: encoder/decoder agree on cache-key mapping; overseer = matcher on cache entries. Probe-only collusion: encoder chooses inputs, decoder reads bottleneck activation directly (linear probe), overseer reads AV — isolates bottleneck channel from AV channel.
**MISSED FLAW:** Assumes overseer sees only AV text, colluders use bottleneck. Rudimentary bottleneck may not reliably transmit ANY signal (near-random AR). "Recovery collapses when activation perturbed" needs working probe. "Codebook complexity" sweep assumes continuous control — but low-dim + few examples = discrete, noisy transmission.

### P070 — Robustness to unseen obfuscation families
**SIGNAL:** Usable (with caveats). Tests cross-family generalization of evasions. Multiple MutateFn families run today vs MockNLA. For real NLA: needs probe+AV for "success". If probe works (trained linear probe): can measure if family-1 evasion transfers to family-2. Degenerate AV: all families → empty → matcher 0 → transfer=100% (false positive). Retrieval: evasions changing cache key may not transfer.
**FAILURE MODE:** Degenerate AV → all inputs → empty → matcher 0 → success always → transfer=100% (meaningless). Probe untrained → success judgment noise. Random-edit family baseline: on degenerate AV, random edits also succeed → baseline inflated. Budget-to-recover: broken pipeline → "recovery" instant or never.
**FLOOR VERSION:** Probe-only: use trained linear probe as success criterion, test mutator families on *input text* (bypass AV). Measures: does bottleneck probe generalization match mutator family? Coarsen: success = probe-intent ≥ floor (binary). Fixed grid of obfuscations per family instead of adaptive search.
**MISSED FLAW:** Assumes adaptive_evade finds REAL evasions on family-1. Rudimentary NLA → "evasions" = artifacts of broken probe/matcher. Transfer of artifacts ≠ meaningful generalization. Cross-family matrix measures AV/matcher artifacts, not true generalization. "Unseen family" generalization only meaningful if monitor has family-specific weaknesses — on rudimentary NLA, matcher uniformly weak/strong for trivial reasons.

---

## Skip on Rudimentary NLAs
*Pure noise / not worth running until NLA is stronger (working probe + non-degenerate AV)*

- **P063** — Evasion transfer (needs 2 working pipelines)
- **P065** — Steganographic capacity (needs generative AV + encoder/decoder training)
- **P069** — Encoder/decoder collusion (most demanding; needs codebook + probe + AV loop)

---

## Highest-Value for the Rudimentary Regime
*Run these first on a weak/tiny NLA — they yield signal with probe-only floors*

1. **P067** — Ordering/dilution grid (probe-only version): Directly tests position/compression invariance in *activation space*; fixed grid needs no adaptive search; works with trained linear probe on frozen backbone.
2. **P066** — Paraphrase retention boundary (probe-only version): Isolates verbalizer/matcher gating from true bottleneck loss; paraphrase mutations are cheap; intent-band filter works with linear probe.
3. **P070** — Obfuscation generalization (probe-only version): Tests whether bottleneck probe robustness generalizes across perturbation families; fixed grid per family; no adaptive search needed.
4. **P062** — Compositional blind spots (probe-only + cache-aware): Tests if bottleneck composes concepts; retrieval verbalizer gets cache-hit baseline; probe-only version feasible today.

*Common thread:* All four become **probe-only** evaluations using a **trained linear probe on frozen activations** (feasible on rudimentary NLA today) and **skip the AV/verbalize entirely**. The AV is the weak link in the rudimentary regime — it adds noise, not signal.
session_id: 20260605_123026_515550

