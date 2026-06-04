# Peer review request: NLAttack — evaluation tests for Natural Language Autoencoders

You are reviewing the experimental design of a research harness. Be a skeptical
peer reviewer. We want disagreement and holes found, not encouragement.

## The running hypothesis

1. Current Natural Language Autoencoders (NLAs) are **weak**. They cannot reliably
   represent negation, word order, attribution (who-said-what), or epistemic state.
2. Therefore the only measurement that yields signal is a **floor primitive**:
   *did concept C (or a near-neighbor) survive the bottleneck? yes/no.*
3. An NLA is modeled as a **lossy concept filter**. We characterize that filter's
   dropout map and ask whether its shape matters for safety/deception.
4. Architecture reality: a real NLA is `activation -> verbalizer(AV) -> natural
   language description -> reconstructor(AR) -> activation`. NOT text->text. The
   human-readable bottleneck is the AV description; faithfulness is judged in
   activation space. We read concept survival off the AV description.
5. Deception angle (knowledge-asymmetry exploitation): the interesting failure is
   *which* content the filter silently drops or launders into a benign neighbor —
   if safety-relevant content drops while its benign neighbors survive, the
   bottleneck creates an exploitable blind spot. The misuse suite is grounded in
   Anthropic's LLM ATT&CK Navigator (14 tactics, ARiES Threat/Vuln/Impact scoring,
   "agentic scaffolding" as the top risk marker).

## The 20 tests (each a group-by over one per-concept retention table)

General adjacency (1-10): 1 dropout law (retention vs frequency) · 2 adjacency
substitution (dropped concept returns as a neighbor) · 3 category-selective dropout
vs frequency-matched controls · 4 specificity collapse (hypernym drift) · 5 salience
crowding · 6 paired-neighbor asymmetry · 7 co-occurrence hallucination (inserted
concepts) · 8 quantity flattening · 9 blind-spot stability (variance across contexts)
· 10 below-floor calibration control.

ATT&CK misuse (11-20): 11 technique-coverage map across tactics · 12 obfuscation
see-through (retention obfuscated/plain — does the NLA strip obfuscation or get
fooled?) · 13 attack->benign laundering (e.g. "credential dumping" -> "password
management") · 14 risk-weighted dropout (rare high-risk post-compromise terms vs
common pre-compromise) · 15 evasion robustness (leetspeak/spacing perturbations) ·
16 kill-chain co-occurrence (do multiple stages survive jointly) · 17 agentic-
scaffolding markers · 18 ARiES axis separability · 19 dual-use disambiguation ·
20 severity monotonicity.

Concept presence is decided by embedding cosine / wordnet / lexical matching against
the AV description. Verified live on Neuronpedia's Gemma NLA (gemma-3-27b-it,
layer 41): "credential dumping" already verbalizes toward "credential stuffing /
password theft / brute-force" — live laundering.

## What we want from you

1. Is the core hypothesis (NLAs as weak lossy concept filters; concept-survival as
   the only floor-valid measurement) sound, or is it throwing away signal that even
   weak NLAs carry? What measurement are we missing?
2. Which of the 20 tests are actually measuring what they claim ON WEAK NLAs, and
   which are confounded, circular, or below the noise floor? Be specific.
3. The biggest validity threat we know of: a low score can mean "NLA dropped the
   concept" OR "the matcher failed" OR "AV decoder couldn't phrase it." How would
   you disentangle these three? Is test 10 (below-floor control) enough?
4. For the deception/misuse angle specifically: are we testing the right thing?
   What would an attacker actually exploit about an NLA-as-monitor that these tests
   would miss?
5. Anything else a reviewer should flag. Propose concrete additional tests or
   controls if you have them.

Return a structured critique. Prioritize the 2-3 most important problems.
