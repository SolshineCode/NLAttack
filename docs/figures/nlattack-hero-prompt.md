Create a professional, publication-ready INFOGRAPHIC. Render ALL text crisply and spell every label exactly as written (this is a technical README hero image, so legibility is critical).

## Image specifications
- Type: Infographic (hero banner for an open-source ML benchmark README)
- Layout: bento-grid (modular cells of varied sizes; one wide hero cell on top, supporting cells below)
- Style: technical-schematic, BLUEPRINT variant — clean engineering blueprint look
- Aspect ratio: 2.35:1 (cinematic ultra-wide banner)
- Language: English

## Visual style (technical-schematic / blueprint, DARK)
- Dark deep-blue background (#0E1B2A to #1E3A5F) with a faint technical grid.
- White and light-cyan thin vector strokes, geometric precision, consistent stroke weights.
- Primary blue #2563EB, cyan callouts, amber #F59E0B highlights for the most important words.
- Clean sans-serif, mostly ALL-CAPS labels; small monospace for code. Subtle dimension-line / annotation accents. Ample whitespace, clear hierarchy. Modern, calm, trustworthy — NOT flashy neon.

## Title (top, prominent)
"NLAttack" (large) with subtitle "An evaluation suite for Natural Language Autoencoders".
Tiny tagline: "Can you read a model's thoughts in plain English — and trust what you read?"

## Cells / content
HERO CELL (wide, top): "WHAT IS A NATURAL LANGUAGE AUTOENCODER?" A left-to-right pipeline of three nodes joined by labeled arrows:
[ ACTIVATION ] --(AV: activation verbalizer)--> [ PLAIN-LANGUAGE TEXT ] --(AR: activation reconstructor)--> [ ACTIVATION' ]
- Node 1 icon: a small vector / number-grid. Node 2 icon: a speech bubble / quote. Node 3 icon: the same vector grid.
- Caption under node 2: "the human-readable bottleneck".

CELL "TWO QUESTIONS":
- 1. CAPABILITY — "Does the NLA work? Did the concept survive the bottleneck?"
- 2. SAFETY — "Can you read the text to catch LLM misuse?" (small note: grounded in the ATT&CK Navigator)

CELL "TWO ACCESS TIERS":
- API TIER — runs on ANY hosted NLA from text alone · the universal leaderboard
- FULL-ACCESS TIER — adds white-box probes that need raw activations · local NLAs only

CELL "BUILT FOR HONESTY":
- Works on WEAK / TINY / EARLY NLAs
- NULL CONTROLS on every result (beat a permutation floor)
- 118 EVALUATIONS in 13 FAMILIES
- small gloss line under it: "each evaluation = a test with a hypothesis, a metric, and a null-control baseline"

CELL "HEADLINE FINDING" (a tiny labeled bar contrast):
- Hosted NLAs (Gemma-3, Llama-3.3-70B): VERBALIZER-STRONG — concept retention 0.90–0.95
- Local Gemma-4-E2B v0.1: BOTTLENECK-STRONG but VERBALIZER-WEAK — probe 0.99 in-distribution, collapses out-of-domain

CELL "GET STARTED" (numbered, with a small code chip):
1. pip install -r requirements.txt
2. plug in any NLA (one-method adapter)
3. run(nla, dataset, matcher)
small note: no API key needed for hosted NLAs

## Footer (thin strip)
"Apache-2.0  ·  github.com/SolshineCode/NLAttack"

## Rules
- Keep on-image text concise and exactly as written above; do not invent extra words or numbers.
- Emphasize the keywords: AV, AR, BOTTLENECK, CAPABILITY, SAFETY, VERBALIZER-STRONG, BOTTLENECK-STRONG.
- Faithful, legible labels over decoration.
