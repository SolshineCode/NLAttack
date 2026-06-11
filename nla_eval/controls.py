"""Frequency + token-length matched controls.

independent review (P1#1): rare concepts drop more — is a low retention score
risk-sensitivity, or just rarity? Every test that compares a concept class to a
baseline needs controls matched on BOTH frequency band and token length, not just
tests 1/3/14. This module builds neutral control concepts whose (freq_band, len)
distribution mirrors the test concepts, tagged so the existing group-bys treat
them as the `is_control` baseline.

Frequency is estimated from a compact built-in tier list (no dependency). For
serious runs, pass your own `freq_fn` backed by `wordfreq` or a corpus count.
"""
from __future__ import annotations

from typing import List, Callable, Optional, Dict
from .matching import content_words

# Compact high-frequency English set (top ~120 content words) → "high" band.
# Anything not here and short-ish → "mid"; long/rare → "low". Coarse but honest,
# and overridable via freq_fn.
_HIGH = set("""
time year people way day man thing woman life child world school state family
student group country problem hand part place case week company system program
question work government number night point home water room mother area money
story fact month lot right study book eye job word business issue side kind head
house service friend father power hour game line end member law car city community
name president team minute idea body information back parent face others level
office door health person art war history party result change morning reason
research girl guy moment air teacher force education food water road work help
""".split())

# obvious neutral nouns/verbs for the control pool (domain-agnostic, benign)
DEFAULT_CONTROL_POOL = [
    "table", "garden", "weather", "music", "river", "bicycle", "coffee",
    "mountain", "letter", "window", "blanket", "harvest", "lantern", "orchard",
    "calendar", "umbrella", "notebook", "pottery", "trolley", "meadow",
    "saxophone", "cinnamon", "wheelbarrow", "marmalade", "telescope",
    "photosynthesis", "constellation", "embroidery", "cartography", "kaleidoscope",
]


def estimate_freq_band(word: str, freq_fn: Optional[Callable[[str], float]] = None) -> str:
    if freq_fn is not None:
        f = freq_fn(word)
        return "high" if f >= 1e-4 else "mid" if f >= 1e-6 else "low"
    w = word.lower()
    if w in _HIGH:
        return "high"
    return "mid" if len(w) <= 8 else "low"


def concept_len(concept: str) -> int:
    """Token count of the concept's content words (proxy for NLA token span)."""
    return max(1, len(content_words(concept)))


def char_band(concept: str) -> str:
    n = len(concept.replace(" ", ""))
    return "short" if n <= 6 else "mid" if n <= 12 else "long"


def profile(concept: str, freq_fn=None) -> Dict[str, str]:
    """The (freq_band, token_count, char_band) signature we match controls on."""
    head = content_words(concept)
    head = head[-1] if head else concept
    return {
        "freq_band": estimate_freq_band(head, freq_fn),
        "n_tokens": concept_len(concept),
        "char_band": char_band(concept),
    }


def build_matched_controls(
    concepts: List[str],
    pool: Optional[List[str]] = None,
    freq_fn=None,
) -> List[Dict]:
    """For each test concept, pick a neutral control from `pool` matched on
    freq_band + token-count + char_band (relaxing char_band, then n_tokens, if no
    exact match). Returns dicts {concept, control, profile} — use the control as a
    neutral Example with meta.is_control=True so t03's control_baseline and the
    dropout law (t01) compare like with like."""
    pool = list(pool or DEFAULT_CONTROL_POOL)
    used = set()
    out = []
    for c in concepts:
        cp = profile(c, freq_fn)
        # candidate scoring: exact freq_band required; then closeness on tokens/chars
        best, best_score = None, -1
        for w in pool:
            if w in used:
                continue
            wp = profile(w, freq_fn)
            score = 0
            score += 2 if wp["freq_band"] == cp["freq_band"] else 0
            score += 1 if wp["n_tokens"] == cp["n_tokens"] else 0
            score += 1 if wp["char_band"] == cp["char_band"] else 0
            if score > best_score:
                best, best_score = w, score
        if best is not None:
            used.add(best)
        out.append({"concept": c, "control": best, "profile": cp})
    return out


def control_examples(concepts, template="The note simply mentions {}.",
                     pool=None, freq_fn=None, id_prefix="ctrl"):
    """Convenience: turn matched controls into tagged Examples ready for run()."""
    from .core import Example

    pairs = build_matched_controls(concepts, pool, freq_fn)
    exs = []
    for i, p in enumerate(pairs):
        if not p["control"]:
            continue
        exs.append(Example(
            id=f"{id_prefix}-{i}",
            text=template.format(p["control"]),
            concepts=[p["control"]],
            meta={"is_control": True, "freq_band": p["profile"]["freq_band"],
                  "n_tokens": p["profile"]["n_tokens"],
                  "matched_to": p["concept"]},
        ))
    return exs
