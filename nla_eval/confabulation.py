"""Confabulation scorers — the #1 stated limitation of NLAs (Anthropic NLA paper,
surfaced by the open Hermes review 2026-06-08): AV verbalizations are often
"thematically faithful but specifically wrong" — confident specific claims the
context does not support. Concept-survival checks *presence*, not *truth*, so it
misses this. For the misuse / ATT&CK monitor use-case this is critical: a
confabulating or over-specific verbalizer makes the monitor untrustworthy (it may
invent attack details, or miss real ones while sounding faithful).

These operate on `(verbalization, source_text)` pairs (CPU, dependency-free):
  factual_grounding  — of the SPECIFIC claims the AV makes (named entities,
                       numbers/quantities), what fraction are NOT supported by the
                       source? = confabulation rate.
  thematic_fidelity  — content-word overlap of the verbalization with the source
                       (the "theme" the AV got right). Reported alongside grounding
                       so you see the thematic-vs-specific split the paper warns of.
  consistency        — across several verbalizations of the SAME item, do the
                       specific claims agree? (contradiction = confabulation tell).

Running these needs AV verbalizations (an AV → GPU or hosted server); the scoring
is local. Caveat: "specific claim" extraction is heuristic (capitalized tokens +
numerics), an upper bound on detectable confabulation, not a semantic verifier.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence, List, Dict
import re

from .matching import content_words

_NUM = re.compile(r"\b\d[\d,\.]*\b")
_PROPER = re.compile(r"\b([A-Z][a-zA-Z][a-zA-Z'\-]+)\b")  # capitalized mid-text proper nouns


@dataclass
class ConfabResult:
    n: int
    confabulation_rate: float    # specific claims unsupported by source / all specific claims
    mean_thematic_fidelity: float
    specificity: float           # specific claims per verbalization (normalized)
    detail: dict
    def flat(self): return {k: v for k, v in asdict(self).items() if k != "detail"}


def _specific_claims(text: str) -> List[str]:
    """Heuristic 'specific claims' an AV could confabulate: numbers + proper nouns
    (excluding sentence-initial words, which are capitalized by grammar not
    specificity)."""
    claims = set(m.group(0) for m in _NUM.finditer(text or ""))
    for sent in re.split(r"[.!?]\s+", text or ""):
        toks = sent.split()
        for w in toks[1:]:                       # skip sentence-initial cap
            m = _PROPER.match(w)
            if m:
                claims.add(m.group(1))
    return [c for c in claims if len(c) > 1]


def _supported(claim: str, source: str) -> bool:
    return claim.lower() in (source or "").lower()


def factual_grounding(pairs: Sequence) -> ConfabResult:
    """pairs: list of (verbalization, source_text). confabulation_rate = fraction
    of the AV's SPECIFIC claims (entities/numbers) absent from the source."""
    n = 0; total_claims = 0; unsupported = 0; them = []; spec = []
    for verb, src in pairs:
        claims = _specific_claims(verb)
        u = sum(0 if _supported(c, src) else 1 for c in claims)
        total_claims += len(claims); unsupported += u
        spec.append(len(claims))
        cw = content_words(verb); scw = set(content_words(src))
        them.append(sum(1 for w in cw if w in scw) / len(cw) if cw else 0.0)
        n += 1
    import numpy as np
    rate = (unsupported / total_claims) if total_claims else float("nan")
    return ConfabResult(
        n=n, confabulation_rate=rate,
        mean_thematic_fidelity=float(np.mean(them)) if them else float("nan"),
        specificity=float(np.mean(spec)) if spec else 0.0,
        detail={"total_specific_claims": total_claims, "unsupported_claims": unsupported,
                "note": "heuristic specific-claim extraction (entities+numbers); "
                        "upper bound on detectable confabulation"})


def consistency(verbalization_groups: Sequence[Sequence[str]]) -> Dict:
    """Each group = several verbalizations of the SAME item. Measures whether the
    SPECIFIC claims agree across them; low agreement = the AV is making things up.
    Returns mean cross-sample claim Jaccard (1.0 = perfectly consistent)."""
    import numpy as np
    jac = []
    for group in verbalization_groups:
        sets = [set(_specific_claims(v)) for v in group if v]
        if len(sets) < 2:
            continue
        pair_j = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = len(sets[i] | sets[j])
                pair_j.append((len(sets[i] & sets[j]) / u) if u else 1.0)
        if pair_j:
            jac.append(float(np.mean(pair_j)))
    return {"mean_claim_consistency": float(np.mean(jac)) if jac else float("nan"),
            "n_groups": len(jac)}
