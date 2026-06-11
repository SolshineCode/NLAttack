"""Core: turn (NLA, dataset) into the per-concept retention table.

Every one of the 20 tests is a group-by over this table. An Example carries
the input text, the controlled concept list, and arbitrary `meta` tags
(frequency band, category, ATT&CK tactic, obfuscation flag, perturbation level,
risk dimension, ...). The tests slice on those tags.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import csv

from .adapters import NLA
from .matching import Matcher, content_words


@dataclass
class Example:
    id: str
    text: str
    concepts: List[str]                      # controlled concepts present in input
    meta: Dict[str, Any] = field(default_factory=dict)
    # optional partner id for paired tests (obfuscated/plain, perturbed, etc.)
    pair_id: Optional[str] = None


@dataclass
class ConceptRow:
    example_id: str
    nla: str
    concept: str
    status: str            # retained | substituted | dropped
    matched_term: Optional[str]
    similarity: float
    mode: str
    cos_sim: Optional[float] = None   # AR activation-space faithfulness (Neuronpedia)
    mse: Optional[float] = None       # AR reconstruction error
    fallback_full: bool = False       # concept token not locatable -> matched whole text
    agreement: Optional[float] = None # ensemble matcher agreement fraction (if used)
    meta: Dict[str, Any] = field(default_factory=dict)

    def flat(self) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "meta"}
        d.update({f"meta.{k}": v for k, v in self.meta.items()})
        return d


@dataclass
class InsertionRow:
    """Concepts that appear in the output but were not in the input (hallucination)."""
    example_id: str
    nla: str
    inserted_term: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    nla: str
    rows: List[ConceptRow]
    insertions: List[InsertionRow]
    reconstructions: Dict[str, str]  # example_id -> output text

    def to_csv(self, path: str):
        flat = [r.flat() for r in self.rows]
        keys = sorted({k for r in flat for k in r})
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)


def run(
    nla: NLA,
    dataset: List[Example],
    matcher: Optional[Matcher] = None,
    detect_insertions: bool = True,
    concept_targeted: bool = True,
) -> RunResult:
    """Run a dataset through an NLA into the retention table.

    If the adapter exposes `verbalize_concepts(text, concepts)` and
    `concept_targeted` is on, each concept is matched against the verbalization
    of ITS OWN token positions (the faithful floor test: does the concept's
    activation verbalize as the concept, or launder into a neighbor?). The
    whole-example text (`__full__`) is still used for insertion detection.
    Otherwise we fall back to one reconstruct() per example.
    """
    matcher = matcher or Matcher()
    use_targeted = concept_targeted and hasattr(nla, "verbalize_concepts")

    rows: List[ConceptRow] = []
    insertions: List[InsertionRow] = []
    recon: Dict[str, str] = {}

    def _text_of(entry, default=""):
        return entry.get("text", default) if isinstance(entry, dict) else (entry or default)

    for ex in dataset:
        if use_targeted:
            per = nla.verbalize_concepts(ex.text, ex.concepts)
            full = _text_of(per.get("__full__"))
        else:
            full = nla.reconstruct(ex.text)
            per = {c: full for c in ex.concepts}
        recon[ex.id] = full

        for c in ex.concepts:
            entry = per.get(c, full)
            text = _text_of(entry, full)
            cos = entry.get("cos") if isinstance(entry, dict) else None
            mse = entry.get("mse") if isinstance(entry, dict) else None
            fb = entry.get("fallback_full", False) if isinstance(entry, dict) else False
            m = matcher.match(c, text)
            rows.append(
                ConceptRow(
                    example_id=ex.id,
                    nla=nla.name,
                    concept=c,
                    status=m.status,
                    matched_term=m.matched_term,
                    similarity=m.similarity,
                    mode=m.mode,
                    cos_sim=cos,
                    mse=mse,
                    fallback_full=fb,
                    agreement=getattr(m, "agreement", None),
                    meta=ex.meta,
                )
            )
        if detect_insertions:
            input_concept_words = {w for c in ex.concepts for w in content_words(c)}
            for w in set(content_words(full)) - input_concept_words:
                insertions.append(InsertionRow(ex.id, nla.name, w, ex.meta))

    return RunResult(nla.name, rows, insertions, recon)


# ---- shared aggregation helpers used by the test suites -------------------

def retention_rate(rows: List[ConceptRow], predicate=lambda r: True) -> float:
    sel = [r for r in rows if predicate(r)]
    if not sel:
        return float("nan")
    kept = sum(1 for r in sel if r.status in ("retained", "substituted"))
    return kept / len(sel)


def substitution_rate(rows: List[ConceptRow], predicate=lambda r: True) -> float:
    sel = [r for r in rows if predicate(r)]
    if not sel:
        return float("nan")
    return sum(1 for r in sel if r.status == "substituted") / len(sel)


def group_retention(rows: List[ConceptRow], key) -> Dict[Any, float]:
    groups: Dict[Any, List[ConceptRow]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    return {k: retention_rate(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def contested_rate(rows: List[ConceptRow]) -> float:
    """Fraction of concepts where matcher topologies DISAGREE (0 < agreement < 1).
    High = the retention/laundering signal is matcher-dependent and should not be
    claimed as an NLA property without probe ground truth (review note P0 #2)."""
    scored = [r for r in rows if r.agreement is not None]
    if not scored:
        return float("nan")
    return sum(1 for r in scored if 0.0 < r.agreement < 1.0) / len(scored)


def mean_faithfulness(rows: List[ConceptRow], predicate=lambda r: True) -> float:
    vals = [r.cos_sim for r in rows if predicate(r) and r.cos_sim is not None]
    return (sum(vals) / len(vals)) if vals else float("nan")


def faithfulness_weighted_retention(rows: List[ConceptRow], default: float = 1.0) -> float:
    """Retention with each concept weighted by AR faithfulness (cos_sim, clamped
    to [0,1]). A 'retained' concept whose activation was reconstructed poorly
    (low cos) counts less — it may be a verbalizer artifact, not real survival.
    Rows without a faithfulness score use `default`. Addresses review note P1#5."""
    num = den = 0.0
    for r in rows:
        w = default if r.cos_sim is None else max(0.0, min(1.0, r.cos_sim))
        den += w
        if r.status in ("retained", "substituted"):
            num += w
    return (num / den) if den else float("nan")
