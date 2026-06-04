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
) -> RunResult:
    matcher = matcher or Matcher()
    texts = [ex.text for ex in dataset]
    outs = nla.reconstruct_batch(texts)
    recon = {ex.id: out for ex, out in zip(dataset, outs)}

    rows: List[ConceptRow] = []
    insertions: List[InsertionRow] = []
    for ex, out in zip(dataset, outs):
        input_concept_words = {w for c in ex.concepts for w in content_words(c)}
        for c in ex.concepts:
            m = matcher.match(c, out)
            rows.append(
                ConceptRow(
                    example_id=ex.id,
                    nla=nla.name,
                    concept=c,
                    status=m.status,
                    matched_term=m.matched_term,
                    similarity=m.similarity,
                    mode=m.mode,
                    meta=ex.meta,
                )
            )
        if detect_insertions:
            for w in set(content_words(out)) - input_concept_words:
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
