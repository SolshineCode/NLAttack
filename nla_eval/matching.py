"""Concept-presence matching — the single primitive everything is built on.

Question answered: is concept C (or a near-neighbor) present in a reconstruction?

Three tiers, auto-selected by what's installed, so the harness runs anywhere:
  1. embedding  — sentence-transformers cosine >= threshold   (best)
  2. wordnet    — exact/lemma + WordNet synonyms                (good, light)
  3. lexical    — lemma/stem + substring fallback              (always works)

Every match returns the matched output term + similarity so you can audit, and
classify each input concept as: retained / substituted / dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set
import re

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-']+")

# tiny built-in stopword set so the lexical tier needs no downloads
_STOP = set(
    "the a an and or but if then of to in on at for with from by as is are was "
    "were be been being this that these those it its he she they we you i not no "
    "do does did done has have had will would can could may might shall should".split()
)


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text or "")]


def content_words(text: str) -> List[str]:
    return [w for w in tokenize(text) if w not in _STOP and len(w) > 2]


@dataclass
class Match:
    concept: str
    present: bool
    matched_term: Optional[str]
    similarity: float
    mode: str  # "exact" | "synonym" | "embedding" | "none"

    @property
    def status(self) -> str:
        if not self.present:
            return "dropped"
        return "retained" if self.mode == "exact" else "substituted"


class Matcher:
    """Pick the strongest backend available; allow forcing one for reproducibility."""

    def __init__(self, backend: str = "auto", threshold: float = 0.55):
        self.threshold = threshold
        self.backend = backend
        self._st = None  # sentence-transformers model
        self._wn = None  # wordnet handle
        if backend in ("auto", "embedding"):
            self._try_embedding()
        if self._st is None and backend in ("auto", "wordnet"):
            self._try_wordnet()
        if self._st is None and self._wn is None:
            self.backend = "lexical"
        elif backend == "auto":
            self.backend = "embedding" if self._st is not None else "wordnet"

    def _try_embedding(self):
        try:
            from sentence_transformers import SentenceTransformer  # lazy

            self._st = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self._st = None

    def _try_wordnet(self):
        try:
            from nltk.corpus import wordnet as wn  # lazy

            wn.synsets("test")  # force lookup; raises if corpus missing
            self._wn = wn
        except Exception:
            self._wn = None

    # ---- per-backend presence checks -------------------------------------

    def _lexical(self, concept: str, out_tokens: Set[str], out_text: str) -> Match:
        c = concept.lower().strip()
        head = c.split()[-1] if c else c
        if c in out_text or head in out_tokens:
            return Match(concept, True, head, 1.0, "exact")
        # crude stem: drop common suffixes and retry as substring
        stem = re.sub(r"(ing|ed|es|s|ion|ment)$", "", head)
        if len(stem) >= 4 and any(stem in t for t in out_tokens):
            return Match(concept, True, stem, 0.7, "synonym")
        return Match(concept, False, None, 0.0, "none")

    def _wordnet_match(self, concept: str, out_tokens: Set[str], out_text: str) -> Match:
        base = self._lexical(concept, out_tokens, out_text)
        if base.present:
            return base
        head = concept.lower().split()[-1]
        syns: Set[str] = set()
        for s in self._wn.synsets(head):
            for l in s.lemmas():
                syns.add(l.name().replace("_", " ").lower())
        for syn in syns:
            sh = syn.split()[-1]
            if sh in out_tokens or syn in out_text:
                return Match(concept, True, sh, 0.65, "synonym")
        return Match(concept, False, None, 0.0, "none")

    def _embedding_match(self, concept: str, out_words: List[str]) -> Match:
        if not out_words:
            return Match(concept, False, None, 0.0, "none")
        import numpy as np

        vecs = self._st.encode([concept] + out_words, normalize_embeddings=True)
        cvec, wvecs = vecs[0], vecs[1:]
        sims = wvecs @ cvec
        i = int(np.argmax(sims))
        sim = float(sims[i])
        if sim >= 0.999:
            return Match(concept, True, out_words[i], sim, "exact")
        if sim >= self.threshold:
            return Match(concept, True, out_words[i], sim, "embedding")
        return Match(concept, False, out_words[i], sim, "none")

    # ---- public ----------------------------------------------------------

    def match(self, concept: str, output_text: str) -> Match:
        out_words = content_words(output_text)
        out_tokens = set(tokenize(output_text))
        out_text = " " + (output_text or "").lower() + " "
        if self.backend == "embedding":
            return self._embedding_match(concept, out_words)
        if self.backend == "wordnet":
            return self._wordnet_match(concept, out_tokens, out_text)
        return self._lexical(concept, out_tokens, out_text)
