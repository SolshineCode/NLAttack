"""Verbalizer-side (AV-conditioning) axes — the levers the activation-side
emergence axes structurally cannot provide.

WHY THIS MODULE EXISTS (from collaborator feedback, 2026-06-08):
The emergence axes (`decodability`, `selectivity`, `stability`, `effective_rank`)
read the FROZEN base-model activation. They characterize what the activation
*carries* — which is ~constant across AV-LoRA checkpoints, so they CANNOT track
AV training progress. The thing the conditioning / prior-deviation / contrastive
work is stuck on is the **verbalizer's reading of the activation**. These axes
measure exactly that: they RUN THE AV (`verbalize_fn`), so they move as the AV is
trained.

Every function takes `verbalize_fn: Callable[[activation_vec], str]` — e.g.
`LocalGemmaE2BNLA.verbalize_activation`, or a Neuronpedia-AV wrapper. The scoring
is CPU + dependency-free (hashed char-n-gram cosine for retrieval).

The two axes:
  * minimal_pair_discrimination — does the AV verbalize a hard-negative minimal
    pair DIFFERENTLY? (the verbalizer-facing `content_adjacency`). Unconditioned
    verbalizer -> 0; conditioning -> rises with training.
  * doc_retrieval — can the AV's verbalization retrieve its own source doc among
    distractors? (the held-out doc-retrieval signal). Chance = 1/N; rises with
    AV conditioning.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence, Optional, List, Tuple
import re

from .matching import EnsembleMatcher, content_words

VerbalizeFn = Callable[[object], str]


@dataclass
class VerbAxisResult:
    name: str
    side: str                      # always "verbalizer"
    available: bool
    raw: float = float("nan")      # the headline metric
    null: float = float("nan")     # chance / unconditioned floor
    margin: float = float("nan")
    n: int = 0
    detail: dict = None

    def flat(self):
        d = {k: v for k, v in asdict(self).items() if k != "detail"}
        return d


# ---------------------------------------------------------------------------
# minimal-pair discrimination (the verbalizer-facing content_adjacency)
# ---------------------------------------------------------------------------

def minimal_pair_discrimination(pairs, verbalize_fn: VerbalizeFn,
                                matcher=None) -> VerbAxisResult:
    """`pairs`: list of (concept, act_pos, act_neg) where act_pos is an activation
    for which `concept` is present and act_neg is a HARD NEGATIVE (concept absent,
    e.g. minimal edit / contrastive). Discrimination is correct when the AV
    verbalization of act_pos surfaces `concept` AND that of act_neg does NOT.

    raw = discrimination accuracy. null = 0.0: an AV that ignores the activation
    emits the same text for both -> pos_hit == neg_hit -> accuracy 0. So any
    margin > 0 is genuine verbalizer conditioning, and the metric RISES as the AV
    LoRA learns to read the activation (unlike the activation-side axes)."""
    if not pairs:
        return VerbAxisResult("minimal_pair_discrimination", "verbalizer", False, detail={})
    matcher = matcher or EnsembleMatcher()
    correct = pos_hits = neg_hits = 0
    n = 0
    for concept, act_pos, act_neg in pairs:
        v_pos = verbalize_fn(act_pos)
        v_neg = verbalize_fn(act_neg)
        ph = matcher.match(concept, v_pos).present
        nh = matcher.match(concept, v_neg).present
        correct += int(ph and not nh)
        pos_hits += int(ph); neg_hits += int(nh); n += 1
    acc = correct / n if n else float("nan")
    return VerbAxisResult(
        "minimal_pair_discrimination", "verbalizer", True,
        raw=acc, null=0.0, margin=acc, n=n,
        detail={"pos_recall": pos_hits / n if n else float("nan"),
                "neg_false_alarm": neg_hits / n if n else float("nan")},
    )


# ---------------------------------------------------------------------------
# held-out doc retrieval (the AV-conditioning signal the team uses)
# ---------------------------------------------------------------------------

def _char_ngram_vec(text: str, dim=1024, n=3):
    import numpy as np
    v = np.zeros(dim, dtype="float32")
    s = re.sub(r"\s+", " ", (text or "").lower())
    for i in range(max(0, len(s) - n + 1)):
        v[hash(s[i:i + n]) % dim] += 1.0
    nrm = (v @ v) ** 0.5
    return v / nrm if nrm > 0 else v


def doc_retrieval(items: Sequence[Tuple[object, str]], verbalize_fn: VerbalizeFn,
                  embed_fn=None) -> VerbAxisResult:
    """`items`: list of (activation, source_doc_text). The AV verbalizes each
    activation; we then rank ALL source docs by similarity to that verbalization
    and ask whether the activation's OWN source doc is retrieved top-1 (and MRR).

    Chance = 1/N. Rises as the AV conditions on the activation (verbalizations
    become doc-specific). `embed_fn(list[str])->matrix` optional (e.g. sentence-
    transformers); default is dependency-free hashed char-3-gram cosine."""
    import numpy as np
    items = list(items)
    N = len(items)
    if N < 3:
        return VerbAxisResult("doc_retrieval", "verbalizer", False,
                              detail={"reason": "need >=3 items"})
    verbs = [verbalize_fn(a) for a, _ in items]
    docs = [d for _, d in items]
    if embed_fn is not None:
        V = np.asarray(embed_fn(verbs), dtype="float32")
        D = np.asarray(embed_fn(docs), dtype="float32")
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    else:
        V = np.stack([_char_ngram_vec(t) for t in verbs])
        D = np.stack([_char_ngram_vec(t) for t in docs])
    sims = V @ D.T                       # [N verbs, N docs]
    top1 = 0; rr = 0.0
    for i in range(N):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == i)[0][0]) + 1
        top1 += int(rank == 1); rr += 1.0 / rank
    p1 = top1 / N; mrr = rr / N; chance = 1.0 / N
    return VerbAxisResult("doc_retrieval", "verbalizer", True,
                          raw=p1, null=chance, margin=p1 - chance, n=N,
                          detail={"mrr": mrr, "chance": chance})


def verbalizer_report(verbalize_fn, *, pairs=None, retrieval_items=None,
                      matcher=None, embed_fn=None) -> List[VerbAxisResult]:
    """Run whichever verbalizer-side axes have inputs. These are the AV-training-
    tracking axes — sweep them across AV-LoRA checkpoints to see conditioning
    emerge (the activation-side emergence index will be ~flat across those same
    checkpoints, by construction)."""
    out = []
    if pairs:
        out.append(minimal_pair_discrimination(pairs, verbalize_fn, matcher))
    if retrieval_items:
        out.append(doc_retrieval(retrieval_items, verbalize_fn, embed_fn))
    return out
