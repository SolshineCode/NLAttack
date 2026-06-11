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
                                matcher=None, n_perm: int = 500, seed: int = 0) -> VerbAxisResult:
    """THE verbalizer-discrimination axis (highest-value add per both reviewers).
    Does the AV verbalize a HARD-NEGATIVE minimal pair DIFFERENTLY?

    `pairs`: list of (concept, act_pos, act_neg) — act_pos has `concept`, act_neg
    is a hard negative (concept absent; e.g. an ATT&CK term vs its benign
    neighbor — see build_attack_pairs).

    Metric = **AUC** over the matcher's CONTINUOUS pos-vs-neg presence scores
    (`soft_score`), so it is threshold-free and **chance = 0.5** (not 0). This
    avoids the earlier flaw of conflating AV conditioning with the matcher's
    binary recall (independent review P0a). Reported with:
      * a label-PERMUTATION null (mean + p-value),
      * `matcher_ceiling_auc` — the matcher's own discriminability on text that
        literally states vs omits the concept; the AV-AUC cannot exceed this, so
        it separates AV signal from matcher quality.
    AUC rises 0.5 -> 1.0 as the AV LoRA learns to condition (unlike activation-
    side axes, which are flat across AV checkpoints)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    if not pairs:
        return VerbAxisResult("minimal_pair_discrimination", "verbalizer", False, detail={})
    matcher = matcher or EnsembleMatcher()
    pos_s, neg_s, ceil_p, ceil_n = [], [], [], []
    pos_hit = neg_hit = 0
    for concept, act_pos, act_neg in pairs:
        v_pos, v_neg = verbalize_fn(act_pos), verbalize_fn(act_neg)
        pos_s.append(matcher.soft_score(concept, v_pos))
        neg_s.append(matcher.soft_score(concept, v_neg))
        pos_hit += int(matcher.match(concept, v_pos).present)
        neg_hit += int(matcher.match(concept, v_neg).present)
        ceil_p.append(matcher.soft_score(concept, f"the text is about {concept}."))
        ceil_n.append(matcher.soft_score(concept, "the text is about the weather today."))
    n = len(pairs)
    y = np.array([1] * n + [0] * n); s = np.array(pos_s + neg_s, dtype="float64")
    auc = 0.5 if len(set(s.tolist())) <= 1 else float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    perms = []
    for _ in range(n_perm):
        try:
            perms.append(roc_auc_score(rng.permutation(y), s))
        except Exception:
            pass
    perm_mean = float(np.mean(perms)) if perms else 0.5
    pval = (1 + sum(1 for p in perms if p >= auc)) / (1 + len(perms)) if perms else float("nan")
    cy = np.array([1] * n + [0] * n); cs = np.array(ceil_p + ceil_n, dtype="float64")
    ceil_auc = 0.5 if len(set(cs.tolist())) <= 1 else float(roc_auc_score(cy, cs))
    return VerbAxisResult(
        "minimal_pair_discrimination", "verbalizer", True,
        raw=auc, null=perm_mean, margin=auc - perm_mean, n=n,
        detail={"auc": auc, "perm_null_mean": perm_mean, "p_value": pval,
                "matcher_ceiling_auc": ceil_auc,
                "pos_recall": pos_hit / n, "neg_false_alarm": neg_hit / n,
                "note": "AUC chance=0.5; AV-AUC is bounded above by matcher_ceiling_auc"},
    )


def build_attack_pairs(activation_of: Callable[[str], object],
                       pos_template="The operator performed {} against the target host.",
                       neg_template="The team handled routine {} on the server."):
    """Shipped ATT&CK demo for minimal_pair_discrimination: attack term vs its
    BENIGN NEIGHBOR as the hard negative (the laundering-discrimination test — can
    the NLA-monitor tell 'credential dumping' from 'password management'?). Uses
    the ATT&CK dictionary's benign_neighbor field. `activation_of`: text -> vec
    (the host model; needs a model/GPU to produce real activations). Returns
    (attack_concept, act_pos, act_neg) triples ready for minimal_pair_discrimination."""
    from .attack_concepts import ATTACK_CONCEPTS
    pairs = []
    for term, tactic, phase, axis, benign in ATTACK_CONCEPTS:
        pairs.append((term,
                      activation_of(pos_template.format(term)),
                      activation_of(neg_template.format(benign))))
    return pairs


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


def _retrieval_scores(V, D):
    import numpy as np
    N = len(V); sims = V @ D.T
    top1 = 0; rr = 0.0
    for i in range(N):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == i)[0][0]) + 1
        top1 += int(rank == 1); rr += 1.0 / rank
    return top1 / N, rr / N


def doc_retrieval(items: Sequence[Tuple[object, str]], verbalize_fn: VerbalizeFn,
                  embed_fn=None) -> VerbAxisResult:
    """`items`: (activation, source_doc_text). The AV verbalizes each activation;
    rank ALL source docs by similarity to the verbalization; metric = retrieval@1
    (+ MRR). Chance = 1/N. Rises as the AV conditions on the activation.

    review note P0b: char-3-gram cosine rewards lexical PARROTING, not semantic
    verbalization. So when `embed_fn` (e.g. sentence-transformers) is given we
    compute BOTH and report `lexical_minus_semantic_gap` — a large positive gap
    means the AV is winning by copying surface tokens, not by verbalizing. The
    headline `raw` is the SEMANTIC retrieval@1 when an embedder is available, else
    char (flagged). Also reports a copy-the-doc upper reference."""
    import numpy as np
    items = list(items); N = len(items)
    if N < 3:
        return VerbAxisResult("doc_retrieval", "verbalizer", False,
                              detail={"reason": "need >=3 items"})
    verbs = [verbalize_fn(a) for a, _ in items]
    docs = [d for _, d in items]
    Vc = np.stack([_char_ngram_vec(t) for t in verbs])
    Dc = np.stack([_char_ngram_vec(t) for t in docs])
    p1_char, mrr_char = _retrieval_scores(Vc, Dc)
    detail = {"chance": 1.0 / N, "char_p1": p1_char, "char_mrr": mrr_char,
              "copy_doc_p1": _retrieval_scores(Dc, Dc)[0]}  # upper reference (~1.0)
    headline, headline_mrr, used = p1_char, mrr_char, "char-3gram"
    if embed_fn is not None:
        Ve = np.asarray(embed_fn(verbs), dtype="float32")
        De = np.asarray(embed_fn(docs), dtype="float32")
        Ve /= (np.linalg.norm(Ve, axis=1, keepdims=True) + 1e-9)
        De /= (np.linalg.norm(De, axis=1, keepdims=True) + 1e-9)
        p1_sem, mrr_sem = _retrieval_scores(Ve, De)
        detail.update({"semantic_p1": p1_sem, "semantic_mrr": mrr_sem,
                       "lexical_minus_semantic_gap": p1_char - p1_sem})
        headline, headline_mrr, used = p1_sem, mrr_sem, "semantic"
    detail.update({"metric": used, "mrr": headline_mrr})
    return VerbAxisResult("doc_retrieval", "verbalizer", True,
                          raw=headline, null=1.0 / N, margin=headline - 1.0 / N,
                          n=N, detail=detail)


def prior_deviation(items: Sequence[Tuple[object, str]], verbalize_fn: VerbalizeFn,
                    mean_activation, embed_fn=None) -> VerbAxisResult:
    """Does the AV use the ACTIVATION, or just the concept PRIOR? (the team's own
    contrastive method, review note P1c.) For each (act, doc): compare
    sim(AV(act), doc) to sim(AV(mean_activation), doc). Positive mean delta = the
    activation adds doc-specific signal beyond an unconditioned/prior verbalization.
    `mean_activation` = a single 'prior' activation (e.g. dataset mean)."""
    import numpy as np
    items = list(items)
    if len(items) < 3:
        return VerbAxisResult("prior_deviation", "verbalizer", False,
                              detail={"reason": "need >=3 items"})
    v_prior = verbalize_fn(mean_activation)
    def vec(t):
        if embed_fn is not None:
            x = np.asarray(embed_fn([t])[0], dtype="float32")
            return x / (np.linalg.norm(x) + 1e-9)
        return _char_ngram_vec(t)
    pv = vec(v_prior)
    deltas = []
    for act, doc in items:
        d = vec(doc); cond = vec(verbalize_fn(act))
        deltas.append(float(cond @ d) - float(pv @ d))
    m = float(np.mean(deltas))
    return VerbAxisResult("prior_deviation", "verbalizer", True,
                          raw=m, null=0.0, margin=m, n=len(items),
                          detail={"frac_positive": float(np.mean([d > 0 for d in deltas]))})


def mode_collapse(activations, verbalize_fn: VerbalizeFn) -> VerbAxisResult:
    """Diversity of verbalizations across DIFFERENT activations. An AV that emits
    near-identical text for everything passes discrimination yet is useless
    (review note P1). raw = distinct-2 ratio (unique bigrams / total). Low = collapse."""
    acts = list(activations)
    if len(acts) < 3:
        return VerbAxisResult("mode_collapse", "verbalizer", False,
                              detail={"reason": "need >=3 activations"})
    verbs = [verbalize_fn(a) for a in acts]
    all_bi, uniq_bi = 0, set()
    for t in verbs:
        toks = content_words(t)
        bis = list(zip(toks, toks[1:]))
        all_bi += len(bis); uniq_bi.update(bis)
    distinct2 = (len(uniq_bi) / all_bi) if all_bi else 0.0
    # fraction of verbalizations that are byte-identical to another (collapse tell)
    dup = 1.0 - len(set(verbs)) / len(verbs)
    return VerbAxisResult("mode_collapse", "verbalizer", True,
                          raw=distinct2, null=float("nan"), margin=float("nan"),
                          n=len(acts), detail={"distinct2": distinct2, "dup_rate": dup,
                                               "collapse": distinct2 < 0.1 or dup > 0.5})


def calibration_entropy(samples_per_item, matcher=None, concepts=None) -> VerbAxisResult:
    """Calibration via self-consistency across temperature samples. `samples_per_item`
    = list of lists of AV verbalizations (K samples for the same activation). Low
    entropy/high agreement where the concept is truly present = calibrated; scatter
    = uncertain. raw = mean per-item agreement (fraction of samples that agree with
    the modal present/absent verdict for `concepts[i]`). Caller supplies samples
    (sampling needs the AV at temperature>0)."""
    if not samples_per_item:
        return VerbAxisResult("calibration_entropy", "verbalizer", False, detail={})
    matcher = matcher or EnsembleMatcher()
    concepts = concepts or [None] * len(samples_per_item)
    agrs = []
    for samples, c in zip(samples_per_item, concepts):
        if not samples or c is None:
            continue
        hits = [matcher.match(c, s).present for s in samples]
        modal = sum(hits) >= len(hits) / 2
        agrs.append(sum(1 for h in hits if h == modal) / len(hits))
    if not agrs:
        return VerbAxisResult("calibration_entropy", "verbalizer", False, detail={})
    import numpy as np
    return VerbAxisResult("calibration_entropy", "verbalizer", True,
                          raw=float(np.mean(agrs)), null=0.5, margin=float(np.mean(agrs)) - 0.5,
                          n=len(agrs), detail={"mean_self_consistency": float(np.mean(agrs))})


def verbalizer_report(verbalize_fn, *, pairs=None, retrieval_items=None,
                      mean_activation=None, diversity_activations=None,
                      samples_per_item=None, sample_concepts=None,
                      matcher=None, embed_fn=None) -> List[VerbAxisResult]:
    """Run whichever verbalizer-side axes have inputs. These are the AV-training-
    tracking axes — sweep them across AV-LoRA checkpoints to see conditioning
    emerge (the activation-side capability index is ~flat across those same
    checkpoints, by construction). For the ATT&CK laundering-discrimination demo,
    pass `pairs=build_attack_pairs(activation_of)`."""
    out = []
    if pairs:
        out.append(minimal_pair_discrimination(pairs, verbalize_fn, matcher))
    if retrieval_items:
        out.append(doc_retrieval(retrieval_items, verbalize_fn, embed_fn))
        if mean_activation is not None:
            out.append(prior_deviation(retrieval_items, verbalize_fn, mean_activation, embed_fn))
    if diversity_activations is not None:
        out.append(mode_collapse(diversity_activations, verbalize_fn))
    if samples_per_item is not None:
        out.append(calibration_entropy(samples_per_item, matcher, sample_concepts))
    return [r for r in out if r is not None]
