"""Emergence dashboard — additive operational definition of "useful
interpretability capability starting to emerge" in a weak/tiny NLA.

Each axis below is an orthogonal facet of the construct, scored in [0,1] and
ALWAYS reported with its own null margin (so a high score can't come from
small-n overfit or a degenerate decoder). The axes sum, ARiES-style, into an
`EmergenceIndex` (a transparent weighted mean of whichever axes are available).

Axes (each from a plan):
  decodability       P081  is anything linearly readable above the permutation null?
  sufficiency        P101  does the bottleneck beat a TRIVIAL input feature baseline?
  selectivity        P102  concept-specific, or riding a freq/length confound?
  content_adjacency  P011  minimal-pair separation (fine distinctions, not topic)
  faithful_rank      P080  AR reconstruction ranks the true concept over a decoy
  stability          P044  probe direction stable across seeds (vs overfit)
  dose_response      P086  decodability tracks training prevalence
  graded_encoding    P103  a continuum (magnitude/intensity) decoded monotonically
  abstraction        P104  probe trained on context A transfers to context B
  effective_rank     P105  representation is structured (not collapsed, not noise)

CPU + numpy/sklearn. AR/AV-dependent axes (faithful_rank, and the optional
verbalization/calibration axes) are skipped automatically when their inputs
aren't supplied, so the whole thing runs on a rudimentary NLA with no working AV.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence, Optional, Dict, List, Callable

import numpy as np

from .bottleneck_probe import _cv_auc_acc, probe_concept, label_by_keyword
from .matching import content_words


@dataclass
class AxisResult:
    name: str
    available: bool
    raw: float = float("nan")        # the natural metric (AUC, d, rho, PR, ...)
    null: float = float("nan")       # the chance/floor reference for `raw`
    margin: float = float("nan")     # raw - null (signed; positive = above floor)
    score: float = float("nan")      # normalized to [0,1] for the additive index
    detail: dict = field(default_factory=dict)

    def flat(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "detail"}
        return d


def _clamp01(x: float) -> float:
    if x != x:  # nan
        return float("nan")
    return float(max(0.0, min(1.0, x)))


# --------------------------------------------------------------------------
# individual axes
# --------------------------------------------------------------------------

def axis_decodability(X, texts, concepts, n_perm=8, seed=0) -> AxisResult:
    """P081: mean per-concept (AUC - permutation_floor), clamped. Score scales the
    margin so 0 margin -> 0 and a +0.4 margin -> ~1."""
    margins, aucs, sig = [], [], 0
    for c in concepts:
        y = label_by_keyword(texts, c)
        pr = probe_concept(X, y, n_permutations=n_perm, seed=seed)
        if pr["auc"] != pr["auc"]:
            continue
        floor = pr["control_auc"] + 2 * pr["control_std"]
        margins.append(pr["auc"] - floor)
        aucs.append(pr["auc"])
        sig += int(pr["signal"])
    if not margins:
        return AxisResult("decodability", False)
    m = float(np.mean(margins))
    return AxisResult("decodability", True, raw=float(np.mean(aucs)),
                      null=0.5, margin=m, score=_clamp01(m / 0.4),
                      detail={"frac_signal": sig / len(margins), "n_concepts": len(margins)})


def _trivial_input_features(texts: Sequence[str], dim: int = 256, seed: int = 0):
    """A deliberately TRIVIAL representation of the input: hashed bag-of-content-
    words. The bottleneck must beat this to count as adding interpretability value
    (use SEMANTIC labels with sufficiency, else the keyword leaks into the bag)."""
    rng = np.random.default_rng(seed)
    feats = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in content_words(t):
            feats[i, hash(w) % dim] += 1.0
    return feats


def axis_sufficiency(X, texts, concepts, labels_by_concept=None, seed=0) -> AxisResult:
    """P101: does the bottleneck beat a trivial bag-of-words-of-input baseline?
    sufficiency = AUC(X) - AUC(trivial). Positive => the bottleneck encodes
    something not trivially readable from surface tokens. NOTE: meaningful only
    with SEMANTIC labels (pass labels_by_concept); with keyword labels the bag
    contains the answer and this floors to ~0 by construction (still a valid,
    honest null — it says 'no value over surface')."""
    triv = _trivial_input_features(texts, seed=seed)
    deltas = []
    for c in concepts:
        y = np.asarray(labels_by_concept[c] if labels_by_concept else label_by_keyword(texts, c), dtype=int)
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        try:
            ax, _ = _cv_auc_acc(np.asarray(X, "float32"), y, 5, seed)
            at, _ = _cv_auc_acc(triv, y, 5, seed)
            deltas.append(ax - at)
        except Exception:
            continue
    if not deltas:
        return AxisResult("sufficiency", False)
    d = float(np.mean(deltas))
    return AxisResult("sufficiency", True, raw=d, null=0.0, margin=d,
                      score=_clamp01((d + 0.1) / 0.3),  # -0.1->0, +0.2->1
                      detail={"n_concepts": len(deltas),
                              "note": "use semantic labels; keyword labels floor this"})


def axis_selectivity(X, texts, concepts, seed=0) -> AxisResult:
    """P102: concept-selective or riding a confound? Confound features = input
    length + token count. selectivity = AUC(X) - AUC(label from confound only).
    High => the probe reads the concept, not just 'longer/denser inputs'."""
    conf = np.array([[len(t or ""), len(content_words(t))] for t in texts], dtype="float32")
    deltas = []
    for c in concepts:
        y = np.asarray(label_by_keyword(texts, c), dtype=int)
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        try:
            ax, _ = _cv_auc_acc(np.asarray(X, "float32"), y, 5, seed)
            ac, _ = _cv_auc_acc(conf, y, 5, seed)
            deltas.append(ax - ac)
        except Exception:
            continue
    if not deltas:
        return AxisResult("selectivity", False)
    d = float(np.mean(deltas))
    return AxisResult("selectivity", True, raw=d, null=0.0, margin=d,
                      score=_clamp01(d / 0.35), detail={"n_concepts": len(deltas)})


def axis_content_adjacency(pairs, seed=0) -> AxisResult:
    """P011: minimal-pair separation. `pairs` = list of (act_a, act_b, neg_act)
    where (a,b) is a minimal pair (e.g. 'X' vs 'not X') and neg_act is a random
    in-topic activation. Score = Cohen's d of ||a-b|| vs ||a-neg||. Skipped if no
    pairs supplied."""
    if not pairs:
        return AxisResult("content_adjacency", False)
    near, far = [], []
    for a, b, neg in pairs:
        a, b, neg = np.asarray(a, "float32"), np.asarray(b, "float32"), np.asarray(neg, "float32")
        near.append(float(np.linalg.norm(a - b)))
        far.append(float(np.linalg.norm(a - neg)))
    near, far = np.array(near), np.array(far)
    pooled = np.sqrt((near.var() + far.var()) / 2) + 1e-9
    d = float((far.mean() - near.mean()) / pooled)  # >0: minimal pair closer than random
    return AxisResult("content_adjacency", True, raw=d, null=0.0, margin=d,
                      score=_clamp01(d / 1.0), detail={"n_pairs": len(pairs)})


def axis_faithful_rank(recon_vectors, true_dirs, decoy_dirs) -> AxisResult:
    """P080: for each reconstructed activation, is it nearer the TRUE concept
    direction than a DECOY (benign neighbor)? Score = (top-1 true-rate - 0.5)*2.
    Needs AR reconstructions; skipped otherwise."""
    if recon_vectors is None or true_dirs is None or decoy_dirs is None:
        return AxisResult("faithful_rank", False)
    R = np.asarray(recon_vectors, "float32")
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    t = np.asarray(true_dirs, "float32"); dch = np.asarray(decoy_dirs, "float32")
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    dch = dch / (np.linalg.norm(dch, axis=1, keepdims=True) + 1e-9)
    wins = ((R * t).sum(1) > (R * dch).sum(1)).mean()
    return AxisResult("faithful_rank", True, raw=float(wins), null=0.5,
                      margin=float(wins - 0.5), score=_clamp01((wins - 0.5) * 2))


def _direction(X, y, seed, subsample=None):
    """Probe weight direction. With `subsample` (fraction in (0,1]) the fit uses a
    different random row-subset per seed, so repeated calls vary — that's what
    makes stability meaningful (a deterministic full-data fit would be identical
    every seed and read 'stable' even on noise)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xf, yf = X, y
    if subsample and subsample < 1.0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max(10, int(len(y) * subsample)), replace=False)
        Xs, ys = X[idx], y[idx]
        if ys.sum() >= 2 and (1 - ys).sum() >= 2:
            Xf, yf = Xs, ys
    sc = StandardScaler().fit(Xf)
    w = LogisticRegression(max_iter=2000).fit(sc.transform(Xf), yf).coef_.ravel()
    return w / (np.linalg.norm(w) + 1e-9)


def axis_stability(X, texts, concepts, n_seeds=5, seed=0) -> AxisResult:
    """P044: mean cross-seed cosine of the probe direction, vs the same statistic
    on shuffled labels (the overfit floor). Margin = real_cos - shuffled_cos."""
    X = np.asarray(X, "float32")
    rng = np.random.default_rng(seed)
    real_cos, shuf_cos = [], []
    for c in concepts:
        y = np.asarray(label_by_keyword(texts, c), dtype=int)
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        dirs = [_direction(X, y, s, subsample=0.7) for s in range(n_seeds)]
        ys = y.copy()
        sdirs = []
        for s in range(n_seeds):
            rng.shuffle(ys)
            if ys.sum() < 2 or (1 - ys).sum() < 2:
                continue
            sdirs.append(_direction(X, ys, s, subsample=0.7))
        real_cos.append(_mean_pairwise_cos(dirs))
        if len(sdirs) >= 2:
            shuf_cos.append(_mean_pairwise_cos(sdirs))
    if not real_cos:
        return AxisResult("stability", False)
    rc = float(np.mean(real_cos)); sc_ = float(np.mean(shuf_cos)) if shuf_cos else 0.0
    return AxisResult("stability", True, raw=rc, null=sc_, margin=rc - sc_,
                      score=_clamp01((rc - sc_) / (1.0 - sc_ + 1e-9)),
                      detail={"n_concepts": len(real_cos)})


def _mean_pairwise_cos(dirs):
    if len(dirs) < 2:
        return float("nan")
    s, c = 0.0, 0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            s += abs(float(np.dot(dirs[i], dirs[j]))); c += 1
    return s / c if c else float("nan")


def axis_dose_response(X, texts, concepts, prevalence=None, seed=0) -> AxisResult:
    """P086: Spearman corr between a concept's training-prevalence proxy and its
    probe AUC. Rising correlation = capacity allocated by exposure (emergence).
    Default proxy = label prevalence in the corpus."""
    from scipy.stats import spearmanr  # lazy
    prevs, aucs = [], []
    n = len(texts) or 1
    for c in concepts:
        y = np.asarray(label_by_keyword(texts, c), dtype=int)
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        pr = probe_concept(X, y, n_permutations=0, seed=seed)
        if pr["auc"] != pr["auc"]:
            continue
        prevs.append(prevalence[c] if prevalence else y.sum() / n)
        aucs.append(pr["auc"])
    if len(aucs) < 4:
        return AxisResult("dose_response", False, detail={"reason": "need >=4 concepts"})
    rho = float(spearmanr(prevs, aucs).statistic)
    return AxisResult("dose_response", True, raw=rho, null=0.0, margin=rho,
                      score=_clamp01((rho + 1) / 2), detail={"n_concepts": len(aucs)})


def axis_graded_encoding(X, graded_values, seed=0) -> AxisResult:
    """P103: can a ridge regression decode a CONTINUUM (magnitude/intensity)
    monotonically? Score from CV R² vs a shuffled-value null. Needs graded labels."""
    if graded_values is None:
        return AxisResult("graded_encoding", False)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(np.asarray(X, "float32"))
    yv = np.asarray(graded_values, "float32")
    if np.std(yv) < 1e-9 or len(yv) < 10:
        return AxisResult("graded_encoding", False)
    r2 = float(np.mean(cross_val_score(Ridge(alpha=1.0), X, yv, cv=5, scoring="r2")))
    rng = np.random.default_rng(seed); ys = yv.copy(); rng.shuffle(ys)
    r2n = float(np.mean(cross_val_score(Ridge(alpha=1.0), X, ys, cv=5, scoring="r2")))
    return AxisResult("graded_encoding", True, raw=r2, null=r2n, margin=r2 - r2n,
                      score=_clamp01(r2 - r2n))


def axis_abstraction(X, texts, concepts, group_ids, seed=0) -> AxisResult:
    """P104: train a probe on context-group A, test on group B. Above-chance
    transfer = abstraction, not context-bound memorization. group_ids: per-row
    group label; uses the two largest groups as A/B."""
    if group_ids is None:
        return AxisResult("abstraction", False)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    X = np.asarray(X, "float32"); g = np.asarray(group_ids)
    uniq, counts = np.unique(g, return_counts=True)
    if len(uniq) < 2:
        return AxisResult("abstraction", False)
    A, B = uniq[np.argsort(-counts)[:2]]
    aucs = []
    for c in concepts:
        y = np.asarray(label_by_keyword(texts, c), dtype=int)
        ia, ib = (g == A), (g == B)
        if y[ia].sum() < 5 or (1 - y[ia]).sum() < 5 or len(np.unique(y[ib])) < 2:
            continue
        sc = StandardScaler().fit(X[ia])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[ia]), y[ia])
        p = clf.predict_proba(sc.transform(X[ib]))[:, 1]
        try:
            aucs.append(roc_auc_score(y[ib], p))
        except Exception:
            continue
    if not aucs:
        return AxisResult("abstraction", False)
    a = float(np.mean(aucs))
    return AxisResult("abstraction", True, raw=a, null=0.5, margin=a - 0.5,
                      score=_clamp01((a - 0.5) * 2), detail={"n_concepts": len(aucs)})


def axis_effective_rank(X, texts, concepts, seed=0) -> AxisResult:
    """P105: participation ratio of the set of concept probe directions. A
    collapsed representation -> PR~1; isotropic noise -> PR~n_concepts (and
    directions unstable). Emergence = a structured low-but->1 rank. Score peaks in
    a mid band and is gated by stability (reported in detail)."""
    X = np.asarray(X, "float32")
    dirs = []
    for c in concepts:
        y = np.asarray(label_by_keyword(texts, c), dtype=int)
        if y.sum() >= 5 and (1 - y).sum() >= 5:
            dirs.append(_direction(X, y, seed))
    if len(dirs) < 3:
        return AxisResult("effective_rank", False, detail={"reason": "need >=3 concepts"})
    D = np.stack(dirs)
    s = np.linalg.svd(D, compute_uv=False)
    pr = float((s.sum() ** 2) / (np.square(s).sum() + 1e-12))
    k = len(dirs)
    # structured if PR is meaningfully below k (shared structure) but above 1
    norm = (pr - 1) / (k - 1 + 1e-9)          # 0 = collapsed, 1 = fully independent/noise
    score = _clamp01(1.0 - abs(norm - 0.5) * 2)  # peaks at mid-rank structure
    return AxisResult("effective_rank", True, raw=pr, null=float(k), margin=float(k - pr),
                      score=score, detail={"n_dirs": k, "norm_rank": norm})


# --------------------------------------------------------------------------
# additive index (P106)
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "decodability": 1.0, "sufficiency": 1.0, "selectivity": 1.0,
    "content_adjacency": 1.0, "faithful_rank": 1.0, "stability": 1.0,
    "dose_response": 0.75, "graded_encoding": 0.75, "abstraction": 1.0,
    "effective_rank": 0.5,
}


@dataclass
class EmergenceReport:
    axes: List[AxisResult]
    index: float                 # weighted mean of available axis scores, [0,1]
    n_available: int
    confident: bool              # at least decodability present AND >0 margin
    label: str                   # human verdict tier

    def table(self) -> str:
        rows = [f"{'axis':18s} {'avail':5s} {'raw':>7s} {'null':>7s} {'margin':>7s} {'score':>6s}"]
        for a in self.axes:
            rows.append(f"{a.name:18s} {str(a.available):5s} {a.raw:7.3f} {a.null:7.3f} "
                        f"{a.margin:7.3f} {a.score:6.3f}")
        rows.append(f"\nEMERGENCE INDEX = {self.index:.3f}  ({self.n_available} axes)  "
                    f"-> {self.label}")
        return "\n".join(rows)


def emergence_index(axes: List[AxisResult], weights: Optional[Dict[str, float]] = None) -> EmergenceReport:
    weights = weights or DEFAULT_WEIGHTS
    avail = [a for a in axes if a.available and a.score == a.score]
    if not avail:
        return EmergenceReport(axes, float("nan"), 0, False, "no axes available")
    wsum = sum(weights.get(a.name, 1.0) for a in avail)
    idx = sum(weights.get(a.name, 1.0) * a.score for a in avail) / (wsum or 1.0)
    deco = next((a for a in avail if a.name == "decodability"), None)
    confident = bool(deco and deco.margin == deco.margin and deco.margin > 0)
    if not confident:
        label = "no signal above the noise floor (not emerging / pre-emergence)"
    elif idx < 0.25:
        label = "faint: a concept is decodable but little structure yet"
    elif idx < 0.5:
        label = "early emergence: real, selective, partly stable signal"
    elif idx < 0.75:
        label = "established: stable, selective, generalizing representation"
    else:
        label = "strong: broad structured interpretable capability"
    return EmergenceReport(axes, idx, len(avail), confident, label)


def run_emergence(X, texts, concepts, *, pairs=None, recon=None, true_dirs=None,
                  decoy_dirs=None, prevalence=None, graded_values=None,
                  group_ids=None, weights=None, seed=0) -> EmergenceReport:
    """Compute every axis whose inputs are available, then the additive index.
    Only X+texts+concepts are required; optional args unlock the richer axes."""
    axes = [
        axis_decodability(X, texts, concepts, seed=seed),
        axis_sufficiency(X, texts, concepts, seed=seed),
        axis_selectivity(X, texts, concepts, seed=seed),
        axis_content_adjacency(pairs, seed=seed),
        axis_faithful_rank(recon, true_dirs, decoy_dirs),
        axis_stability(X, texts, concepts, seed=seed),
        axis_dose_response(X, texts, concepts, prevalence=prevalence, seed=seed),
        axis_graded_encoding(X, graded_values, seed=seed),
        axis_abstraction(X, texts, concepts, group_ids, seed=seed),
        axis_effective_rank(X, texts, concepts, seed=seed),
    ]
    return emergence_index(axes, weights)
