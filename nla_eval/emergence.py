"""Interpretability-capability measurement for weak/tiny NLAs — with a
longitudinal EMERGENCE criterion on top.

DESIGN NOTE (revised after expert critique, Hermes nemotron-3-ultra 2026-06-05,
and grounded in the probing/emergence literature — see
`docs/LITERATURE.md`):

The first version summed axes into a weighted-mean "Emergence Index." That was
wrong in two ways the critique (and our own data) exposed:
  * The weighted mean MASKS failure: our cross-domain run had abstraction AUC 0.52
    (chance) yet scored 0.49 "early" — a measure that calls a model that cannot
    generalize "emerging" is invalid. Aggregation is now a **hierarchical
    conjunctive tier** (phase-transition framing, cf. Schaeffer et al. 2304.15004),
    not an average. We report the highest tier *fully* passed.
  * Four axes (decodability/sufficiency/selectivity/stability) shared the same
    probe-AUC numerator -> probe AUC got ~4x weight. Collapsed into ONE
    `decoding_quality = AUC - max(permutation_floor, BoW_AUC, length_AUC)`
    (Hewitt & Liang control-task selectivity, 1909.03368); sufficiency/selectivity
    are kept as DIAGNOSTICS, not summed.

Also: a single snapshot is "capability," not "emergence." `emergence_from_curve`
adds the dynamical criterion (monotone rise across checkpoints + first checkpoint
to pass Tier 1). And a `min_concepts` gate + concept-bootstrap CI guard the n=2-4
statistical-invalidity failure mode. With keyword labels + no minimal pairs / AR
recon, the honest ceiling is Tier 1 — by design.

CPU + numpy/sklearn/scipy. AR/AV-dependent axes are skipped when inputs absent.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence, Optional, Dict, List

import numpy as np

from .bottleneck_probe import _cv_auc_acc, probe_concept, label_by_keyword
from .matching import content_words


@dataclass
class AxisResult:
    name: str
    available: bool
    raw: float = float("nan")
    null: float = float("nan")
    margin: float = float("nan")
    score: float = float("nan")
    role: str = "axis"            # "axis" | "diagnostic"
    detail: dict = field(default_factory=dict)

    def flat(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "detail"}


def _clamp01(x: float) -> float:
    return float("nan") if x != x else float(max(0.0, min(1.0, x)))


def _trivial_input_features(texts, dim=256):
    feats = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in content_words(t):
            feats[i, hash(w) % dim] += 1.0
    return feats


def _confound_features(texts):
    return np.array([[len(t or ""), len(content_words(t))] for t in texts], dtype="float32")


def _direction(X, y, seed, subsample=None):
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


def _mean_pairwise_cos(dirs):
    if len(dirs) < 2:
        return float("nan")
    s, c = 0.0, 0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            s += abs(float(np.dot(dirs[i], dirs[j]))); c += 1
    return s / c if c else float("nan")


# ===================== the gate axis (collapsed AUC) =====================

def axis_decoding_quality(X, texts, concepts, n_perm=8, seed=0) -> AxisResult:
    """ONE decoding axis = probe AUC minus the STRONGEST of three nulls
    (label-permutation floor, bag-of-words-of-input, length/density confound).
    Folds in what used to be three correlated axes; also stashes the sufficiency
    and selectivity deltas as diagnostics. This is the Tier-1 gate."""
    X = np.asarray(X, "float32")
    triv = _trivial_input_features(texts)
    conf = _confound_features(texts)
    margins, aucs, suf, sel = [], [], [], []
    for c in concepts:
        y = np.asarray(label_by_keyword(texts, c), dtype=int)
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        pr = probe_concept(X, y, n_permutations=n_perm, seed=seed)
        if pr["auc"] != pr["auc"]:
            continue
        perm = pr["control_auc"] + 2 * pr["control_std"]
        try:
            bow, _ = _cv_auc_acc(triv, y, 5, seed)
            length, _ = _cv_auc_acc(conf, y, 5, seed)
        except Exception:
            bow, length = 0.5, 0.5
        margins.append(pr["auc"] - max(perm, bow, length))
        aucs.append(pr["auc"]); suf.append(pr["auc"] - bow); sel.append(pr["auc"] - length)
    if not margins:
        return AxisResult("decoding_quality", False)
    m = float(np.mean(margins))
    return AxisResult("decoding_quality", True, raw=float(np.mean(aucs)),
                      null=float(np.mean(aucs) - m), margin=m, score=_clamp01(m / 0.3),
                      detail={"n_concepts": len(margins),
                              "sufficiency": float(np.mean(suf)),
                              "selectivity": float(np.mean(sel))})


# ===================== other (genuinely distinct) axes =====================

def axis_stability(X, texts, concepts, n_seeds=5, seed=0) -> AxisResult:
    X = np.asarray(X, "float32"); rng = np.random.default_rng(seed)
    real, shuf = [], []
    for c in concepts:
        y = np.asarray(label_by_keyword(texts, c), dtype=int)
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        real.append(_mean_pairwise_cos([_direction(X, y, s, subsample=0.7) for s in range(n_seeds)]))
        ys = y.copy(); sd = []
        for s in range(n_seeds):
            rng.shuffle(ys)
            if ys.sum() >= 2 and (1 - ys).sum() >= 2:
                sd.append(_direction(X, ys, s, subsample=0.7))
        if len(sd) >= 2:
            shuf.append(_mean_pairwise_cos(sd))
    if not real:
        return AxisResult("stability", False)
    rc = float(np.mean(real)); sc_ = float(np.mean(shuf)) if shuf else 0.0
    return AxisResult("stability", True, raw=rc, null=sc_, margin=rc - sc_,
                      score=_clamp01((rc - sc_) / (1 - sc_ + 1e-9)),
                      detail={"n_concepts": len(real)})


def axis_abstraction(X, texts, concepts, group_ids, seed=0) -> AxisResult:
    if group_ids is None:
        return AxisResult("abstraction", False)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    X = np.asarray(X, "float32"); g = np.asarray(group_ids)
    uniq, counts = np.unique(g, return_counts=True)
    if len(uniq) < 2:
        return AxisResult("abstraction", False)
    A, B = uniq[np.argsort(-counts)[:2]]; aucs = []
    for c in concepts:
        y = np.asarray(label_by_keyword(texts, c), dtype=int)
        ia, ib = (g == A), (g == B)
        if y[ia].sum() < 5 or (1 - y[ia]).sum() < 5 or len(np.unique(y[ib])) < 2:
            continue
        sc = StandardScaler().fit(X[ia])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[ia]), y[ia])
        try:
            aucs.append(roc_auc_score(y[ib], clf.predict_proba(sc.transform(X[ib]))[:, 1]))
        except Exception:
            pass
    if not aucs:
        return AxisResult("abstraction", False)
    a = float(np.mean(aucs))
    return AxisResult("abstraction", True, raw=a, null=0.5, margin=a - 0.5,
                      score=_clamp01((a - 0.5) * 2), detail={"n_concepts": len(aucs)})


def axis_content_adjacency(pairs) -> AxisResult:
    if not pairs:
        return AxisResult("content_adjacency", False)
    near, far = [], []
    for a, b, neg in pairs:
        a, b, neg = (np.asarray(z, "float32") for z in (a, b, neg))
        near.append(float(np.linalg.norm(a - b))); far.append(float(np.linalg.norm(a - neg)))
    near, far = np.array(near), np.array(far)
    pooled = np.sqrt((near.var() + far.var()) / 2) + 1e-9
    d = float((far.mean() - near.mean()) / pooled)
    return AxisResult("content_adjacency", True, raw=d, null=0.0, margin=d,
                      score=_clamp01(d / 1.0), detail={"n_pairs": len(pairs)})


def axis_faithful_rank(recon, true_dirs, decoy_dirs) -> AxisResult:
    if recon is None or true_dirs is None or decoy_dirs is None:
        return AxisResult("faithful_rank", False)
    R = np.asarray(recon, "float32"); R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    t = np.asarray(true_dirs, "float32"); t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    d = np.asarray(decoy_dirs, "float32"); d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    wins = float(((R * t).sum(1) > (R * d).sum(1)).mean())
    return AxisResult("faithful_rank", True, raw=wins, null=0.5, margin=wins - 0.5,
                      score=_clamp01((wins - 0.5) * 2))


def axis_dose_response(X, texts, concepts, prevalence=None, seed=0) -> AxisResult:
    from scipy.stats import spearmanr
    prevs, aucs = [], []; n = len(texts) or 1
    for c in concepts:
        y = np.asarray(label_by_keyword(texts, c), dtype=int)
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        pr = probe_concept(X, y, n_permutations=0, seed=seed)
        if pr["auc"] == pr["auc"]:
            prevs.append(prevalence[c] if prevalence else y.sum() / n); aucs.append(pr["auc"])
    if len(aucs) < 6:  # raised: <6 is meaningless (critique #3)
        return AxisResult("dose_response", False, detail={"reason": "need >=6 concepts"})
    rho = float(spearmanr(prevs, aucs).statistic)
    return AxisResult("dose_response", True, raw=rho, null=0.0, margin=rho,
                      score=_clamp01((rho + 1) / 2), detail={"n_concepts": len(aucs)})


def axis_graded_encoding(X, graded_values, seed=0) -> AxisResult:
    if graded_values is None:
        return AxisResult("graded_encoding", False)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(np.asarray(X, "float32"))
    yv = np.asarray(graded_values, "float32")
    if np.std(yv) < 1e-9 or len(yv) < 10:
        return AxisResult("graded_encoding", False)
    r2 = float(np.mean(cross_val_score(Ridge(1.0), X, yv, cv=5, scoring="r2")))
    rng = np.random.default_rng(seed); ys = yv.copy(); rng.shuffle(ys)
    r2n = float(np.mean(cross_val_score(Ridge(1.0), X, ys, cv=5, scoring="r2")))
    return AxisResult("graded_encoding", True, raw=r2, null=r2n, margin=r2 - r2n,
                      score=_clamp01(r2 - r2n))


def axis_effective_rank(X, texts, concepts, seed=0) -> AxisResult:
    X = np.asarray(X, "float32"); dirs = []
    for c in concepts:
        y = np.asarray(label_by_keyword(texts, c), dtype=int)
        if y.sum() >= 5 and (1 - y).sum() >= 5:
            dirs.append(_direction(X, y, seed))
    if len(dirs) < 6:  # PR needs k >> 1 (critique #3)
        return AxisResult("effective_rank", False, detail={"reason": "need >=6 concepts"})
    s = np.linalg.svd(np.stack(dirs), compute_uv=False)
    pr = float((s.sum() ** 2) / (np.square(s).sum() + 1e-12)); k = len(dirs)
    norm = (pr - 1) / (k - 1 + 1e-9)
    return AxisResult("effective_rank", True, raw=pr, null=float(k), margin=float(k - pr),
                      score=_clamp01(1.0 - abs(norm - 0.5) * 2),
                      detail={"n_dirs": k, "norm_rank": norm})


# ===================== hierarchical tier verdict =====================

@dataclass
class CapabilityReport:
    axes: List[AxisResult]
    tier: int                    # 0..3, or -1 = insufficient coverage
    tier_label: str
    n_concepts: int
    profile: float               # SECONDARY diagnostic mean (NOT the verdict)
    profile_ci: Optional[tuple] = None
    reasons: List[str] = field(default_factory=list)

    def get(self, name):
        return next((a for a in self.axes if a.name == name), None)

    def table(self) -> str:
        rows = [f"{'axis':18s} {'role':10s} {'avail':5s} {'raw':>7s} {'margin':>7s} {'score':>6s}"]
        for a in self.axes:
            rows.append(f"{a.name:18s} {a.role:10s} {str(a.available):5s} {a.raw:7.3f} "
                        f"{a.margin:7.3f} {a.score:6.3f}")
        ci = f"  CI95={self.profile_ci}" if self.profile_ci else ""
        rows.append(f"\nVERDICT: Tier {self.tier} — {self.tier_label}  "
                    f"(n_concepts={self.n_concepts})")
        rows.append(f"diagnostic profile score = {self.profile:.3f}{ci}  "
                    f"(secondary; the TIER is the verdict)")
        if self.reasons:
            rows.append("why: " + "; ".join(self.reasons))
        return "\n".join(rows)


TIER_LABELS = {
    -1: "INSUFFICIENT CONCEPT COVERAGE — no verdict",
    0: "no signal above the noise floor (pre-emergence)",
    1: "capability present: decodable, stable" + " (+generalizing if testable)",
    2: "selective: + concept-specific over surface/confound",
    3: "established: + faithful reconstruction & fine adjacency",
}


def _pos(ax: Optional[AxisResult], thr=0.0):
    """available AND margin strictly above threshold (unavailable -> None = unknown)."""
    if ax is None or not ax.available or ax.margin != ax.margin:
        return None
    return ax.margin > thr


def capability_tier(axes: List[AxisResult], n_concepts: int, min_concepts: int = 8) -> CapabilityReport:
    by = {a.name: a for a in axes}
    reasons = []
    if n_concepts < min_concepts:
        return CapabilityReport(axes, -1, TIER_LABELS[-1], n_concepts, float("nan"),
                                reasons=[f"only {n_concepts} concepts (<{min_concepts}); "
                                         "every axis is statistically unreliable — use semantic "
                                         "labels to raise concept count"])
    dq = by.get("decoding_quality"); stab = by.get("stability"); ab = by.get("abstraction")
    suf = sel = None
    if dq and dq.available:
        suf = dq.detail.get("sufficiency"); sel = dq.detail.get("selectivity")
    fr = by.get("faithful_rank"); ca = by.get("content_adjacency")

    tier = 0
    # Tier 1: decodable AND stable AND (abstraction passes OR not testable)
    t1 = (_pos(dq) is True) and (_pos(stab) is True)
    if t1 and _pos(ab) is False:  # testable and FAILED -> block
        t1 = False; reasons.append("abstraction at chance (testable, failed) -> capped below Tier 1")
    if t1:
        tier = 1
        # Tier 2: concept-selective over surface + confound (needs semantic labels for sufficiency)
        if (sel is not None and sel > 0.05) and (suf is not None and suf > 0.05):
            tier = 2
            # Tier 3: faithful reconstruction + fine adjacency (need AR recon + minimal pairs)
            if (_pos(fr, 0.05) is True) and (_pos(ca, 0.2) is True):
                tier = 3
        else:
            if suf is not None and suf <= 0.05:
                reasons.append("sufficiency ~0 (keyword labels leak into BoW; use semantic labels) "
                               "-> Tier 2 not awarded")
    else:
        if _pos(dq) is not True:
            reasons.append("decoding_quality does not clear the strongest null")
        elif _pos(stab) is not True:
            reasons.append("probe direction unstable across resamples")

    # secondary diagnostic profile (distinct axes only; NOT the verdict)
    prof_axes = [by.get(n) for n in ("decoding_quality", "stability", "abstraction",
                                     "faithful_rank", "content_adjacency", "graded_encoding",
                                     "effective_rank")]
    scores = [a.score for a in prof_axes if a and a.available and a.score == a.score]
    profile = float(np.mean(scores)) if scores else float("nan")
    return CapabilityReport(axes, tier, TIER_LABELS[tier], n_concepts, profile, reasons=reasons)


def run_capability(X, texts, concepts, *, pairs=None, recon=None, true_dirs=None,
                   decoy_dirs=None, prevalence=None, graded_values=None, group_ids=None,
                   min_concepts=8, seed=0) -> CapabilityReport:
    """Snapshot capability verdict (hierarchical tier) for one activation set."""
    dq = axis_decoding_quality(X, texts, concepts, seed=seed)
    # diagnostics surfaced separately (not summed): pure decodability for reference
    axes = [
        dq,
        axis_stability(X, texts, concepts, seed=seed),
        axis_abstraction(X, texts, concepts, group_ids, seed=seed),
        axis_content_adjacency(pairs),
        axis_faithful_rank(recon, true_dirs, decoy_dirs),
        axis_dose_response(X, texts, concepts, prevalence=prevalence, seed=seed),
        axis_graded_encoding(X, graded_values, seed=seed),
        axis_effective_rank(X, texts, concepts, seed=seed),
    ]
    n_concepts = dq.detail.get("n_concepts", 0) if dq.available else 0
    return capability_tier(axes, n_concepts, min_concepts=min_concepts)


def bootstrap_profile(X, texts, concepts, n_boot=20, min_concepts=8, seed=0):
    """Concept-level bootstrap CI for the diagnostic profile (critique #3): with
    few concepts the CI is wide -> correctly 'unknown'. Returns (lo, hi)."""
    if len(concepts) < 3:
        return None
    rng = np.random.default_rng(seed); vals = []
    for b in range(n_boot):
        cs = list(rng.choice(concepts, size=len(concepts), replace=True))
        rep = run_capability(X, texts, cs, min_concepts=min_concepts, seed=seed + b)
        if rep.profile == rep.profile:
            vals.append(rep.profile)
    if len(vals) < 3:
        return None
    return (round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3))


# ===================== longitudinal EMERGENCE criterion =====================

@dataclass
class EmergenceVerdict:
    emerged: bool
    emergence_point: Optional[str]   # checkpoint label where Tier 1 first passes
    monotone_rise: bool
    detail: dict


def emergence_from_curve(curve: List[dict]) -> EmergenceVerdict:
    """A snapshot is CAPABILITY; emergence is a dynamical claim. `curve` is an
    ordered list of per-checkpoint dicts with keys 'label','tier','profile'.
    Emergence = the tier crosses from 0 to >=1 (the emergence point) AND the
    profile rises monotonically (non-decreasing) up to/through it. Mirrors the
    'is it a real transition or a metric mirage' question (Schaeffer 2304.15004)."""
    if not curve:
        return EmergenceVerdict(False, None, False, {"reason": "empty curve"})
    point = next((c["label"] for c in curve if c.get("tier", 0) >= 1), None)
    profs = [c.get("profile", float("nan")) for c in curve]
    clean = [p for p in profs if p == p]
    monotone = all(b >= a - 1e-6 for a, b in zip(clean, clean[1:])) if len(clean) >= 2 else False
    emerged = point is not None and monotone
    return EmergenceVerdict(emerged, point, monotone,
                            {"tiers": [c.get("tier") for c in curve],
                             "profiles": profs,
                             "labels": [c.get("label") for c in curve]})


# backward-compat alias
run_emergence = run_capability
