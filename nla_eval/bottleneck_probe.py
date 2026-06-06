"""Bottleneck probes — ground truth concept presence (Hermes review P0 #1).

The harness's normal signal (does concept C appear in the AV verbalization) is
confounded: the AV verbalizer + the matcher are two extra lossy filters. To know
whether the *NLA bottleneck itself* carries concept C, probe the bottleneck
representation directly.

For an activation→text→activation NLA, the bottleneck is the **activation vector**
fed to the verbalizer (= the AR's reconstruction target). If a linear probe can
read concept C off that vector, the concept IS present at the bottleneck —
independent of whether the verbalizer lexicalized it or the matcher caught it.

This gives the disentanglement Hermes asked for:

    probe_auc(C)            — concept present at the bottleneck? (ground truth)
    av_matcher_acc(C)       — did the end-to-end AV+matcher report it?
    gap = probe_auc - av_matcher_acc   — verbalizer + matcher loss

A high probe_auc with low av_matcher_acc = the NLA kept it but the verbalizer/
matcher lost it (NOT an NLA dropout). Only when probe_auc is ALSO low is the
bottleneck genuinely dropping the concept.

Dependency: scikit-learn (lazy-imported). The probe runs on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Sequence
import csv


@dataclass
class ProbeResult:
    concept: str
    n_pos: int
    n_neg: int
    probe_auc: float          # cross-val ROC-AUC of linear probe on activations
    probe_acc: float          # cross-val balanced accuracy
    baseline: float           # majority-class accuracy (floor)
    av_matcher_acc: Optional[float] = None   # filled when AV verbalizations exist
    gap: Optional[float] = None              # probe_acc - av_matcher_acc

    def flat(self) -> Dict:
        return asdict(self)


def label_by_keyword(texts: Sequence[str], concept: str, aliases: Sequence[str] = ()) -> List[int]:
    """Ground-truth concept presence from the INPUT text (not the NLA output).
    Concept is present if the concept head or any alias appears (case-insensitive)."""
    terms = [concept.lower()] + [a.lower() for a in aliases]
    # also match the concept's last word (head) to catch multiword concepts
    terms.append(concept.lower().split()[-1])
    out = []
    for t in texts:
        tl = (t or "").lower()
        out.append(1 if any(term in tl for term in terms if term) else 0)
    return out


def auto_select_concepts(texts: Sequence[str], candidates: Sequence[str],
                         min_prev: float = 0.15, max_prev: float = 0.85) -> List[str]:
    """Keep candidate concepts whose prevalence is balanced enough to probe."""
    n = len(texts) or 1
    keep = []
    for c in candidates:
        prev = sum(label_by_keyword(texts, c)) / n
        if min_prev <= prev <= max_prev:
            keep.append(c)
    return keep


def probe_concept(activations, labels, n_splits: int = 5, seed: int = 0) -> Dict:
    """Cross-validated linear probe (logistic regression) on activations → label.
    Returns dict with auc, balanced_acc, baseline. CPU, scikit-learn."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(activations, dtype="float32")
    y = np.asarray(labels, dtype=int)
    pos, neg = int(y.sum()), int((1 - y).sum())
    baseline = max(pos, neg) / len(y)
    if pos < n_splits or neg < n_splits:
        return {"auc": float("nan"), "balanced_acc": float("nan"),
                "baseline": baseline, "n_pos": pos, "n_neg": neg}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, accs = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        accs.append(balanced_accuracy_score(y[te], (p >= 0.5).astype(int)))
    return {"auc": float(np.mean(aucs)), "balanced_acc": float(np.mean(accs)),
            "baseline": baseline, "n_pos": pos, "n_neg": neg}


def run_probe_suite(activations, texts: Sequence[str], concepts: Sequence[str],
                    aliases: Optional[Dict[str, Sequence[str]]] = None,
                    seed: int = 0) -> List[ProbeResult]:
    """For each concept: label from text, probe the bottleneck activations.
    `activations` is an (n, d) array aligned with `texts`."""
    aliases = aliases or {}
    results = []
    for c in concepts:
        y = label_by_keyword(texts, c, aliases.get(c, ()))
        pr = probe_concept(activations, y, seed=seed)
        results.append(ProbeResult(
            concept=c, n_pos=pr["n_pos"], n_neg=pr["n_neg"],
            probe_auc=pr["auc"], probe_acc=pr["balanced_acc"], baseline=pr["baseline"],
        ))
    return results


def attach_av_accuracy(results: List[ProbeResult],
                       av_matcher_acc: Dict[str, float]) -> List[ProbeResult]:
    """Fold in end-to-end AV+matcher accuracy (computed when GPU verbalizations
    exist) and compute the probe−AV gap = verbalizer+matcher loss."""
    for r in results:
        a = av_matcher_acc.get(r.concept)
        if a is not None:
            r.av_matcher_acc = a
            r.gap = (r.probe_acc - a) if r.probe_acc == r.probe_acc else None
    return results


def save_csv(results: List[ProbeResult], path: str):
    rows = [r.flat() for r in results]
    keys = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
