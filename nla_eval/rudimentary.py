"""Floor tooling for evaluating RUDIMENTARY NLAs.

Both external reviews (Hermes nemotron-3-ultra, Antigravity gemini-3.5-flash)
converged on the same point: on a weak/tiny NLA the AV is degenerate and the AR
near-random, so end-to-end metrics collapse and high probe AUCs can be overfit.
The fix is a small set of probe-only / activation-space floor checks that still
yield signal — and that answer the two existential questions FIRST:

    1. Does a usable bottleneck exist at all?           -> bottleneck_exists (plan P081)
    2. Is the AR/AV even conditioned on the input?      -> identity_leakage    (plan P077)

Plus the floor substitute for a broken verbalizer:

    3. Forced-probe verbalizer (read the bottleneck via a probe, emit a fixed
       template) so the pipeline runs even with an empty/degenerate AV  (plan P046)

Everything here is CPU + numpy/sklearn; no GPU and no working AV required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, List, Callable

from .bottleneck_probe import probe_concept, label_by_keyword
from .matching import content_words


# ---------------------------------------------------------------------------
# P081 — does a usable bottleneck exist?
# ---------------------------------------------------------------------------

@dataclass
class ExistenceResult:
    concept: str
    probe_auc: float
    control_auc: float
    control_std: float
    margin: float          # probe_auc - (control_auc + 2*std)
    exists: bool           # margin > 0
    n_pos: int
    n_neg: int


def bottleneck_exists(activations, texts: Sequence[str], concept: str,
                      aliases: Sequence[str] = (), n_permutations: int = 10,
                      seed: int = 0) -> ExistenceResult:
    """Probe the activations for `concept` and judge the result against its own
    label-permutation floor. `exists` is True only when the real AUC clears the
    noise floor by >2 std — the conservative "is there any encodable signal here"
    test that doesn't get fooled by small-n overfit."""
    y = label_by_keyword(texts, concept, aliases)
    pr = probe_concept(activations, y, n_permutations=n_permutations, seed=seed)
    floor = (pr["control_auc"] + 2 * pr["control_std"]) if pr["control_auc"] == pr["control_auc"] else float("nan")
    margin = pr["auc"] - floor if pr["auc"] == pr["auc"] else float("nan")
    return ExistenceResult(
        concept=concept, probe_auc=pr["auc"], control_auc=pr["control_auc"],
        control_std=pr["control_std"], margin=margin,
        exists=bool(pr["signal"]), n_pos=pr["n_pos"], n_neg=pr["n_neg"],
    )


# ---------------------------------------------------------------------------
# P077 — is the AR/AV conditioned on the input, or emitting a generic constant?
# ---------------------------------------------------------------------------

@dataclass
class IdentityResult:
    n: int
    mean_pairwise: float       # mean pairwise similarity of outputs (cos or jaccard)
    distinct_ratio: float      # for text: unique outputs / n
    conditional: bool          # outputs vary with input (not a near-constant)
    mode: str                  # "vector" | "text"


def identity_leakage_vectors(vectors, threshold: float = 0.97) -> IdentityResult:
    """Cross-row cosine of reconstructed activations. If the AR emits ~the same
    vector regardless of input (mean pairwise cosine ~1), it is NOT conditional —
    the most fundamental rudimentary-AR failure. `conditional` is False when the
    mean off-diagonal cosine exceeds `threshold`."""
    import numpy as np

    V = np.asarray(vectors, dtype="float32")
    n = len(V)
    if n < 2:
        return IdentityResult(n, float("nan"), float("nan"), True, "vector")
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    S = Vn @ Vn.T
    off = S[~np.eye(n, dtype=bool)]
    mean_pair = float(off.mean())
    return IdentityResult(n, mean_pair, float("nan"), mean_pair < threshold, "vector")


def identity_leakage_text(texts: Sequence[str], threshold: float = 0.9) -> IdentityResult:
    """Text analog for AV verbalizations: are the outputs distinct across inputs,
    or is the AV emitting a near-constant boilerplate string? Uses content-word
    Jaccard. `conditional` is False when outputs are near-identical."""
    sets = [set(content_words(t)) for t in texts]
    n = len(sets)
    if n < 2:
        return IdentityResult(n, float("nan"), float("nan"), True, "text")
    distinct = len({" ".join(sorted(s)) for s in sets}) / n
    sims, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            u = len(a | b)
            sims += (len(a & b) / u) if u else 1.0
            cnt += 1
    mean_pair = sims / cnt if cnt else float("nan")
    return IdentityResult(n, mean_pair, distinct, mean_pair < threshold, "text")


# ---------------------------------------------------------------------------
# P046 — forced-probe verbalizer (floor substitute for a degenerate AV)
# ---------------------------------------------------------------------------

def fit_forced_verbalizer(activations, texts: Sequence[str], concepts: Sequence[str],
                          seed: int = 0):
    """Train one linear probe per concept on the activations, then return a
    function vec -> verbalization that emits a fixed template listing the concepts
    the probe reads as present. This BYPASSES the (possibly empty) trained AV, so
    the rest of the harness can run on a rudimentary NLA. Returns (verbalize_fn,
    info) where info maps concept -> whether its probe cleared the noise floor."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(activations, dtype="float32")
    models = {}
    info = {}
    for c in concepts:
        y = np.asarray(label_by_keyword(texts, c), dtype=int)
        pr = probe_concept(X, y, seed=seed)
        info[c] = {"signal": pr["signal"], "auc": pr["auc"],
                   "control_auc": pr["control_auc"]}
        if pr["n_pos"] < 3 or pr["n_neg"] < 3:
            continue
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
        models[c] = (sc, clf)

    def verbalize(vec) -> str:
        v = np.asarray(vec, dtype="float32").reshape(1, -1)
        present = []
        for c, (sc, clf) in models.items():
            if clf.predict_proba(sc.transform(v))[0, 1] >= 0.5:
                present.append(c)
        # fixed template the ensemble matcher can read deterministically
        return "Concepts present: " + (", ".join(present) if present else "none") + "."

    return verbalize, info
