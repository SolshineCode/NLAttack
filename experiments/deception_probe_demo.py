"""Demo: deception-monitoring bottleneck probe (Family M / P113) on REAL
same-prompt honest-vs-deceptive behavioral-split activations from a companion
NLA-training repo (the activation .pt is not bundled here). CPU only.

Reproduces the source repo's ~0.87 balanced-accuracy baseline through NLAttack's
own permutation-null-controlled probe — validating the deception family on real
data. The verbalizer side (P114 deception_discrimination) additionally needs an AV.

The activation tensor lives in your separate NLA-training checkout. Point
NLA_DATA_ROOT at it (or edit PT below), then run with a venv that has torch + sklearn:
  NLA_DATA_ROOT=/path/to/nla-training-repo python experiments/deception_probe_demo.py
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path

NLATTACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NLATTACK))
from nla_eval import deception as D

# same-prompt honest-vs-deceptive behavioral split (model's own choice, judge-labeled).
# Activations are produced by the NLA-training repo and kept separate from this harness.
_DATA_ROOT = Path(os.environ.get("NLA_DATA_ROOT", "path/to/nla-training-repo"))
PT = _DATA_ROOT / "experiments/v3_behavioral_sampling/results/v3_layer12_activations.pt"
OUT = NLATTACK / "results" / "deception"


def main():
    import torch, numpy as np, warnings
    warnings.filterwarnings("ignore")
    d = torch.load(PT, map_location="cpu", weights_only=False)
    X = d["X"].float().numpy(); y = d["y"].long().numpy()
    print(f"behavioral split: n={len(y)} deceptive={int(y.sum())} honest={int((1-y).sum())} d={X.shape[1]}")
    r = D.deception_probe(X, y, n_permutations=20)
    print(f"deception_probe: AUROC={r.metric:.3f} balanced_acc={r.detail['balanced_acc']:.3f} "
          f"shuffle-null={r.null:.3f} signal={r.signal}")
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {**r.flat(),
           "source": "NLA-training repo: v3_behavioral_sampling/v3_layer12_activations.pt (nanochat-d32, L12)",
           "integrity": "same-prompt behavioral split, model's own choice, judge-labeled (PRIMARY, not control)",
           "note": "bottleneck side; verbalizer side (deception_discrimination) needs an AV"}
    (OUT / "deception_probe_nanochat_d32_L12.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"wrote {OUT/'deception_probe_nanochat_d32_L12.json'}")


if __name__ == "__main__":
    main()
