"""Emergence dashboard — is useful interpretability capability starting to emerge
in a weak/tiny NLA? Computes the additive emergence axes (each with its null
margin) and the composite Emergence Index, for one activation set or across a
sweep of checkpoints/layers (the emergence curve).

CPU only (numpy/sklearn/scipy). Needs activation parquets with columns
`detokenized_text_truncated` + `activation_vector` (the v8 stage0 format).

Single set:
  .venv-gemma4/Scripts/python.exe experiments/emergence_dashboard.py \
      --parquet .../data/stage0/gemma4_deception_chunk1.parquet

Checkpoint/condition sweep (emergence curve across several parquets):
  ... emergence_dashboard.py --sweep a.parquet b.parquet c.parquet --labels c50 c100 c200

Abstraction axis: pass several parquets via --sweep AND --pool to concatenate
them with per-source group ids (probe trains on one source, tests on another).
"""
from __future__ import annotations

import sys, json, argparse
from pathlib import Path

NLATTACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NLATTACK))

from nla_eval import emergence as E
from nla_eval.bottleneck_probe import auto_select_concepts
from experiments.local_gemma_e2b.probe_bottleneck import CANDIDATES, GENERIC  # reuse concept pools

OUT = NLATTACK / "results" / "emergence"


def load(parquet):
    import numpy as np, pyarrow.parquet as pq
    t = pq.read_table(parquet, columns=["detokenized_text_truncated", "activation_vector"])
    texts = t.column("detokenized_text_truncated").to_pylist()
    X = np.array(t.column("activation_vector").to_pylist(), dtype="float32")
    return texts, X


def one(parquet, generic, seed=0, group_ids=None, X=None, texts=None, min_concepts=8):
    if X is None:
        texts, X = load(parquet)
    cands = (CANDIDATES + GENERIC) if generic else CANDIDATES
    concepts = auto_select_concepts(texts, cands, min_prev=0.12, max_prev=0.88)
    rep = E.run_capability(X, texts, concepts, group_ids=group_ids,
                           min_concepts=min_concepts, seed=seed)
    rep.profile_ci = E.bootstrap_profile(X, texts, concepts, min_concepts=min_concepts, seed=seed)
    return rep, concepts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path)
    ap.add_argument("--sweep", type=Path, nargs="+", help="multiple parquets = emergence curve")
    ap.add_argument("--labels", nargs="+", help="labels for --sweep entries")
    ap.add_argument("--pool", action="store_true",
                    help="concat --sweep parquets with per-source group ids (enables abstraction axis)")
    ap.add_argument("--generic", action="store_true", help="use the broad concept pool (news/web)")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.pool and args.sweep:
        import numpy as np
        allX, alltext, groups = [], [], []
        for p in args.sweep:
            tx, X = load(p)
            allX.append(X); alltext += tx; groups += [p.stem] * len(tx)
        X = np.concatenate(allX); rep, concepts = one(None, args.generic, group_ids=groups, X=X, texts=alltext)
        print(f"[pooled {len(args.sweep)} sources, n={len(alltext)}, concepts={concepts}]\n")
        print(rep.table())
        _save(rep, OUT / f"emergence_pooled{args.out_suffix}")
        return

    if args.sweep:
        labels = args.labels or [p.stem for p in args.sweep]
        curve = []
        for lab, p in zip(labels, args.sweep):
            rep, concepts = one(p, args.generic)
            row = {"label": lab, "tier": rep.tier, "profile": rep.profile,
                   "verdict": rep.tier_label}
            for a in rep.axes:
                row[a.name] = a.score
            curve.append(row)
            print(f"== {lab} ==  Tier {rep.tier} ({rep.tier_label})  profile={rep.profile:.3f}")
            print(rep.table()); print()
        verdict = E.emergence_from_curve(curve)
        (OUT / f"emergence_curve{args.out_suffix}.json").write_text(
            json.dumps({"curve": curve, "emergence": E.asdict_safe(verdict)
                        if hasattr(E, "asdict_safe") else verdict.__dict__}, indent=2, default=str))
        print("\nEMERGENCE CURVE (tier / profile by checkpoint):")
        for r in curve:
            bar = "#" * int(max(0, r["profile"]) * 40 if r["profile"] == r["profile"] else 0)
            print(f"  {r['label']:10s} T{r['tier']} {r['profile']:.3f} |{bar}")
        print(f"\nEMERGENCE VERDICT: emerged={verdict.emerged}  point={verdict.emergence_point}  "
              f"monotone_rise={verdict.monotone_rise}")
        print(f"wrote {OUT/('emergence_curve'+args.out_suffix+'.json')}")
        return

    if not args.parquet:
        ap.error("give --parquet or --sweep")
    rep, concepts = one(args.parquet, args.generic)
    print(f"[{args.parquet.name}  concepts={concepts}]\n")
    print(rep.table())
    _save(rep, OUT / f"emergence_{args.parquet.stem}{args.out_suffix}")


def _save(rep, stem: Path):
    import csv
    with open(str(stem) + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rep.axes[0].flat().keys()))
        w.writeheader()
        for a in rep.axes:
            w.writerow(a.flat())
    Path(str(stem) + ".json").write_text(json.dumps(
        {"tier": rep.tier, "tier_label": rep.tier_label, "profile": rep.profile,
         "profile_ci": rep.profile_ci, "n_concepts": rep.n_concepts, "reasons": rep.reasons,
         "axes": [a.flat() for a in rep.axes]}, indent=2))
    print(f"\nwrote {stem}.csv / .json")


if __name__ == "__main__":
    main()
