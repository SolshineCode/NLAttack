"""Bottleneck-probe experiment on the local Gemma-4-E2B NLA (review note P0 #1).

Reads the deception corpus parquet READ-ONLY (activation_vector @ L23 =
the NLA bottleneck; detokenized_text_truncated = ground-truth source) and asks,
for each concept: can a linear probe read the concept off the bottleneck?
That is ground-truth concept presence, independent of the AV verbalizer + matcher.

CPU only (sklearn) — does NOT touch the GPU, so it coexists with other GPU jobs.

The corpus parquet lives in your separate NLA-training checkout. Point NLA_DATA_ROOT
at it (or pass --parquet), then run with a venv that has pyarrow/sklearn/numpy:
  NLA_DATA_ROOT=/path/to/nla-training-repo python experiments/local_gemma_e2b/probe_bottleneck.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# make nla_eval importable regardless of cwd
NLATTACK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NLATTACK))

from nla_eval.bottleneck_probe import (
    auto_select_concepts, run_probe_suite, save_csv,
)

# Deception corpus parquet (read-only). L23 activations + source text.
# Produced by the NLA-training repo, kept separate from this harness.
_DATA_ROOT = Path(os.environ.get("NLA_DATA_ROOT", "path/to/nla-training-repo"))
PARQUET = _DATA_ROOT / "experiments/v8_nla_local/data/stage0/gemma4_deception_chunk1.parquet"
OUT_DIR = NLATTACK / "results" / "local_gemma_e2b"

# Candidate concepts plausibly present in a business/deception scenario corpus.
# auto_select keeps only those with balanced prevalence (probeable signal).
CANDIDATES = [
    "company", "money", "report", "earnings", "customer", "decision", "risk",
    "manager", "employee", "financial", "quarter", "product", "contract",
    "client", "team", "market", "profit", "data", "email", "price", "sales",
    "investor", "deadline", "quality", "safety", "budget", "project", "meeting",
]


# Broader generic concepts for diverse corpora (news / web / arxiv).
GENERIC = [
    "people", "government", "science", "health", "water", "history", "music",
    "war", "technology", "animal", "school", "energy", "city", "country",
    "computer", "disease", "child", "police", "court", "election", "research",
    "climate", "food", "money", "company", "student", "family", "game",
]


def main():
    import argparse
    import numpy as np
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument("--generic", action="store_true",
                    help="use the generic concept list (for news/web/arxiv corpora)")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    parquet = args.parquet
    candidates = (CANDIDATES + GENERIC) if args.generic else CANDIDATES

    print(f"[reading {parquet.name} ...]")
    t = pq.read_table(parquet, columns=["detokenized_text_truncated",
                                        "activation_vector", "activation_layer"])
    texts = t.column("detokenized_text_truncated").to_pylist()
    acts = np.array(t.column("activation_vector").to_pylist(), dtype="float32")
    layer = t.column("activation_layer").to_pylist()[0]
    print(f"  rows={len(texts)}  activation dim={acts.shape[1]}  layer={layer}")

    concepts = auto_select_concepts(texts, candidates, min_prev=0.15, max_prev=0.85)
    print(f"  probeable concepts ({len(concepts)}): {concepts}")

    print("[training linear probes on the bottleneck (5-fold CV)...]")
    results = run_probe_suite(acts, texts, concepts)
    results.sort(key=lambda r: (r.probe_auc if r.probe_auc == r.probe_auc else -1),
                 reverse=True)

    print(f"\n{'concept':14s} {'auc':>6s} {'bal_acc':>8s} {'baseline':>9s} {'n_pos':>6s}")
    for r in results:
        print(f"{r.concept:14s} {r.probe_auc:6.3f} {r.probe_acc:8.3f} "
              f"{r.baseline:9.3f} {r.n_pos:6d}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"probe_bottleneck_results{args.out_suffix}.csv"
    save_csv(results, str(out))
    # headline: mean AUC over concepts the bottleneck genuinely encodes
    valid = [r.probe_auc for r in results if r.probe_auc == r.probe_auc]
    print(f"\nmean probe AUC = {sum(valid)/len(valid):.3f} over {len(valid)} concepts")
    print(f"wrote {out}")
    print("\nNEXT (GPU, deferred): run AV verbalization on the same rows, compute "
          "av_matcher_acc per concept, then probe_acc - av_matcher_acc = the "
          "verbalizer+matcher loss the review asked us to subtract.")


if __name__ == "__main__":
    main()
