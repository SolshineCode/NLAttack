"""AV-side of the bottleneck disentanglement (GPU; run when the 4GB card is free).

Completes Hermes P0 #1: for the same rows the bottleneck probe used, run the AV
verbalizer on each activation, score whether the matcher finds each concept in
the verbalization (av_matcher_acc), then:

    gap = probe_acc - av_matcher_acc   = verbalizer + matcher loss

A concept with high probe_acc but low av_matcher_acc was KEPT by the NLA but lost
by the verbalizer/matcher — NOT an NLA dropout. This is the number that converts
NLAttack's AV-text observations into NLA-bottleneck claims.

Run (needs GPU free; uses the deception repo's venv for transformers/peft/bnb):
  cd C:/Users/caleb/deception-nanochat-sae-research
  KMP_DUPLICATE_LIB_OK=TRUE .venv-gemma4/Scripts/python.exe \
    C:/Users/caleb/nla-eval-harness/experiments/local_gemma_e2b/verbalize_av.py \
    --av-checkpoint experiments/v8_nla_local/checkpoints/av_v0_1_aux_readout/block_to000100/final \
    --limit 300
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

NLATTACK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NLATTACK))

from nla_eval.local_gemma_e2b import LocalGemmaE2BNLA
from nla_eval.matching import EnsembleMatcher
from nla_eval.bottleneck_probe import (
    auto_select_concepts, run_probe_suite, attach_av_accuracy, label_by_keyword, save_csv,
)

PARQUET = Path(
    r"C:\Users\caleb\deception-nanochat-sae-research\experiments\v8_nla_local"
    r"\data\stage0\gemma4_deception_chunk1.parquet"
)
OUT = NLATTACK / "results" / "local_gemma_e2b"
CANDIDATES = ["report", "team", "company", "money", "earnings", "customer",
              "decision", "risk", "financial", "product", "client", "market"]


def main():
    import numpy as np
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--av-checkpoint", required=True)
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    t = pq.read_table(args.parquet,
                      columns=["detokenized_text_truncated", "activation_vector"])
    texts = t.column("detokenized_text_truncated").to_pylist()[:args.limit]
    acts = np.array(t.column("activation_vector").to_pylist(), dtype="float32")[:args.limit]

    concepts = auto_select_concepts(texts, CANDIDATES)
    print(f"concepts: {concepts}")

    # 1) probe side (CPU, ground truth at the bottleneck)
    probe_results = run_probe_suite(acts, texts, concepts)

    # 2) AV side (GPU): verbalize each activation, then matcher-detect each concept
    nla = LocalGemmaE2BNLA(av_checkpoint=args.av_checkpoint).load()
    matcher = EnsembleMatcher()
    verbalizations = nla.verbalize_batch(acts)
    nla.close()

    av_acc = {}
    for c in concepts:
        truth = label_by_keyword(texts, c)              # ground-truth presence in source
        found = [1 if matcher.match(c, v).present else 0 for v in verbalizations]
        # accuracy of the AV+matcher pipeline at recovering true concept presence
        correct = sum(1 for y, f in zip(truth, found) if y == f)
        av_acc[c] = correct / len(truth)

    results = attach_av_accuracy(probe_results, av_acc)

    print(f"\n{'concept':12s} {'probe_acc':>10s} {'av_acc':>8s} {'gap':>7s}")
    for r in results:
        gap = f"{r.gap:.3f}" if r.gap is not None else "  -  "
        print(f"{r.concept:12s} {r.probe_acc:10.3f} {(r.av_matcher_acc or 0):8.3f} {gap:>7s}")

    OUT.mkdir(parents=True, exist_ok=True)
    save_csv(results, str(OUT / "probe_vs_av_gap.csv"))
    # persist the raw verbalizations too (data-permanence)
    import json
    with open(OUT / "av_verbalizations.jsonl", "w", encoding="utf-8") as f:
        for txt, vb in zip(texts, verbalizations):
            f.write(json.dumps({"text": txt[:200], "verbalization": vb}) + "\n")
    print(f"\nwrote {OUT/'probe_vs_av_gap.csv'} and av_verbalizations.jsonl")


if __name__ == "__main__":
    main()
