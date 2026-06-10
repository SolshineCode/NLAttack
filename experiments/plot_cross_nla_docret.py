"""Cross-NLA comparison on the ONE metric measured identically across the tested
NLAs: held-out document retrieval (same docs, 12 distractors so chance = 0.077,
MiniLM embedder). Series are labeled by NLA NAME, not base model.

Reads results/cross_nla/llama_doc_retrieval.json (it carries both the Llama-3.3-70B
result and the Gemma-4-E2B reference) and the local-Gemma-4 probe README numbers,
and writes results/cross_nla/cross_nla_docret_comparison.png.

This is the honest cross-model picture. The Gemma-3-27B NLA is not plotted: its
hosted inference server has returned 502 since ~2026-06-05, so it has no data yet.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "results", "cross_nla", "llama_doc_retrieval.json")
OUT = os.path.join(ROOT, "results", "cross_nla", "cross_nla_docret_comparison.png")


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        d = json.load(f)
    chance = d["chance"]
    gemma4 = d["gemma4_e2b_v01_reference"]

    # NLA name -> (char/tfidf retrieval, semantic retrieval)
    nlas = [
        ("Llama-3.3-70B-NLA-av@L53", d["char_top1"], d["semantic_top1"]),
        ("Gemma-4-E2B-NLA@L23", gemma4["char/tfidf"], gemma4["semantic"]),
    ]
    # Bottleneck-probe AUC where we have activation access (Gemma-4 local).
    probe = {"Gemma-4-E2B-NLA@L23": {"in-distribution": 0.988, "out-of-distribution": 0.695}}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- left: verbalizer side, held-out doc retrieval (shared protocol) ---
    labels = [n[0] for n in nlas]
    x = np.arange(len(labels))
    w = 0.38
    axL.bar(x - w / 2, [n[1] for n in nlas], w, label="char / tf-idf", color="#4C72B0")
    axL.bar(x + w / 2, [n[2] for n in nlas], w, label="semantic (MiniLM)", color="#55A868")
    axL.axhline(chance, ls="--", lw=1.3, color="#555555")
    axL.text(len(labels) - 0.5, chance + 0.012, f"chance = {chance:.3f}",
             ha="right", va="bottom", fontsize=9, color="#555555")
    axL.set_xticks(x)
    axL.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    axL.set_ylabel("held-out doc retrieval (top-1, 12 distractors)")
    axL.set_title("Verbalizer side: does the AV text identify the source doc?\n"
                  "(higher = better; identical protocol across NLAs)", fontsize=10)
    axL.set_ylim(0, 0.6)
    axL.legend(fontsize=9)

    # --- right: bottleneck side, probe AUC (where activations are available) ---
    splits = ["in-distribution", "out-of-distribution"]
    vals = [probe["Gemma-4-E2B-NLA@L23"][s] for s in splits]
    bx = np.arange(len(splits))
    axR.bar(bx, vals, 0.5, color=["#4C72B0", "#C44E52"])
    axR.axhline(0.5, ls="--", lw=1.3, color="#555555")
    axR.text(len(splits) - 0.5, 0.5 + 0.012, "permutation null ≈ 0.50",
             ha="right", va="bottom", fontsize=9, color="#555555")
    for i, v in enumerate(vals):
        axR.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    axR.set_xticks(bx)
    axR.set_xticklabels(splits, fontsize=9)
    axR.set_ylabel("linear-probe AUC on the bottleneck activation")
    axR.set_title("Bottleneck side: Gemma-4-E2B-NLA@L23\n"
                  "(activation access; the bottleneck is not a uniform filter)", fontsize=10)
    axR.set_ylim(0, 1.05)

    fig.suptitle("NLAttack cross-NLA results (attributed by NLA name)\n"
                 "Llama hosted on Neuronpedia, Gemma-4-E2B run locally; no shared GPU run",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
