"""Headline NLAttack benchmark figure: all evaluated NLAs across the parts of the
suite where data exists, attributed by NLA name. Reads the committed result JSONs.

The story the three panels tell together:
  - Hosted NLAs (Gemma-3, Llama) are verbalizer-strong: high concept retention.
  - Gemma-4-E2B v0.1 is bottleneck-strong but verbalizer-weak AND domain-specific:
    its bottleneck probe is near-perfect in-distribution, yet its verbalizer
    retrieves documents only weakly in-domain and collapses to 0 concept retention
    on the OUT-OF-DOMAIN cross_nla dataset (per its model card: at chance OOD).

Writes results/cross_nla/benchmark_overview.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CN = os.path.join(ROOT, "results", "cross_nla")


def load(name):
    with open(os.path.join(CN, name), "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    summary = load("cross_nla_summary.json")              # Gemma-3, Llama
    by = {r["nla_label"]: r for r in summary}
    g4 = load("Gemma-4-E2B-NLA@L23.json")                  # Gemma-4-E2B v0.1 (OOD)
    dr = load("llama_doc_retrieval.json")

    gemma3 = "nla-gemma3-27b-av@L41"
    llama = "Llama-3.3-70B-NLA-av@L53"
    C = {"gemma3": "#4C72B0", "llama": "#55A868", "g4": "#C44E52"}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.2))

    # --- Panel 1: concept retention on the identical 20-example dataset (API tier) ---
    labels = ["nla-gemma3-27b-av@L41", "Llama-3.3-70B-NLA-av@L53", "Gemma-4-E2B-NLA@L23"]
    vals = [by[gemma3]["retention_rate"], by[llama]["retention_rate"], g4["retention_rate"]]
    cols = [C["gemma3"], C["llama"], C["g4"]]
    x = np.arange(len(labels))
    ax1.bar(x, vals, 0.6, color=cols)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax1.text(2, 0.06, "out-of-domain\nfor v0.1", ha="center", va="bottom", fontsize=8.5, color="#C44E52")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha="right", fontsize=8.5)
    ax1.set_ylabel("concept retention")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Concept retention, identical 20-example dataset\n(API tier; higher = survives the bottleneck)", fontsize=10)

    # --- Panel 2: held-out doc retrieval, in-domain protocol (chance fixed) ---
    chance = dr["chance"]
    dl = [("Llama-3.3-70B-NLA-av@L53", dr["char_top1"], dr["semantic_top1"], C["llama"]),
          ("Gemma-4-E2B-NLA@L23", dr["gemma4_e2b_v01_reference"]["char/tfidf"],
           dr["gemma4_e2b_v01_reference"]["semantic"], C["g4"])]
    dx = np.arange(len(dl))
    w = 0.36
    ax2.bar(dx - w / 2, [d[1] for d in dl], w, label="char / tf-idf", color="#8c8c8c")
    ax2.bar(dx + w / 2, [d[2] for d in dl], w, label="semantic (MiniLM)", color=[d[3] for d in dl])
    ax2.axhline(chance, ls="--", lw=1.3, color="#555555")
    ax2.text(len(dl) - 0.5, chance + 0.012, f"chance = {chance:.3f}", ha="right", va="bottom", fontsize=9, color="#555555")
    ax2.set_xticks(dx)
    ax2.set_xticklabels([d[0] for d in dl], rotation=12, ha="right", fontsize=8.5)
    ax2.set_ylabel("held-out doc retrieval (top-1)")
    ax2.set_ylim(0, 0.6)
    ax2.legend(fontsize=8.5)
    ax2.set_title("Verbalizer doc retrieval, in-domain protocol\n(Gemma-3 not run; higher = AV text identifies source)", fontsize=10)

    # --- Panel 3: bottleneck probe AUC (full-access; Gemma-4-E2B only) ---
    splits = ["in-distribution", "out-of-distribution"]
    pv = [0.988, 0.695]
    px = np.arange(len(splits))
    ax3.bar(px, pv, 0.5, color=["#C44E52", "#e0a0a0"])
    ax3.axhline(0.5, ls="--", lw=1.3, color="#555555")
    ax3.text(len(splits) - 0.5, 0.5 + 0.012, "permutation null = 0.50", ha="right", va="bottom", fontsize=9, color="#555555")
    for i, v in enumerate(pv):
        ax3.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax3.set_xticks(px)
    ax3.set_xticklabels(splits, fontsize=9)
    ax3.set_ylabel("linear-probe AUC on bottleneck activation")
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Bottleneck probe, Gemma-4-E2B-NLA@L23\n(full-access tier; needs raw activations)", fontsize=10)

    fig.suptitle("NLAttack benchmark overview, by NLA name. Hosted NLAs are verbalizer-strong; "
                 "Gemma-4-E2B v0.1 is bottleneck-strong but verbalizer-weak and domain-specific.",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(CN, "benchmark_overview.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
