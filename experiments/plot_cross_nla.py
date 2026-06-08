"""Side-by-side visuals comparing hosted NLAs on the API-runnable evals.
Reads results/cross_nla/cross_nla_summary.json (from cross_nla_eval.py) and writes
a grouped-bar comparison PNG. Series are labeled by NLA NAME (not base model).

    python experiments/plot_cross_nla.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path

NLATTACK = Path(__file__).resolve().parents[1]
OUT = NLATTACK / "results" / "cross_nla"


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((OUT / "cross_nla_summary.json").read_text())
    data = [d for d in data if "error" not in d]
    if not data:
        print("no successful NLA results to plot"); return
    labels = [d["nla_label"] for d in data]
    n = len(data)

    panels = [
        ("Concept retention (higher = survives bottleneck)",
         [("overall", "retention_rate"), ("general", "retention_general"),
          ("attack", "retention_attack")]),
        ("Loss modes (lower better)",
         [("dropped", "drop_rate"), ("substituted/laundered", "substitution_rate")]),
        ("AV faithfulness — activation space (cos higher better)",
         [("mean cosine_sim", "mean_cos_sim")]),
        ("Misuse-eval signals",
         [("obfuscation see-through", "seethrough_ratio"),
          ("attack→benign laundering", "laundering_rate"),
          ("matcher contested", "contested_rate")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, n)))
    for ax, (title, series) in zip(axes, panels):
        m = len(series); x = np.arange(m); w = 0.8 / n
        for i, d in enumerate(data):
            vals = [float(d.get(k, float("nan"))) for _, k in series]
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=labels[i], color=colors[i])
        ax.set_xticks(x); ax.set_xticklabels([s[0] for s in series], rotation=15, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Cross-NLA evaluation (Neuronpedia hosted NLAs, API; no local GPU)\n"
                 "attributed by NLA name", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "cross_nla_comparison.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
