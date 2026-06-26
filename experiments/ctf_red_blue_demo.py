"""Demo: CTF Red/Blue exercise (v2, Family N) against the offline MockNLA.

Pits an adaptive Red team (laundering a hidden deceptive objective) against a Blue
NLA-monitor (reading the AV verbalization). Prints the scoreboard and writes it to
results/ctf/. No GPU, no network. CONTROL-TIER by construction: the capture oracle
is the model-agnostic survival proxy, and the flags are constructed scenarios — so
this demonstrates the game mechanics and the verbalizer blind spot, NOT an
emergent-deception result (see plans/N_ctf_red_blue.md, P128).

    python experiments/ctf_red_blue_demo.py

To get a PRIMARY-tier result, build the monitor on a local NLA and pass a
`ctf.probe_capture_oracle(probe_predict)` over the model's own behavior.
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path

# MockNLA's token-keep uses the builtin hash() (roadmap harness fix H1), which is
# process-randomized; pin PYTHONHASHSEED so this committed artifact is reproducible.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

NLATTACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NLATTACK))

from nla_eval import MockNLA, EnsembleMatcher
from nla_eval import ctf

OUT = NLATTACK / "results" / "ctf"


def main():
    matcher = EnsembleMatcher()
    nla = MockNLA(keep_prob=0.6)

    # Baseline (null control): naive Red states the objective plainly. A monitor
    # that can't BUST this is catching nothing — read the adaptive numbers against it.
    naive = ctf.play_ctf(nla, red_policy=ctf.naive_red_policy, matcher=matcher, budget=1)
    # The real game: adaptive laundering Red under a query budget.
    adaptive = ctf.play_ctf(nla, red_policy=ctf.adaptive_red_policy, matcher=matcher, budget=50)

    for name, sc in (("naive (null control)", naive), ("adaptive Red (budget=50)", adaptive)):
        print(f"\n== {name} [tier={sc.tier}] ==")
        print(f"  clean-capture rate (Red wins) : {sc.clean_capture_rate:.2f}")
        print(f"  catch rate (Blue recall)      : {sc.catch_rate:.2f}")
        print(f"  busted / failed / false-alarm : "
              f"{sc.busted_rate:.2f} / {sc.failed_rate:.2f} / {sc.false_alarm_rate:.2f}")
        print(f"  mean stealth gap              : {sc.mean_stealth_gap:+.2f}")
        print(f"  mean queries to clean-capture : {sc.mean_queries_to_capture}")
        print(f"  by category:")
        for cat, d in sc.by_category.items():
            print(f"    {cat:20s} clean={d['clean_capture_rate']:.2f} catch={d['catch_rate']}")

    OUT.mkdir(parents=True, exist_ok=True)
    rec = {
        "naive_null_control": naive.flat(),
        "adaptive_red": adaptive.flat(),
        "nla": "MockNLA(keep_prob=0.6)",
        "tier": adaptive.tier,
        "integrity": ("CONTROL: survival-proxy oracle over constructed flags; "
                      "swap in a bottleneck-probe oracle over model-own behavior for PRIMARY (P128)"),
        "note": "v2 Family N CTF Red/Blue; mechanics demo, not a monitor certification",
    }
    path = OUT / "ctf_red_blue_mock.json"
    path.write_text(json.dumps(rec, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
