"""Wire the harness to a REAL NLA. Two paths; pick one.

Path A — Neuronpedia-hosted (no GPU, needs an API key):
    NEURONPEDIA_API_KEY=... python real_nla_example.py neuronpedia

Path B — local kitft NLA checkpoints (needs GPU + the host model):
    python real_nla_example.py kitft

Both run the same attack dataset + the same 20 tests as run_example.py.
"""
import os
import sys

from nla_eval import Matcher, run, NeuronpediaNLA, KitftNLA
from nla_eval import tests as T
from run_example import build_attack_dataset


def get_nla(which: str):
    if which == "neuronpedia":
        # Confirm the endpoint path/fields in adapters.NeuronpediaNLA against
        # the live Scalar docs at neuronpedia.org/api-doc the first time.
        return NeuronpediaNLA(
            model_id="gemma-2-2b",
            layer=12,
            api_key=os.environ.get("NEURONPEDIA_API_KEY"),
        )
    if which == "kitft":
        # Use the gemma E2B base string once published; pattern is identical.
        return KitftNLA(
            host_model_id="google/gemma-2-2b",
            av_model_id="kitft/nla-gemma3-12b-L32-av",
            layer=32,
            device="cuda",
        )
    raise SystemExit("usage: real_nla_example.py [neuronpedia|kitft]")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "neuronpedia"
    nla = get_nla(which)
    ds = build_attack_dataset()
    matcher = Matcher()  # use embedding backend for real evals: pip install sentence-transformers
    print(f"[nla={nla.name}  matcher={matcher.backend}]")
    res = run(nla, ds, matcher=matcher)

    print("t12 obfuscation see-through :", T.t12_obfuscation_seethrough(res, ds))
    print("t13 attack->benign laundering:", T.t13_attack_to_benign_laundering(res))
    print("t14 risk-weighted dropout   :", T.t14_risk_weighted_dropout(res))
    print("t15 evasion robustness      :", T.t15_evasion_robustness(res))
    print("t11 technique coverage      :", T.t11_technique_coverage(res))
    res.to_csv(f"retention_{nla.name.replace(':','_').replace('/','_')}.csv")


if __name__ == "__main__":
    main()
