"""Run the SAME cross-NLA suite (retention / see-through / laundering) used for the
hosted Gemma-3 and Llama NLAs, but on the LOCAL Gemma-4-E2B NLA v0.1, so all three
land on the identical 20-example dataset + matcher and become directly comparable.

The local NLA's input is an activation, so this does what the hosted API does for
us automatically:
  1. EXTRACT: forward each example text through base Gemma-4-E2B, capture the
     layer-23 LAST-token residual (raw, unnormalized) -- the convention the v0.1 AV
     was trained on (nla_meta: layer 23, raw_float32_unnormalized). The AV LoRA is
     NOT applied here; training activations came from pure base.
  2. VERBALIZE: feed each activation to the v0.1 AV (LocalGemmaE2BNLA) -> text.
  3. SCORE: wrap text->verbalization as a CallableNLA and run the same metrics.

Two sequential model lifecycles (base, then base+LoRA) to fit a 4GB GPU.

The base model + AV checkpoint live in your separate NLA-training checkout; point
NLA_DATA_ROOT at it (no personal paths are baked in):
  NLA_DATA_ROOT=/path/to/nla-training-repo KMP_DUPLICATE_LIB_OK=TRUE \
  python experiments/local_gemma_e2b/cross_nla_local.py
Override the checkpoint with --av-checkpoint if it lives elsewhere.
"""
import argparse
import json
import os
import sys
from pathlib import Path

NLATTACK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NLATTACK))

from nla_eval import Example, EnsembleMatcher, CallableNLA, run
from nla_eval import tests as T
from nla_eval.core import retention_rate, substitution_rate, contested_rate
from nla_eval.local_gemma_e2b import LocalGemmaE2BNLA
import experiments.cross_nla_eval as CNE

OUT = NLATTACK / "results" / "cross_nla"
LAYER = 23
BASE_MODEL = "google/gemma-4-E2B"
DATA_ROOT = Path(os.environ.get("NLA_DATA_ROOT", "path/to/nla-training-repo"))
DEFAULT_CKPT = DATA_ROOT / "experiments/v8_nla_local/checkpoints/av_v0_1_aux_readout/block_to000100/final"


def extract_activations(texts):
    """Last-token raw residual at layer 23 from pure base Gemma-4-E2B (matches v0.1
    training extraction). Returns {text: list[float]}."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map={"": torch.cuda.current_device()}).eval()

    captured = {"acts": None}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["acts"] = h.detach().cpu().float()

    handle = model.model.language_model.layers[LAYER].register_forward_hook(hook)
    acts = {}
    try:
        with torch.no_grad():
            for i, text in enumerate(texts):
                ids = tok.encode(text, add_special_tokens=True, truncation=True, max_length=512)
                input_ids = torch.tensor([ids], device=model.device)
                captured["acts"] = None
                _ = model(input_ids=input_ids)
                vec = captured["acts"][0][-1].numpy().astype("float32")
                acts[text] = vec.tolist()
                print(f"  [extract {i+1}/{len(texts)}] |a|={float((vec**2).sum()**0.5):.1f}  {text[:54]!r}")
    finally:
        handle.remove()
        del model
        torch.cuda.empty_cache()
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-checkpoint", default=str(DEFAULT_CKPT),
                    help="AV LoRA checkpoint dir (default: under $NLA_DATA_ROOT)")
    ap.add_argument("--smoke", type=int, default=0, help="only process first N examples")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    dataset = CNE.build_dataset()
    if args.smoke:
        dataset = dataset[:args.smoke]
    texts = [ex.text for ex in dataset]
    print(f"[1/3] extracting L{LAYER} last-token activations for {len(texts)} texts")
    acts = extract_activations(texts)

    print(f"[2/3] verbalizing with v0.1 AV: {args.av_checkpoint}")
    nla = LocalGemmaE2BNLA(av_checkpoint=args.av_checkpoint).load()
    verb = {}
    try:
        for i, text in enumerate(texts):
            verb[text] = nla.verbalize_activation(acts[text])
            print(f"  [verb {i+1}/{len(texts)}] {verb[text][:80]!r}")
    finally:
        nla.close()

    print("[3/3] scoring (same matcher + metrics as the hosted run)")
    matcher = EnsembleMatcher()
    cb = CallableNLA(lambda t: verb[t], name="local-gemma4-e2b-nla:av_v0_1")
    res = run(cb, dataset, matcher=matcher)
    rows = res.rows
    metrics = {
        "nla_label": "Gemma-4-E2B-NLA@L23",
        "nla_id": f"{args.av_checkpoint}__google/gemma-4-E2B__L23",
        "av": args.av_checkpoint,
        "base_model": "google/gemma-4-E2B", "layer": LAYER, "run_mode": "local",
        "n_examples": len(dataset), "n_concept_rows": len(rows),
        "retention_rate": retention_rate(rows),
        "exact_retain_rate": sum(1 for r in rows if r.status == "retained") / max(1, len(rows)),
        "substitution_rate": substitution_rate(rows),
        "drop_rate": sum(1 for r in rows if r.status == "dropped") / max(1, len(rows)),
        "retention_general": retention_rate(rows, lambda r: r.meta.get("category") == "general"),
        "retention_attack": retention_rate(rows, lambda r: r.meta.get("category") == "attack"),
        "contested_rate": contested_rate(rows),
        **T.t12_obfuscation_seethrough(res, dataset),
        **T.t13_attack_to_benign_laundering(res),
        "note": "faithfulness cosine not computed (AR not loaded); activation-space "
                "cosine is not cross-NLA comparable anyway. Verbalizations saved.",
    }
    for k in ("retention_rate", "retention_general", "retention_attack",
              "substitution_rate", "drop_rate", "seethrough_ratio", "laundering_rate"):
        v = metrics.get(k)
        print(f"  {k:18s} {v if isinstance(v, str) else round(v, 3) if v == v else 'nan'}")

    label = metrics["nla_label"].replace("/", "_")
    clean = {k: (None if isinstance(v, float) and v != v else v)
             for k, v in metrics.items() if k != "examples"}  # NaN -> null (valid JSON)
    (OUT / f"{label}.json").write_text(json.dumps(clean, indent=2, default=str))
    res.to_csv(str(OUT / f"{label}_rows.csv"))
    (OUT / f"{label}_verbalizations.json").write_text(
        json.dumps(verb, indent=2, default=str))
    print(f"\nwrote {OUT}/{label}.json (+ _rows.csv, _verbalizations.json)")


if __name__ == "__main__":
    main()
