"""Adapter for the LOCAL Gemma-4-E2B NLA (trained in a companion repo, v8).

This NLA is `activation → AV verbalizer → text` (the AR reconstructs the
activation). Unlike a text→text model, its natural input is an activation
vector, so this adapter exposes `verbalize_activation(vec) -> str` rather than
the text→text `reconstruct`. Pair it with precomputed (text, activation_vector)
rows to feed NLAttack's concept-retention harness, and with
`nla_eval.bottleneck_probe` for the ground-truth probe side.

Faithful port of the proven, OOM-safe inference path in
`experiments/v8_nla_local/eval_av_smoke.py` (forward-hook injection, NOT
inputs_embeds — Gemma-4's per_layer_inputs OOMs otherwise). The deception repo
is treated READ-ONLY; this code lives in NLAttack.

GPU-DEFERRED: validated by construction against eval_av_smoke.py but not yet run
live here (the 4GB GPU was occupied by another session's job at build time). Run
`experiments/local_gemma_e2b/verbalize_av.py` when the GPU frees.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# constants copied from eval_av_smoke.py (must match the trained checkpoint)
INJECTION_TOKEN_ID = 249568
INJECTION_LEFT_NEIGHBOR_ID = 236813
INJECTION_RIGHT_NEIGHBOR_ID = 954
INJECTION_CHAR = chr(0x3297)
D_MODEL = 1536

AV_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into activation vectors from a language model. Your overall task is to "
    "describe the semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. You "
    "must then produce an explanation for the vector, enclosed within "
    "<explanation> tags. The explanation consists of 2-3 text snippets "
    "describing that vector.\n\nHere is the vector:\n\n<concept>{c}</concept>"
)


class LocalGemmaE2BNLA:
    """Loads base Gemma-4-E2B (4-bit) + the AV LoRA and verbalizes activations.

    Example:
        nla = LocalGemmaE2BNLA(
            av_checkpoint=r"...\\checkpoints\\av_v0_1_aux_readout\\block_to000100\\final")
        nla.load()
        text = nla.verbalize_activation(vec)   # vec: np.ndarray (d_model,)
    """

    name = "local-gemma4-e2b-nla"

    def __init__(self, av_checkpoint: str, base_model: str = "google/gemma-4-E2B",
                 max_new_tokens: int = 120):
        self.av_checkpoint = str(av_checkpoint)
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.name = f"local-gemma4-e2b:{Path(av_checkpoint).parent.parent.name}"
        self._tok = None
        self._av = None
        self._pending = {"input_ids": None, "vec": None}
        self._handle = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_quant_type="nf4")
        self._tok = AutoTokenizer.from_pretrained(self.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, quantization_config=bnb,
            device_map={"": torch.cuda.current_device()})
        self._av = PeftModel.from_pretrained(base, self.av_checkpoint).eval()
        self._handle = self._av.get_input_embeddings().register_forward_hook(self._inject)
        return self

    def _inject(self, module, args_in, output):
        # inject only during prompt encoding (seq>1), at the marker trio
        if output.shape[1] <= 1:
            return output
        ids = self._pending["input_ids"]
        vec = self._pending["vec"]
        if ids is None or vec is None:
            return output
        h = output.clone()
        for b in range(min(ids.shape[0], h.shape[0])):
            for p in range(1, min(ids.shape[1], h.shape[1]) - 1):
                if (ids[b, p].item() == INJECTION_TOKEN_ID
                        and ids[b, p - 1].item() == INJECTION_LEFT_NEIGHBOR_ID
                        and ids[b, p + 1].item() == INJECTION_RIGHT_NEIGHBOR_ID):
                    h[b, p] = vec[b].to(h.dtype)
                    break
        return h

    def verbalize_activation(self, vec) -> str:
        """vec: 1-D activation (d_model,). Returns the AV description text."""
        import numpy as np
        import torch

        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9) * float(np.sqrt(D_MODEL))  # match training scale
        vec_t = torch.from_numpy(v).to(self._av.device).unsqueeze(0)

        prompt = AV_TEMPLATE.format(c=INJECTION_CHAR)
        ids = self._tok.encode(prompt, return_tensors="pt").to(self._av.device)
        self._pending = {"input_ids": ids, "vec": vec_t}
        try:
            with torch.no_grad():
                out = self._av.generate(input_ids=ids, max_new_tokens=self.max_new_tokens,
                                        do_sample=False, pad_token_id=self._tok.eos_token_id)
        finally:
            self._pending = {"input_ids": None, "vec": None}
        return self._tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def verbalize_batch(self, vecs) -> List[str]:
        return [self.verbalize_activation(v) for v in vecs]

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
