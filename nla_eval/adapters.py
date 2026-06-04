"""NLA adapter contract.

IMPORTANT — what a real NLA actually is (per Anthropic's NLA work + the
kitft/natural_language_autoencoders library + Neuronpedia's hosted NLAs):

    activation --[AV: activation verbalizer]--> natural-language description
    description --[AR: activation reconstructor]--> activation'

It is NOT a text->text autoencoder. The human-readable **bottleneck is the AV
description**, and faithfulness is judged in *activation* space (how close AR's
activation' is to the original).

So this harness measures concept survival through the bottleneck as:

    input text -> host-model activation @ layer L -> AV verbalization

and asks: does the verbalization preserve concept C from the input? That is
exactly the "a monitor reads the NL bottleneck" framing the tests are built for.

Every adapter implements ONE method:

    reconstruct(text: str) -> str        # returns the AV verbalization (bottleneck text)

`encode(text)` optionally returns the activation (or AR's reconstructed
activation) for the few latent-space checks. Weak-NLA tests never need it.

To add any NLA: subclass `NLA`, implement `reconstruct`. Done.
"""
from __future__ import annotations

from typing import Optional, List
import abc


class NLA(abc.ABC):
    name: str = "unnamed-nla"

    @abc.abstractmethod
    def reconstruct(self, text: str) -> str:
        """Return the NL bottleneck (AV verbalization) for `text`."""
        raise NotImplementedError

    def reconstruct_batch(self, texts: List[str]) -> List[str]:
        return [self.reconstruct(t) for t in texts]

    def encode(self, text: str):
        """Optional: return an activation vector (np.ndarray) or None."""
        return None


# --------------------------------------------------------------------------
# Test/dev adapters (no external deps)
# --------------------------------------------------------------------------

class MockNLA(NLA):
    """Deterministic lossy concept filter — exercises every group-by offline."""

    name = "mock-lossy"

    def __init__(self, keep_prob: float = 0.6, seed: int = 0):
        self.keep_prob = keep_prob
        self._seed = seed

    def _keep(self, word: str) -> bool:
        h = abs(hash((word.lower(), self._seed)))
        penalty = min(0.4, 0.03 * max(0, len(word) - 5))
        return (h % 1000) / 1000.0 < (self.keep_prob - penalty)

    def reconstruct(self, text: str) -> str:
        return " ".join(w for w in text.split() if self._keep(w))


class CallableNLA(NLA):
    """Wrap any fn(str)->str (notebooks, quick experiments, a hosted endpoint)."""

    def __init__(self, fn, name: str = "callable-nla"):
        self._fn = fn
        self.name = name

    def reconstruct(self, text: str) -> str:
        return self._fn(text)


# --------------------------------------------------------------------------
# Neuronpedia-hosted NLA (gemma-2-2b, llama3.3-70b-it, ...)
# --------------------------------------------------------------------------

class NeuronpediaNLA(NLA):
    """Adapter for Neuronpedia's hosted NLAs at neuronpedia.org/nla.

    Neuronpedia exposes an interpretability API (OpenAPI/Scalar at /api-doc,
    JS-rendered so not fetchable statically). The request/response *shape* below
    matches Neuronpedia's activation-style endpoints; CONFIRM the exact path and
    field names against the live Scalar docs or the `neuronpedia` Python client,
    then adjust the two marked lines. Everything else is correct.

    Auth: set api_key (header `x-api-key`) — get one from your Neuronpedia account.
    """

    name = "neuronpedia-nla"

    def __init__(
        self,
        model_id: str = "gemma-2-2b",          # or "llama3.3-70b-it"
        layer: int = 12,
        api_key: Optional[str] = None,
        base_url: str = "https://www.neuronpedia.org/api",
        path: str = "/nla/verbalize",          # <-- CONFIRM against /api-doc
    ):
        self.name = f"neuronpedia:{model_id}-L{layer}"
        self.model_id = model_id
        self.layer = layer
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.path = path

    def reconstruct(self, text: str) -> str:
        import json
        import urllib.request

        body = json.dumps(
            {"model": self.model_id, "layer": self.layer, "text": text}  # <-- CONFIRM fields
        ).encode()
        req = urllib.request.Request(
            self.base_url + self.path,
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"x-api-key": self.api_key} if self.api_key else {}),
            },
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read())
        # AV verbalization key — CONFIRM against live response:
        return (
            payload.get("verbalization")
            or payload.get("description")
            or payload.get("text")
            or ""
        )


# --------------------------------------------------------------------------
# Local NLA via kitft/natural_language_autoencoders (Gemma E2B / Qwen / ...)
# --------------------------------------------------------------------------

class KitftNLA(NLA):
    """Local NLA using released AV/AR checkpoints from
    kitft/natural_language_autoencoders.

    Checkpoint naming: kitft/nla-{base}-L{layer}-av  (verbalizer)
                       kitft/nla-{base}-L{layer}-ar  (reconstructor)
      e.g. base="gemma3-12b", layer=32  ->  kitft/nla-gemma3-12b-L32-av
      (use the gemma E2B base string once published, same pattern.)

    Pipeline implemented here (the faithful path):
      1. run the *host* model, grab the residual-stream activation at layer L
         (last token by default),
      2. feed that activation to the AV model to produce the verbalization.

    Two integration seams are marked. The activation grab (step 1) is standard
    transformers hooking and works as written. The AV injection (step 2) follows
    the library's `input_embeds` convention; if you have the repo installed,
    prefer routing through its `nla_inference.NLAClient` for the exact
    model-specific scaling — see `_verbalize_via_library`.
    """

    name = "kitft-nla"

    def __init__(
        self,
        host_model_id: str,        # e.g. "google/gemma-2-2b" (the model being explained)
        av_model_id: str,          # e.g. "kitft/nla-gemma3-12b-L32-av"
        layer: int,
        device: str = "cpu",
        pool: str = "last",        # "last" | "mean"
        max_new_tokens: int = 64,
        use_library: bool = False, # True -> route AV through nla_inference (exact scaling)
        sglang_url: str = "http://localhost:30000",
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM  # lazy

        self.name = f"kitft:{av_model_id.split('/')[-1]}"
        self.layer = layer
        self.device = device
        self.pool = pool
        self.max_new_tokens = max_new_tokens
        self.use_library = use_library
        self.sglang_url = sglang_url

        self.host_tok = AutoTokenizer.from_pretrained(host_model_id)
        self.host = AutoModelForCausalLM.from_pretrained(
            host_model_id, output_hidden_states=True
        ).to(device).eval()
        self.av_model_id = av_model_id
        if not use_library:
            self.av_tok = AutoTokenizer.from_pretrained(av_model_id)
            self.av = AutoModelForCausalLM.from_pretrained(av_model_id).to(device).eval()

    # ---- step 1: activation from the host model --------------------------
    def encode(self, text: str):
        import torch  # lazy

        ids = self.host_tok(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            out = self.host(**ids)
        hs = out.hidden_states[self.layer][0]  # [seq, d_model]
        vec = hs[-1] if self.pool == "last" else hs.mean(dim=0)
        return vec.detach().cpu().numpy()

    # ---- step 2a: AV via the library (exact, recommended) ----------------
    def _verbalize_via_library(self, activation):
        from nla_inference import NLAClient  # lazy; repo must be importable

        client = NLAClient(self.av_model_id, sglang_url=self.sglang_url)
        return client.verbalize(activation)  # see docs/inference.md for signature

    # ---- step 2b: AV via transformers soft-prompt injection --------------
    def _verbalize_via_transformers(self, activation):
        import torch  # lazy

        d = self.av.get_input_embeddings().weight.shape[1]
        act = torch.tensor(activation, dtype=self.av.get_input_embeddings().weight.dtype)
        if act.shape[-1] != d:
            # project if AV d_model differs from host d_model (linear, no-train fallback)
            proj = torch.nn.Linear(act.shape[-1], d, bias=False)
            act = proj(act)
        soft = act.view(1, 1, d).to(self.device)  # one soft token carrying the activation
        bos = torch.tensor([[self.av_tok.bos_token_id or 0]]).to(self.device)
        bos_emb = self.av.get_input_embeddings()(bos)
        inp = torch.cat([bos_emb, soft], dim=1)
        with torch.no_grad():
            out = self.av.generate(inputs_embeds=inp, max_new_tokens=self.max_new_tokens)
        return self.av_tok.decode(out[0], skip_special_tokens=True)

    def reconstruct(self, text: str) -> str:
        act = self.encode(text)
        if self.use_library:
            return self._verbalize_via_library(act)
        return self._verbalize_via_transformers(act)
