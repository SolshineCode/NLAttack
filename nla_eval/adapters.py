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
    """Adapter for Neuronpedia's hosted NLAs (POST /api/nla/explain).

    VERIFIED against the live API (no key required; rate-limited 120 req/hr/IP).
    Discover (modelId, nlaSourceId) pairs via GET /api/nla/sources. As of the
    last check:
        gemma-3-27b-it  / kitft-l41   (Gemma NLA, layer 41, kitft/nla-gemma3-27b)
        llama3.3-70b-it / kitft-l53   (Llama NLA, layer 53)

    /api/nla/explain returns a natural-language `description` (the AV
    verbalization) per requested token position. The harness "bottleneck text"
    is the concatenation of those per-position descriptions; concept-survival =
    does that text mention concept C.

    Because explain takes <=16 positions and is rate-limited, reconstruct() does
    one bootstrap call (position 0) to learn prompt_length, then explains up to
    `max_positions` evenly-spaced token positions in one batched call.

    encode() returns nothing useful here (no activation vector is exposed), but
    per-position `cosine_similarity`/`mse` (activation-space faithfulness) are
    captured in `self.last_faithfulness` for optional confidence weighting.
    """

    name = "neuronpedia-nla"

    def __init__(
        self,
        model_id: str = "gemma-3-27b-it",
        nla_source_id: str = "kitft-l41",
        api_key: Optional[str] = None,
        base_url: str = "https://www.neuronpedia.org/api",
        max_positions: int = 16,
        temperature: float = 0.7,
        timeout: int = 120,
        retry_on_429: int = 2,
        retry_on_gateway: int = 4,
    ):
        self.name = f"neuronpedia:{model_id}/{nla_source_id}"
        self.model_id = model_id
        self.nla_source_id = nla_source_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_positions = min(max_positions, 16)
        self.temperature = temperature
        self.timeout = timeout
        self.retry_on_429 = retry_on_429
        self.retry_on_gateway = retry_on_gateway
        self.last_faithfulness: list = []
        self.last_truncated = 0

    # gateway codes worth retrying — the NLA inference server is on RunPod and
    # cold-starts or transiently faults with 500/502/503/504; 429 is the documented
    # rate limit. 500 is included because the hosted backend emits transient
    # "NLA server error: 500" responses that clear on retry (seen on Gemma-3,
    # 2026-06-10); a genuine persistent 500 still raises after the retry budget.
    _RETRY_CODES = (429, 500, 502, 503, 504)

    def _explain(self, text: str, positions):
        import json
        import time
        import urllib.request
        import urllib.error

        body = json.dumps({
            "modelId": self.model_id,
            "nlaSourceId": self.nla_source_id,
            "text": text,
            "positions": list(positions),
            "temperature": self.temperature,
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/nla/explain",
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"x-api-key": self.api_key} if self.api_key else {}),
            },
        )
        attempt = 0
        max_attempts = max(self.retry_on_429, self.retry_on_gateway)
        while True:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in self._RETRY_CODES and attempt < max_attempts:
                    attempt += 1
                    time.sleep(min(30, 3 * (2 ** (attempt - 1))))  # 3,6,12,24,30s
                    continue
                raise
            except urllib.error.URLError:
                if attempt < max_attempts:
                    attempt += 1
                    time.sleep(min(30, 3 * (2 ** (attempt - 1))))
                    continue
                raise

    def _explain_body_positions(self, text: str):
        """One bootstrap (pos 0) to learn prompt_length, then explain the body
        positions 1..min(prompt_length-1, max_positions). Position 0 is the BOS
        token whose verbalization is reliably global-prior noise, so it is
        excluded. Returns the list of result records (token + description +
        faithfulness per position)."""
        boot = self._explain(text, [0])
        plen = int(boot.get("prompt_length", 1))
        if plen <= 1:
            return boot.get("results", [])
        positions = list(range(1, min(plen, 1 + self.max_positions)))
        if plen - 1 > self.max_positions:
            self.last_truncated = plen - 1 - self.max_positions  # logged by caller
        data = self._explain(text, positions)
        results = data.get("results", [])
        self.last_faithfulness = [
            {"position": r.get("position"),
             "cosine_similarity": r.get("cosine_similarity"), "mse": r.get("mse")}
            for r in results
        ]
        return results

    def reconstruct(self, text: str) -> str:
        """Whole-example bottleneck text (all body positions). Used for
        insertion/hallucination detection and reconstruct-only callers."""
        results = self._explain_body_positions(text)
        return "\n".join(r.get("description", "") for r in results)

    @staticmethod
    def _norm_tok(tok: str) -> str:
        return "".join(ch for ch in (tok or "").lower() if ch.isalnum())

    @staticmethod
    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    def _pack(self, recs):
        """Bundle a set of position-records into {text, cos, mse, positions}.
        cos/mse are the AR activation-space faithfulness at those positions —
        how trustworthy this verbalization is (Hermes review P1#5 / item B)."""
        return {
            "text": "\n".join(r.get("description", "") for r in recs),
            "cos": self._mean([r.get("cosine_similarity") for r in recs]),
            "mse": self._mean([r.get("mse") for r in recs]),
            "positions": [r.get("position") for r in recs],
        }

    def verbalize_concepts(self, text: str, concepts):
        """Concept-targeted verbalization: explain the body once, then for each
        concept return the descriptions at the token positions that make up that
        concept (so we test whether the concept's OWN activations verbalize as
        the concept, vs launder into a neighbor). Falls back to the full text for
        any concept whose tokens can't be located.

        Returns {concept: {text, cos, mse, positions[, fallback_full]}} plus a
        "__full__" entry over all body positions (used for insertion detection).
        """
        from .matching import content_words

        results = self._explain_body_positions(text)
        out = {}
        for c in concepts:
            cwords = [self._norm_tok(w) for w in content_words(c)]
            hits = [r for r in results
                    if (lambda nt: nt and any(
                        (nt in w or w in nt) and min(len(nt), len(w)) >= 3
                        for w in cwords))(self._norm_tok(r.get("token", "")))]
            if hits:
                out[c] = self._pack(hits)
            else:
                entry = self._pack(results)      # whole-text fallback
                entry["fallback_full"] = True
                out[c] = entry
        out["__full__"] = self._pack(results)
        return out


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
