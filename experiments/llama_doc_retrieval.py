"""Held-out doc-retrieval on the Llama-3.3-70B NLA (Neuronpedia API, no GPU) over
the SAME eval docs the Gemma-4 E2B session used (indomain_eval_cmp), with the SAME
12-distractor grouping (chance = 1/13 = 0.077) and the SAME embedder (MiniLM) — so
the number is directly comparable to Gemma-4 E2B v0.1's tfidf/semantic 0.135.

Method note (honest): Gemma-4 verbalized a single doc-level activation; here the
Neuronpedia AV verbalizes up to 16 token positions of the doc (concatenated) — the
hosted API's nature. Same docs, same distractor count, same embedder otherwise.

The eval parquet lives in your separate NLA-training checkout. Point NLA_DATA_ROOT
at it (or pass --parquet), then run with a venv that has sentence-transformers + pyarrow:
  NLA_DATA_ROOT=/path/to/nla-training-repo python experiments/llama_doc_retrieval.py
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

NLATTACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NLATTACK))
from nla_eval import NeuronpediaNLA
from nla_eval.verbalizer_axes import _char_ngram_vec

# The eval docs are produced by the NLA-training repo, kept separate from this harness.
_DATA_ROOT = Path(os.environ.get("NLA_DATA_ROOT", "path/to/nla-training-repo"))
PARQUET = _DATA_ROOT / "experiments/v8_nla_local/data/stage3_v0_4_fineweb/indomain_eval_cmp.parquet"
OUT = NLATTACK / "results" / "cross_nla"
N_DOCS = 40           # cap to stay well under the 120 req/hr rate limit
K_DISTRACTORS = 12    # -> chance 1/(12+1) = 0.077, matching Gemma-4's eval
R_REPEATS = 200


def retrieval_at1(qvecs, dvecs, k_distractors, repeats, seed=0):
    """Mean retrieval@1 where each query competes against its true doc + k random
    distractors (chance = 1/(k+1)). Matches the Gemma-4 eval's grouping."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(qvecs); hits = 0; trials = 0
    idx_all = np.arange(n)
    for _ in range(repeats):
        for i in range(n):
            pool = [i] + list(rng.choice(idx_all[idx_all != i],
                                         size=min(k_distractors, n - 1), replace=False))
            sims = [float(qvecs[i] @ dvecs[j]) for j in pool]
            hits += int(int(np.argmax(sims)) == 0); trials += 1
    return hits / trials if trials else float("nan")


def main():
    import numpy as np
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

    docs = pq.read_table(PARQUET, columns=["detokenized_text_truncated"]) \
             .column(0).to_pylist()[:N_DOCS]
    print(f"[docs={len(docs)}  chance(1/{K_DISTRACTORS+1})={1/(K_DISTRACTORS+1):.3f}]")

    nla = NeuronpediaNLA(model_id="llama3.3-70b-it", nla_source_id="kitft-l53")
    verbs = []
    for i, d in enumerate(docs):
        try:
            verbs.append(nla.reconstruct(d))
        except Exception as e:
            print(f"  doc {i} failed: {e}"); verbs.append("")
        if (i + 1) % 10 == 0:
            print(f"  verbalized {i+1}/{len(docs)}")

    # normalize embeddings
    def norm(M):
        M = np.asarray(M, dtype="float32")
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    Vc = norm([_char_ngram_vec(v) for v in verbs]); Dc = norm([_char_ngram_vec(d) for d in docs])
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    Vs = norm(st.encode(verbs)); Ds = norm(st.encode(docs))

    res = {
        "nla": "kitft/Llama-3.3-70B-NLA-av__llama3.3-70b-it__L53__kitft-l53",
        "n_docs": len(docs), "k_distractors": K_DISTRACTORS,
        "chance": 1 / (K_DISTRACTORS + 1),
        "char_top1": retrieval_at1(Vc, Dc, K_DISTRACTORS, R_REPEATS),
        "semantic_top1": retrieval_at1(Vs, Ds, K_DISTRACTORS, R_REPEATS),
        "gemma4_e2b_v01_reference": {"char/tfidf": 0.135, "semantic": 0.135,
                                     "chance": 0.077, "p": 0.08},
    }
    print(f"\nLlama-3.3-70B NLA doc-retrieval@1 (k=12 distractors, chance 0.077):")
    print(f"  char/tfidf : {res['char_top1']:.3f}")
    print(f"  semantic   : {res['semantic_top1']:.3f}")
    print(f"  vs Gemma-4 E2B v0.1: 0.135 / 0.135 (chance 0.077, p~0.08)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "llama_doc_retrieval.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT/'llama_doc_retrieval.json'}")


if __name__ == "__main__":
    main()
