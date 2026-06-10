"""Access tiers: what you can run on an NLA you only reach through an API, versus
one you have full (white-box) access to.

This is the distinction that lets NLAttack work as a public benchmark. A hosted
NLA (for example one served by Neuronpedia) hands you the AV's verbalization text,
and sometimes a scalar reconstruction faithfulness (cosine / mse). It does NOT hand
you the raw activation vector, the model weights, or training checkpoints. So the
suite splits into two tiers:

  TIER 1 -- "api"  (black-box, universal):
      Everything computable from {AV verbalization text, + the API's own scalar
      faithfulness if it returns one}. Runs on ANY NLA, hosted or local, so this is
      the universal leaderboard tier: every NLA can be scored on it. Local NLAs are
      scored here too (they can obviously produce text), they just ALSO unlock the
      full-access tier.

  TIER 2 -- "full_access"  (white-box, extended):
      Anything that needs the raw activation vector, the model internals, or
      training checkpoints: linear probes on the bottleneck, activation-space
      faithfulness/rank, the emergence/longitudinal capability index, and the
      probe-grounded misuse/deception plans. Only runs on NLAs you have weights for.

A few evals are genuinely MIXED (a verbalizer-side part is api, an
activation-side part is full_access); those are marked "mixed" and the per-eval
helpers below resolve the actual function you call.

Cross-NLA comparability is a SEPARATE axis from the access tier. Of the api-tier
metrics, only the PROTOCOL-ANCHORED ones (fixed null regardless of model/dataset)
are honestly comparable across different NLAs on a leaderboard: see
`LEADERBOARD_METRICS`. Activation-space scalars like raw cosine are not on a common
scale across base models, so they are reported per-NLA, not ranked across NLAs.
"""
from __future__ import annotations

API = "api"
FULL = "full_access"
MIXED = "mixed"

# --- by implemented module (nla_eval/*.py) ---
MODULE_TIER = {
    "core": API,              # per-concept retention table from AV text
    "matching": API,          # concept matching over text
    "tests": API,             # 20 retention + ATT&CK misuse tests (text)
    "verbalizer_axes": API,   # minimal-pair AUC, doc retrieval, mode collapse, calibration
    "confabulation": API,     # factual grounding / thematic fidelity from text
    "controls": API,          # frequency/length-matched text controls
    "redteam": API,           # text-level evasion + compositional blind-spot search
    "attack_concepts": API,   # ATT&CK technique dictionary (data)
    "bottleneck_probe": FULL, # linear probe on the raw activation
    "emergence": FULL,        # activation-side axes + longitudinal capability index
    "local_gemma_e2b": FULL,  # the full-access local adapter
    "deception": MIXED,       # discrimination=api; probe/transfer=full_access
    "rudimentary": MIXED,     # AV-input-conditioned check=api; bottleneck-exists probe=full
}

# --- by plan family (plans/, A..M) ---
FAMILY_TIER = {
    "A": API,    # concept survival / dropout map
    "B": API,    # content adjacency / laundering
    "C": MIXED,  # deception & knowledge asymmetry (verbalizer api, epistemic probe full)
    "D": MIXED,  # ATT&CK misuse (survival/see-through/laundering api; P037/P038/P040 full)
    "E": FULL,   # bottleneck probes / ground truth
    "F": API,    # matcher & verbalizer confound
    "G": API,    # adaptive red team (text-level)
    "H": MIXED,  # faithfulness: scalar cos/mse api if exposed; activation-space analysis full
    "I": FULL,   # distributional / OOD / training dependence
    "J": API,    # calibration / uncertainty (verbalizer side)
    "K": FULL,   # emergence & additive capability index
    "L": MIXED,  # literature axes: simulatability api, steerability/intervention full
    "M": MIXED,  # deception monitoring: discrimination api, probe/transfer full
}

# --- by specific eval / axis (the names that resolve a MIXED module) ---
EVAL_TIER = {
    # api-tier evals (verbalizer side)
    "concept_retention": API,
    "doc_retrieval": API,
    "minimal_pair_discrimination": API,
    "prior_deviation": API,
    "mode_collapse": API,
    "calibration_entropy": API,
    "obfuscation_seethrough": API,
    "attack_to_benign_laundering": API,
    "deception_discrimination": API,        # deception.py, verbalizer side
    "av_input_conditioned": API,            # rudimentary.py
    # full-access evals (activation side)
    "bottleneck_probe": FULL,
    "probe_vs_verbalizer_gap": FULL,
    "deception_probe": FULL,                # deception.py, bottleneck side
    "cross_scenario_transfer": FULL,
    "aries_axis_probe": FULL,               # plan P038
    "downstream_monitor_classifier": FULL,  # plan P040
    "bottleneck_exists": FULL,              # rudimentary.py
    "emergence_capability_index": FULL,
    "activation_faithfulness_rank": FULL,
}

# Protocol-anchored api-tier metrics: fixed null regardless of model or dataset, so
# these are the ones honestly comparable ACROSS NLAs (the leaderboard columns).
LEADERBOARD_METRICS = {
    "doc_retrieval":               {"null": "1/(k+1) distractors", "example_chance": 0.077},
    "minimal_pair_discrimination": {"null": "AUC 0.5"},
    "deception_discrimination":    {"null": "AUC 0.5"},
    "obfuscation_seethrough":      {"null": "ratio 1.0 = fully strips obfuscation"},
    "attack_to_benign_laundering": {"null": "rate 0.0 = no laundering"},
}


def tier(name: str) -> str:
    """Resolve the access tier for a module, family letter, or specific eval name.
    Returns "api", "full_access", or "mixed" (or "unknown")."""
    if name in EVAL_TIER:
        return EVAL_TIER[name]
    if name in MODULE_TIER:
        return MODULE_TIER[name]
    if name in FAMILY_TIER:
        return FAMILY_TIER[name]
    return "unknown"


def requires_full_access(name: str) -> bool:
    """True if `name` needs raw activations / weights / checkpoints (cannot run on a
    text-only hosted NLA). MIXED resolves to False here (its api part is runnable);
    use EVAL_TIER for the exact sub-eval."""
    return tier(name) == FULL


def api_runnable(name: str) -> bool:
    """True if `name` can be scored on a hosted, text-only NLA (api or mixed)."""
    return tier(name) in (API, MIXED)


def leaderboard_metrics() -> list:
    """The protocol-anchored api-tier metrics that are comparable across NLAs."""
    return sorted(LEADERBOARD_METRICS)
