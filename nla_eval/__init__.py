"""nla_eval — capability-floor evaluation harness for Natural Language Autoencoders.

The whole thing rests on one primitive: did concept C survive the bottleneck?
Plug in any NLA via the `NLA` adapter contract (one method: reconstruct), run a
tagged dataset through `core.run`, then read off the 20 tests as group-bys.
"""
__version__ = "2.0.0"  # v2 release; v1 benchmark content frozen — see docs/VERSIONING.md
from .adapters import NLA, MockNLA, CallableNLA, NeuronpediaNLA, KitftNLA
from .matching import Matcher, EnsembleMatcher
from .core import Example, run, RunResult
from . import (tests, controls, bottleneck_probe, redteam, rudimentary, emergence,
               verbalizer_axes, confabulation, deception, access, ctf)

__all__ = [
    "NLA", "MockNLA", "CallableNLA", "NeuronpediaNLA", "KitftNLA",
    "Matcher", "EnsembleMatcher", "Example", "run", "RunResult",
    "tests", "controls", "bottleneck_probe", "redteam", "rudimentary", "emergence",
    "verbalizer_axes", "confabulation", "deception", "access", "ctf",
]
