"""nla_eval — capability-floor evaluation harness for Natural Language Autoencoders.

The whole thing rests on one primitive: did concept C survive the bottleneck?
Plug in any NLA via the `NLA` adapter contract (one method: reconstruct), run a
tagged dataset through `core.run`, then read off the 20 tests as group-bys.
"""
from .adapters import NLA, MockNLA, CallableNLA, NeuronpediaNLA, KitftNLA
from .matching import Matcher
from .core import Example, run, RunResult
from . import tests

__all__ = [
    "NLA", "MockNLA", "CallableNLA", "NeuronpediaNLA", "KitftNLA",
    "Matcher", "Example", "run", "RunResult", "tests",
]
