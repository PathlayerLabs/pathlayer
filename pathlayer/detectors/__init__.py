"""Detection layer.

Holds extraction (canonicalization, action tokenization), derived signals,
and the detectors that consume them. All detectors produce a continuous
confidence in [0, 1] rather than a binary verdict.

V1 detectors:
- LoopDetector       (class i: repetition; n-gram/exact-repeat + entropy)
- RetryDetector      (repeated same error / tool retries)
- NoProgressDetector (class ii: no-progress-without-repetition; CUSUM on a
                      domain progress signal -- the differentiated case)
- BudgetDetector     (cost ceiling)
- ResourceDetector   (class iii: spawns/depth/restarts/wall-clock/peak RSS; detect-only)

Deferred to V2: Markov spectral-gap analysis, TF-IDF/ROUGE coverage signal.
"""

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.budget import BudgetState, advance_budget_state, budget_delta
from pathlayer.detectors.progress import (
    ExitCodeProgressAnchor,
    GitProgressAnchor,
    ProgressAnchor,
    TestProgressAnchor,
    coding_anchors,
)

__all__ = [
    "BudgetState",
    "Detector",
    "ExitCodeProgressAnchor",
    "GitProgressAnchor",
    "ProgressAnchor",
    "Signal",
    "TestProgressAnchor",
    "advance_budget_state",
    "budget_delta",
    "coding_anchors",
]
