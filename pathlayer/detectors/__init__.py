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
- ResourceDetector   (class iii: spawns/depth/restarts/wall-clock; detect-only)

Deferred to V2: Markov spectral-gap analysis, TF-IDF/ROUGE coverage signal.
"""

from pathlayer.detectors.base import Detector, Signal
from pathlayer.detectors.progress import (
    ExitCodeProgressAnchor,
    ProgressAnchor,
    TestProgressAnchor,
    coding_anchors,
)

__all__ = [
    "Detector",
    "ExitCodeProgressAnchor",
    "ProgressAnchor",
    "Signal",
    "TestProgressAnchor",
    "coding_anchors",
]
