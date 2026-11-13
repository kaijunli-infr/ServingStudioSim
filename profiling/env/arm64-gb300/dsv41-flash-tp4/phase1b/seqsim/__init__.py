"""Phase 1B: opt-in TP4 sequence simulator built on the Phase 1A `costmodel` components.

The Phase 1A package lives in ../phase1a and is imported unchanged; this package adds the
iteration-sequence layer (host timelines, scheduler constraints, launch-queue throttling, per-rank
GPU timelines with explicit collective barriers) and its own calibration / evaluation.
"""
from __future__ import annotations

import os
import sys

__version__ = "0.1.0"

_PHASE1A = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "phase1a"))
if _PHASE1A not in sys.path:
    sys.path.insert(0, _PHASE1A)

import costmodel  # noqa: E402,F401  (Phase 1A, unchanged)
