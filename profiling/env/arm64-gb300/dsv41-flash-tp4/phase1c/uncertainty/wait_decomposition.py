"""Measured per-rank collective wait (Phase 1A decode tables) versus the three composition rules.

For each batch size: the table's measured mean wait per rank, the wait implied by the median path (all ranks at their
segment medians), by independent per-(rank, segment) quantile draws (engine behaviour in pred-mc) and by common-mode
draws (same quantile on every rank). The position of the measured wait between the common-mode and the independent
value is the rank-independent fraction of the per-segment variation (reported, not fitted into the model).
Usage: python uncertainty/wait_decomposition.py   (needs only the committed coefficients)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1B = os.path.abspath(os.path.join(HERE, "..", "..", "phase1b"))
P1A = os.path.abspath(os.path.join(HERE, "..", "..", "phase1a"))
sys.path.insert(0, P1B)
import seqsim  # noqa: E402,F401
from costmodel.compose import critical_path_batch  # noqa: E402
from costmodel.decode import _sample  # noqa: E402
from costmodel.predict import Predictor  # noqa: E402

R, S = 4, 8192


def main():
    d = Predictor(json.load(open(os.path.join(P1A, "out", "coefficients.json")))).decode
    print("b | status n K | measured wait/rank ms | median path | independent draws | common-mode draws | independent fraction (mean over ranks) | wall median/mean ms (table) | median-path wall | first-segment arrival spread p15-p85 ms")
    for b in (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 16, 17, 24, 25, 27, 48):
        p, st = d._lookup(b)
        K = p["K"]
        rng = np.random.default_rng(0)
        med = critical_path_batch(np.asarray(p["start"], float)[None], np.asarray(p["seg"], float)[None], np.asarray(p["intrinsic"], float)[None])
        ind = critical_path_batch(_sample(p["start_q"], rng.random((S, R))), _sample(p["seg_q"], rng.random((S, R, K + 1))), _sample(p["intrinsic_q"], rng.random((S, K))))
        ug = np.repeat(rng.random((S, 1, K + 1)), R, axis=1); us = np.repeat(rng.random((S, 1)), R, axis=1)
        com = critical_path_batch(_sample(p["start_q"], us), _sample(p["seg_q"], ug), _sample(p["intrinsic_q"], rng.random((S, K))))
        meas = np.asarray(p["wait"], float) / 1e6
        wi = ind["rank_wait"].mean(axis=0) / 1e6; wc = com["rank_wait"].mean(axis=0) / 1e6; wm = med["rank_wait"].mean(axis=0) / 1e6
        frac = np.clip((meas - wc) / np.maximum(wi - wc, 1e-9), -1, 2)
        segq = np.asarray(p["seg_q"]) / 1e6
        print(f"{b} | {st} n={p.get('n', 0)} K={K} | {np.round(meas, 3)} | {np.round(wm, 3)} | {np.round(wi, 3)} | {np.round(wc, 3)} | {frac.mean():.2f} "
              f"| {p['wall_median'] / 1e6:.3f}/{p['wall_mean'] / 1e6:.3f} | {med['wall'][0] / 1e6:.3f} | {(segq[:, 0, 8] - segq[:, 0, 1]).mean():.3f}")


if __name__ == "__main__":
    main()
