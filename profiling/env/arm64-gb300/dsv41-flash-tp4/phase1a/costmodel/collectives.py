"""Component 5: TP4 collective communication.

Two sub-models:
* decode (graph path): per-call intrinsic cost per kind at batch b lives in the decode table
  (DecodeModel.table[b]["intrinsic"]); this module only summarises it.
* eager / ring (prefill chunks and mixed iterations): intrinsic cost per call is linear in the
  payload tokens (P + b) per kind, fitted on calibration mixed iterations; call counts per kind are
  structural (81 all-reduce + 5 all-gather per forward, measured).

Intrinsic cost = min over ranks of the call duration (no rank waits).  Waiting time is NOT part of
this component: it is produced by the composition (component 6) from rank arrival skew, or, for
eager iterations, by the host model (component 2/7).  Measured wait medians are kept as evidence.
"""
from __future__ import annotations

import numpy as np

from . import groups as G
from .data import Feat, PassData

EAGER_KINDS = ("twoshot", "nccl_allreduce", "nccl_allgather")


def _lstsq(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    rmse = float(np.sqrt(np.mean(resid ** 2))) if len(y) else float("nan")
    return coef, rmse


class CollectiveModel:
    def __init__(self):
        self.eager: dict[str, dict] = {}        # kind -> {a_us, c_us_per_token, tokens_min, tokens_max, n, rmse_us}
        self.counts: dict[str, dict] = {}       # mixed kind -> {comm kind: calls}
        self.wait_ms: dict[str, list] = {}      # mixed kind -> per-rank median Σ wait (evidence only)

    def fit(self, passes: list[PassData], partition) -> "CollectiveModel":
        rows: dict[str, list] = {k: [] for k in EAGER_KINDS}
        counts: dict[str, list] = {}
        waits: dict[str, list] = {}
        for pd in passes:
            if pd.label not in partition.calibration_passes:
                continue
            ids = [f.iteration for f in pd.feats if f.stage == "mixed" and not f.excluded and f.role == "calibration"]
            partition.calibration_only(pd.label, ids)
            byf = {f.iteration: f for f in pd.feats}
            for it in ids:
                c = pd.collectives(it)
                if c is None or c.get("_kind_mismatch"):
                    continue
                f = byf[it]
                dur = c["end_rel_ns"] - c["start_rel_ns"]
                mn = dur.min(axis=0)
                kinds = c["kind"][0]
                cnt = {}
                for ki, kname in enumerate(G.COMM_KINDS):
                    m = kinds == ki
                    if not m.any():
                        continue
                    cnt[kname] = int(m.sum())
                    if kname in rows:
                        rows[kname].append((f.total_tokens, float(mn[m].mean()) / 1e3, f.mixed_kind))
                counts.setdefault(f.mixed_kind, []).append(cnt)
                waits.setdefault(f.mixed_kind, []).append((dur - mn).sum(axis=1) / 1e6)
        for kname, rs in rows.items():
            if len(rs) < 2:
                continue
            t = np.array([r[0] for r in rs], dtype=np.float64)
            y = np.array([r[1] for r in rs], dtype=np.float64)
            X = np.stack([np.ones_like(t), t], axis=1)
            coef, rmse = _lstsq(X, y)
            self.eager[kname] = dict(a_us=float(coef[0]), c_us_per_token=float(coef[1]), tokens_min=int(t.min()), tokens_max=int(t.max()),
                                     n=len(rs), rmse_us=rmse, status="measured",
                                     note="per-call intrinsic (min over ranks) vs payload tokens P+b; payload bytes not in trace (inferred from tokens x hidden)")
        for mk, cs in counts.items():
            keys = set().union(*[set(c) for c in cs])
            self.counts[mk] = {k: int(np.median([c.get(k, 0) for c in cs])) for k in sorted(keys)}
        for mk, ws in waits.items():
            self.wait_ms[mk] = [float(x) for x in np.median(np.stack(ws), axis=0)]
        return self

    def intrinsic_us(self, kind: str, tokens: int) -> tuple[float, str]:
        p = self.eager[kind]
        status = "in-range" if p["tokens_min"] <= tokens <= p["tokens_max"] else "extrapolation"
        return max(0.0, p["a_us"] + p["c_us_per_token"] * tokens), status

    def forward_comm_ns(self, mixed_kind: str, tokens: int) -> tuple[float, str]:
        """Σ intrinsic over all collective calls of one eager/ring forward."""
        allreduce = "nccl_allreduce" if mixed_kind == "ring" else "twoshot"
        cnt = self.counts.get(mixed_kind) or self.counts.get("eager")
        total, status = 0.0, "in-range"
        for kname in (allreduce, "nccl_allgather"):
            if kname not in self.eager:
                continue
            us, st = self.intrinsic_us(kname, tokens)
            total += us * cnt.get(kname, 0) * 1e3
            if st != "in-range":
                status = st
        return total, status

    def to_dict(self) -> dict:
        return dict(eager=self.eager, counts=self.counts, wait_ms=self.wait_ms)

    @classmethod
    def from_dict(cls, d: dict) -> "CollectiveModel":
        m = cls()
        m.eager, m.counts, m.wait_ms = d["eager"], d["counts"], d["wait_ms"]
        return m
