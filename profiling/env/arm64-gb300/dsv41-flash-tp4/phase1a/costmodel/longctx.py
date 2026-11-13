"""Component 4: prefill GPU compute per kernel group, incl. the long-context sparse indexer and
sparse-metadata KV-offset dependence.

Per kernel group g (fmha, indexer, sparse_meta, moe, other) and per rank r the kernel-duration sum
of an eager/ring iteration is modelled as a linear function of structural features of the
iteration:

    y_{g,r} = a + c_T · T + c_Q · Q + c_K · KVW + c_b · b

    T   = prefill tokens in the iteration
    Q   = Σ_i t_i² / 16384 over prefill chunks (causal attention inside a chunk grows with t²)
    KVW = Σ_i t_i · kv0_i / 16384  (KV depth seen by the indexer, weighted by chunk tokens;
          for a single 16k chunk at offset kv0 this is ≈ kv0, so c_K is the ms-per-KV-token slope)
    b   = decode requests processed in the same eager forward

Fitted by least squares on calibration mixed iterations only (graph-path iterations excluded: they
run different kernels).  The dedicated `chunk_slope_fit` fits y = a + s·kv0 on calibration clean
16k chunks alone; that is the direct test of the "0.26 ms per 1k KV" planning result and the model
used for the held-out chain / offset-band evaluation.

Status: `in-range` when T, KVW and b lie inside the calibration ranges; `extrapolation` otherwise
(e.g. kv0 beyond the deepest calibration chunk).  No prediction is made silently outside the range.
"""
from __future__ import annotations

import numpy as np

from . import groups as G
from .data import NRANK, Feat, PassData

COMPUTE_GROUPS = ("fmha", "indexer", "sparse_meta", "moe", "other")
FEATURES = ("const", "T", "Q", "KVW", "b")


def design_row(f: Feat) -> np.ndarray:
    # KVW = Σ t_i·kv0_i / 16384 / 1000: thousands of KV tokens at 16k-chunk weight (single 16k chunk: kv0 / 1000)
    return np.array([1.0, f.P, f.sum_t2 / 16384.0, f.kv_weighted / 1e3, f.b], dtype=np.float64)


def _lstsq(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, float(np.sqrt(np.mean(resid ** 2))) if len(y) else float("nan")


class PrefillComputeModel:
    def __init__(self):
        self.coef: dict[str, np.ndarray] = {}       # group -> (NRANK, 5) coefficients in ns
        self.rmse: dict[str, list] = {}
        self.ranges: dict[str, tuple] = {}
        self.n_fit = 0
        self.idle_ns: dict[str, list] = {}           # mixed kind -> per-rank median idle
        self.overlap_cc_ns: dict[str, list] = {}
        self.wait_ns: dict[str, list] = {}           # mixed kind -> per-rank median collective wait (ring only is used)
        self.chunk: dict = {}                        # chunk_slope_fit result

    def fit(self, passes: list[PassData], partition) -> "PrefillComputeModel":
        X, Y, kinds = [], [], []
        acc: dict[str, list] = {}
        for pd in passes:
            if pd.label not in partition.calibration_passes:
                continue
            ids = [f.iteration for f in pd.feats if f.stage == "mixed" and not f.excluded and f.role == "calibration" and f.mixed_kind != "graph"]
            partition.calibration_only(pd.label, ids)
            byf = {f.iteration: f for f in pd.feats}
            for it in ids:
                f = byf[it]
                X.append(design_row(f))
                Y.append(pd.group_sums(it))                   # (NRANK, 6) ns
                kinds.append(f.mixed_kind)
                rows = pd.rank_rows(it)
                c = pd.collectives(it)
                wait = None
                if c is not None and not c.get("_kind_mismatch"):
                    dur = c["end_rel_ns"] - c["start_rel_ns"]
                    wait = (dur - dur.min(axis=0)).sum(axis=1)
                acc.setdefault(f.mixed_kind, []).append((pd.ir["idle_ns"][rows], pd.ir["overlap_comp_comp_ns"][rows], wait))
        X = np.array(X)
        Y = np.array(Y)
        self.n_fit = len(X)
        for gi, g in enumerate(G.GROUPS):
            if g not in COMPUTE_GROUPS:
                continue
            co = np.zeros((NRANK, len(FEATURES)))
            rm = []
            for r in range(NRANK):
                c, e = _lstsq(X, Y[:, r, gi].astype(np.float64))
                co[r] = c
                rm.append(e)
            self.coef[g] = co
            self.rmse[g] = rm
        self.ranges = {"T": (float(X[:, 1].min()), float(X[:, 1].max())), "Q": (float(X[:, 2].min()), float(X[:, 2].max())),
                       "KVW_k": (float(X[:, 3].min()), float(X[:, 3].max())), "b": (float(X[:, 4].min()), float(X[:, 4].max()))}
        for mk, vals in acc.items():
            self.idle_ns[mk] = [float(x) for x in np.median(np.stack([v[0] for v in vals]), axis=0)]
            self.overlap_cc_ns[mk] = [float(x) for x in np.median(np.stack([v[1] for v in vals]), axis=0)]
            ws = [v[2] for v in vals if v[2] is not None]
            if ws:
                self.wait_ns[mk] = [float(x) for x in np.median(np.stack(ws), axis=0)]
        self.chunk = chunk_slope_fit(passes, partition)
        return self

    def status(self, f: Feat) -> str:
        x = design_row(f)
        r = self.ranges
        if not (r["T"][0] <= x[1] <= r["T"][1]):
            return "extrapolation"
        if not (r["KVW_k"][0] <= x[3] <= r["KVW_k"][1] * 1.0001):
            return "extrapolation"
        if not (r["b"][0] <= x[4] <= r["b"][1]):
            return "extrapolation"
        if len(f.chunks) > 1 and f.kv_weighted > 0:
            return "inferred-structure"      # multi-chunk with KV offsets: only a handful of calibration rows (gap G1)
        return "in-range"

    def predict_groups(self, f: Feat) -> np.ndarray:
        """(NRANK, len(GROUPS)) predicted kernel-duration sums in ns (comm column left 0)."""
        x = design_row(f)
        out = np.zeros((NRANK, len(G.GROUPS)))
        for g, co in self.coef.items():
            out[:, G.GROUP_INDEX[g]] = np.clip(co @ x, 0, None)
        return out

    def to_dict(self) -> dict:
        return dict(features=list(FEATURES), coef_ns={g: c.tolist() for g, c in self.coef.items()}, rmse_ns=self.rmse, ranges=self.ranges,
                    n_fit=self.n_fit, idle_ns=self.idle_ns, overlap_cc_ns=self.overlap_cc_ns, wait_ns=self.wait_ns, chunk=self.chunk)

    @classmethod
    def from_dict(cls, d: dict) -> "PrefillComputeModel":
        m = cls()
        m.coef = {g: np.array(c) for g, c in d["coef_ns"].items()}
        m.rmse, m.ranges, m.n_fit = d["rmse_ns"], {k: tuple(v) for k, v in d["ranges"].items()}, d["n_fit"]
        m.idle_ns, m.overlap_cc_ns, m.wait_ns, m.chunk = d["idle_ns"], d["overlap_cc_ns"], d["wait_ns"], d["chunk"]
        return m


def chunk_slope_fit(passes: list[PassData], partition) -> dict:
    """y_g = a_g + s_g · kv0 on calibration clean 16k chunks (slowest-rank group sums and per-rank busy)."""
    rows = []
    for pd in passes:
        if pd.label not in partition.calibration_passes:
            continue
        ids = [f.iteration for f in pd.feats if f.is_chunk16k and not f.excluded and f.role == "calibration"]
        partition.calibration_only(pd.label, ids)
        byf = {f.iteration: f for f in pd.feats}
        for it in ids:
            f = byf[it]
            gs = pd.group_sums(it)                       # (R, 6)
            r = pd.rank_rows(it)
            rows.append(dict(kv0=f.kv0, groups=gs, busy=pd.ir["busy_ns"][r], idle=pd.ir["idle_ns"][r], wall=f.wall_ns, chain=f.chain, label=pd.label))
    if len(rows) < 3:
        return {"n": len(rows)}
    kv = np.array([r["kv0"] for r in rows], dtype=np.float64) / 1e3          # thousands of KV tokens
    X = np.stack([np.ones_like(kv), kv], axis=1)
    out = {"n": len(rows), "kv0_min": float(kv.min() * 1e3), "kv0_max": float(kv.max() * 1e3), "chains": sorted({r["chain"] for r in rows}), "groups": {}}
    for gi, g in enumerate(G.GROUPS):
        y = np.array([r["groups"][:, gi].max() for r in rows]) / 1e6          # slowest-rank group time, ms
        coef, rmse = _lstsq(X, y)
        out["groups"][g] = dict(a_ms=float(coef[0]), s_ms_per_1k=float(coef[1]), rmse_ms=rmse)
    y = np.array([r["busy"].max() for r in rows]) / 1e6
    coef, rmse = _lstsq(X, y)
    out["busy_slowest"] = dict(a_ms=float(coef[0]), s_ms_per_1k=float(coef[1]), rmse_ms=rmse)
    y = np.array([r["wall"] for r in rows]) / 1e6
    coef, rmse = _lstsq(X, y)
    out["wall"] = dict(a_ms=float(coef[0]), s_ms_per_1k=float(coef[1]), rmse_ms=rmse)
    # quadratic check
    X2 = np.stack([np.ones_like(kv), kv, kv ** 2], axis=1)
    coef2, rmse2 = _lstsq(X2, y)
    out["wall_quadratic"] = dict(a_ms=float(coef2[0]), s_ms_per_1k=float(coef2[1]), q_ms_per_1k2=float(coef2[2]), rmse_ms=rmse2,
                                 q_term_at_max_ms=float(coef2[2] * kv.max() ** 2))
    out["idle_ms_median_per_rank"] = [float(x) for x in np.median(np.stack([r["idle"] for r in rows]), axis=0) / 1e6]
    return out


def chunk_predict(chunk: dict, kv0: int, what: str = "wall") -> tuple[float, str]:
    p = chunk[what]
    status = "in-range" if chunk["kv0_min"] <= kv0 <= chunk["kv0_max"] else "extrapolation"
    return p["a_ms"] + p["s_ms_per_1k"] * kv0 / 1e3, status
