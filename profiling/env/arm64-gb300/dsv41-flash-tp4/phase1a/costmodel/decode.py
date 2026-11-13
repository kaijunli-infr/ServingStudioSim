"""Component 1: decode GPU execution (CUDA-graph path), three kernel-path regimes.

Model: for each decode batch size b the per-rank ordered timeline (K+1 compute windows between the
K collective calls, per-rank first-kernel offset) and the per-call intrinsic collective cost are
measured as per-element empirical quantiles over the calibration iterations of that batch size.
The iteration time is the expected critical path over timelines sampled from those quantiles
(stochastic composition, fixed seed): at each of the K synchronization points the collective waits
for whichever rank is late, so the expectation exceeds the deterministic path through the medians
(reported too, as `wall_det_ns`).  Independence of the sampled windows across ranks and segments is
an assumption (classified `inferred`); the gap between the two estimates is the synchronization
loss the deterministic model misses.

Batch sizes that were not calibrated are linearly interpolated element-wise between the nearest
calibrated batch sizes of the same regime (`interpolated`); outside the calibrated range of a
regime the nearest table entry is used and the prediction is flagged `extrapolation`.  Entries with
fewer than MIN_TABLE_N calibration iterations are evidence only, never used for prediction.

KV independence is an assumption of this component, stated only for the observed KV range
(kv_max <= observed maximum); it is tested in evaluate.py (residual vs kv).

Overlap is explicit: the windows are measured spans (compute union + intra-window idle), and the
multi-stream overlap (Σ kernel durations − union) is carried as a separate measured quantity so that
the sum of kernel-group durations is never used as a time.
"""
from __future__ import annotations

import numpy as np

from . import groups as G
from .compose import critical_path, critical_path_batch
from .data import NRANK, Feat, PassData, decode_regime

REGIMES = ("b<=16", "b17-24", "b>=25")
MIN_TABLE_N = 5          # entries with fewer calibration iterations are kept as evidence but not used for prediction
QUANTILES = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
MC_DRAWS = 256
MC_SEED = 20261113
STAT_KEYS = ("start", "seg", "intrinsic", "wait", "groups", "compute_union", "comm_union", "idle", "busy", "overlap_cc", "overlap_cm")


def _median_stack(arrs: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(arrs, axis=0), axis=0)


def _quantiles(arrs: list[np.ndarray]) -> np.ndarray:
    """(..., len(QUANTILES)) empirical quantiles along the iteration axis."""
    a = np.stack(arrs, axis=0).astype(np.float64)
    return np.moveaxis(np.quantile(a, QUANTILES, axis=0), 0, -1)


def _sample(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF sampling: q (..., Q) quantile values, u (N, ...) uniforms → (N, ...) samples."""
    idx = np.clip(np.searchsorted(QUANTILES, u, side="right") - 1, 0, len(QUANTILES) - 2)
    q0 = np.take_along_axis(np.broadcast_to(q, u.shape + (len(QUANTILES),)), idx[..., None], axis=-1)[..., 0]
    q1 = np.take_along_axis(np.broadcast_to(q, u.shape + (len(QUANTILES),)), idx[..., None] + 1, axis=-1)[..., 0]
    w = (u - QUANTILES[idx]) / (QUANTILES[idx + 1] - QUANTILES[idx])
    return q0 + np.clip(w, 0, 1) * (q1 - q0)


class DecodeModel:
    def __init__(self):
        self.table: dict[int, dict] = {}      # b -> parameters
        self.kv_max_observed = 0
        self.kv_sum_observed = 0
        self.n_fit = 0

    # ---- fit -----------------------------------------------------------------------------------
    def fit(self, passes: list[PassData], partition) -> "DecodeModel":
        by_b: dict[int, list] = {}
        for pd in passes:
            if pd.label not in partition.calibration_passes:
                continue
            ids = [f.iteration for f in pd.feats if f.stage == "decode" and not f.excluded and f.role == "calibration"]
            partition.calibration_only(pd.label, ids)       # leakage guard
            byf = {f.iteration: f for f in pd.feats}
            for it in ids:
                seg = pd.segments(it)
                if seg is None or pd.collectives(it).get("_kind_mismatch"):
                    continue
                start_rel, seg_len, intrinsic, dur = seg
                f = byf[it]
                rows = pd.rank_rows(it)
                rec = dict(
                    start=start_rel, seg=seg_len, intrinsic=intrinsic, wait=(dur - intrinsic).sum(axis=1),
                    groups=pd.group_sums(it), kinds=pd.comm_kind_counts(it)[0],
                    compute_union=pd.ir["compute_union_ns"][rows], comm_union=pd.ir["comm_union_ns"][rows],
                    idle=pd.ir["idle_ns"][rows], busy=pd.ir["busy_ns"][rows],
                    overlap_cc=pd.ir["overlap_comp_comp_ns"][rows], overlap_cm=pd.ir["overlap_comp_comm_ns"][rows],
                    wall=f.wall_ns, kv_max=f.kv_max, kv_sum=f.kv_sum, label=pd.label,
                )
                by_b.setdefault(f.b, []).append(rec)
                self.kv_max_observed = max(self.kv_max_observed, f.kv_max)
                self.kv_sum_observed = max(self.kv_sum_observed, f.kv_sum)
                self.n_fit += 1
        for b, recs in sorted(by_b.items()):
            Ks = {r["seg"].shape[1] for r in recs}
            if len(Ks) != 1:
                K_modal = max(Ks, key=lambda k: sum(1 for r in recs if r["seg"].shape[1] == k))
                recs = [r for r in recs if r["seg"].shape[1] == K_modal]
            entry = dict(n=len(recs), regime=decode_regime(b), K=int(recs[0]["seg"].shape[1] - 1), kinds=recs[0]["kinds"],
                         wall_median=float(np.median([r["wall"] for r in recs])), wall_mean=float(np.mean([r["wall"] for r in recs])),
                         passes=sorted({r["label"] for r in recs}))
            for key in STAT_KEYS:
                entry[key] = _median_stack([r[key] for r in recs])
            for key in ("start", "seg", "intrinsic"):
                entry[key + "_q"] = _quantiles([r[key] for r in recs])
            self.table[b] = entry
        return self

    # ---- predict -------------------------------------------------------------------------------
    def _lookup(self, b: int) -> tuple[dict, str]:
        if b in self.table and self.table[b]["n"] >= MIN_TABLE_N:
            return self.table[b], "measured"
        reg = decode_regime(b)
        bs = sorted(k for k, v in self.table.items() if v["regime"] == reg and v["n"] >= MIN_TABLE_N)
        if not bs:
            raise KeyError(f"no calibrated batch size in regime {reg}")
        lo = [k for k in bs if k < b]
        hi = [k for k in bs if k > b]
        if lo and hi:
            a, c = self.table[lo[-1]], self.table[hi[0]]
            if a["K"] != c["K"]:
                return (a if (b - lo[-1]) <= (hi[0] - b) else c), "interpolated-nearest"
            w = (b - lo[-1]) / (hi[0] - lo[-1])
            out = {}
            for key in STAT_KEYS + ("start_q", "seg_q", "intrinsic_q"):
                out[key] = (1 - w) * a[key] + w * c[key]
            out.update(n=0, regime=reg, K=a["K"], kinds=a["kinds"], wall_median=float((1 - w) * a["wall_median"] + w * c["wall_median"]),
                       wall_mean=float((1 - w) * a["wall_mean"] + w * c["wall_mean"]),
                       passes=sorted(set(a["passes"]) | set(c["passes"])), interpolated_between=(lo[-1], hi[0]))
            return out, "interpolated"
        near = lo[-1] if lo else hi[0]
        return self.table[near], "extrapolation"

    def _stochastic(self, p: dict) -> dict:
        rng = np.random.default_rng(MC_SEED)
        R, K1 = p["seg"].shape
        start = _sample(p["start_q"], rng.random((MC_DRAWS, R)))
        seg = _sample(p["seg_q"], rng.random((MC_DRAWS, R, K1)))
        intr = _sample(p["intrinsic_q"], rng.random((MC_DRAWS, K1 - 1)))
        res = critical_path_batch(start, seg, intr)
        return dict(wall=float(res["wall"].mean()), wall_p10=float(np.percentile(res["wall"], 10)), wall_p90=float(np.percentile(res["wall"], 90)),
                    rank_end=res["rank_end"].mean(axis=0), rank_wait=res["rank_wait"].mean(axis=0), comm_intrinsic=float(intr.sum(axis=1).mean()),
                    slowest_rank=int(np.argmax(res["critical_count"].sum(axis=0))))

    def predict(self, f: Feat) -> dict:
        p, status = self._lookup(f.b)
        det = critical_path(p["start"], p["seg"], p["intrinsic"])
        mc = self._stochastic(p)
        comm_union = mc["comm_intrinsic"] + mc["rank_wait"]          # per rank: intrinsic + wait
        compute_union = p["compute_union"]
        busy = compute_union + comm_union - p["overlap_cm"]
        kv_status = "in-range" if f.kv_max <= self.kv_max_observed else "kv-extrapolation"
        return dict(
            component="decode", regime=p["regime"], status=status, kv_status=kv_status, K=p["K"],
            wall_ns=mc["wall"], wall_det_ns=det.wall, wall_p10_ns=mc["wall_p10"], wall_p90_ns=mc["wall_p90"],
            rank_end_ns=mc["rank_end"], rank_wait_ns=mc["rank_wait"], rank_wait_det_ns=det.rank_wait, slowest_rank=mc["slowest_rank"],
            rank_compute_union_ns=compute_union, rank_comm_union_ns=comm_union, rank_idle_ns=p["idle"],
            rank_busy_ns=busy, rank_overlap_cc_ns=p["overlap_cc"], rank_overlap_cm_ns=p["overlap_cm"],
            groups_ns=p["groups"], comm_intrinsic_ns=mc["comm_intrinsic"], kinds=p["kinds"],
            table_wall_median_ns=p["wall_median"], table_wall_mean_ns=p["wall_mean"], n_table=p["n"],
        )

    # ---- serialization -------------------------------------------------------------------------
    def to_dict(self) -> dict:
        out = {"kv_max_observed": self.kv_max_observed, "kv_sum_observed": self.kv_sum_observed, "n_fit": self.n_fit, "min_table_n": MIN_TABLE_N,
               "quantiles": QUANTILES.tolist(), "mc_draws": MC_DRAWS, "mc_seed": MC_SEED, "table": {}}
        for b, p in self.table.items():
            out["table"][str(b)] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "DecodeModel":
        m = cls()
        m.kv_max_observed, m.kv_sum_observed, m.n_fit = d["kv_max_observed"], d["kv_sum_observed"], d["n_fit"]
        for b, p in d["table"].items():
            m.table[int(b)] = {k: (np.array(v) if isinstance(v, list) and k not in ("passes",) else v) for k, v in p.items()}
        return m

    def summary_rows(self) -> list[dict]:
        rows = []
        for b, p in sorted(self.table.items()):
            det = critical_path(p["start"], p["seg"], p["intrinsic"])
            mc = self._stochastic(p)
            rows.append(dict(b=b, regime=p["regime"], n=p["n"], K=p["K"], wall_mc_ms=mc["wall"] / 1e6, wall_det_ms=det.wall / 1e6,
                             wall_median_ms=p["wall_median"] / 1e6, wall_mean_ms=p["wall_mean"] / 1e6,
                             compute_union_ms=[round(x / 1e6, 3) for x in p["compute_union"]],
                             comm_intrinsic_ms=mc["comm_intrinsic"] / 1e6, wait_mc_ms=[round(x / 1e6, 3) for x in mc["rank_wait"]],
                             wait_measured_ms=[round(x / 1e6, 3) for x in p["wait"]],
                             idle_ms=[round(x / 1e6, 3) for x in p["idle"]], overlap_cc_ms=[round(x / 1e6, 3) for x in p["overlap_cc"]],
                             groups_r0_ms={g: round(p["groups"][0][i] / 1e6, 3) for i, g in enumerate(G.GROUPS)},
                             kinds={k: int(p["kinds"][i]) for i, k in enumerate(G.COMM_KINDS) if p["kinds"][i]}, passes=p["passes"]))
        return rows
