"""Component 2 (+ the eager part of component 7): mixed prefill/decode iterations.

Three observed kinds (data.mixed_kind):
* graph  — P + b <= 256 tokens: the cudagraph path is used; modelled as measured class medians only.
* eager  — 256 < P + b < 8192: every kernel is launched eagerly (≈1,500 launches per rank, cute-DSL
           twoshot all-reduce); the iteration is normally host-launch-bound.
* ring   — P + b >= 8192 (NCCL ring all-reduce; incl. the 16k chunks): normally GPU-bound.

For eager and ring the iteration time is composed from two paths:
    host path  H = max_r [ c_r · L_base(P, b, chunks) + c_kv · L_kv(KV) ] + tail
    GPU path   G = max_r [ compute_r (component 4) − overlap + Σ intrinsic (component 5) + wait + idle ]
    wall       = max(G, H − lead)
`lead` is how far the host had already run ahead when the previous iteration's GPU work finished
(previous GPU end − this iteration's earliest host start); it is a property of the iteration
sequence, measured from the timeline when evaluating single iterations (reported next to the
standalone prediction with lead = 0) and to be produced by a sequence simulator in Phase 1B.

Host parameters: per-rank per-launch cost c_r (forward window / launches on host-bound calibration
iterations), per-launch CUDA API cost c_api, GPU tail after the last launch, and the launch count
L = l0 + l1·P + l2·(chunks − 1) + l3·KV16 (KV16 = Σ tokens_i·kv0_i / 16384², the KV depth scanned by
the indexer in 16k-chunk equivalents).  The cost c_kv of the KV-dependent launches is inferred
from GPU-bound iterations after removing launch back-pressure (launch API time inflates from ≈14 µs
to >30 µs per launch when the GPU is behind), so it is classified `inferred`.

Host decomposition inside the device's NVTX window (measured per rank, reported per class):
  CPU-side preparation = host window − forward window, CUDA API time = Σ API call durations
  (launch / sync / query / memcpy / other), host launch gaps = forward window − API time inside the
  forward (Python between launches), GPU execution = kernel union, synchronization = Σ *Synchronize
  durations, host/GPU overlap = kernel union inside the host window.
"""
from __future__ import annotations

import numpy as np

from . import groups as G
from .collectives import CollectiveModel
from .data import NRANK, Feat, PassData
from .longctx import PrefillComputeModel

HOST_BOUND_RATIO = 0.8      # eager iteration counts as host-bound when max_r compute_union < ratio × min_r forward window


def _lstsq(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, float(np.sqrt(np.mean(resid ** 2))) if len(y) else float("nan")


def kv16(f: Feat) -> float:
    return float(sum(c[1] * c[0] for c in f.chunks)) / (16384.0 * 16384.0)


def host_decomposition(pd: PassData, it: int) -> dict:
    rows = pd.rank_rows(it)
    ir = pd.ir
    g = lambda c: ir[c][rows].astype(np.float64)
    fwd = g("fwd_window_ns")
    api = g("api_sum_ns")
    return dict(
        host_window_ns=g("host_window_ns"), fwd_window_ns=fwd, cpu_prep_ns=g("host_window_ns") - fwd,
        api_ns=api, api_launch_ns=g("fwd_launch_sum_ns"), api_sync_ns=g("api_sync_sum_ns"), api_query_ns=g("api_query_sum_ns"),
        api_memcpy_ns=g("api_memcpy_sum_ns"), api_other_ns=g("api_other_sum_ns"), launch_gaps_ns=fwd - g("fwd_api_sum_ns"),
        gpu_busy_ns=g("busy_ns"), gpu_compute_ns=g("compute_union_ns"), gpu_comm_ns=g("comm_union_ns"), gpu_idle_ns=g("idle_ns"),
        host_gpu_overlap_ns=g("gpu_in_host_window_ns"), gpu_tail_after_host_ns=g("gpu_tail_after_host_ns"),
        launches=g("fwd_launch_n"), api_us_per_launch=g("fwd_launch_sum_ns") / np.maximum(g("fwd_launch_n"), 1) / 1e3,
        host_us_per_launch=fwd / np.maximum(g("fwd_launch_n"), 1) / 1e3,
    )


class MixedModel:
    def __init__(self):
        self.launch = {}
        self.c_host_ns = None        # (NRANK,) per-launch host cost
        self.c_api_ns = None         # per-launch CUDA API cost (unthrottled)
        self.outside_fwd_ns = None   # (NRANK,) host window − forward window
        self.tail_ns = None          # wall − max_r forward window on host-bound iterations
        self.c_kv_ns = None          # inferred per-launch cost of KV-dependent launches
        self.c_kv_evidence = {}
        self.graph_table = {}
        self.host_bound_ids = []
        self.n_fit = 0
        self.ranges = {}
        self.lead_evidence = {}
        self.decomposition = {}      # per kind: median host decomposition per rank (ms)

    def fit(self, passes: list[PassData], partition) -> "MixedModel":
        L_rows, hb, ring, graph, leads = [], [], [], [], {}
        decomp: dict[str, list] = {}
        for pd in passes:
            if pd.label not in partition.calibration_passes:
                continue
            ids = [f.iteration for f in pd.feats if f.stage == "mixed" and not f.excluded and f.role == "calibration"]
            partition.calibration_only(pd.label, ids)
            byf = {f.iteration: f for f in pd.feats}
            for it in ids:
                f = byf[it]
                h = host_decomposition(pd, it)
                decomp.setdefault(f.mixed_kind, []).append(h)
                if f.prev_stage:
                    leads.setdefault(f"{f.mixed_kind} after {f.prev_stage}", []).append(f.lead_ns)
                if f.mixed_kind == "graph":
                    graph.append((f, h, pd.label))
                    continue
                L = float(h["launches"][0])
                L_rows.append((f.P, len(f.chunks) - 1, kv16(f), L))
                host_bound = h["gpu_compute_ns"].max() < HOST_BOUND_RATIO * h["fwd_window_ns"].min()
                if f.mixed_kind == "eager" and host_bound and kv16(f) == 0:
                    hb.append((f, h, pd.label))
                    self.host_bound_ids.append((pd.label, it))
                if f.mixed_kind == "ring":
                    ring.append((f, h, pd.label))
        self.n_fit = len(L_rows)
        A = np.array(L_rows, dtype=np.float64)
        X = np.stack([np.ones(len(A)), A[:, 0], A[:, 1], A[:, 2]], axis=1)
        coef, rmse = _lstsq(X, A[:, 3])
        self.launch = dict(l0=float(coef[0]), l1_per_P=float(coef[1]), l2_per_extra_chunk=float(coef[2]), l3_per_kv16=float(coef[3]), rmse=rmse, n=len(A),
                           status="measured (structural launch count; identical on all ranks)")
        self.ranges = dict(P=(float(A[:, 0].min()), float(A[:, 0].max())), chunks=(float(A[:, 1].min() + 1), float(A[:, 1].max() + 1)),
                           kv16=(float(A[:, 2].min()), float(A[:, 2].max())), L=(float(A[:, 3].min()), float(A[:, 3].max())))
        if hb:
            self.c_host_ns = np.median(np.stack([h["fwd_window_ns"] / h["launches"] for _, h, _ in hb]), axis=0)
            self.c_api_ns = float(np.median(np.concatenate([h["api_launch_ns"] / h["launches"] for _, h, _ in hb])))
            self.outside_fwd_ns = np.median(np.stack([h["cpu_prep_ns"] for _, h, _ in hb]), axis=0)
            self.tail_ns = float(np.median([f.wall_ns - h["fwd_window_ns"].max() for f, h, _ in hb]))
        ests = []
        for f, h, lab in ring:
            k16 = kv16(f)
            if k16 <= 0 or self.c_host_ns is None:
                continue
            L = h["launches"][0]
            L_base = self.launch_count(f.P, len(f.chunks), 0.0)
            extra = L - L_base
            if extra < 20:
                continue
            inflation = np.clip(h["api_launch_ns"] - self.c_api_ns * L, 0, None)
            F_clean = h["fwd_window_ns"] - inflation
            ests.extend(((F_clean - self.c_host_ns * L_base) / extra).tolist())
        if ests:
            ests = np.array(ests)
            self.c_kv_ns = float(max(0.0, np.median(ests)))
            self.c_kv_evidence = dict(n=len(ests), median_ns=float(np.median(ests)), p10_ns=float(np.percentile(ests, 10)), p90_ns=float(np.percentile(ests, 90)),
                                      status="inferred (de-confounded from launch back-pressure; the host is not on the critical path in these iterations)")
        else:
            self.c_kv_ns = float(self.c_host_ns.mean()) if self.c_host_ns is not None else 0.0
            self.c_kv_evidence = dict(n=0, status="not identifiable (no KV>0 ring iteration in calibration); upper bound c_host used")
        groups = {}
        for f, h, lab in graph:
            key = (f.P // 64 * 64, f.b)
            groups.setdefault(key, []).append((f.wall_ns, h["gpu_busy_ns"], h["fwd_window_ns"], lab))
        for key, v in groups.items():
            self.graph_table[f"{key[0]},{key[1]}"] = dict(n=len(v), wall_ns=float(np.median([x[0] for x in v])),
                                                          busy_ns=np.median(np.stack([x[1] for x in v]), axis=0).tolist(),
                                                          fwd_ns=np.median(np.stack([x[2] for x in v]), axis=0).tolist(), passes=sorted({x[3] for x in v}))
        for k, v in leads.items():
            a = np.array(v) / 1e6
            self.lead_evidence[k] = dict(n=len(a), median_ms=float(np.median(a)), p90_ms=float(np.percentile(a, 90)), max_ms=float(a.max()), min_ms=float(a.min()),
                                         share_positive=float((a > 0).mean()))
        for k, hs in decomp.items():
            keys = ("host_window_ns", "fwd_window_ns", "cpu_prep_ns", "api_ns", "api_launch_ns", "api_sync_ns", "api_query_ns", "api_memcpy_ns", "api_other_ns",
                    "launch_gaps_ns", "gpu_busy_ns", "gpu_compute_ns", "gpu_comm_ns", "gpu_idle_ns", "host_gpu_overlap_ns", "gpu_tail_after_host_ns")
            self.decomposition[k] = {kk.replace("_ns", "_ms"): (np.median(np.stack([h[kk] for h in hs]), axis=0) / 1e6).round(2).tolist() for kk in keys}
            self.decomposition[k]["launches"] = float(np.median([h["launches"][0] for h in hs]))
            self.decomposition[k]["api_us_per_launch"] = np.median(np.stack([h["api_us_per_launch"] for h in hs]), axis=0).round(1).tolist()
            self.decomposition[k]["host_us_per_launch"] = np.median(np.stack([h["host_us_per_launch"] for h in hs]), axis=0).round(1).tolist()
            self.decomposition[k]["n"] = len(hs)
        return self

    # ---- predict -------------------------------------------------------------------------------
    def launch_count(self, P: int, n_chunks: int, k16: float) -> float:
        l = self.launch
        return l["l0"] + l["l1_per_P"] * P + l["l2_per_extra_chunk"] * max(0, n_chunks - 1) + l["l3_per_kv16"] * k16

    def host_path(self, f: Feat) -> dict:
        k16 = kv16(f)
        L_base = self.launch_count(f.P, len(f.chunks), 0.0)
        L = self.launch_count(f.P, len(f.chunks), k16)
        F = self.c_host_ns * L_base + self.c_kv_ns * max(0.0, L - L_base)
        status = "in-range"
        if not (self.ranges["P"][0] <= f.P <= self.ranges["P"][1]) or not (self.ranges["chunks"][0] <= len(f.chunks) <= self.ranges["chunks"][1]):
            status = "extrapolation"
        elif k16 > 0:
            status = "inferred"
        return dict(launches=L, launches_base=L_base, fwd_ns=F, host_path_ns=float(F.max() + self.tail_ns), slowest_host_rank=int(np.argmax(F)), status=status)

    def predict(self, f: Feat, prefill: PrefillComputeModel, coll: CollectiveModel, lead_ns: float = 0.0) -> dict:
        if f.mixed_kind == "graph":
            key = f"{f.P // 64 * 64},{f.b}"
            if key in self.graph_table:
                t, st = self.graph_table[key], "measured-class"
            elif self.graph_table:
                near = min(self.graph_table, key=lambda k: abs(int(k.split(",")[0]) - f.P) + abs(int(k.split(",")[1]) - f.b))
                t, st = self.graph_table[near], "extrapolation"
            else:
                return dict(component="mixed-graph", status="not-identifiable", wall_ns=float("nan"), wall_standalone_ns=float("nan"), regime="mixed-graph")
            return dict(component="mixed-graph", status=st, wall_ns=t["wall_ns"], wall_standalone_ns=t["wall_ns"], rank_busy_ns=np.array(t["busy_ns"]),
                        host_path_ns=float(max(t["fwd_ns"])), gpu_path_ns=float(max(t["busy_ns"])), regime="mixed-graph", bound="n/a")
        hp = self.host_path(f)
        groups = prefill.predict_groups(f)                         # (R, 6) ns, comm = 0
        compute_sum = groups.sum(axis=1)
        mk = f.mixed_kind
        overlap = np.array(prefill.overlap_cc_ns.get(mk) or prefill.overlap_cc_ns.get("ring") or [0] * NRANK)
        idle = np.array(prefill.idle_ns.get("ring") or [0] * NRANK)   # GPU-inherent idle (measured when GPU-bound); host-induced idle is not a GPU property
        compute_union = np.clip(compute_sum - overlap, 0, None)
        comm_ns, cstat = coll.forward_comm_ns(mk, f.total_tokens)
        wait = np.array(prefill.wait_ns.get("ring") or [0] * NRANK)
        gpu_rank = compute_union + comm_ns + wait + idle
        gpu_path = float(gpu_rank.max())
        host_eff = hp["host_path_ns"] - max(0.0, float(lead_ns))
        wall_standalone = max(gpu_path, hp["host_path_ns"])
        wall = max(gpu_path, host_eff)
        bound = "gpu" if gpu_path >= host_eff else "host"
        pstat = prefill.status(f)
        status = "in-range"
        for s in (pstat, hp["status"], cstat):
            if s == "extrapolation":
                status = "extrapolation"
            elif s in ("inferred", "inferred-structure") and status == "in-range":
                status = "inferred"
        return dict(component=f"mixed-{mk}", regime=f"mixed-{mk}", status=status, wall_ns=wall, wall_standalone_ns=wall_standalone, bound=bound,
                    lead_ns=float(lead_ns), host_path_ns=hp["host_path_ns"], host_path_eff_ns=host_eff, gpu_path_ns=gpu_path, host_fwd_ns=hp["fwd_ns"],
                    launches=hp["launches"], groups_ns=groups, rank_compute_union_ns=compute_union, comm_intrinsic_ns=comm_ns, rank_idle_ns=idle,
                    rank_wait_ns=wait, rank_gpu_path_ns=gpu_rank, prefill_status=pstat, host_status=hp["status"], comm_status=cstat)

    def to_dict(self) -> dict:
        return dict(launch=self.launch, c_host_ns=None if self.c_host_ns is None else self.c_host_ns.tolist(), c_api_ns=self.c_api_ns,
                    outside_fwd_ns=None if self.outside_fwd_ns is None else self.outside_fwd_ns.tolist(), tail_ns=self.tail_ns, c_kv_ns=self.c_kv_ns,
                    c_kv_evidence=self.c_kv_evidence, graph_table=self.graph_table, host_bound_ids=self.host_bound_ids, n_fit=self.n_fit, ranges=self.ranges,
                    lead_evidence_ms=self.lead_evidence, decomposition=self.decomposition,
                    status={"c_host per rank": "measured (host-bound eager calibration iterations)", "c_api": "measured", "tail": "measured",
                            "launch count": "measured (linear fit, rank-independent)", "c_kv": self.c_kv_evidence.get("status"),
                            "lead": "measured from the timeline when evaluating; sequence simulation is Phase 1B",
                            "root cause of rank-2 host slowness": "not identifiable from GPU/API traces (gap G4)"})

    @classmethod
    def from_dict(cls, d: dict) -> "MixedModel":
        m = cls()
        m.launch, m.c_api_ns, m.tail_ns, m.c_kv_ns = d["launch"], d["c_api_ns"], d["tail_ns"], d["c_kv_ns"]
        m.c_host_ns = np.array(d["c_host_ns"]) if d["c_host_ns"] is not None else None
        m.outside_fwd_ns = np.array(d["outside_fwd_ns"]) if d["outside_fwd_ns"] is not None else None
        m.c_kv_evidence, m.graph_table, m.host_bound_ids, m.n_fit = d["c_kv_evidence"], d["graph_table"], d["host_bound_ids"], d["n_fit"]
        m.ranges = {k: tuple(v) for k, v in d["ranges"].items()}
        m.lead_evidence, m.decomposition = d.get("lead_evidence_ms", {}), d.get("decomposition", {})
        return m
