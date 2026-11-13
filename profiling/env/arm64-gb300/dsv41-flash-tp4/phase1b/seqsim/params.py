"""Sequence-level parameters (Phase 1B), estimated on calibration-role iterations only.

Everything here is about *how consecutive iterations couple*; per-iteration GPU costs come from the
Phase 1A coefficients. All quantities are measured directly from the host (NVTX/API) and GPU
timelines of the frozen traces; each carries a status (measured / inferred / not identifiable).

Rules implemented by the engine (see engine.py) and the parameters they need:

  host_start_0(i) = max(host_end_0(i-1) + a_host,  GPU_end(i-2) + eps_sched[kind_i, b_{i-2}])
  host_start_r(i) = max(host_end_r(i-1) + a_host,  host_start_0(i))              (scheduler broadcast)
  first GPU kernel of i on rank r  >= host_start_r(i) + f_idle                    (preprocess copies)
  forward launches start at         host_start_r(i) + h_pre[kind][r]
  each launch j completes no earlier than the GPU completion of launch j - Q      (launch queue)
  host_end_r(i) = last launch + p_post[kind][r]
  decode (CUDA graph): host forward window = hostwin_dec[b][r], n_launch_dec[b]; GPU path from Phase 1A tables
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from costmodel import groups as G
from costmodel.data import NRANK, PassData, mixed_kind

MIN_N = 5                 # table entries below this many calibration rows are interpolated, not used
LEAD_DENSE_NS = 20e6      # ring iterations whose host ran >= 20 ms ahead: GPU timeline is launch-independent
QUANTILES = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])


def _med(a):
    return float(np.median(np.asarray(a, dtype=np.float64)))


def _q(a):
    return np.quantile(np.asarray(a, dtype=np.float64), QUANTILES)


def kind_of(f) -> str:
    if f.stage == "decode":
        return "decode"
    return mixed_kind(f.P, f.b)


@dataclass
class SeqParams:
    eps_sched_ns: dict = field(default_factory=dict)      # kind -> {b_prev2: {median, n}} (decode) or {"*": {...}}
    a_host_ns: list = field(default_factory=list)         # per rank
    f_idle_ns: float = 0.0
    delta_launch_ns: float = 20_000.0                     # host launch -> GPU start latency for an individual launch (inferred)
    h_pre_ns: dict = field(default_factory=dict)          # kind -> per rank
    p_post_ns: dict = field(default_factory=dict)         # kind -> per rank
    hostwin_dec_ns: dict = field(default_factory=dict)    # b -> per rank host forward window (graph launch)
    n_launch_dec: dict = field(default_factory=dict)      # b -> launches per decode iteration
    queue_depth: float = 0.0                              # Q, launches
    queue_evidence: dict = field(default_factory=dict)
    profile: dict = field(default_factory=dict)           # mixed kind -> {K, kinds (K,), share (R,K+1), share_q (R,K+1,10), n}
    idle_share: dict = field(default_factory=dict)        # mixed kind -> per-rank inherent GPU idle fraction of the forward
    graph_mixed: dict = field(default_factory=dict)       # mixed-graph classes: host window per rank, launches
    disp_q: dict = field(default_factory=dict)            # measured common-mode dispersion: decode -> {b: {q, n}}, ring -> {q, n} (wall / reference)
    status: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    n_fit: int = 0

    # ------------------------------------------------------------------------------------------
    @classmethod
    def fit(cls, passes: list[PassData], partition, coeffs: dict) -> "SeqParams":
        p = cls()
        eps: dict[str, dict] = {}
        a_host = [[] for _ in range(NRANK)]
        f_idle = []
        h_pre: dict[str, list] = {}
        p_post: dict[str, list] = {}
        hostwin: dict[int, list] = {}
        nl_dec: dict[int, list] = {}
        q_rows = []
        prof: dict[str, list] = {}
        idle_rows: dict[str, list] = {}
        graph_rows: dict[str, list] = {}
        disp_dec: dict[int, list] = {}
        disp_ring: list = []
        from costmodel.longctx import chunk_predict
        chunk_fit = (coeffs.get("prefill") or {}).get("chunk") or {}
        for pd in passes:
            if pd.label not in partition.calibration_passes:
                continue
            feats = {f.iteration: f for f in pd.feats}
            cal = [f.iteration for f in pd.feats if f.role == "calibration" and not f.excluded]
            partition.calibration_only(pd.label, cal)
            cal_set = set(cal)
            for it in cal:
                f = feats[it]
                k = kind_of(f)
                rows = pd.rank_rows(it)
                hs = pd.ir["host_start_ns"][rows].astype(np.float64)
                he = pd.ir["host_end_ns"][rows].astype(np.float64)
                fs = pd.ir["fwd_start_ns"][rows].astype(np.float64)
                fe = pd.ir["fwd_end_ns"][rows].astype(np.float64)
                nl = pd.ir["fwd_launch_n"][rows].astype(np.float64)
                h_pre.setdefault(k, []).append(fs - hs)
                p_post.setdefault(k, []).append(he - fe)
                if k == "decode":
                    hostwin.setdefault(f.b, []).append(fe - fs)
                    nl_dec.setdefault(f.b, []).append(nl)
                    disp_dec.setdefault(f.b, []).append(float(f.wall_ns))
                if k == "ring" and f.is_chunk16k and chunk_fit:
                    ref, _ = chunk_predict(chunk_fit, f.kv0, "wall")
                    if ref and ref > 0:
                        disp_ring.append(float(f.wall_ns) / float(ref))
                if k == "graph":
                    key = f"{f.P // 64 * 64},{f.b}"
                    graph_rows.setdefault(key, []).append(dict(hostwin=(fe - fs).tolist(), launches=nl.tolist(), wall=f.wall_ns))
                # ---- coupling to the previous iterations (all three must be calibration, non-excluded) ----
                f1 = feats.get(it - 1)
                f2 = feats.get(it - 2)
                if f1 is None or f1.iteration not in cal_set:
                    continue
                r1 = pd.rank_rows(f1.iteration)
                he1 = pd.ir["host_end_ns"][r1].astype(np.float64)
                prev_kind = kind_of(f1)
                if prev_kind != "decode":
                    # host serial constraint is the binding one after a mixed iteration
                    for r in range(NRANK):
                        a_host[r].append(hs[r] - he1[r])
                if f2 is not None and f2.iteration in cal_set and prev_kind == "decode":
                    # scheduler constraint binds after a decode iteration
                    d = eps.setdefault(k, {})
                    key = int(f2.b) if k == "decode" else "*"
                    d.setdefault(key, []).append(hs[0] - f2.t1_ns)
                if f.t0_ns - f1.t1_ns > 0.5e6:
                    f_idle.append(f.t0_ns - hs.min())
                if k == "ring":
                    wall = max(1.0, float(f.wall_ns))
                    for r in range(NRANK):
                        d2 = f.t1_ns - he[r]
                        q_rows.append((it, r, d2 * nl[r] / wall, d2 / 1e6, (f1.t1_ns - hs[r]) / 1e6))
                if k in ("ring", "eager"):
                    seg = pd.segments(it)
                    c = pd.collectives(it)
                    if seg is None or c.get("_kind_mismatch"):
                        continue
                    _, seg_len, _, _ = seg
                    cu = np.concatenate([c["seg_compute_union_ns"], c["tail_compute_union_ns"][:, :1]], axis=1).astype(np.float64)
                    lead = f1.t1_ns - hs.min()
                    kinds = c["kind"][0]
                    comp_union = pd.ir["compute_union_ns"][rows].astype(np.float64)
                    idle = pd.ir["idle_ns"][rows].astype(np.float64)
                    prof.setdefault(k, []).append(dict(seg=seg_len.astype(np.float64), cu=cu, kinds=kinds, lead=lead, idle=idle, comp=comp_union))
        p.n_fit = sum(len(v) for v in h_pre.values())
        # ---- scheduler epsilon ------------------------------------------------------------------
        for k, d in eps.items():
            p.eps_sched_ns[k] = {str(b): dict(median=_med(v), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)), n=len(v))
                                 for b, v in sorted(d.items(), key=lambda kv: (str(type(kv[0])), kv[0]))}
        p.a_host_ns = [_med(a) if a else 0.0 for a in a_host]
        p.f_idle_ns = _med(f_idle) if f_idle else 0.0
        p.h_pre_ns = {k: np.median(np.stack(v), axis=0).tolist() for k, v in h_pre.items()}
        p.p_post_ns = {k: np.median(np.stack(v), axis=0).tolist() for k, v in p_post.items()}
        p.hostwin_dec_ns = {str(b): dict(median=np.median(np.stack(v), axis=0).tolist(), n=len(v)) for b, v in sorted(hostwin.items())}
        p.n_launch_dec = {str(b): dict(median=float(np.median(np.stack(v))), n=len(v)) for b, v in sorted(nl_dec.items())}
        # ---- launch queue depth: the host can be at most Q launches ahead of the GPU --------------------
        if q_rows:
            qa = np.array([r[2] for r in q_rows])
            lead = np.array([r[4] for r in q_rows])
            sat = qa[lead >= 60.0]
            p.queue_depth = float(qa.max())
            p.queue_evidence = dict(n_rank_rows=len(qa), implied_p50=float(np.median(qa)), implied_p90=float(np.percentile(qa, 90)),
                                    implied_max=float(qa.max()), saturated_n=int(len(sat)),
                                    saturated_p50=float(np.median(sat)) if len(sat) else None, saturated_p90=float(np.percentile(sat, 90)) if len(sat) else None,
                                    rule="Q = max over calibration ring rank-rows of (GPU_end - host_end_r) * launches_r / wall (lower bound of the hard cap); nominal CUDA queue 1024")
        # ---- mixed segment profiles --------------------------------------------------------------------
        for k, recs in prof.items():
            Ks = {r["seg"].shape[1] for r in recs}
            K1 = max(Ks, key=lambda x: sum(1 for r in recs if r["seg"].shape[1] == x))
            recs = [r for r in recs if r["seg"].shape[1] == K1]
            dense = [r for r in recs if r["lead"] >= LEAD_DENSE_NS] if k == "ring" else recs
            src = dense if len(dense) >= MIN_N else recs
            # share of the forward's compute per segment: use the GPU-dense windows (span) for ring with the host ahead,
            # compute-union shares otherwise (host-paced windows contain host-induced idle, which is not a GPU property)
            use_span = (k == "ring" and len(dense) >= MIN_N)
            mats = []
            for r in src:
                m = r["seg"] if use_span else r["cu"]
                tot = m.sum(axis=1, keepdims=True)
                mats.append(m / np.where(tot > 0, tot, 1))
            mats = np.stack(mats)                                   # (n, R, K+1)
            share = np.median(mats, axis=0)
            share = share / share.sum(axis=1, keepdims=True)
            share_q = np.quantile(mats, QUANTILES, axis=0).transpose(1, 2, 0)   # (R, K+1, 10)
            kinds = np.array([r["kinds"] for r in recs])
            kinds_modal = np.array([np.bincount(kinds[:, j]).argmax() for j in range(kinds.shape[1])], dtype=np.int64)
            idle_share = np.median(np.stack([r["idle"] / np.maximum(1.0, r["comp"] + r["idle"]) for r in dense or recs]), axis=0)
            p.profile[k] = dict(K=int(K1 - 1), kinds=kinds_modal.tolist(), share=share.tolist(), share_q=share_q.tolist(), n=len(src),
                                source="span windows, host >= 20 ms ahead" if use_span else "compute-union windows", n_dense=len(dense))
            p.idle_share[k] = idle_share.tolist()
        # ---- measured common-mode dispersion (for the stochastic bands only; mean-1 factors in the engine) ---------------
        p.disp_q["decode"] = {}
        for b, w in sorted(disp_dec.items()):
            w = np.asarray(w, float)
            if len(w) >= MIN_N:
                p.disp_q["decode"][str(b)] = dict(q=(np.quantile(w, QUANTILES) / np.median(w)).tolist(), n=len(w))
        if len(disp_ring) >= MIN_N:
            p.disp_q["ring"] = dict(q=np.quantile(np.asarray(disp_ring), QUANTILES).tolist(), n=len(disp_ring), reference="Phase 1A 16k-chunk fit wall(kv0)")
        for key, rows in graph_rows.items():
            p.graph_mixed[key] = dict(hostwin=np.median(np.array([r["hostwin"] for r in rows]), axis=0).tolist(),
                                      launches=float(np.median([np.median(r["launches"]) for r in rows])), n=len(rows))
        p.status = {
            "eps_sched[kind, b_prev2]": "measured (host_start_0(i) - GPU_end(i-2) after a decode iteration; per b table, MIN_N rule)",
            "a_host": "measured (host_start_r(i) - host_end_r(i-1) after mixed iterations, per rank)",
            "f_idle": f"measured (first kernel - host start when the GPU was idle; n={len(f_idle)})",
            "delta_launch": "inferred (individual launch -> GPU start latency, 20 us; not separately identifiable, second order)",
            "h_pre / p_post": "measured (forward start - host start; host end - forward end; per kind and rank)",
            "hostwin_dec / n_launch_dec": "measured per b (decode host forward window, launches)",
            "queue_depth": "measured lower bound (max implied depth on calibration ring rows)",
            "segment profile": "measured (median normalized compute window per rank and segment, quantiles for stochastic draws)",
            "launch distribution over segments": "assumed uniform (launch index proportional to segment index); not identifiable per segment from the API trace",
            "rank-2 host lateness": "not identifiable (gap G4): carried as the measured per-rank c_host of Phase 1A, no cause modelled",
            "c_kv (KV-dependent launch cost)": "not identifiable: Phase 1A value 0 kept; launch count still grows with KV16 (measured)",
            "dispersion": "measured (calibration quantiles of wall / reference; used only to widen the stochastic band, mean-1 factor)",
            "mixed segment jitter": "not sampled: per-segment jitter is common-mode across ranks; independent sampling over-predicted ring walls by 6 %",
        }
        p.evidence = dict(a_host_n=[len(a) for a in a_host], f_idle_n=len(f_idle), f_idle_values_ms=[round(x / 1e6, 3) for x in f_idle])
        return p

    # ------------------------------------------------------------------------------------------
    def eps_sched(self, kind: str, b_prev2: int) -> tuple[float, str]:
        d = self.eps_sched_ns.get(kind) or self.eps_sched_ns.get("decode")
        if kind != "decode":
            e = d.get("*")
            if e and e["n"] >= MIN_N:
                return e["median"], "measured"
            dd = self.eps_sched_ns["decode"]
            return max(v["median"] for v in dd.values() if v["n"] >= MIN_N), "inferred"
        e = d.get(str(b_prev2))
        if e and e["n"] >= MIN_N:
            return e["median"], "measured"
        bs = sorted(int(k) for k, v in d.items() if v["n"] >= MIN_N)
        lo = [x for x in bs if x < b_prev2]
        hi = [x for x in bs if x > b_prev2]
        if lo and hi:
            w = (b_prev2 - lo[-1]) / (hi[0] - lo[-1])
            return (1 - w) * d[str(lo[-1])]["median"] + w * d[str(hi[0])]["median"], "interpolated"
        return d[str(lo[-1] if lo else hi[0])]["median"], "extrapolation"

    def decode_host(self, b: int) -> tuple[np.ndarray, float, str]:
        e = self.hostwin_dec_ns.get(str(b))
        n = self.n_launch_dec.get(str(b))
        if e and e["n"] >= MIN_N:
            return np.array(e["median"]), n["median"], "measured"
        bs = sorted(int(k) for k, v in self.hostwin_dec_ns.items() if v["n"] >= MIN_N)
        near = min(bs, key=lambda x: abs(x - b))
        return np.array(self.hostwin_dec_ns[str(near)]["median"]), self.n_launch_dec[str(near)]["median"], "interpolated"

    # ------------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "SeqParams":
        return cls(**d)

    def save(self, path: str):
        json.dump(self.to_dict(), open(path, "w"), indent=1)

    @classmethod
    def load(cls, path: str) -> "SeqParams":
        return cls.from_dict(json.load(open(path)))
