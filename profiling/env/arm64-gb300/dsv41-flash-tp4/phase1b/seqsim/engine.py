"""TP4 sequence engine (Phase 1B).

Each rank has two timelines: a host timeline (scheduler hand-off, preprocessing, kernel launches,
post-processing) and a GPU timeline (ordered compute windows separated by collectives). Iterations
are coupled through

  * the per-rank host serial order            host_start_r(i) >= host_end_r(i-1) + a_host
  * the scheduler one-step-ahead constraint   host_start_0(i) >= GPU_end(i-2) + eps_sched
  * the scheduler broadcast                   host_start_r(i) >= host_start_0(i)
  * the launch queue                          launch j completes >= GPU completion of launch j - Q
  * the GPU order per rank                    a kernel starts >= max(GPU free, its launch + delta)
  * collectives                               completion = max_r arrival_r + intrinsic (explicit barrier)

Iteration completion is the maximum over ranks of the tail window end (critical path). Rank
durations are never summed; nothing is scaled by a global multiplier.

The engine is vectorised over N Monte-Carlo draws (N = 1 is the deterministic median path).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from costmodel import groups as G
from costmodel.compose import critical_path_batch
from costmodel.data import NRANK, Feat, mixed_kind
from costmodel.decode import STAT_KEYS, _sample
from costmodel.predict import Predictor

from .params import SeqParams, kind_of

R = NRANK
NEG = -1e30


@dataclass
class Measured:
    """Measured per-rank timeline of one iteration (used for excluded iterations and reconstruction)."""
    start_rel: np.ndarray            # (R,) first kernel offsets
    seg: np.ndarray                  # (R, K+1) windows
    intrinsic: np.ndarray            # (K,)
    host_start: np.ndarray           # (R,) absolute ns
    host_end: np.ndarray             # (R,)
    fwd_start: np.ndarray            # (R,)
    fwd_end: np.ndarray              # (R,)
    launches: np.ndarray             # (R,)
    t0: float
    t1: float


class _QueueLog:
    """Per-rank history of launched blocks: (cum_start, cum_end, gpu_start, gpu_done), arrays over draws."""

    def __init__(self, n: int):
        self.recs: list[deque] = [deque() for _ in range(R)]
        self.n = n

    def push(self, r: int, cs, ce, gs, gd):
        # copies: callers pass views of arrays that are advanced in place afterwards
        self.recs[r].append((np.array(cs, dtype=float, copy=True), np.array(ce, dtype=float, copy=True), np.array(gs, dtype=float, copy=True), np.array(gd, dtype=float, copy=True)))
        while len(self.recs[r]) > 96:
            self.recs[r].popleft()

    def done_of_launch(self, r: int, x: np.ndarray) -> np.ndarray:
        """GPU completion time of launch index x (per draw); -inf when x precedes the retained history."""
        out = np.full(self.n, NEG)
        if not self.recs[r]:
            return out
        for cs, ce, gs, gd in reversed(self.recs[r]):
            m = (x >= cs) & (x < ce) & (out <= NEG)
            if m.any():
                frac = np.where(ce > cs, (x - cs) / np.maximum(ce - cs, 1e-9), 1.0)
                out = np.where(m, gs + frac * (gd - gs), out)
            if (x >= ce).all():
                break
        return out


@dataclass
class IterRecord:
    iteration: int
    kind: str
    status: str
    kv_status: str
    t0: np.ndarray            # (N,)
    t1: np.ndarray            # (N,)
    host_start: np.ndarray    # (N, R)
    host_end: np.ndarray      # (N, R)
    rank_end: np.ndarray      # (N, R)
    rank_wait: np.ndarray     # (N, R)
    lead: np.ndarray          # (N,) GPU end of i-1 minus earliest host start
    binding: dict             # which constraint set host_start_0 (counts over draws)
    host_limited_segments: np.ndarray   # (N, R) number of segments whose GPU start waited for the host
    extra: dict = field(default_factory=dict)

    @property
    def wall(self):
        return self.t1 - self.t0


class Engine:
    def __init__(self, coeffs: dict, params: SeqParams, draws: int = 1, seed: int = 0, stochastic: bool = False,
                 moe_band: bool = False, throttle: bool = True):
        self.pred = Predictor(coeffs)
        self.params = params
        self.N = int(draws)
        self.stochastic = bool(stochastic)
        self.moe_band = bool(moe_band)
        self.throttle = bool(throttle)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self._pool: dict[int, dict] = {}
        self.mixed = self.pred.mixed
        self.prefill = self.pred.prefill
        self.coll = self.pred.coll
        self.decode = self.pred.decode
        self.moe = self.pred.moe

    # ---- decode sample pool ----------------------------------------------------------------------
    def _decode_pool(self, b: int) -> dict:
        if b in self._pool:
            return self._pool[b]
        p, status = self.decode._lookup(b)
        S = 256 if self.stochastic else 1
        rng = np.random.default_rng(self.seed * 1000003 + b)
        if self.stochastic:
            start = _sample(p["start_q"], rng.random((S, R)))
            seg = _sample(p["seg_q"], rng.random((S, R, p["K"] + 1)))
            intr = _sample(p["intrinsic_q"], rng.random((S, p["K"])))
        else:
            start = np.asarray(p["start"], float)[None]
            seg = np.asarray(p["seg"], float)[None]
            intr = np.asarray(p["intrinsic"], float)[None]
        res = critical_path_batch(start, seg, intr)
        arr0 = start + seg[:, :, 0]
        done0 = arr0.max(axis=1) + intr[:, 0]
        groups = np.asarray(p["groups"], float)
        moe_share = float(groups[:, G.GROUP_INDEX["moe"]].sum() / max(1.0, groups[:, :4].sum() + groups[:, 5].sum()))
        mp = self.moe.predict(b) if self.moe_band else None
        band = (mp["low_ns"] / mp["central_ns"], mp["high_ns"] / mp["central_ns"]) if mp and mp.get("central_ns") else (1.0, 1.0)
        pool = dict(S=S, status=status, start=start, arr0=arr0, intr0=intr[:, 0], done0=done0, rank_end=res["rank_end"], wall=res["wall"],
                    rank_wait=res["rank_wait"], seg0=seg[:, :, 0], moe_share=moe_share, moe_band=band, K=p["K"])
        self._pool[b] = pool
        return pool

    def _dispersion(self, kind: str, b) -> np.ndarray:
        """Common-mode multiplicative factor per draw (mean 1) from the calibration dispersion quantiles; 1 when unavailable."""
        d = self.params.disp_q.get(kind)
        if d is None:
            return np.ones(self.N)
        if kind == "decode":
            e = d.get(str(b))
            if e is None or e["n"] < 5:
                return np.ones(self.N)
            q = np.asarray(e["q"], float)
        else:
            if d.get("n", 0) < 5:
                return np.ones(self.N)
            q = np.asarray(d["q"], float)
        u = self.rng.random(self.N)
        x = np.interp(u, np.linspace(0.05, 0.95, len(q)), q)
        return x / max(q.mean(), 1e-9)

    # ---- one run -----------------------------------------------------------------------------------
    def run(self, schedule: list[Feat], measured: dict[int, Measured] | None = None, recon: str | None = None,
            progress: int = 0) -> list[IterRecord]:
        """recon=None: fully predicted; 'gpu': measured segments + measured host times; 'seq': measured segments and
        host durations, host start times from the sequence rules. `measured` supplies the per-iteration timelines
        (required for excluded iterations, which are replayed from their measured timeline in every mode)."""
        N = self.N
        P = self.params
        measured = measured or {}
        gpu_free = None
        host_free = None
        gend1 = gend2 = None
        b_hist = [0, 0]            # [b_{i-1}, b_{i-2}] of the previous two iterations (any kind)
        cum = np.zeros((N, R))
        qlog = _QueueLog(N)
        out: list[IterRecord] = []
        for n_i, f in enumerate(schedule):
            kind = kind_of(f)
            m = measured.get(f.iteration)
            use_meas = f.excluded or recon is not None
            if use_meas and m is None:
                raise ValueError(f"iteration {f.iteration} needs a measured timeline (excluded or reconstruction mode)")
            # ---------------- host start (scheduler + serial) ----------------
            if gpu_free is None:
                base = float(m.host_start.min()) if m is not None else 0.0
                host_start = np.full((N, R), base)
                binding = {"first": N}
                gpu_free = np.full((N, R), base)
                host_free = np.full((N, R), base)
                gend1 = np.full(N, base)
                gend2 = np.full(N, base)
            else:
                eps, _ = P.eps_sched(kind, b_hist[1])
                a = np.array(P.a_host_ns)
                c_serial = host_free[:, 0] + a[0]
                c_sched = gend2 + eps
                h0 = np.maximum(c_serial, c_sched)
                binding = {"serial": int((c_serial >= c_sched).sum()), "scheduler": int((c_sched > c_serial).sum())}
                host_start = np.maximum(host_free + a[None, :], h0[:, None])
            if use_meas and recon == "gpu":
                host_start = np.broadcast_to(m.host_start.astype(float), (N, R)).copy()
            lead = gend1 - host_start.min(axis=1)
            # ---------------- step ----------------
            if use_meas:
                rec = self._step_measured(f, kind, m, host_start, gpu_free, cum, qlog, recon)
            elif kind == "decode":
                rec = self._step_decode(f, host_start, gpu_free, cum, qlog)
            elif kind == "graph":
                rec = self._step_graph(f, host_start, gpu_free, cum, qlog)
            else:
                rec = self._step_mixed(f, kind, host_start, gpu_free, cum, qlog)
            rec.lead = lead
            rec.binding = binding
            out.append(rec)
            # ---------------- advance ----------------
            gpu_free = rec.rank_end.copy()
            host_free = rec.host_end.copy()
            gend2 = gend1
            gend1 = rec.t1.copy()
            b_hist = [f.b, b_hist[0]]
            if progress and n_i % progress == 0:
                print(f"  [{n_i}/{len(schedule)}]", flush=True)
        return out

    # ---- decode (CUDA graph) ------------------------------------------------------------------------
    def _step_decode(self, f: Feat, host_start, gpu_free, cum, qlog) -> IterRecord:
        N, P = self.N, self.params
        pool = self._decode_pool(f.b)
        idx = self.rng.integers(0, pool["S"], size=N) if pool["S"] > 1 else np.zeros(N, dtype=int)
        hostwin, n_launch, hst = P.decode_host(f.b)
        h = host_start + np.array(P.h_pre_ns["decode"])[None, :]
        h_end_unthr = h + hostwin[None, :]
        h_end = h_end_unthr.copy()
        if self.throttle:
            for r in range(R):
                thr = qlog.done_of_launch(r, cum[:, r] + n_launch - P.queue_depth)
                h_end[:, r] = np.maximum(h_end[:, r], thr)
        T0 = np.maximum(gpu_free, host_start + P.f_idle_ns)
        start = pool["start"][idx]                 # (N, R)
        arr0 = pool["arr0"][idx]
        # the graph cannot run before it was launched: arrival at the first collective >= launch + delta + first window
        arrival = np.maximum(T0 + arr0, h_end + P.delta_launch_ns + (arr0 - start))
        t0 = np.minimum(T0.min(axis=1), arrival.min(axis=1))
        done0 = arrival.max(axis=1) + pool["intr0"][idx]
        shift = done0 - pool["done0"][idx]          # absolute = relative + shift
        rank_end = pool["rank_end"][idx] + shift[:, None]
        # extra waiting at the first collective beyond what the sampled path already contains (start skew carried over from i-1)
        extra0 = (arrival.max(axis=1)[:, None] - arrival) - (arr0.max(axis=1)[:, None] - arr0)
        wait = pool["rank_wait"][idx] + np.clip(extra0, 0, None)
        if self.moe_band and pool["moe_band"] != (1.0, 1.0):
            lo, hi = pool["moe_band"]
            mfac = self.rng.uniform(lo, hi, size=N) if self.stochastic else np.full(N, 0.5 * (lo + hi))
            rank_end = t0[:, None] + (rank_end - t0[:, None]) * (1.0 + (mfac[:, None] - 1.0) * pool["moe_share"])
        if self.stochastic:
            cm = self._dispersion("decode", f.b)
            rank_end = t0[:, None] + (rank_end - t0[:, None]) * cm[:, None]
        rank_end = np.maximum(rank_end, gpu_free)
        t1 = rank_end.max(axis=1)
        host_end = h_end + np.array(P.p_post_ns["decode"])[None, :]
        for r in range(R):
            qlog.push(r, cum[:, r], cum[:, r] + n_launch, T0[:, r], rank_end[:, r])
        cum += n_launch
        kv_status = "in-range" if f.kv_max <= self.decode.kv_max_observed else "kv-extrapolation"
        host_limited = (arrival > T0 + arr0 + 1.0).astype(int)       # graph launch later than the GPU could have started
        return IterRecord(f.iteration, "decode", pool["status"], kv_status, t0, t1, host_start, host_end, rank_end, wait,
                          np.zeros(N), {}, host_limited, extra=dict(K=pool["K"], host_status=hst))

    # ---- mixed graph-path (P + b <= 256): class medians ------------------------------------------------
    def _step_graph(self, f: Feat, host_start, gpu_free, cum, qlog) -> IterRecord:
        N, P = self.N, self.params
        p = self.mixed.predict(f, self.prefill, self.coll)
        key = f"{f.P // 64 * 64},{f.b}"
        g = P.graph_mixed.get(key) or (P.graph_mixed[min(P.graph_mixed, key=lambda k: abs(int(k.split(',')[0]) - f.P) + abs(int(k.split(',')[1]) - f.b))] if P.graph_mixed else None)
        hostwin = np.array(g["hostwin"]) if g else np.zeros(R)
        n_launch = g["launches"] if g else 0.0
        h = host_start + np.array(P.h_pre_ns.get("graph", P.h_pre_ns["decode"]))[None, :]
        h_end = h + hostwin[None, :]
        T0 = np.maximum(gpu_free, host_start + P.f_idle_ns)
        wall = float(p["wall_ns"]) if np.isfinite(p["wall_ns"]) else 0.0
        rank_end = np.maximum(T0.max(axis=1)[:, None] + wall, h_end + P.delta_launch_ns)
        t0 = T0.min(axis=1)
        t1 = rank_end.max(axis=1)
        host_end = h_end + np.array(P.p_post_ns.get("graph", P.p_post_ns["decode"]))[None, :]
        for r in range(R):
            qlog.push(r, cum[:, r], cum[:, r] + n_launch, T0[:, r], rank_end[:, r])
        cum += n_launch
        return IterRecord(f.iteration, "graph", p["status"], "in-range", t0, t1, host_start, host_end, rank_end, np.zeros((N, R)),
                          np.zeros(N), {}, np.zeros((N, R), int), extra=dict(wall_class=wall))

    # ---- mixed eager / ring: launch-paced per-rank timelines -----------------------------------------
    def _mixed_spec(self, f: Feat, kind: str) -> dict:
        P = self.params
        prof = P.profile.get(kind) or P.profile["ring"]
        K = prof["K"]
        kinds = np.array(prof["kinds"])
        groups = self.prefill.predict_groups(f)                            # (R, 6) ns
        comp = groups.sum(axis=1)
        overlap = np.array(self.prefill.overlap_cc_ns.get(kind) or self.prefill.overlap_cc_ns.get("ring") or [0] * R)
        comp_union = np.clip(comp - overlap, 0, None)
        idle_share = np.array(P.idle_share.get(kind) or P.idle_share.get("ring") or [0] * R)
        tot = comp_union / np.clip(1.0 - idle_share, 0.5, 1.0)              # windows = compute union + inherent idle
        tokens = f.total_tokens
        intr = np.zeros(K)
        cstat = "in-range"
        for k in range(K):
            kname = G.COMM_KINDS[int(kinds[k])]
            if kname in self.coll.eager:
                us, st = self.coll.intrinsic_us(kname, tokens)
            elif kname in ("twoshot", "nccl_allreduce"):
                us, st = self.coll.intrinsic_us("nccl_allreduce" if kind == "ring" else "twoshot", tokens)
            else:
                us, st = 0.0, "inferred"
            intr[k] = us * 1e3
            if st != "in-range":
                cstat = st
        hp = self.mixed.host_path(f)
        L = float(hp["launches"])                                   # all launches enter the queue (measured count model)
        # host time per launch: Phase 1A host path F_r = c_host_r * L_base + c_kv * (L - L_base) with c_kv = 0 (inferred, not identifiable),
        # spread over all L launches, so the total host forward time equals the Phase 1A host path
        c_host = np.asarray(hp["fwd_ns"], float) / max(L, 1.0)
        status = "in-range"
        for s in (self.prefill.status(f), hp["status"], cstat):
            if s == "extrapolation":
                status = "extrapolation"
            elif s in ("inferred", "inferred-structure") and status == "in-range":
                status = "inferred"
        return dict(K=K, kinds=kinds, share=np.array(prof["share"]), share_q=np.array(prof["share_q"]), tot=tot, intr=intr, L=L,
                    c_host=c_host, h_pre=np.array(P.h_pre_ns.get(kind) or P.h_pre_ns["ring"]), p_post=np.array(P.p_post_ns.get(kind) or P.p_post_ns["ring"]),
                    tail=float(self.mixed.tail_ns), status=status, groups=groups)

    def _step_mixed(self, f: Feat, kind: str, host_start, gpu_free, cum, qlog) -> IterRecord:
        N, P = self.N, self.params
        s = self._mixed_spec(f, kind)
        K = s["K"]
        # segment windows at their calibration medians on every draw: per-segment jitter is common-mode across ranks in the
        # traces (same kernels), so sampling it independently per rank would over-predict collective waiting
        share = np.broadcast_to(s["share"], (N, R, K + 1))
        seg = share * s["tot"][None, :, None]                                   # (N, R, K+1)
        if self.stochastic:
            seg = seg * self._dispersion(kind, None)[:, None, None]             # measured common-mode dispersion (calibration)
        lps = s["L"] / (K + 1)
        c = s["c_host"]
        rec = self._launch_paced(f, kind, host_start, gpu_free, cum, qlog, seg, s["intr"], lps, c[None, :].repeat(N, 0), s["h_pre"], s["p_post"], s["tail"],
                                 throttle=self.throttle)
        rec.status = s["status"]
        rec.kv_status = "in-range" if self.prefill.status(f) != "extrapolation" else "kv-extrapolation"
        rec.extra.update(K=K, launches=s["L"], groups=s["groups"].tolist())
        return rec

    def _launch_paced(self, f, kind, host_start, gpu_free, cum, qlog, seg, intr, lps, c_per_launch, h_pre, p_post, tail, throttle) -> IterRecord:
        """Per-rank launch-paced GPU timeline with explicit collectives. seg: (N, R, K+1); c_per_launch: (N, R)."""
        N, P = self.N, self.params
        K = seg.shape[2] - 1
        h = host_start + np.asarray(h_pre, float)[None, :]
        gpu_t = gpu_free.copy()
        T0 = np.maximum(gpu_free, host_start + P.f_idle_ns)
        t0 = T0.min(axis=1)
        wait = np.zeros((N, R))
        host_lim = np.zeros((N, R), int)
        cum_local = cum.copy()
        rank_end = np.zeros((N, R))
        for k in range(K + 1):
            seg_k = seg[:, :, k]
            h_end = h + c_per_launch * lps
            if throttle:
                for r in range(R):
                    thr = qlog.done_of_launch(r, cum_local[:, r] + lps - P.queue_depth)
                    h_end[:, r] = np.maximum(h_end[:, r], thr)
            g_start = np.maximum(gpu_t, h + P.delta_launch_ns)
            kern = seg_k / max(lps, 1.0) if k < K else np.full_like(seg_k, tail)
            g_done = np.maximum(g_start + seg_k, h_end + P.delta_launch_ns + kern)
            host_lim += (h_end + P.delta_launch_ns + kern > g_start + seg_k + 1.0).astype(int)   # GPU completion bound by the host
            for r in range(R):
                qlog.push(r, cum_local[:, r], cum_local[:, r] + lps, g_start[:, r], g_done[:, r])
            cum_local += lps
            h = h_end
            if k < K:
                amax = g_done.max(axis=1)
                done_k = amax + intr[k]
                wait += amax[:, None] - g_done
                gpu_t = np.repeat(done_k[:, None], R, axis=1)
            else:
                rank_end = g_done
        cum[...] = cum_local
        t1 = rank_end.max(axis=1)
        host_end = h + np.asarray(p_post, float)[None, :]
        return IterRecord(f.iteration, kind, "in-range", "in-range", t0, t1, host_start, host_end, rank_end, wait, np.zeros(N), {}, host_lim,
                          extra=dict(fwd_end=h.copy()))

    # ---- measured timelines (excluded iterations; reconstruction) --------------------------------------
    def _step_measured(self, f: Feat, kind: str, m: Measured, host_start, gpu_free, cum, qlog, recon) -> IterRecord:
        N, P = self.N, self.params
        R_, K1 = m.seg.shape
        K = K1 - 1
        hostwin = (m.fwd_end - m.fwd_start).astype(float)
        h_pre = (m.fwd_start - m.host_start).astype(float)
        p_post = (m.host_end - m.fwd_end).astype(float)
        L = float(np.median(m.launches))
        if kind == "decode" or K == 0 or kind == "graph":
            h = host_start + h_pre[None, :]
            h_end = h + hostwin[None, :]
            T0 = np.maximum(gpu_free, host_start + P.f_idle_ns)
            start = np.broadcast_to(m.start_rel.astype(float), (N, R))
            seg = np.broadcast_to(m.seg.astype(float), (N, R, K1))
            intr = np.broadcast_to(m.intrinsic.astype(float), (N, K))
            res = critical_path_batch(start, seg, intr)
            t0 = T0.min(axis=1)
            rank_end = t0[:, None] + res["rank_end"]
            rank_end = np.maximum(rank_end, gpu_free)
            t1 = rank_end.max(axis=1)
            for r in range(R):
                qlog.push(r, cum[:, r], cum[:, r] + L, T0[:, r], rank_end[:, r])
            cum += L
            host_end = h_end + p_post[None, :]
            return IterRecord(f.iteration, kind, "measured", "measured", t0, t1, host_start, host_end, rank_end, res["rank_wait"], np.zeros(N), {},
                              np.zeros((N, R), int), extra=dict(K=K))
        seg = np.broadcast_to(m.seg.astype(float), (N, R, K1))
        lps = max(L, 1.0) / K1
        c = (hostwin / max(L, 1.0))[None, :].repeat(N, 0)          # measured (already throttled) pacing -> no queue rule
        rec = self._launch_paced(f, kind, host_start, gpu_free, cum, qlog, seg, m.intrinsic.astype(float), lps, c, h_pre, p_post, self.mixed.tail_ns,
                                 throttle=False)
        rec.status = "measured"
        rec.kv_status = "measured"
        rec.extra.update(K=K, launches=L)
        return rec


def measured_from_pass(pd, iteration: int) -> Measured | None:
    seg = pd.segments(iteration)
    rows = pd.rank_rows(iteration)
    if seg is None:
        c = None
        start_rel = pd.rank_col("first_start_ns", iteration) - pd.it_value("t0_ns", iteration)
        last = pd.rank_col("last_end_ns", iteration) - pd.it_value("t0_ns", iteration)
        seg_arr = (last - start_rel).astype(float)[:, None]
        intr = np.zeros(0)
    else:
        start_rel, seg_arr, intr, _ = seg
        if pd.collectives(iteration).get("_kind_mismatch"):
            return None
    return Measured(start_rel=np.asarray(start_rel, float), seg=np.asarray(seg_arr, float), intrinsic=np.asarray(intr, float),
                    host_start=pd.ir["host_start_ns"][rows].astype(float), host_end=pd.ir["host_end_ns"][rows].astype(float),
                    fwd_start=pd.ir["fwd_start_ns"][rows].astype(float), fwd_end=pd.ir["fwd_end_ns"][rows].astype(float),
                    launches=pd.ir["fwd_launch_n"][rows].astype(float), t0=float(pd.it_value("t0_ns", iteration)), t1=float(pd.it_value("t1_ns", iteration)))
