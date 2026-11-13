"""Read-only extraction of compact per-iteration / per-rank / per-collective tables from a frozen
Phase 0 profile directory (parsed.json, parsed.kernels.parquet, host_timeline.json).

Nothing under the profile directory is written.  The output tables are the only inputs the
calibration and evaluation steps read, so a Phase 1A result is reproducible from
(frozen trace hashes, this file, the partition file).

Usage:
  python -m costmodel.extract --profile <dir> --label <pass> --out <dir> [--excl <jit_excl.txt>]
                              [--sha256sums <file>] [--sha-prefix run/<pass>/profile]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import groups as G
from .intervals import union_length, windowed_union

INPUT_FILES = ("parsed.json", "parsed.kernels.parquet", "host_timeline.json")

SYNC_TOKENS = ("StreamSynchronize", "EventSynchronize", "DeviceSynchronize", "CtxSynchronize")
QUERY_TOKENS = ("EventQuery", "StreamQuery")
LAUNCH_TOKENS = ("LaunchKernel", "GraphLaunch", "LaunchCooperativeKernel", "cuLaunchKernelEx")
MEMCPY_TOKENS = ("Memcpy", "Memset")


def sha256_file(path: str, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_inputs(profile: str, sums_file: str | None, sha_prefix: str | None) -> dict:
    out = {}
    expected = {}
    if sums_file:
        for line in open(sums_file):
            parts = line.split(None, 1)
            if len(parts) == 2:
                expected[parts[1].strip()] = parts[0]
    for name in INPUT_FILES:
        p = os.path.join(profile, name)
        digest = sha256_file(p)
        rec = {"path": os.path.abspath(p), "sha256": digest, "bytes": os.path.getsize(p)}
        if sums_file:
            key = f"{sha_prefix}/{name}" if sha_prefix else None
            exp = expected.get(key)
            rec["expected_sha256"] = exp
            rec["verified"] = (exp == digest)
            if exp is not None and exp != digest:
                raise SystemExit(f"hash mismatch for {p}: {digest} != {exp}")
        out[name] = rec
    return out


def api_class(strings: list[str]) -> np.ndarray:
    """0 other, 1 launch, 2 sync, 3 query, 4 memcpy/memset."""
    cls = np.zeros(len(strings), dtype=np.int8)
    for i, s in enumerate(strings):
        if any(t in s for t in LAUNCH_TOKENS):
            cls[i] = 1
        elif any(t in s for t in SYNC_TOKENS):
            cls[i] = 2
        elif any(t in s for t in QUERY_TOKENS):
            cls[i] = 3
        elif any(t in s for t in MEMCPY_TOKENS):
            cls[i] = 4
    return cls


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--excl", default=None, help="file with excluded iteration ids (comma/newline separated)")
    ap.add_argument("--sha256sums", default=None)
    ap.add_argument("--sha-prefix", default=None)
    ap.add_argument("--no-host", action="store_true", help="skip host_timeline.json (faster)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N iterations")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    excl = set()
    if a.excl:
        excl = {int(x) for x in open(a.excl).read().replace("\n", ",").split(",") if x.strip()}

    inputs = verify_inputs(a.profile, a.sha256sums, a.sha_prefix)
    print(f"[{a.label}] inputs verified in {time.time()-t_start:.0f}s", flush=True)

    d = json.load(open(os.path.join(a.profile, "parsed.json")))
    det = d["iteration_details"]
    if a.limit:
        det = det[: a.limit]
    names = {int(k): v for k, v in d["kernel_names"].items()}
    grp_of, kind_of = G.build_lookup(names)
    devices = sorted(d["device_ids"])
    ndev = len(devices)
    print(f"[{a.label}] parsed.json: {len(det)} iterations, {len(names)} kernel names, devices {devices}", flush=True)

    tab = pq.read_table(os.path.join(a.profile, "parsed.kernels.parquet"),
                        columns=["detail_index", "range_index", "name_id", "start_ns", "end_ns", "stream_id"])
    di = tab["detail_index"].to_numpy()
    ri = tab["range_index"].to_numpy()
    nid = tab["name_id"].to_numpy()
    s = tab["start_ns"].to_numpy().astype(np.int64)
    e = tab["end_ns"].to_numpy().astype(np.int64)
    sid = tab["stream_id"].to_numpy().astype(np.int64)
    del tab
    o = np.argsort(di, kind="stable")
    di, ri, nid, s, e, sid = di[o], ri[o], nid[o], s[o], e[o], sid[o]
    del o
    if a.limit:
        keep = di < len(det)
        di, ri, nid, s, e, sid = di[keep], ri[keep], nid[keep], s[keep], e[keep], sid[keep]
    bounds_it = np.searchsorted(di, np.arange(len(det) + 1))
    maxr = max(len(x["ranges"]) for x in det)
    rd = np.full(len(det) * maxr, -1, dtype=np.int8)
    for ai, x in enumerate(det):
        for bi, r in enumerate(x["ranges"]):
            rd[ai * maxr + bi] = r["device_id"]
    dev_k = rd[di.astype(np.int64) * maxr + ri]
    g_arr = grp_of[nid]
    k_arr = kind_of[nid]
    dur = e - s
    print(f"[{a.label}] kernels: {len(di)} rows sorted in {time.time()-t_start:.0f}s", flush=True)

    # ---- host timeline -------------------------------------------------------------------------
    host = None
    if not a.no_host:
        h = json.load(open(os.path.join(a.profile, "host_timeline.json")))
        api = np.array(h["api_calls"], dtype=np.int64)
        order = np.argsort(api[:, 1], kind="stable")
        api = api[order]
        acls = api_class(h["strings"])
        main_thread = {}
        helper_threads = collections.defaultdict(list)
        for i, t in enumerate(h["threads"]):
            if t["device_id"] is None:
                continue
            if t["role"] == "worker main thread":
                main_thread[t["device_id"]] = i
            elif "worker" in t["role"]:
                helper_threads[t["device_id"]].append(i)
        host = dict(api=api, acls=acls, main_thread=main_thread, helper_threads=helper_threads)
        del h
        print(f"[{a.label}] host_timeline: {len(api)} api calls", flush=True)

    # ---- per-iteration tables ------------------------------------------------------------------
    it_rows = []
    ir_rows = []
    coll_cols = collections.defaultdict(list)
    n_unaligned = 0
    for ai, x in enumerate(det):
        it = int(x["iteration"])
        m = x["metrics"]
        kv = m.get("decode_kv_lens", []) or []
        chunks = m.get("prefill_chunk_pairs", []) or []
        jit = int(x["jit_module_loads"]) > 0 or int(x["jit_stall_ns"]) > 0
        sl = slice(int(bounds_it[ai]), int(bounds_it[ai + 1]))
        S, E, Dv, Gk, Kk, Sid, Du = s[sl], e[sl], dev_k[sl], g_arr[sl], k_arr[sl], sid[sl], dur[sl]
        t0 = int(S.min()) if len(S) else 0
        t1 = int(E.max()) if len(E) else 0
        it_rows.append(dict(
            iteration=it, stage=x["stage"], excluded=bool(jit or it in excl), jit_module_loads=int(x["jit_module_loads"]),
            jit_stall_ns=int(x["jit_stall_ns"]), decode_requests=int(m.get("decode_requests", 0) or 0),
            prefill_tokens=int(m.get("prefill_tokens", 0) or 0), n_prefill_chunks=len(chunks),
            chunk_pairs=json.dumps(chunks), kv_max=int(max(kv)) if kv else 0, kv_sum=int(sum(kv)) if kv else 0,
            kv_n=len(kv), t0_ns=t0, t1_ns=t1, wall_ns=t1 - t0, n_kernels=int(len(S)),
        ))
        ep_counts = []
        for r in devices:
            msk = Dv == r
            if not msk.any():
                ep_counts.append(-1)
                continue
            Sr, Er, Gr, Kr, Sr_id, Dr = S[msk], E[msk], Gk[msk], Kk[msk], Sid[msk], Du[msk]
            oo = np.argsort(Sr, kind="stable")
            Sr, Er, Gr, Kr, Sr_id, Dr = Sr[oo], Er[oo], Gr[oo], Kr[oo], Sr_id[oo], Dr[oo]
            comm = Gr == G.GROUP_INDEX["comm"]
            first, last = int(Sr[0]), int(Er.max())
            busy = union_length(Sr, Er)
            cu = union_length(Sr[~comm], Er[~comm])
            mu = union_length(Sr[comm], Er[comm])
            gsum = np.bincount(Gr, weights=Dr, minlength=len(G.GROUPS)).astype(np.int64)
            gcnt = np.bincount(Gr, minlength=len(G.GROUPS)).astype(np.int64)
            ksum = np.bincount(Kr[comm], weights=Dr[comm], minlength=len(G.COMM_KINDS)).astype(np.int64)
            kcnt = np.bincount(Kr[comm], minlength=len(G.COMM_KINDS)).astype(np.int64)
            # stream structure
            streams, scount = np.unique(Sr_id, return_counts=True)
            primary = streams[np.argmax(scount)]
            side = Sr_id != primary
            side_sum = int(Dr[side].sum())
            side_union = union_length(Sr[side], Er[side]) if side.any() else 0
            # collective calls = individual communication kernels in start order (the kernel order is
            # identical on all ranks: same program; alignment across ranks is by index k and checked
            # later by comparing the per-rank kind sequences).  Overlapping comm kernels (side-stream
            # all-gather during an all-reduce) collapse the compute window between them to zero.
            comm_idx = np.nonzero(comm)[0]
            nep = int(len(comm_idx))
            ep_start = Sr[comm_idx]
            ep_end = Er[comm_idx]
            ep_kind = Kr[comm_idx]
            ep_n = np.ones(nep, dtype=np.int64)
            ep_sum = Dr[comm_idx].astype(np.int64)
            ep_counts.append(nep)
            # windows: [first, ep_start0, ep_end0, ep_start1, ..., ep_end(K-1), last]
            wb = np.empty(2 * nep + 2, dtype=np.int64)
            wb[0] = first
            wb[1:-1:2] = ep_start
            wb[2:-1:2] = ep_end
            wb[-1] = last + 1
            wb = np.maximum.accumulate(wb)  # overlapping collectives -> zero-length compute windows
            comp_u = windowed_union(Sr[~comm], Er[~comm], wb)
            all_u = windowed_union(Sr, Er, wb)
            seg_u = comp_u[0::2]            # compute segments k = 0..nep (nep + 1 segments)
            seg_all = all_u[0::2]
            comp_in_comm = comp_u[1::2]     # compute kernels launched while collective k runs
            seg_len = np.diff(wb)[0::2]
            ir_rows.append(dict(
                iteration=it, rank=int(r), n_kernels=int(len(Sr)), first_start_ns=first, last_end_ns=last,
                span_ns=last - first, busy_ns=busy, compute_union_ns=cu, comm_union_ns=mu,
                compute_sum_ns=int(Dr[~comm].sum()), comm_sum_ns=int(Dr[comm].sum()), idle_ns=last - first - busy,
                overlap_total_ns=int(Dr.sum()) - busy, overlap_comp_comm_ns=cu + mu - busy,
                overlap_comp_comp_ns=int(Dr[~comm].sum()) - cu, n_streams=int(len(streams)),
                side_stream_sum_ns=side_sum, side_stream_union_ns=side_union, n_comm_episodes=nep,
                seg_compute_union_sum_ns=int(seg_u.sum()), seg_idle_sum_ns=int((seg_len - seg_all).sum()),
                comp_in_comm_sum_ns=int(comp_in_comm.sum()),
                **{f"g_{g}_sum_ns": int(gsum[i]) for i, g in enumerate(G.GROUPS)},
                **{f"g_{g}_count": int(gcnt[i]) for i, g in enumerate(G.GROUPS)},
                **{f"c_{k}_sum_ns": int(ksum[i]) for i, k in enumerate(G.COMM_KINDS)},
                **{f"c_{k}_count": int(kcnt[i]) for i, k in enumerate(G.COMM_KINDS)},
                **host_stats(host, x, r, first, last, Sr, Er),
            ))
            if nep:
                n = nep
                coll_cols["iteration"].append(np.full(n, it, dtype=np.int32))
                coll_cols["rank"].append(np.full(n, r, dtype=np.int8))
                coll_cols["k"].append(np.arange(n, dtype=np.int16))
                coll_cols["kind"].append(ep_kind.astype(np.int8))
                coll_cols["n_kernels"].append(ep_n.astype(np.int16))
                coll_cols["start_rel_ns"].append((ep_start - t0).astype(np.int64))
                coll_cols["end_rel_ns"].append((ep_end - t0).astype(np.int64))
                coll_cols["comm_sum_ns"].append(ep_sum)
                coll_cols["seg_len_ns"].append(seg_len[:n].astype(np.int64))         # compute window before collective k
                coll_cols["seg_compute_union_ns"].append(seg_u[:n].astype(np.int64))
                coll_cols["seg_idle_ns"].append((seg_len[:n] - seg_all[:n]).astype(np.int64))
                coll_cols["comp_in_comm_ns"].append(comp_in_comm.astype(np.int64))
                coll_cols["tail_len_ns"].append(np.full(n, int(seg_len[n]), dtype=np.int64))  # compute after the last collective
                coll_cols["tail_compute_union_ns"].append(np.full(n, int(seg_u[n]), dtype=np.int64))
        if len(set(c for c in ep_counts if c >= 0)) > 1:
            n_unaligned += 1
        if ai % 2000 == 0:
            print(f"[{a.label}] {ai}/{len(det)} iterations, {time.time()-t_start:.0f}s", flush=True)

    it_tab = pa.Table.from_pylist(it_rows)
    ir_tab = pa.Table.from_pylist(ir_rows)
    pq.write_table(it_tab, os.path.join(a.out, "iterations.parquet"), compression="zstd")
    pq.write_table(ir_tab, os.path.join(a.out, "iter_rank.parquet"), compression="zstd")
    if coll_cols:
        ct = pa.table({k: np.concatenate(v) for k, v in coll_cols.items()})
        pq.write_table(ct, os.path.join(a.out, "collectives.parquet"), compression="zstd")
    meta = dict(
        label=a.label, profile_dir=os.path.abspath(a.profile), inputs=inputs, excluded_ids_file=a.excl,
        n_iterations=len(det), n_excluded=int(sum(r["excluded"] for r in it_rows)), n_unaligned_episode_counts=n_unaligned,
        devices=devices, groups=list(G.GROUPS), comm_kinds=list(G.COMM_KINDS), kernel_name_count=len(names),
        schema_version=1, extract_sha256=sha256_file(__file__), groups_sha256=sha256_file(G.__file__),
        intervals_sha256=sha256_file(os.path.join(os.path.dirname(__file__), "intervals.py")),
        extracted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seconds=round(time.time() - t_start, 1),
        kernel_groups_of_names={str(k): G.group(v) for k, v in names.items()},
    )
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"[{a.label}] done: {len(it_rows)} iterations, {len(ir_rows)} rank rows, "
          f"{sum(len(v) for v in coll_cols.get('k', []))} collective rows, unaligned {n_unaligned}, {time.time()-t_start:.0f}s", flush=True)


def host_stats(host, x, r, first, last, Sr, Er) -> dict:
    """Per-rank host decomposition inside the device's NVTX iteration window."""
    rr = [rg for rg in x["ranges"] if rg["device_id"] == r]
    if not rr:
        return {}
    w0 = min(rg["start_ns"] for rg in rr)
    w1 = max(rg["end_ns"] for rg in rr)
    fwd = [rg for rg in rr if rg["phase"] == "forward"]
    f0 = min(rg["start_ns"] for rg in fwd) if fwd else w0
    f1 = max(rg["end_ns"] for rg in fwd) if fwd else w0
    out = dict(host_start_ns=int(w0), host_end_ns=int(w1), host_window_ns=int(w1 - w0), fwd_start_ns=int(f0), fwd_end_ns=int(f1),
               fwd_window_ns=int(f1 - f0), gpu_tail_after_host_ns=int(last - w1),
               gpu_in_host_window_ns=int(union_length(np.maximum(Sr, w0), np.maximum(np.minimum(Er, w1), np.maximum(Sr, w0)))))
    if host is None:
        return out
    api, acls = host["api"], host["acls"]
    tix = host["main_thread"].get(r)
    lo, hi = np.searchsorted(api[:, 1], [w0, w1])
    seg = api[lo:hi]
    main = seg[seg[:, 0] == tix] if tix is not None else seg[:0]
    helpers = host["helper_threads"].get(r, [])
    helper = seg[np.isin(seg[:, 0], helpers)] if helpers else seg[:0]
    c = acls[main[:, 3]]
    d_ = main[:, 2] - main[:, 1]
    infwd = (main[:, 1] >= f0) & (main[:, 1] < f1)
    out.update(
        api_n=int(len(main)), api_sum_ns=int(d_.sum()), api_launch_n=int((c == 1).sum()), api_launch_sum_ns=int(d_[c == 1].sum()),
        api_sync_n=int((c == 2).sum()), api_sync_sum_ns=int(d_[c == 2].sum()), api_query_n=int((c == 3).sum()),
        api_query_sum_ns=int(d_[c == 3].sum()), api_memcpy_n=int((c == 4).sum()), api_memcpy_sum_ns=int(d_[c == 4].sum()),
        api_other_sum_ns=int(d_[c == 0].sum()), fwd_api_n=int(infwd.sum()), fwd_api_sum_ns=int(d_[infwd].sum()),
        fwd_launch_n=int((infwd & (c == 1)).sum()), fwd_launch_sum_ns=int(d_[infwd & (c == 1)].sum()),
        fwd_sync_sum_ns=int(d_[infwd & (c == 2)].sum()),
        helper_api_n=int(len(helper)), helper_api_sum_ns=int((helper[:, 2] - helper[:, 1]).sum()),
    )
    # longest single sync call (e.g. the end-of-step cudaEventSynchronize)
    out["api_sync_max_ns"] = int(d_[c == 2].max()) if (c == 2).any() else 0
    return out


if __name__ == "__main__":
    main()
