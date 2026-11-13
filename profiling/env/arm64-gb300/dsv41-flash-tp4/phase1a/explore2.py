import json, sys, collections, numpy as np, pyarrow.parquet as pq
G = ["fmha","indexer","sparse_meta","moe","comm","other"]; KINDS = ["mhc","oneshot","twoshot","nccl_allreduce","nccl_allgather","nccl_other"]
def load(T):
    it = pq.read_table(f"{T}/iterations.parquet").to_pydict(); ir = pq.read_table(f"{T}/iter_rank.parquet").to_pydict()
    I = {k: np.array(v) for k, v in it.items() if k not in ("chunk_pairs","stage")}; I["stage"] = np.array(it["stage"]); I["chunk_pairs"] = it["chunk_pairs"]
    R = {k: np.array(v) for k, v in ir.items()}
    pos = {(int(a), int(r)): i for i, (a, r) in enumerate(zip(R["iteration"], R["rank"]))}
    return I, R, pos
for T in sys.argv[1:]:
    I, R, pos = load(T); n = len(I["iteration"]); ok = ~I["excluded"]
    def rk(k, a): return np.array([R[k][pos[(a, r)]] for r in range(4)])
    print(f"\n######## {T.split('/')[-1]}")
    g = collections.defaultdict(list)
    for i in range(n - 1):
        if I["iteration"][i + 1] == I["iteration"][i] + 1 and ok[i] and ok[i + 1]:
            c = f"decode b={I['decode_requests'][i+1]:3d}" if I["stage"][i + 1] == "decode" else f"mixed P={I['prefill_tokens'][i+1]//4000*4:3d}k"
            g[c].append((I["t0_ns"][i + 1] - I["t1_ns"][i]) / 1e6)
    print("== gap to next iteration by next class (ms)")
    for c in sorted(g):
        v = np.array(g[c]); print(f"{c:16s} n={len(v):5d} med={np.median(v):7.2f} p10={np.percentile(v,10):7.2f} p90={np.percentile(v,90):7.2f} max={v.max():7.2f}")
    byb = collections.defaultdict(list)
    for i in range(n):
        if ok[i] and I["stage"][i] == "decode": byb[int(I["decode_requests"][i])].append(i)
    print("\n== decode per b: per-rank medians (ms): busy | compute_union | comm_union | idle | ovl_cc | ovl_cm | groups(fmha,idx,smeta,moe,other) r0 | side_union r0 | host_window r0 | gpu_tail r0 | sync_max r0 | fwd_launch_n r0")
    for b in sorted(byb):
        idx = byb[b]; a = [int(I["iteration"][i]) for i in idx[:400]]
        def med(k): return np.median(np.array([rk(k, x) for x in a]), axis=0) / 1e6
        gs = [np.median([R[f"g_{gg}_sum_ns"][pos[(x,0)]] for x in a]) / 1e6 for gg in ("fmha","indexer","sparse_meta","moe","other")]
        print(f"b={b:3d} n={len(idx):5d} busy={np.round(med('busy_ns'),2)} cu={np.round(med('compute_union_ns'),2)} mu={np.round(med('comm_union_ns'),2)} idle={np.round(med('idle_ns'),2)} ovl_cc={np.round(med('overlap_comp_comp_ns'),2)} ovl_cm={np.round(med('overlap_comp_comm_ns'),3)} g={np.round(gs,2)} side={med('side_stream_union_ns')[0]:.2f} host={med('host_window_ns')[0]:.2f} tail={med('gpu_tail_after_host_ns')[0]:.2f} syncmax={med('api_sync_max_ns')[0]:.2f} L={np.median([R['fwd_launch_n'][pos[(x,0)]] for x in a]):.0f} kinds={[int(np.median([R[f'c_{k}_count'][pos[(x,0)]] for x in a])) for k in KINDS]}")
    # segment structure: sum_k max_r seg vs max_r sum_k seg; intrinsic per kind
    co = pq.read_table(f"{T}/collectives.parquet", columns=["iteration","rank","k","kind","start_rel_ns","end_rel_ns","seg_len_ns","seg_compute_union_ns","seg_idle_ns","tail_len_ns"]).to_pydict()
    C = {k: np.array(v) for k, v in co.items()}; oc = np.lexsort((C["k"], C["rank"], C["iteration"])); C = {k: v[oc] for k, v in C.items()}
    imin = C["iteration"].min(); cb = np.searchsorted(C["iteration"], np.arange(imin, C["iteration"].max() + 2))
    def coll(a):
        lo, hi = cb[a - imin], cb[a - imin + 1]; s = slice(lo, hi); K = (hi - lo) // 4
        return {k: C[k][s].reshape(4, K) for k in C}
    print("\n== decode per b: sum_k max_r seg_len vs max_r sum_k seg_len (ms), intrinsic per kind per call (us), wait per rank (ms), seg0 idle per rank (ms)")
    for b in sorted(byb):
        idx = byb[b][:120]; smx = []; mxs = []; intr = collections.defaultdict(list); waits = []; seg0 = []
        for i in idx:
            c = coll(int(I["iteration"][i])); sl = c["seg_len_ns"]; smx.append(sl.max(axis=0).sum() / 1e6); mxs.append(sl.sum(axis=1).max() / 1e6)
            d = c["end_rel_ns"] - c["start_rel_ns"]; mn = d.min(axis=0)
            for kk in range(len(KINDS)):
                m = c["kind"][0] == kk
                if m.any(): intr[KINDS[kk]].append(mn[m].mean() / 1e3)
            waits.append((d - mn).sum(axis=1) / 1e6); seg0.append(c["seg_idle_ns"][:, 0] / 1e6)
        print(f"b={b:3d} sum_max={np.median(smx):.3f} max_sum={np.median(mxs):.3f} intr_us={{{', '.join(f'{k}:{np.median(v):.1f}' for k, v in intr.items())}}} wait={np.round(np.median(waits,axis=0),2)} seg0_idle={np.round(np.median(seg0,axis=0),2)}")
    print("\n== mixed (clean): it, P, nchunks, b, chunks | L | H | F | api_launch | busy | tail | wall | intrinsic twoshot/nccl per call us")
    for i in range(n):
        if ok[i] and I["stage"][i] == "mixed":
            a = int(I["iteration"][i]); c = coll(a); d = c["end_rel_ns"] - c["start_rel_ns"]; mn = d.min(axis=0)
            ks = {KINDS[kk]: f"{mn[c['kind'][0]==kk].mean()/1e3:.0f}x{int((c['kind'][0]==kk).sum())}" for kk in range(len(KINDS)) if (c['kind'][0]==kk).any()}
            print(f"{a:6d} P={I['prefill_tokens'][i]:6d} nc={I['n_prefill_chunks'][i]} b={I['decode_requests'][i]:3d} ch={I['chunk_pairs'][i][:34]:34s} L={rk('fwd_launch_n',a)[0]} H={np.round(rk('host_window_ns',a)/1e6,0)} F={np.round(rk('fwd_window_ns',a)/1e6,0)} apiL={np.round(rk('fwd_launch_sum_ns',a)/1e6,0)} busy={np.round(rk('busy_ns',a)/1e6,0)} cu={np.round(rk('compute_union_ns',a)/1e6,0)} tail={np.round(rk('gpu_tail_after_host_ns',a)/1e6,0)} wall={I['wall_ns'][i]/1e6:.1f} {ks}")
