import json, sys, collections, numpy as np, pyarrow.parquet as pq
T = sys.argv[1]
it = pq.read_table(f"{T}/iterations.parquet").to_pydict(); ir = pq.read_table(f"{T}/iter_rank.parquet").to_pydict()
n = len(it["iteration"]); print("iterations", n, "ir cols", len(ir), [k for k in ir if k.startswith("api") or k.startswith("fwd") or k.startswith("host")][:30])
I = {k: np.array(v) for k, v in it.items() if k != "chunk_pairs" and k != "stage"}; I["stage"] = np.array(it["stage"]); I["chunk_pairs"] = it["chunk_pairs"]
R = {k: np.array(v) for k, v in ir.items()}
# index rank rows by iteration
pos = {}
for i, (a, r) in enumerate(zip(R["iteration"], R["rank"])): pos[(int(a), int(r))] = i
def rk(k, a): return np.array([R[k][pos[(a, r)]] for r in range(4)])
ok = ~I["excluded"]
# 1. decode: rank start offsets & segment counts & reconstruction rule check on a sample
co = pq.read_table(f"{T}/collectives.parquet", columns=["iteration","rank","k","kind","start_rel_ns","end_rel_ns","seg_len_ns"]).to_pydict()
C = {k: np.array(v) for k, v in co.items()}
oc = np.lexsort((C["k"], C["rank"], C["iteration"])); C = {k: v[oc] for k, v in C.items()}
cb = np.searchsorted(C["iteration"], np.arange(C["iteration"].min(), C["iteration"].max() + 2))
def coll(a):
    lo, hi = cb[a - C["iteration"].min()], cb[a - C["iteration"].min() + 1]; s = slice(lo, hi)
    K = (hi - lo) // 4
    return C["kind"][s].reshape(4, K), C["start_rel_ns"][s].reshape(4, K), C["end_rel_ns"][s].reshape(4, K), C["seg_len_ns"][s].reshape(4, K)
print("\n== decode by batch: n, K collectives, kinds, median rank start offsets (ms), wall, max_r busy, reconstruction err (ms)")
byb = collections.defaultdict(list)
for i in range(n):
    if ok[i] and I["stage"][i] == "decode": byb[int(I["decode_requests"][i])].append(i)
for b in sorted(byb):
    idx = byb[b]; sm = idx[:: max(1, len(idx) // 40)][:40]
    offs = []; walls = []; busy = []; rec = []; Ks = []; kindseq_same = 0; wait_r = []
    for i in sm:
        a = int(I["iteration"][i]); fs = rk("first_start_ns", a); t0 = I["t0_ns"][i]; offs.append((fs - t0) / 1e6)
        walls.append(I["wall_ns"][i] / 1e6); busy.append(rk("busy_ns", a).max() / 1e6)
        kinds, cs, ce, sl = coll(a); Ks.append(kinds.shape[1]); kindseq_same += int((kinds == kinds[0]).all())
        # reconstruction: completion_k = max_r arrival_k + intrinsic_k, where arrival = prev completion + seg_len_r,k
        intr = (ce - cs).min(axis=0); t = fs - t0  # per rank current time
        for k in range(kinds.shape[1]):
            arr = t + sl[:, k]; done = arr.max() + intr[k]; t = np.full(4, done)
        tail = rk("last_end_ns", a) - t0 - ce[:, -1]
        rec.append(((t + tail).max() - I["wall_ns"][i]) / 1e6)
        wait_r.append(((ce - cs) - intr).sum(axis=1) / 1e6)
    offs = np.median(np.array(offs), axis=0); wr = np.median(np.array(wait_r), axis=0)
    print(f"b={b:3d} n={len(idx):5d} K={collections.Counter(Ks).most_common(2)} same_kinds={kindseq_same}/{len(sm)} start_off={np.round(offs,2)} wall={np.median(walls):.2f} maxbusy={np.median(busy):.2f} recon_err med={np.median(rec):+.3f} p90abs={np.percentile(np.abs(rec),90):.3f} wait_r={np.round(wr,2)}")
# 2. mixed: launches vs structure
print("\n== mixed iterations (clean): it, P, nchunks, b, chunks, per-rank fwd_launch_n, host_window ms, fwd_window, fwd_api_sum, fwd_launch_sum, sync, busy, gpu_tail, wall")
for i in range(n):
    if ok[i] and I["stage"][i] == "mixed":
        a = int(I["iteration"][i])
        print(f"{a:6d} P={I['prefill_tokens'][i]:6d} nc={I['n_prefill_chunks'][i]} b={I['decode_requests'][i]:3d} ch={I['chunk_pairs'][i][:40]:40s} L={rk('fwd_launch_n',a)} H={np.round(rk('host_window_ns',a)/1e6,1)} F={np.round(rk('fwd_window_ns',a)/1e6,1)} api={np.round(rk('fwd_api_sum_ns',a)/1e6,1)} lsum={np.round(rk('fwd_launch_sum_ns',a)/1e6,1)} sync={np.round(rk('api_sync_sum_ns',a)/1e6,1)} busy={np.round(rk('busy_ns',a)/1e6,1)} tail={np.round(rk('gpu_tail_after_host_ns',a)/1e6,1)} wall={I['wall_ns'][i]/1e6:.1f}")
# 3. gap to next iteration by class
print("\n== gap to next iteration (t0[i+1]-t1[i]) by next iteration class (ms): median, p10, p90")
g = collections.defaultdict(list)
for i in range(n - 1):
    if I["iteration"][i + 1] == I["iteration"][i] + 1 and ok[i] and ok[i + 1]:
        c = f"decode b={I['decode_requests'][i+1]}" if I["stage"][i + 1] == "decode" else f"mixed P={I['prefill_tokens'][i+1]//4000*4}k"
        g[c].append((I["t0_ns"][i + 1] - I["t1_ns"][i]) / 1e6)
for c in sorted(g, key=lambda s: (s[0], len(s), s)):
    v = np.array(g[c]); print(f"{c:16s} n={len(v):5d} med={np.median(v):6.2f} p10={np.percentile(v,10):6.2f} p90={np.percentile(v,90):6.2f}")
# 4. decode host: run-ahead & sync
print("\n== decode host stats (b=3 and modal): host_window, fwd_window, api_n, api_launch_n, api_sync_sum, api_sync_max, gpu_tail, gpu_in_host")
for b in sorted(byb)[:3] + [max(byb, key=lambda k: len(byb[k]))]:
    idx = byb[b][:200]; a = [int(I["iteration"][i]) for i in idx]
    for key in ("host_window_ns","fwd_window_ns","api_n","api_launch_n","api_sync_sum_ns","api_sync_max_ns","gpu_tail_after_host_ns","gpu_in_host_window_ns","api_launch_sum_ns"):
        v = np.array([rk(key, x) for x in a]); m = np.median(v, axis=0); print(f"b={b} {key:24s} {np.round(m/1e6 if 'ns' in key else m, 3)}")
