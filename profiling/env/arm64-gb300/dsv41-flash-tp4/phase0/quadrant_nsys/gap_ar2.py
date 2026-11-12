# Follow-up: per-rank skew in mixed iterations (which rank lags, host-launch vs GPU), NCCL per rank, which thread syncs, decode per-rank busy.
import json, re, sys, collections
import numpy as np, pyarrow.parquet as pq
P = sys.argv[1]; EXCL = set(range(239, 247))
d = json.load(open(f"{P}/parsed.json")); det = d["iteration_details"]; names = {int(k): v for k, v in d["kernel_names"].items()}
tab = pq.read_table(f"{P}/parsed.kernels.parquet", columns=["detail_index","range_index","name_id","start_ns","end_ns"])
di = tab["detail_index"].to_numpy(); ri = tab["range_index"].to_numpy(); nid = tab["name_id"].to_numpy()
s = tab["start_ns"].to_numpy().astype(np.int64); e = tab["end_ns"].to_numpy().astype(np.int64); dur = e - s
rd_arr = np.zeros(len(det) * 64, dtype=np.int8) - 1
for a, x in enumerate(det):
    for b, r in enumerate(x["ranges"]): rd_arr[a * 64 + b] = r["device_id"]
dev_k = rd_arr[di.astype(np.int64) * 64 + ri]
coll = np.array([k for k, v in names.items() if any(t in v for t in ("twoshotAllreduce", "oneshotAllreduceFusion", "ncclDevKernel", "all_reduce_mhc"))])
two = np.array([k for k, v in names.items() if "twoshotAllreduce" in v]); nccl_ar = np.array([k for k, v in names.items() if "ncclDevKernel_AllReduce" in v])
h = json.load(open(f"{P}/host_timeline.json")); thr = h["threads"]; strings = h["strings"]
api = np.array(h["api_calls"], dtype=np.int64)
tid_of_dev = {t["device_id"]: i for i, t in enumerate(thr) if t["role"] == "worker main thread"}
launch_ids = {i for i, n in enumerate(strings) if n.startswith(("cuLaunchKernel", "cudaLaunchKernel", "cudaGraphLaunch"))}
def busy_union(m):
    if not m.any(): return 0
    o = np.argsort(s[m]); ss = s[m][o]; ee = e[m][o]; tot = 0; cs, ce = ss[0], ee[0]
    for i in range(1, len(ss)):
        if ss[i] > ce: tot += ce - cs; cs, ce = ss[i], ee[i]
        else: ce = max(ce, ee[i])
    return tot + ce - cs
print("== per-rank breakdown of mixed/prefill iterations (ms). compute = kernels that are not collectives; coll = twoshot+oneshot+nccl+mhc_allreduce;")
print("   host_fwd = NVTX forward range on the worker main thread; launches = CUDA launch API calls inside the iteration host span; gpu_first = first kernel start relative to rank 0 (us)")
print(f"{'iter':>5} {'pre_tok':>7} rank {'compute':>8} {'coll':>8} {'twoshot':>8} {'ncclAR':>8} {'gpu_busy':>8} {'gpu_span':>8} {'host_span':>9} {'host_fwd':>8} {'launches':>8} {'launch_ms':>9} {'gpu_first_us':>12} {'gpu_last_us':>11}")
lag_count = collections.Counter(); lag_count_clean = collections.Counter()
for a, x in enumerate(det):
    if x["stage"] == "decode": continue
    pre = x["metrics"].get("prefill_tokens", 0)
    base_first = None; base_last = None; rows = []
    for r in range(4):
        m = (di == a) & (dev_k == r)
        if not m.any(): continue
        comp = dur[m & ~np.isin(nid, coll)].sum() / 1e6; cl = dur[m & np.isin(nid, coll)].sum() / 1e6
        tw = dur[m & np.isin(nid, two)].sum() / 1e6; na = dur[m & np.isin(nid, nccl_ar)].sum() / 1e6
        rr = sorted([q for q in x["ranges"] if q["device_id"] == r], key=lambda q: q["start_ns"])
        h0, h1 = rr[0]["start_ns"], rr[-1]["end_ns"]; fwd = sum(q["duration_ns"] for q in rr if q["phase"] == "forward") / 1e6
        tix = tid_of_dev[r]
        ma = (api[:, 0] == tix) & (api[:, 1] >= h0) & (api[:, 1] < h1)
        la = sum(1 for row in api[ma] if row[3] in launch_ids); lt = sum(row[2] - row[1] for row in api[ma] if row[3] in launch_ids) / 1e6
        g0, g1 = s[m].min(), e[m].max()
        if base_first is None: base_first, base_last = g0, g1
        rows.append((r, comp, cl, tw, na, busy_union(m) / 1e6, (g1 - g0) / 1e6, (h1 - h0) / 1e6, fwd, la, lt, (g0 - base_first) / 1e3, (g1 - base_last) / 1e3))
    # laggard = rank with the smallest collective time
    lag = min(rows, key=lambda z: z[2])[0]; lag_count[lag] += 1
    if x["iteration"] not in EXCL: lag_count_clean[lag] += 1
    for z in rows:
        print(f"{x['iteration']:>5} {pre:>7}   r{z[0]} {z[1]:8.1f} {z[2]:8.1f} {z[3]:8.1f} {z[4]:8.1f} {z[5]:8.1f} {z[6]:8.1f} {z[7]:9.1f} {z[8]:8.1f} {z[9]:8d} {z[10]:9.2f} {z[11]:12.0f} {z[12]:11.0f}" + ("  <- laggard (least waiting)" if z[0] == lag else "") + ("  EXCL" if x["iteration"] in EXCL and z[0] == 0 else ""))
print("\n laggard rank histogram: all mixed", dict(lag_count), " clean", dict(lag_count_clean))
# i-th twoshot call skew, for a clean 3k-4k iteration
cand = [a for a, x in enumerate(det) if 1000 <= x["metrics"].get("prefill_tokens", 0) < 10000 and x["iteration"] not in EXCL]
for a in cand[:2]:
    x = det[a]; print(f"\n== call-level skew, clean iteration {x['iteration']} (prefill_tokens={x['metrics']['prefill_tokens']}): i-th twoshot call across ranks")
    per = []
    for r in range(4):
        m = (di == a) & (dev_k == r) & np.isin(nid, two); o = np.argsort(s[m]); per.append((s[m][o], e[m][o], dur[m][o]))
    n = min(len(p[0]) for p in per); tot_sk = 0; tot_mx = 0; tot_mn = 0
    for i in range(n):
        st_ = [p[0][i] for p in per]; en = [p[1][i] for p in per]; ds = [p[2][i] for p in per]
        tot_sk += max(st_) - min(st_); tot_mx += max(ds); tot_mn += min(ds)
        if i < 5 or i % 20 == 0: print(f"   call {i:3d}: start skew {(max(st_)-min(st_))/1e3:8.1f} us (last to arrive: r{int(np.argmax(st_))})  dur us {[round(v/1e3,1) for v in ds]}  end skew {(max(en)-min(en))/1e3:6.1f} us")
    print(f"   calls={n}: sum max-dur {tot_mx/1e6:.1f} ms, sum min-dur (≈ true collective cost) {tot_mn/1e6:.1f} ms, sum start-skew {tot_sk/1e6:.1f} ms -> skew explains {100*tot_sk/max(tot_mx,1):.0f}% of the max-rank time")
# which thread issues the long cudaEventSynchronize in decode gaps
print("\n== cudaEventSynchronize by thread (role) over whole window:")
sync_ids = {i for i, n in enumerate(strings) if n.startswith("cudaEventSynchronize")}
cnt = collections.Counter(); tt = collections.Counter()
for row in api:
    if row[3] in sync_ids: cnt[row[0]] += 1; tt[row[0]] += row[2] - row[1]
for k in sorted(cnt): print(f"   thread {k} {thr[k]['process']:16s} {thr[k]['role']:22s} dev={thr[k]['device_id']}  calls={cnt[k]}  total {tt[k]/1e9:.2f} s  mean {tt[k]/cnt[k]/1e6:.3f} ms")
# decode per-rank GPU busy and host span (is rank 2 slower in graph mode too?)
print("\n== steady decode (48 reqs), per rank: GPU busy / GPU span / host span / collective ms, medians over iterations")
acc = {r: collections.defaultdict(list) for r in range(4)}
for a, x in enumerate(det):
    if x["stage"] != "decode" or x["metrics"].get("decode_requests", 0) != 48: continue
    g0s = {}
    for r in range(4):
        m = (di == a) & (dev_k == r)
        if not m.any(): continue
        rr = [q for q in x["ranges"] if q["device_id"] == r]
        acc[r]["busy"].append(busy_union(m) / 1e6); acc[r]["span"].append((e[m].max() - s[m].min()) / 1e6)
        acc[r]["host"].append((max(q["end_ns"] for q in rr) - min(q["start_ns"] for q in rr)) / 1e6)
        acc[r]["coll"].append(dur[m & np.isin(nid, coll)].sum() / 1e6); acc[r]["two"].append(dur[m & np.isin(nid, two)].sum() / 1e6)
        g0s[r] = s[m].min()
    for r in g0s: acc[r]["first_rel_us"].append((g0s[r] - min(g0s.values())) / 1e3)
for r in range(4):
    print(f"   r{r}: " + "  ".join(f"{k}={np.median(v):.3f}" for k, v in acc[r].items()))
