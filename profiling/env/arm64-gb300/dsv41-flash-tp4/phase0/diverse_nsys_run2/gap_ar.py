# Investigate (A) mixed-iteration twoshotAllreduce duration and (B) inter-iteration host gap / duty cycle.
import json, re, sys, collections, statistics as st
import numpy as np, pyarrow.parquet as pq
P = sys.argv[1]
EXCL = set(range(239, 247)) if len(sys.argv) < 3 else set(int(x) for x in sys.argv[2].split(",") if x)
d = json.load(open(f"{P}/parsed.json"))
det = d["iteration_details"]; names = {int(k): v for k, v in d["kernel_names"].items()}
tab = pq.read_table(f"{P}/parsed.kernels.parquet", columns=["detail_index","range_index","name_id","start_ns","end_ns"])
di = tab["detail_index"].to_numpy(); ri = tab["range_index"].to_numpy(); nid = tab["name_id"].to_numpy()
s = tab["start_ns"].to_numpy().astype(np.int64); e = tab["end_ns"].to_numpy().astype(np.int64); dur = e - s
# range -> device
rdev = {}
for a, x in enumerate(det):
    for b, r in enumerate(x["ranges"]): rdev[(a, b)] = r["device_id"]
key = di.astype(np.int64) * 64 + ri
rd_arr = np.zeros(len(det) * 64, dtype=np.int8) - 1
for (a, b), v in rdev.items(): rd_arr[a * 64 + b] = v
dev_k = rd_arr[key]
it_arr = np.array([x["iteration"] for x in det]); stage_arr = np.array([x["stage"] for x in det])
it_k = it_arr[di]; st_k = stage_arr[di]
def pre_tok(x): return x["metrics"].get("prefill_tokens", 0)
# ---------------- (A) twoshot all-reduce in mixed iterations ----------------
two = np.array([k for k, v in names.items() if "twoshotAllreduce" in v]); one = np.array([k for k, v in names.items() if "oneshotAllreduceFusion" in v])
nccl_ar = np.array([k for k, v in names.items() if "ncclDevKernel_AllReduce" in v]); nccl_ag = np.array([k for k, v in names.items() if "ncclDevKernel_AllGather" in v])
mhc_ar = np.array([k for k, v in names.items() if "all_reduce_mhc" in v])
print("== (A) mnnvl twoshotAllreduce per mixed/prefill iteration, per rank (ms, calls). excl = JIT-contaminated iterations", sorted(EXCL))
print(f"{'iter':>5} {'stage':>7} {'pre_tok':>7} {'chunks':>22} | {'r0 ms':>7} {'r1 ms':>7} {'r2 ms':>7} {'r3 ms':>7} | {'calls/r':>7} {'max_call_ms':>11} {'median_call_us':>14} | {'nccl_AR/r':>9} {'nccl_AR_ms_r0':>13} {'host_span_s_r0':>14} {'jit_stall_s':>11}")
rows = []
for a, x in enumerate(det):
    if x["stage"] == "decode": continue
    m = (di == a)
    m2 = m & np.isin(nid, two); mn = m & np.isin(nid, nccl_ar)
    per = [dur[m2 & (dev_k == r)].sum() / 1e6 for r in range(4)]
    calls = m2.sum() / 4
    mx = dur[m2].max() / 1e6 if m2.any() else 0; med = np.median(dur[m2]) / 1e3 if m2.any() else 0
    r0 = [r for r in x["ranges"] if r["device_id"] == 0]
    span = (max(r["end_ns"] for r in r0) - min(r["start_ns"] for r in r0)) / 1e9
    chunks = str([p[1] if isinstance(p, list) else p for p in x["metrics"].get("prefill_chunk_pairs", [])])[:22]
    rows.append((x["iteration"], per, calls, mx, med, mn.sum() / 4, dur[mn & (dev_k == 0)].sum() / 1e6, span, x["jit_stall_ns"] / 1e9, pre_tok(x)))
    print(f"{x['iteration']:>5} {x['stage']:>7} {pre_tok(x):>7} {chunks:>22} | " + " ".join(f"{v:7.1f}" for v in per) + f" | {calls:7.1f} {mx:11.1f} {med:14.1f} | {mn.sum()/4:9.1f} {dur[mn & (dev_k == 0)].sum()/1e6:13.2f} {span:14.3f} {x['jit_stall_ns']/1e9:11.2f}" + ("  EXCL" if x["iteration"] in EXCL else ""))
clean = [r for r in rows if r[0] not in EXCL]; dirty = [r for r in rows if r[0] in EXCL]
def avg(rs): return sum(sum(r[1]) / 4 for r in rs) / max(1, len(rs))
print(f"\n twoshot ms/iter/rank: all mixed {avg(rows):.1f} (n={len(rows)}) | clean {avg(clean):.1f} (n={len(clean)}) | excluded {avg(dirty):.1f} (n={len(dirty)})")
# per-call: twoshot call durations in clean mixed iterations vs decode, binned by prefill tokens
print("\n twoshot per-call duration (us) by iteration class, clean iterations only:")
for lo, hi, lab in [(0, 1, "decode (0 prefill tok)"), (1, 1000, "mixed <1k prefill tok"), (1000, 10000, "mixed 1k-10k"), (10000, 10**9, "mixed >=10k")]:
    sel = [a for a, x in enumerate(det) if lo <= pre_tok(x) < hi and x["iteration"] not in EXCL]
    if not sel: continue
    m = np.isin(di, sel) & np.isin(nid, two)
    if not m.any(): print(f"  {lab:28s} no twoshot calls"); continue
    q = np.percentile(dur[m] / 1e3, [50, 90, 99, 100])
    print(f"  {lab:28s} iters={len(sel):5d} calls/iter/rank={m.sum()/len(sel)/4:6.1f}  p50={q[0]:8.1f} p90={q[1]:8.1f} p99={q[2]:8.1f} max={q[3]:9.1f}  sum/iter/rank={dur[m].sum()/1e6/len(sel)/4:7.2f} ms")
    mn = np.isin(di, sel) & np.isin(nid, nccl_ar)
    if mn.any():
        q = np.percentile(dur[mn] / 1e3, [50, 90, 99, 100])
        print(f"  {'   nccl AllReduce':28s} calls/iter/rank={mn.sum()/len(sel)/4:6.1f}  p50={q[0]:8.1f} p90={q[1]:8.1f} p99={q[2]:8.1f} max={q[3]:9.1f}  sum/iter/rank={dur[mn].sum()/1e6/len(sel)/4:7.2f} ms")
# skew: for one clean 16k iteration, compare the i-th twoshot call start across ranks (arrival skew) vs its duration
big_clean = [a for a, x in enumerate(det) if pre_tok(x) >= 10000 and x["iteration"] not in EXCL]
big_dirty = [a for a, x in enumerate(det) if pre_tok(x) >= 10000 and x["iteration"] in EXCL]
for lab, lst in [("clean", big_clean[:1]), ("excluded", big_dirty[:1])]:
    for a in lst:
        print(f"\n per-call skew analysis, {lab} iteration {det[a]['iteration']} (prefill_tokens={pre_tok(det[a])}): i-th twoshot call: start skew across ranks vs duration")
        per_rank = []
        for r in range(4):
            m = (di == a) & (dev_k == r) & np.isin(nid, two)
            o = np.argsort(s[m]); per_rank.append((s[m][o], e[m][o], dur[m][o]))
        n = min(len(p[0]) for p in per_rank)
        tot_skew = 0; tot_dur = 0
        for i in range(n):
            starts = [p[0][i] for p in per_rank]; ends = [p[1][i] for p in per_rank]; ds = [p[2][i] for p in per_rank]
            skew = (max(starts) - min(starts)) / 1e3; tot_skew += skew; tot_dur += max(ds) / 1e3
            if i < 6 or max(ds) > 2e6: print(f"   call {i:3d}: start skew {skew:9.1f} us  dur per rank us {[round(x/1e3,1) for x in ds]}  end skew {(max(ends)-min(ends))/1e3:8.1f} us")
        print(f"   calls={n}  sum(max dur)={tot_dur/1e3:.1f} ms  sum(start skew)={tot_skew/1e3:.1f} ms  -> skew-explained share {100*tot_skew/max(tot_dur,1):.0f}%")
# ---------------- (B) duty cycle / inter-iteration gap ----------------
print("\n== (B) decode duty cycle, rank 0 (ms per iteration). host = NVTX ranges on worker thread; gpu = kernel timestamps")
rank0 = []
for a, x in enumerate(det):
    r0 = sorted([r for r in x["ranges"] if r["device_id"] == 0], key=lambda r: r["start_ns"])
    if not r0: continue
    m = (di == a) & (dev_k == 0)
    g0 = s[m].min() if m.any() else None; g1 = e[m].max() if m.any() else None
    busy = 0
    if m.any():
        o = np.argsort(s[m]); ss = s[m][o]; ee = e[m][o]
        cur_s, cur_e = ss[0], ee[0]
        for i in range(1, len(ss)):
            if ss[i] > cur_e: busy += cur_e - cur_s; cur_s, cur_e = ss[i], ee[i]
            else: cur_e = max(cur_e, ee[i])
        busy += cur_e - cur_s
    ph = {p: 0 for p in ["preprocess", "forward", "postprocess", "sample", "bookkeep", "eplb"]}
    for r in r0: ph[r["phase"]] += r["duration_ns"]
    rank0.append(dict(it=x["iteration"], stage=x["stage"], h0=r0[0]["start_ns"], h1=r0[-1]["end_ns"], g0=g0, g1=g1, busy=busy, ph=ph, nreq=x["metrics"].get("decode_requests", 0), pre=pre_tok(x), ksum=dur[m].sum()))
rank0.sort(key=lambda z: z["it"])
dec = [(z, nz) for z, nz in zip(rank0, rank0[1:]) if z["stage"] == "decode" and nz["stage"] == "decode" and z["nreq"] == 48 and nz["nreq"] == 48 and z["it"] not in EXCL and nz["it"] not in EXCL and nz["it"] == z["it"] + 1]
print(f" steady decode pairs (48 reqs, consecutive): {len(dec)}")
def med(v): return np.median(v)
wall = [(nz["h0"] - z["h0"]) / 1e6 for z, nz in dec]
host_span = [(z["h1"] - z["h0"]) / 1e6 for z, nz in dec]
host_gap = [(nz["h0"] - z["h1"]) / 1e6 for z, nz in dec]
gpu_span = [(z["g1"] - z["g0"]) / 1e6 for z, nz in dec]
gpu_busy = [z["busy"] / 1e6 for z, nz in dec]
gpu_ksum = [z["ksum"] / 1e6 for z, nz in dec]
gpu_gap = [(nz["g0"] - z["g1"]) / 1e6 for z, nz in dec]
lag = [(z["g0"] - z["h0"]) / 1e6 for z, nz in dec]          # GPU behind host at iteration start
lag_end = [(z["g1"] - z["h1"]) / 1e6 for z, nz in dec]      # GPU work still outstanding when host leaves the iteration
print(f" {'metric':34s} {'median':>8} {'p10':>8} {'p90':>8}")
for lab, v in [("step wall (host start->next start)", wall), ("host NVTX span of iteration", host_span), ("host gap to next iteration", host_gap),
               ("GPU span first->last kernel", gpu_span), ("GPU busy (union of kernels)", gpu_busy), ("GPU kernel sum (all streams)", gpu_ksum),
               ("GPU idle gap to next iteration", gpu_gap), ("GPU lag behind host at iter start", lag), ("GPU outstanding at host iter end", lag_end)]:
    print(f" {lab:34s} {med(v):8.3f} {np.percentile(v,10):8.3f} {np.percentile(v,90):8.3f}")
print(" host phase durations (median ms):", {k: round(np.median([z["ph"][k] for z, _ in dec]) / 1e6, 3) for k in dec[0][0]["ph"]})
w = np.median(wall); b = np.median(gpu_busy); gs = np.median(gpu_span)
print(f" duty cycle: GPU busy / wall = {100*b/w:.1f}%  ; intra-iteration GPU gaps (span-busy) = {gs-b:.3f} ms ({100*(gs-b)/w:.1f}%) ; inter-iteration GPU idle = {np.median(gpu_gap):.3f} ms ({100*np.median(gpu_gap)/w:.1f}%)")
# where does the host wait? CUDA sync API calls on worker thread of device 0 between h1 of iter N and h0 of N+1 (host_timeline)
h = json.load(open(f"{P}/host_timeline.json"))
print(" host_timeline row_format:", h["row_format"])
thr = h["threads"]; strings = h["strings"]
print(" threads:", [(i, t["process"], t["role"], t["device_id"]) for i, t in enumerate(thr)])
api = np.array(h["api_calls"], dtype=np.int64); nv = np.array(h["nvtx_ranges"], dtype=np.int64)
# pick thread index of worker main thread device 0
w0 = [i for i, t in enumerate(thr) if t["device_id"] == 0 and "worker" in t["role"]]
sch = [i for i, t in enumerate(thr) if t["device_id"] is None]
print(" worker0 thread idx", w0, "scheduler thread idx", sch)
# sample: 20 steady decode iterations in the middle — api calls on worker0 during the gap, and nvtx on scheduler thread
mid = dec[len(dec)//2: len(dec)//2 + 200]
gap_api = collections.Counter(); gap_api_t = collections.Counter(); in_api = collections.Counter(); in_api_t = collections.Counter()
sched_nv = collections.Counter(); sched_nv_t = collections.Counter()
for z, nz in mid:
    for tix in w0:
        m = (api[:, 0] == tix) & (api[:, 1] >= z["h1"]) & (api[:, 1] < nz["h0"])
        for row in api[m]: gap_api[strings[row[3]]] += 1; gap_api_t[strings[row[3]]] += row[2] - row[1]
        m = (api[:, 0] == tix) & (api[:, 1] >= z["h0"]) & (api[:, 1] < z["h1"])
        for row in api[m]: in_api[strings[row[3]]] += 1; in_api_t[strings[row[3]]] += row[2] - row[1]
    for tix in sch:
        m = (nv[:, 0] == tix) & (nv[:, 1] >= z["h0"]) & (nv[:, 1] < nz["h0"])
        for row in nv[m]:
            nm = re.sub(r"\(\d+\)", "(N)", strings[row[3]]); sched_nv[nm] += 1; sched_nv_t[nm] += row[2] - row[1]
n = len(mid)
print(f"\n CUDA API calls on worker0 thread INSIDE the iteration NVTX span (per iteration, over {n} steady iterations):")
for k, c in in_api_t.most_common(12): print(f"   {k:45s} {in_api[k]/n:7.1f} calls  {c/n/1e6:8.3f} ms")
print(f" CUDA API calls on worker0 thread in the GAP between iterations (per iteration):")
for k, c in gap_api_t.most_common(12): print(f"   {k:45s} {gap_api[k]/n:7.1f} calls  {c/n/1e6:8.3f} ms")
print(f" NVTX ranges on the scheduler (EngineCore) thread per step (host ms):")
for k, c in sched_nv_t.most_common(15): print(f"   {k:45s} {sched_nv[k]/n:7.1f}  {c/n/1e6:8.3f} ms")
# synchronization: where is cudaStreamSynchronize / EventSynchronize / Memcpy D2H on worker0, relative to phases
print("\n sync-type API calls on worker0, by containing phase (per steady iteration):")
sync_names = [i for i, sname in enumerate(strings) if any(t in sname for t in ("Synchronize", "cudaMemcpy", "cudaEventQuery", "cudaStreamQuery", "cuMemcpy", "cuStreamSync", "cuEventSync"))]
sync_set = set(sync_names)
byph = collections.Counter(); byph_t = collections.Counter()
for z, nz in mid:
    a = next(i for i, x in enumerate(det) if x["iteration"] == z["it"])
    r0 = sorted([r for r in det[a]["ranges"] if r["device_id"] == 0], key=lambda r: r["start_ns"])
    for tix in w0:
        m = (api[:, 0] == tix) & (api[:, 1] >= z["h0"]) & (api[:, 1] < nz["h0"])
        for row in api[m]:
            if row[3] not in sync_set: continue
            ph = next((r["phase"] for r in r0 if r["start_ns"] <= row[1] < r["end_ns"]), "gap")
            byph[(ph, strings[row[3]])] += 1; byph_t[(ph, strings[row[3]])] += row[2] - row[1]
for k, c in sorted(byph_t.items(), key=lambda kv: -kv[1])[:15]: print(f"   {k[0]:12s} {k[1]:40s} {byph[k]/n:6.2f} calls {c/n/1e6:8.3f} ms")
# all-iteration gap distribution incl. non-steady, for the record
allg = [(nz["h0"] - z["h1"]) / 1e6 for z, nz in zip(rank0, rank0[1:]) if nz["it"] == z["it"] + 1]
print(f"\n host gap all consecutive iterations: n={len(allg)} median {np.median(allg):.3f} p99 {np.percentile(allg,99):.3f} max {max(allg):.1f} ms")
