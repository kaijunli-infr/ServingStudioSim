# Phase 1 planning (CPU-only, read-only on frozen Phase 0 traces).
# TP4 per-rank critical-path decomposition for clean iterations by class:
#   rank-local compute (union of non-collective kernels), collective kernel time, intrinsic collective cost (min over ranks per call),
#   rank-arrival skew (start spread per call), collective wait (per-rank duration minus intrinsic), GPU idle (span - busy),
#   compute/comm overlap (kernel sum - union), slowest-rank critical path vs sum over ranks, host launch skew across ranks.
# Timestamps: all 4 ranks are processes on the same host; nsys/CUPTI timestamps share one clock, so cross-rank comparison is direct.
import json, sys, collections
import numpy as np, pyarrow.parquet as pq
P = sys.argv[1]; EXCL = set(int(x) for x in sys.argv[2].split(",") if x) if len(sys.argv) > 2 else set(); LABEL = sys.argv[3] if len(sys.argv) > 3 else P
OUT = sys.argv[4] if len(sys.argv) > 4 else None
d = json.load(open(f"{P}/parsed.json")); det = d["iteration_details"]; names = {int(k): v for k, v in d["kernel_names"].items()}
tab = pq.read_table(f"{P}/parsed.kernels.parquet", columns=["detail_index", "range_index", "name_id", "start_ns", "end_ns"])
di = tab["detail_index"].to_numpy(); ri = tab["range_index"].to_numpy(); nid = tab["name_id"].to_numpy()
s = tab["start_ns"].to_numpy().astype(np.int64); e = tab["end_ns"].to_numpy().astype(np.int64); dur = e - s
_o = np.argsort(di, kind="stable"); di = di[_o]; ri = ri[_o]; nid = nid[_o]; s = s[_o]; e = e[_o]; dur = dur[_o]
_b = np.searchsorted(di, np.arange(len(det) + 1))
def SL(a): return slice(int(_b[a]), int(_b[a + 1]))
rd = np.zeros(len(det) * 64, dtype=np.int8) - 1
for a, x in enumerate(det):
    for b, r in enumerate(x["ranges"]): rd[a * 64 + b] = r["device_id"]
dev_k = rd[di.astype(np.int64) * 64 + ri]
def is_comm(n): return any(t in n for t in ("twoshotAllreduce", "oneshotAllreduceFusion", "ncclDevKernel", "all_reduce_mhc"))
def comm_kind(n):
    for t, lab in (("twoshotAllreduce", "twoshot"), ("oneshotAllreduceFusion", "oneshot"), ("all_reduce_mhc", "ar_mhc"), ("ncclDevKernel_AllReduce", "nccl_ar"), ("ncclDevKernel_AllGather", "nccl_ag"), ("ncclDevKernel", "nccl_other")):
        if t in n: return lab
    return None
ck = np.zeros(max(names) + 1, dtype=np.int8)
for k, v in names.items(): ck[k] = 1 if is_comm(v) else 0
comm_k = ck[nid].astype(bool)

def union(ss, ee):
    if len(ss) == 0: return 0
    o = np.argsort(ss); ss = ss[o]; ee = ee[o]; tot = 0; cs, ce = ss[0], ee[0]
    for i in range(1, len(ss)):
        if ss[i] > ce: tot += ce - cs; cs, ce = ss[i], ee[i]
        else: ce = max(ce, ee[i])
    return tot + (ce - cs)

def analyze(a):
    x = det[a]; S = SL(a); dk = dev_k[S]; cm = comm_k[S]; sS = s[S]; eS = e[S]; du = dur[S]; nS = nid[S]
    ranks = []; coll = []
    for r in range(4):
        m = dk == r
        if not m.any(): return None
        busy = union(sS[m], eS[m]); comp = union(sS[m & ~cm], eS[m & ~cm]); comu = union(sS[m & cm], eS[m & cm])
        t0 = sS[m].min(); t1 = eS[m].max()
        mc = m & cm; o = np.argsort(sS[mc]); coll.append((sS[mc][o], eS[mc][o], du[mc][o], nS[mc][o]))
        ranks.append(dict(t0=int(t0), t1=int(t1), span=(t1 - t0) / 1e6, busy=busy / 1e6, compute=comp / 1e6, comm_union=comu / 1e6, comm_sum=du[mc].sum() / 1e6,
                          ksum=du[m].sum() / 1e6, idle=(t1 - t0 - busy) / 1e6, overlap=(du[m].sum() - busy) / 1e6, ncoll=int(mc.sum())))
    ns = [len(c[0]) for c in coll]; aligned = len(set(ns)) == 1
    skew = intrinsic = 0.0; wait = [0.0] * 4; by_kind = collections.Counter(); by_kind_int = collections.Counter(); by_kind_skew = collections.Counter(); first = []
    if aligned and ns[0] > 0:
        st = np.stack([c[0] for c in coll]); du4 = np.stack([c[2] for c in coll]); kinds = [comm_kind(names[int(k)]) for k in coll[0][3]]
        sk = st.max(0) - st.min(0); mn = du4.min(0); w = du4 - mn[None, :]
        skew = sk.sum() / 1e6; intrinsic = mn.sum() / 1e6; wait = list(w.sum(1) / 1e6)
        for i, kd in enumerate(kinds): by_kind[kd] += 1; by_kind_int[kd] += mn[i] / 1e3; by_kind_skew[kd] += sk[i] / 1e3
        for i in range(min(5, ns[0])): first.append((kinds[i], float(sk[i] / 1e3), [float(v / 1e3) for v in du4[:, i]]))
    t0 = min(r["t0"] for r in ranks); t1 = max(r["t1"] for r in ranks)
    slow = int(np.argmax([r["t1"] for r in ranks])); gpu_wall = (t1 - t0) / 1e6
    start_skew = (max(r["t0"] for r in ranks) - t0) / 1e6; end_skew = (t1 - min(r["t1"] for r in ranks)) / 1e6
    # host launch skew: 'forward' NVTX range start per device
    fw = {rr["device_id"]: rr["start_ns"] for rr in x["ranges"] if rr["phase"] == "forward"}
    host_skew = (max(fw.values()) - min(fw.values())) / 1e6 if len(fw) == 4 else float("nan")
    return dict(a=a, it=x["iteration"], stage=x["stage"], nreq=x["metrics"].get("decode_requests", 0), pre=x["metrics"].get("prefill_tokens", 0),
                chunks=x["metrics"].get("prefill_chunk_pairs", []), ranks=ranks, aligned=aligned, ncoll=ns, skew=skew, intrinsic=intrinsic, wait=wait,
                by_kind={k: (by_kind[k], by_kind_int[k], by_kind_skew[k]) for k in by_kind}, first=first, slow=slow, gpu_wall=gpu_wall,
                start_skew=start_skew, end_skew=end_skew, host_skew=host_skew, sum_busy=sum(r["busy"] for r in ranks), slow_busy=ranks[slow]["busy"])

# classes
def cls(x):
    if x["stage"] == "decode": return f"decode nreq={x['metrics'].get('decode_requests', 0)}"
    pt = x["metrics"].get("prefill_tokens", 0); ch = x["metrics"].get("prefill_chunk_pairs", [])
    if len(ch) == 1 and ch[0][1] >= 16000: return "prefill single 16k chunk"
    if pt >= 10000: return "mixed >=10k prefill tok (multi-chunk)"
    if pt >= 1000: return "mixed 1k-10k prefill tok"
    return "mixed <1k prefill tok"
groups = collections.defaultdict(list)
for a, x in enumerate(det):
    if x["iteration"] in EXCL or x["jit_module_loads"]: continue
    groups[cls(x)].append(a)
# host wall per iteration (rank 0 ranges start -> next iteration start)
h0 = {}
for a, x in enumerate(det):
    r0 = [r for r in x["ranges"] if r["device_id"] == 0]
    if r0: h0[x["iteration"]] = min(r["start_ns"] for r in r0)
def host_wall(it): return (h0[it + 1] - h0[it]) / 1e6 if (it + 1) in h0 and it in h0 else float("nan")

print(f"== {LABEL}: clean iterations by class (JIT/module-load iterations excluded)")
results = {}
med = lambda v: float(np.median(v)) if len(v) else float("nan")
def keep(c, n): return n >= 20 or not c.startswith("decode")
hdr = f"{'class':38s} {'n':>5} | {'gpu_wall':>8} {'host_wall':>9} {'sumbusy':>7} {'slowbusy':>8} | {'compute':>7} {'comm_u':>6} {'intrins':>7} {'wait':>6} {'skew':>6} {'idle':>5} {'ovlp':>5} | {'hostskw':>7} {'gpu0skw':>7} {'slow rank hist'}"
print(hdr)
for c in sorted(groups, key=lambda k: (not k.startswith("decode"), k)):
    al = groups[c]
    if not keep(c, len(al)): continue
    # sample at most 400 iterations evenly for speed
    idx = al if len(al) <= 400 else [al[i] for i in np.linspace(0, len(al) - 1, 400).astype(int)]
    rs = [analyze(a) for a in idx]; rs = [r for r in rs if r]
    if not rs: continue
    slowhist = collections.Counter(r["slow"] for r in rs)
    agg = dict(n=len(al), sampled=len(rs), gpu_wall=med([r["gpu_wall"] for r in rs]), host_wall=med([host_wall(r["it"]) for r in rs if not np.isnan(host_wall(r["it"]))]),
               sum_busy=med([r["sum_busy"] for r in rs]), slow_busy=med([r["slow_busy"] for r in rs]),
               compute=[med([r["ranks"][k]["compute"] for r in rs]) for k in range(4)], comm_union=[med([r["ranks"][k]["comm_union"] for r in rs]) for k in range(4)],
               intrinsic=med([r["intrinsic"] for r in rs if r["aligned"]]), wait=[med([r["wait"][k] for r in rs if r["aligned"]]) for k in range(4)],
               skew=med([r["skew"] for r in rs if r["aligned"]]), idle=[med([r["ranks"][k]["idle"] for r in rs]) for k in range(4)],
               overlap=[med([r["ranks"][k]["overlap"] for r in rs]) for k in range(4)], host_skew=med([r["host_skew"] for r in rs]), gpu_start_skew=med([r["start_skew"] for r in rs]),
               slow_hist=dict(slowhist), aligned_frac=sum(r["aligned"] for r in rs) / len(rs), ncoll=rs[0]["ncoll"],
               by_kind={k: [med([r["by_kind"][k][j] for r in rs if k in r["by_kind"]]) for j in range(3)] for k in set().union(*[r["by_kind"].keys() for r in rs])})
    results[c] = agg
    sl = int(np.argmax(agg["compute"]))
    print(f"{c:38s} {len(al):>5} | {agg['gpu_wall']:8.3f} {agg['host_wall']:9.3f} {agg['sum_busy']:7.2f} {agg['slow_busy']:8.2f} | {agg['compute'][sl]:7.2f} {agg['comm_union'][sl]:6.2f} {agg['intrinsic']:7.2f} {agg['wait'][sl]:6.2f} {agg['skew']:6.2f} {agg['idle'][sl]:5.2f} {agg['overlap'][sl]:5.2f} | {agg['host_skew']:7.3f} {agg['gpu_start_skew']:7.3f} {dict(sorted(slowhist.items()))}")
    print(f"{'':38s} {'':>5}   per-rank compute {['%.2f' % v for v in agg['compute']]} comm_union {['%.2f' % v for v in agg['comm_union']]} wait {['%.2f' % v for v in agg['wait']]} idle {['%.2f' % v for v in agg['idle']]} ovlp {['%.2f' % v for v in agg['overlap']]} coll/rank {agg['ncoll'][0]} aligned {agg['aligned_frac']:.2f}")
    print(f"{'':38s} {'':>5}   collectives by kind (calls, intrinsic us/iter, start-skew us/iter): " + "; ".join(f"{k}: {int(v[0])}, {v[1]:.0f}, {v[2]:.0f}" for k, v in sorted(agg['by_kind'].items())))
print("\nColumns: gpu_wall = max over ranks of last kernel end - min first kernel start; sumbusy = sum of 4 ranks' busy (NOT a critical path);"
      " slowbusy = busy of the rank that finishes last; compute/comm_u/wait/idle/ovlp shown for the rank with the largest compute;"
      " intrins = sum over collective calls of min-over-ranks duration; wait = per-rank collective time above intrinsic; skew = sum over calls of start spread;"
      " hostskw = spread of the 'forward' NVTX start across ranks; gpu0skw = spread of first kernel start across ranks.")

# representative iterations: median gpu_wall per class
print("\n== representative clean iterations (median gpu_wall of class), per-rank timeline")
for c in sorted(results, key=lambda k: (not k.startswith("decode"), k)):
    al = groups[c]
    if c.startswith("decode") and results[c]["n"] < 200: continue
    idx = al if len(al) <= 400 else [al[i] for i in np.linspace(0, len(al) - 1, 400).astype(int)]
    rs = [r for r in (analyze(a) for a in idx) if r]
    walls = sorted(rs, key=lambda r: r["gpu_wall"]); r = walls[len(walls) // 2]
    print(f"\n-- {c}: iteration {r['it']} (nreq={r['nreq']}, prefill_tokens={r['pre']}, chunks={str(r['chunks'])[:40]}), gpu_wall {r['gpu_wall']:.3f} ms, host wall {host_wall(r['it']):.3f} ms, slowest rank {r['slow']}, host launch skew {r['host_skew']:.3f} ms")
    t0 = min(k["t0"] for k in r["ranks"])
    print(f"   {'rank':>4} {'first_k':>8} {'last_k':>8} {'span':>7} {'busy':>7} {'compute':>7} {'comm_u':>6} {'comm_sum':>8} {'wait':>6} {'idle':>5} {'ovlp':>5} {'ncoll':>5}")
    for k, rr in enumerate(r["ranks"]):
        print(f"   {k:>4} {(rr['t0']-t0)/1e6:8.3f} {(rr['t1']-t0)/1e6:8.3f} {rr['span']:7.3f} {rr['busy']:7.3f} {rr['compute']:7.3f} {rr['comm_union']:6.3f} {rr['comm_sum']:8.3f} {r['wait'][k]:6.3f} {rr['idle']:5.3f} {rr['overlap']:5.3f} {rr['ncoll']:>5}")
    print(f"   collective intrinsic sum {r['intrinsic']:.3f} ms, start-skew sum {r['skew']:.3f} ms; first calls (kind, start skew us, per-rank dur us): {[(k, round(sk,1), [round(v,1) for v in dd]) for k, sk, dd in r['first']]}")
if OUT: json.dump(results, open(OUT, "w"), indent=1)
