# Phase 1 planning: per-rank host launch throughput in eager (mixed) iterations vs graph (decode) iterations, from host_timeline.json.
# For each clean iteration of a class, per worker thread (device): #CUDA runtime/driver API calls inside the device's own NVTX
# iteration window, sum of their durations, launch-type call count, window length -> mean cost per launch and host-bound share.
import json, sys, collections, numpy as np
P = sys.argv[1]; EXCL = set(int(x) for x in sys.argv[2].split(",") if x) if len(sys.argv) > 2 else set()
d = json.load(open(f"{P}/parsed.json")); det = d["iteration_details"]
h = json.load(open(f"{P}/host_timeline.json")); thr = h["threads"]; strings = h["strings"]
api = np.array(h["api_calls"], dtype=np.int64); api = api[np.argsort(api[:, 1], kind="stable")]
is_launch = np.array([("Launch" in s_) for s_ in strings]); is_sync = np.array([any(t in s_ for t in ("Synchronize", "cudaMemcpy", "EventQuery", "StreamQuery")) for s_ in strings])
dev_threads = collections.defaultdict(list)
for i, t in enumerate(thr):
    if t["device_id"] is not None and "worker" in t["role"]: dev_threads[t["device_id"]].append(i)
print("worker threads per device:", {k: v for k, v in dev_threads.items()})
def cls(x):
    if x["stage"] == "decode": return f"decode nreq={x['metrics'].get('decode_requests', 0)}"
    pt = x["metrics"].get("prefill_tokens", 0); ch = x["metrics"].get("prefill_chunk_pairs", [])
    if len(ch) == 1 and ch[0][1] >= 16000: return "prefill single 16k chunk"
    return "mixed >=10k tok" if pt >= 10000 else ("mixed 1k-10k tok" if pt >= 1000 else "mixed <1k tok")
groups = collections.defaultdict(list)
for a, x in enumerate(det):
    if x["iteration"] in EXCL or x["jit_module_loads"]: continue
    groups[cls(x)].append(a)
print(f"{'class':28s} {'n':>4} | per device: api_calls launches api_sum_ms window_ms  us/launch  (median over iterations)")
for c in sorted(groups, key=lambda k: (not k.startswith("decode"), k)):
    al = groups[c]
    if c.startswith("decode") and len(al) < 300: continue
    idx = al if len(al) <= 60 else [al[i] for i in np.linspace(0, len(al) - 1, 60).astype(int)]
    per = collections.defaultdict(list)
    for a in idx:
        x = det[a]
        for dev, tix in dev_threads.items():
            rr = [r for r in x["ranges"] if r["device_id"] == dev]
            if not rr: continue
            w0 = min(r["start_ns"] for r in rr); w1 = max(r["end_ns"] for r in rr)
            lo, hi = np.searchsorted(api[:, 1], [w0, w1]); seg = api[lo:hi]; seg = seg[np.isin(seg[:, 0], tix)]
            n = len(seg); nl = int(is_launch[seg[:, 3]].sum()); tot = (seg[:, 2] - seg[:, 1]).sum() / 1e6; syn = (seg[is_sync[seg[:, 3]], 2] - seg[is_sync[seg[:, 3]], 1]).sum() / 1e6
            per[dev].append((n, nl, tot, (w1 - w0) / 1e6, syn))
    line = f"{c:28s} {len(al):>4} |"
    for dev in sorted(per):
        v = np.array(per[dev]); m = np.median(v, axis=0)
        line += f" d{dev}: {m[0]:6.0f} {m[1]:5.0f} {m[2]:7.2f} {m[3]:8.2f} {1e3*m[2]/max(m[1],1):7.1f} (sync {m[4]:6.2f}) |"
    print(line)
# API names dominating eager mixed iterations, device 2 vs device 0
for c in ("mixed <1k tok", "mixed 1k-10k tok"):
    al = groups.get(c, [])[:6]
    for dev in (0, 2):
        cnt = collections.Counter(); tm = collections.Counter()
        for a in al:
            x = det[a]; rr = [r for r in x["ranges"] if r["device_id"] == dev]; w0 = min(r["start_ns"] for r in rr); w1 = max(r["end_ns"] for r in rr)
            lo, hi = np.searchsorted(api[:, 1], [w0, w1]); seg = api[lo:hi]; seg = seg[np.isin(seg[:, 0], dev_threads[dev])]
            for row in seg: cnt[strings[row[3]]] += 1; tm[strings[row[3]]] += row[2] - row[1]
        n = max(1, len(al))
        print(f"\n {c}, device {dev}, top API calls per iteration (count, ms):", "; ".join(f"{k}: {cnt[k]/n:.0f}, {tm[k]/n/1e6:.2f}" for k, _ in tm.most_common(6)))
