# Phase 1 planning (CPU-only, read-only on frozen Phase 0 traces).
# 16k-chunk prefill cost as a function of KV offset, per long-context chain, with kernel groups split into
# FMHA / indexer / sparse metadata / MoE / communication / other, plus GPU idle (span-busy) and multi-stream overlap (ksum-busy).
# Fits per chain and pooled; linear vs quadratic; piecewise slopes; per-kernel growth table.
import json, re, sys, collections
import numpy as np, pyarrow.parquet as pq
P = sys.argv[1]; EXCL = set(int(x) for x in sys.argv[2].split(",") if x) if len(sys.argv) > 2 else set(); LABEL = sys.argv[3] if len(sys.argv) > 3 else P
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

def group(n):
    if "twoshotAllreduce" in n or "oneshotAllreduceFusion" in n or "ncclDevKernel_AllReduce" in n or "all_reduce_mhc" in n or "ncclDevKernel_AllGather" in n or "ncclDevKernel" in n: return "comm"
    if n.startswith("fmhaSm100f") or "flashmla" in n or "core_attn" in n: return "fmha"
    if "mqa_logits" in n or "cooperative_topk" in n or "topKPerRow" in n or "_dsv4_topk" in n or "fp8_fp4_gemm" in n or "indexer" in n.lower(): return "indexer"
    if "sparse_indices" in n or "swa_indices" in n or "lookback" in n or "ring_slot" in n or "radixSort" in n or "cub::" in n or "DeviceScan" in n or "DeviceRadixSort" in n or "FullCacheKernel" in n or "_q_kv_norm_quant" in n: return "sparse_meta"
    if n.startswith("bmm_MxE4m3") or n.startswith("bmm_Bfloat16") or "routing" in n or "finalize" in n or "act_and_mul" in n: return "moe"
    return "other"
grp_of = {k: group(v) for k, v in names.items()}
G = ["fmha", "indexer", "sparse_meta", "moe", "comm", "other"]; gi = {g: i for i, g in enumerate(G)}
gk = np.zeros(max(names) + 1, dtype=np.int8)
for k, g in grp_of.items(): gk[k] = gi[g]
g_arr = gk[nid]

def union(ss, ee):
    if len(ss) == 0: return 0
    o = np.argsort(ss); ss = ss[o]; ee = ee[o]; tot = 0; cs, ce = ss[0], ee[0]
    for i in range(1, len(ss)):
        if ss[i] > ce: tot += ce - cs; cs, ce = ss[i], ee[i]
        else: ce = max(ce, ee[i])
    return tot + (ce - cs)

rows = []
for a, x in enumerate(det):
    ch = x["metrics"].get("prefill_chunk_pairs", [])
    if len(ch) != 1 or ch[0][1] < 16000: continue
    S = SL(a); dk = dev_k[S]; ga = g_arr[S]; du = dur[S]; sS = s[S]; eS = e[S]
    per_rank = []
    for r in range(4):
        m = dk == r
        busy = union(sS[m], eS[m]); span = (eS[m].max() - sS[m].min()) if m.any() else 0; ksum = du[m].sum()
        per = {g: du[m & (ga == gi[g])].sum() / 1e6 for g in G}
        per_rank.append(dict(busy=busy / 1e6, span=span / 1e6, ksum=ksum / 1e6, idle=(span - busy) / 1e6, overlap=(ksum - busy) / 1e6, **per))
    rows.append(dict(a=a, it=x["iteration"], kv0=ch[0][0], ln=ch[0][1], dec=x["metrics"].get("decode_requests", 0),
                     dec_kv=sum(x["metrics"].get("decode_kv_lens", [])), excl=x["iteration"] in EXCL, jit=x["jit_module_loads"], ranks=per_rank))
rows.sort(key=lambda z: z["it"])
# chain identification: consecutive iterations whose kv0 == previous kv0 + previous len
chain_id = 0; last = None
for z in rows:
    if last is not None and z["kv0"] == last["kv0"] + last["ln"] and z["it"] - last["it"] <= 3: z["chain"] = last["chain"]
    else: chain_id += 1; z["chain"] = chain_id
    last = z
chains = collections.defaultdict(list)
for z in rows: chains[z["chain"]].append(z)
print(f"== {LABEL}: single-chunk >=16k prefill iterations: {len(rows)} ({sum(1 for z in rows if z['excl'])} excluded), chains: {len(chains)}")
print(f"{'it':>5} {'chain':>5} {'kv0':>7} {'len':>5} {'dec':>3} {'x':>1} | {'busy_r0':>7} {'span_r0':>7} {'idle':>5} {'ovlp':>5} | " + " ".join(f"{g:>8}" for g in G) + f" | {'comm_min':>8} {'comm_max':>8} {'slow_r':>6} {'busy_slow':>9}")
for z in rows:
    r0 = z["ranks"][0]; cm = [r["comm"] for r in z["ranks"]]; slow = int(np.argmax([r["span"] for r in z["ranks"]]))
    print(f"{z['it']:>5} {z['chain']:>5} {z['kv0']:>7} {z['ln']:>5} {z['dec']:>3} {'E' if z['excl'] else ' '} | {r0['busy']:7.1f} {r0['span']:7.1f} {r0['idle']:5.1f} {r0['overlap']:5.1f} | " + " ".join(f"{r0[g]:8.2f}" for g in G) + f" | {min(cm):8.2f} {max(cm):8.2f} {slow:>6} {z['ranks'][slow]['busy']:9.1f}")

clean = [z for z in rows if not z["excl"]]
def fit(xs, ys, deg=1):
    xs = np.asarray(xs, float) / 1000.0; ys = np.asarray(ys, float)
    if len(xs) <= deg: return None
    c = np.polyfit(xs, ys, deg); res = ys - np.polyval(c, xs); return c, float(np.sqrt(np.mean(res ** 2))), float(np.max(np.abs(res)))
print("\n== per-chain linear fit of rank-0 busy (ms) vs kv0 (per 1k tokens), clean iterations only")
print(f"{'chain':>5} {'n':>2} {'kv0 range':>16} {'slope ms/1k':>11} {'icpt ms':>8} {'rmse':>6} {'maxres':>6} | slopes per group ms/1k: " + " ".join(f"{g:>9}" for g in G) + f" {'idle':>6} {'ovlp':>6}")
for cid, zs in sorted(chains.items()):
    zs = [z for z in zs if not z["excl"]]
    if len(zs) < 3: continue
    xs = [z["kv0"] for z in zs]; f = fit(xs, [z["ranks"][0]["busy"] for z in zs])
    gs = [fit(xs, [z["ranks"][0][g] for z in zs])[0][0] for g in G]; fi = fit(xs, [z["ranks"][0]["idle"] for z in zs])[0][0]; fo = fit(xs, [z["ranks"][0]["overlap"] for z in zs])[0][0]
    print(f"{cid:>5} {len(zs):>2} {min(xs):>7}-{max(xs):<8} {f[0][0]:11.3f} {f[0][1]:8.1f} {f[1]:6.2f} {f[2]:6.2f} | " + " ".join(f"{v:9.3f}" for v in gs) + f" {fi:6.3f} {fo:6.3f}")
xs = [z["kv0"] for z in clean]; ys = [z["ranks"][0]["busy"] for z in clean]
f1 = fit(xs, ys, 1); f2 = fit(xs, ys, 2)
print(f"\n== pooled clean (n={len(clean)}): linear busy = {f1[0][1]:.1f} + {f1[0][0]:.3f}*kv_k ms  rmse {f1[1]:.2f} max|res| {f1[2]:.2f}")
print(f"   quadratic: {f2[0][2]:.1f} + {f2[0][1]:.3f}*kv_k + {f2[0][0]:.5f}*kv_k^2  rmse {f2[1]:.2f} max|res| {f2[2]:.2f}  (quadratic term over 0..131k = {f2[0][0]*131**2:.1f} ms)")
for lo, hi in [(0, 40000), (40000, 80000), (80000, 140000), (0, 65536), (65536, 140000)]:
    sel = [z for z in clean if lo <= z["kv0"] < hi]
    if len(sel) >= 3:
        f = fit([z["kv0"] for z in sel], [z["ranks"][0]["busy"] for z in sel]); fg = {g: fit([z["kv0"] for z in sel], [z["ranks"][0][g] for z in sel])[0][0] for g in G}
        print(f"   piecewise kv0 in [{lo:>6},{hi:>6}): n={len(sel):2d} slope {f[0][0]:.3f} ms/1k  (" + ", ".join(f"{g} {fg[g]:+.3f}" for g in G) + ")")
# slowest-rank busy pooled
f1s = fit(xs, [max(r["busy"] for r in z["ranks"]) for z in clean], 1)
print(f"   slowest-rank busy: {f1s[0][1]:.1f} + {f1s[0][0]:.3f}*kv_k ms (rmse {f1s[1]:.2f})")
# decode-load confound: residual vs decode tokens
res = np.array(ys) - np.polyval(f1[0], np.array(xs) / 1000.0); dec = np.array([z["dec"] for z in clean]); dkv = np.array([z["dec_kv"] for z in clean])
if len(set(dec)) > 1:
    c = np.corrcoef(res, dec)[0, 1]; c2 = np.corrcoef(res, dkv)[0, 1] if dkv.std() > 0 else float("nan")
    print(f"   residual correlation with #decode requests in the iteration: r={c:.2f}; with decode kv sum: r={c2:.2f}; per-decode-request slope {np.polyfit(dec, res, 1)[0]:.3f} ms/req")
# per-kernel growth: compare shallowest vs deepest clean chunks
print("\n== per-kernel growth (rank 0): mean time in the 3 deepest clean chunks minus mean in the 3 shallowest; top 20 by growth")
sh = sorted(clean, key=lambda z: z["kv0"])[:3]; dp = sorted(clean, key=lambda z: -z["kv0"])[:3]
def per_name(zs):
    acc = collections.Counter(); cnt = collections.Counter()
    for z in zs:
        S = SL(z["a"]); m = dev_k[S] == 0
        for k, v in zip(nid[S][m], dur[S][m]): acc[int(k)] += v; cnt[int(k)] += 1
    return {k: v / len(zs) / 1e6 for k, v in acc.items()}, {k: v / len(zs) for k, v in cnt.items()}
A1, C1 = per_name(sh); A2, C2 = per_name(dp)
grow = sorted(((A2.get(k, 0) - A1.get(k, 0), k) for k in set(A1) | set(A2)), reverse=True)
print(f"   shallow kv0={[z['kv0'] for z in sh]}  deep kv0={[z['kv0'] for z in dp]}")
print(f"{'growth ms':>9} {'shallow ms':>10} {'deep ms':>8} {'calls':>6} {'group':>11} kernel")
for gval, k in grow[:20]: print(f"{gval:9.2f} {A1.get(k,0):10.2f} {A2.get(k,0):8.2f} {C2.get(k,0):6.0f} {grp_of[k]:>11} {names[k][:95]}")
tot_growth = sum(v for v, _ in grow); print(f"   total growth {tot_growth:.1f} ms; top-20 explain {sum(v for v,_ in grow[:20]):.1f} ms")
json.dump(dict(label=LABEL, groups=G, rows=[{k: v for k, v in z.items() if k != "a"} for z in rows]), open(f"{sys.argv[4]}", "w") if len(sys.argv) > 4 else open("/dev/null", "w"))
