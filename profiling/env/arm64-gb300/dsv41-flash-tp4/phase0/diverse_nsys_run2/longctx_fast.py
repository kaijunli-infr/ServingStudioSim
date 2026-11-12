# Long-context evidence from a diverse capture: 16k-chunk prefill cost by kv position (per kernel family group, laggard-rank collectives),
# decode cost vs total kv, folded kernel sequence of a deep-kv prefill chunk and of a deep-kv decode step.
import json, re, sys, collections
import numpy as np, pyarrow.parquet as pq
P = sys.argv[1]; EXCL = set(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else set()
d = json.load(open(f"{P}/parsed.json")); det = d["iteration_details"]; names = {int(k): v for k, v in d["kernel_names"].items()}
tab = pq.read_table(f"{P}/parsed.kernels.parquet", columns=["detail_index","range_index","name_id","start_ns","end_ns"])
di = tab["detail_index"].to_numpy(); ri = tab["range_index"].to_numpy(); nid = tab["name_id"].to_numpy()
s = tab["start_ns"].to_numpy().astype(np.int64); e = tab["end_ns"].to_numpy().astype(np.int64); dur = e - s

# --- fast path: sort kernel rows by detail_index and address each iteration by slice instead of a full-array mask ---
_o = np.argsort(di, kind="stable"); di = di[_o]; ri = ri[_o]; nid = nid[_o]; s = s[_o]; e = e[_o]; dur = dur[_o]
_b = np.searchsorted(di, np.arange(len(det) + 1))
def SL(a): return slice(int(_b[a]), int(_b[a + 1]))
rd = np.zeros(len(det) * 64, dtype=np.int8) - 1; ph = {}
for a, x in enumerate(det):
    for b, r in enumerate(x["ranges"]): rd[a * 64 + b] = r["device_id"]; ph[(a, b)] = r["phase"]
dev_k = rd[di.astype(np.int64) * 64 + ri]
def group(n):
    if "twoshotAllreduce" in n or "oneshotAllreduceFusion" in n or "ncclDevKernel_AllReduce" in n or "all_reduce_mhc" in n: return "allreduce"
    if "ncclDevKernel_AllGather" in n: return "allgather"
    if n.startswith("fmhaSm100f") or "flashmla" in n or "core_attn" in n: return "attn_core(fmha)"
    if "mqa_logits" in n or "cooperative_topk" in n or "topKPerRow" in n or "sparse_indices" in n or "indexer" in n.lower() or "swa_indices" in n or "lookback" in n or "ring_slot" in n: return "indexer+sparse_meta"
    if n.startswith("bmm_MxE4m3") or n.startswith("bmm_Bfloat16"): return "moe_expert_gemm"
    if "routing" in n or "finalize" in n or "_dsv4_topk" in n or "act_and_mul" in n: return "moe_route/finalize/act"
    if "dense_blockscaled_gemm" in n or "nvjet" in n or "splitKreduce" in n or "fp8_gemm" in n or "fp8_fp4_gemm" in n or "cutlass_device_kernel" in n: return "dense_gemm"
    if "mega_mhc" in n or "mhc" in n or "hc_prenorm" in n: return "mhc"
    if "engram" in n.lower() or "hash_ids" in n: return "engram"
    if "Quantize" in n or "quant" in n.lower(): return "quantize"
    if "FullCacheKernel" in n or "insert" in n.lower() or "Insert" in n: return "kv_insert"
    return "other"
grp_of = {k: group(v) for k, v in names.items()}
G = sorted(set(grp_of.values())); gi = {g: i for i, g in enumerate(G)}
gk = np.zeros(max(names) + 1, dtype=np.int8)
for k, g in grp_of.items(): gk[k] = gi[g]
g_arr = gk[nid]
def kv_max(x): return max((a + b for a, b in x["metrics"].get("prefill_chunk_pairs", [])), default=0)
print("== 16k-chunk prefill iterations (single chunk, kv_start >= 0): GPU time per group (ms, rank 0; allreduce = min over ranks), sorted by kv position")
print(f"{'it':>5} {'kv_start':>8} {'chunk':>6} {'dec':>4} {'busy_r0':>8} | " + " ".join(f"{g[:14]:>14}" for g in G))
rows = []
for a, x in enumerate(det):
    ch = x["metrics"].get("prefill_chunk_pairs", [])
    if len(ch) != 1 or ch[0][1] < 16000 or x["iteration"] in EXCL: continue
    S = SL(a); dk = dev_k[S]; ga = g_arr[S]; du = dur[S]
    m0 = (dk == 0)
    per = {}
    for g in G:
        mm = m0 & (ga == gi[g])
        if g == "allreduce":
            per[g] = min(du[(dk == r) & (ga == gi[g])].sum() for r in range(4)) / 1e6
        else: per[g] = du[mm].sum() / 1e6
    r0 = [q for q in x["ranges"] if q["device_id"] == 0]; busy = sum(q["kernel_busy_union_ns"] for q in r0) / 1e6
    rows.append((ch[0][0], x["iteration"], ch[0][1], x["metrics"].get("decode_requests", 0), busy, per))
for kv0, it, ln, dec, busy, per in sorted(rows):
    print(f"{it:>5} {kv0:>8} {ln:>6} {dec:>4} {busy:8.1f} | " + " ".join(f"{per[g]:14.2f}" for g in G))
# decode: kernel time vs total kv (sum over requests) and max kv, by group
print("\n== decode iterations: GPU busy and group time vs kv (rank 0), binned by sum(kv_lens)/1000 and #requests")
bins = collections.defaultdict(list)
for a, x in enumerate(det):
    if x["stage"] != "decode" or x["iteration"] in EXCL: continue
    kv = x["metrics"].get("decode_kv_lens", []); n = len(kv)
    if n == 0: continue
    S = SL(a); dk = dev_k[S]; ga = g_arr[S]; du = dur[S]
    m0 = (dk == 0)
    r0 = [q for q in x["ranges"] if q["device_id"] == 0]; busy = sum(q["kernel_busy_union_ns"] for q in r0) / 1e6
    key = (n, int(sum(kv) // 100000) * 100)
    bins[key].append((busy, max(kv), sum(kv), {g: du[m0 & (ga == gi[g])].sum() / 1e6 for g in ("attn_core(fmha)", "indexer+sparse_meta", "moe_expert_gemm", "dense_gemm", "mhc", "allreduce")}))
print(f"{'nreq':>4} {'sumkv_k>=':>9} {'n':>4} {'busy_med':>8} {'maxkv_med':>9} | {'fmha':>6} {'indexer':>7} {'moe_gemm':>8} {'dense':>6} {'mhc':>6} {'allred':>6}")
for key in sorted(bins):
    v = bins[key]
    if len(v) < 5: continue
    med = lambda f: np.median([f(z) for z in v])
    print(f"{key[0]:>4} {key[1]:>9} {len(v):>4} {med(lambda z: z[0]):8.2f} {med(lambda z: z[1]):9.0f} | " + " ".join(f"{med(lambda z, g=g: z[3][g]):6.2f}" if g != 'indexer+sparse_meta' else f"{med(lambda z, g=g: z[3][g]):7.2f}" for g in ("attn_core(fmha)", "indexer+sparse_meta", "moe_expert_gemm", "dense_gemm", "mhc", "allreduce")))
# folded sequences
def fam(n):
    n = n.strip(); n = n[5:] if n.startswith("void ") else n
    n = re.sub(r"<.*", "", n); n = re.sub(r"\(.*", "", n); n = re.sub(r"_object_at__.*", "", n)
    n = re.sub(r"^kernel_cutlass_kernel_", "cute:", n); n = re.sub(r"^kernel_cutlass_device_kernel_.*", "cute:device_kernel", n)
    n = re.sub(r"^(bmm_)(MxE4m3|Bfloat16)_MxE2m1.*", r"moe_\1\2", n); n = re.sub(r"^fmhaSm100fKernel_.*", "fmhaSm100f(sparse MLA)", n)
    n = re.sub(r"^nvjet_sm103.*", "nvjet_sm103", n); n = re.sub(r"^deep_gemm::sched::.*", "dg::sched", n)
    return n[:60]
def folded(a, dev=0, phase="forward"):
    rix = [j for j, r in enumerate(det[a]["ranges"]) if r["device_id"] == dev and r["phase"] == phase]
    S = SL(a); m = np.isin(ri[S], rix); o = np.argsort(s[S][m]); seq = [fam(names[int(k)]) for k in nid[S][m][o]]; dd = dur[S][m][o]
    out = []; i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]: j += 1
        out.append((seq[i], j - i, dd[i:j].sum() / 1e3)); i = j
    # fold repeated layer blocks: find the most common pattern of length L repeated
    return out, len(seq), dd.sum() / 1e6
for label, pick in [("deepest 16k prefill chunk", max(rows, key=lambda z: z[0])[1] if rows else None),
                    ("decode step with the largest max kv", max(((max(x["metrics"].get("decode_kv_lens", [0])), x["iteration"]) for x in det if x["stage"] == "decode" and x["iteration"] not in EXCL))[1])]:
    if pick is None: continue
    a = next(i for i, x in enumerate(det) if x["iteration"] == pick)
    out, n, tot = folded(a)
    print(f"\n== folded forward kernel sequence, {label}: iteration {pick} (rank 0): {n} kernels, {tot:.1f} ms; metrics={ {k: (v if not isinstance(v, list) else (str(v)[:60])) for k, v in det[a]['metrics'].items() if k in ('prefill_tokens','prefill_chunk_pairs','decode_requests','decode_kv_lens')} }")
    # print first 70 folded entries and the entries of the last 40
    for t in out[:70]: print(f"   {t[1]:>3}x {t[0]:60s} {t[2]:9.1f} us")
    if len(out) > 110: print("   ...", len(out) - 110, "folded entries omitted ...")
    for t in out[-40:] if len(out) > 110 else out[70:]: print(f"   {t[1]:>3}x {t[0]:60s} {t[2]:9.1f} us")
    print(f"   distinct families in this forward: {len(set(t[0] for t in out))}")
