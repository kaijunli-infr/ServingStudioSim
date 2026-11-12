import json, re, sys, sqlite3, collections, statistics as st
import pyarrow.parquet as pq, pyarrow.compute as pc, numpy as np
P = sys.argv[1]
d = json.load(open(f"{P}/parsed.json"))
det = d["iteration_details"]; names = {int(k): v for k, v in d["kernel_names"].items()}
# detail_index -> (iteration, stage); range_index -> (device, phase)
it_of = {}; stage_of = {}; dev_of_range = {}; phase_of_range = {}
for di, x in enumerate(det):
    it_of[di] = x["iteration"]; stage_of[di] = x["stage"]
    for ri, r in enumerate(x["ranges"]):
        dev_of_range[(di, ri)] = r["device_id"]; phase_of_range[(di, ri)] = r["phase"]
tab = pq.read_table(f"{P}/parsed.kernels.parquet", columns=["detail_index","range_index","name_id","start_ns","end_ns","correlation_id"])
di = tab["detail_index"].to_numpy(); ri = tab["range_index"].to_numpy(); nid = tab["name_id"].to_numpy()
dur = (tab["end_ns"].to_numpy() - tab["start_ns"].to_numpy()).astype(np.int64)
stage_arr = np.array([stage_of[i] for i in range(len(det))])
st_k = stage_arr[di]
dev_lookup = {}
for (a, b), v in dev_of_range.items(): dev_lookup.setdefault(a, {})[b] = v
dev_k = np.fromiter((dev_lookup[a][b] for a, b in zip(di, ri)), dtype=np.int8, count=len(di))
def family(n):
    n = n.strip()
    if n.startswith("void "): n = n[5:]
    n = re.sub(r"<.*", "", n); n = re.sub(r"\(.*", "", n)
    n = re.sub(r"_object_at__.*", "", n)
    n = re.sub(r"^kernel_cutlass_kernel_", "cute:", n); n = re.sub(r"^kernel_cutlass_device_kernel_.*", "cute:device_kernel(cuBLASLt-like)", n)
    n = re.sub(r"^(bmm_)(MxE4m3|Bfloat16)_MxE2m1MxE4m3.*", r"trtllm_moe_\1\2_MxE2m1 (mxfp4 expert GEMM)", n)
    n = re.sub(r"^fmhaSm100fKernel_.*", "trtllm_fmha Sm100f (sparse MLA)", n)
    n = re.sub(r"^nvjet_sm103.*", "nvjet_sm103 (cuBLAS)", n)
    n = re.sub(r"^deep_gemm::sched::.*", "deep_gemm::sched metadata", n)
    return n[:90]
fam_of_id = {k: family(v) for k, v in names.items()}
fam_ids = sorted(set(fam_of_id.values())); fam_index = {f: i for i, f in enumerate(fam_ids)}
fam_k = np.array([fam_index[fam_of_id[k]] for k in range(max(names)+1) if k in fam_of_id] + [0], dtype=np.int32)
# name ids may not be contiguous; build array sized max+1
fk = np.zeros(max(names)+1, dtype=np.int32)
for k, f in fam_of_id.items(): fk[k] = fam_index[f]
fam_arr = fk[nid]
tot_ms = dur.sum()/1e6
print(f"== kernel rows {len(dur)} total GPU kernel time (sum over 4 devices) {tot_ms:.0f} ms; per device avg {tot_ms/4:.0f} ms; capture window kernel span {d['step_alignment']['steps']} iterations")
dec_mask = st_k == "decode"; mix_mask = ~dec_mask
agg = collections.defaultdict(lambda: [0,0,0,0,0,0])
for f in range(len(fam_ids)):
    m = fam_arr == f
    if not m.any(): continue
    agg[f][0] = dur[m].sum(); agg[f][1] = m.sum()
    md = m & dec_mask; mm = m & mix_mask
    agg[f][2] = dur[md].sum(); agg[f][3] = md.sum(); agg[f][4] = dur[mm].sum(); agg[f][5] = mm.sum()
rank = sorted(agg.items(), key=lambda kv: -kv[1][0])
n_dec = sum(1 for x in det if x["stage"] == "decode"); n_mix = len(det) - n_dec
print(f"\n== TOP 25 kernel families by cumulative GPU time (all 4 devices; per-iteration columns are per device). decode iters={n_dec} mixed/prefill iters={n_mix}")
print(f"{'rank':>4} {'share%':>6} {'total_ms':>9} {'calls':>9} | {'decode_ms':>9} {'ms/dec_it':>9} {'calls/dec_it':>12} | {'mixed_ms':>9} {'ms/mix_it':>9} {'calls/mix_it':>12}  family")
cum = 0
for i, (f, (t, c, td, cd, tm, cm)) in enumerate(rank[:25], 1):
    cum += t
    print(f"{i:>4} {100*t/dur.sum():6.2f} {t/1e6:9.1f} {c:9d} | {td/1e6:9.1f} {td/1e6/n_dec/4:9.4f} {cd/n_dec/4:12.1f} | {tm/1e6:9.1f} {tm/1e6/n_mix/4:9.3f} {cm/n_mix/4:12.1f}  {fam_ids[f]}")
print(f"  top-25 cumulative share {100*cum/dur.sum():.2f}%  ({len(rank)} families total)")
# Collective / MoE / attention per decode iteration counts (device 0)
print("\n== per decode iteration (device 0) counts of key families")
d0 = dec_mask & (dev_k == 0)
for pat in ["twoshotAllreduce", "oneshotAllreduceFusion", "ncclDevKernel_AllReduce", "ncclDevKernel_AllGather", "all_reduce_mhc", "trtllm_moe_bmm_MxE4m3", "trtllm_moe_bmm_Bfloat16", "routing", "finalize", "_dsv4_topk", "trtllm_fmha", "paged_mqa_logits", "sparse_mqa_logits", "sm100_mega_mhc", "_engram_lookup", "_hash_ids", "fused_engram_post", "sm100_fp8_fp4_gemm", "tf32_hc_prenorm"]:
    fs = [fam_index[f] for f in fam_ids if pat in f]
    m = d0 & np.isin(fam_arr, fs)
    print(f"  {pat:28s} {m.sum()/n_dec:8.2f} calls/it  {dur[m].sum()/1e6/n_dec:8.4f} ms/it")
print("\n== same for mixed/prefill iterations (device 0)")
m0 = mix_mask & (dev_k == 0)
for pat in ["twoshotAllreduce", "oneshotAllreduceFusion", "ncclDevKernel_AllReduce", "ncclDevKernel_AllGather", "all_reduce_mhc", "trtllm_moe_bmm_MxE4m3", "trtllm_moe_bmm_Bfloat16", "routing", "trtllm_fmha", "paged_mqa_logits", "sparse_mqa_logits", "sm100_mega_mhc", "sm100_fp8_fp4_gemm", "_engram_lookup"]:
    fs = [fam_index[f] for f in fam_ids if pat in f]
    m = m0 & np.isin(fam_arr, fs)
    print(f"  {pat:28s} {m.sum()/n_mix:8.2f} calls/it  {dur[m].sum()/1e6/n_mix:8.3f} ms/it")
# Ordered sequence of one decode iteration forward (device 0), folded
def folded_sequence(it_index, dev=0, phase="forward", limit=400):
    dix = next(i for i, x in enumerate(det) if x["iteration"] == it_index)
    rix = [j for j, r in enumerate(det[dix]["ranges"]) if r["device_id"] == dev and r["phase"] == phase]
    m = (di == dix) & np.isin(ri, rix)
    order = np.argsort(tab["start_ns"].to_numpy()[m])
    seq = [fam_of_id[int(k)] for k in nid[m][order]]
    out = []; i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]: j += 1
        out.append(f"{seq[i]}" + (f" x{j-i}" if j - i > 1 else "")); i = j
    return len(seq), out[:limit]
for it_index, lbl in ((3000, "decode it 3000"), (det[0]["iteration"], f"first iteration {det[0]['iteration']} ({det[0]['stage']})")):
    n, seq = folded_sequence(it_index)
    print(f"\n== forward kernel sequence, {lbl}, device 0: {n} kernels; first 2 layers approx (folded):")
    # print up to the 3rd occurrence of _dsv4_topk_kernel
    cnt = 0; shown = []
    for s in seq:
        shown.append(s)
        if s.startswith("_dsv4_topk"): cnt += 1
        if cnt >= 3: break
    for s in shown: print("   ", s)
# Per-iteration inter-iteration gaps (device 0 forward ranges) and in-iteration GPU idle
starts = {}; ends = {}
for x in det:
    r0 = [r for r in x["ranges"] if r["device_id"] == 0]
    starts[x["iteration"]] = min(r["start_ns"] for r in r0); ends[x["iteration"]] = max(r["end_ns"] for r in r0)
its = [x["iteration"] for x in det]
gaps = [(its[i], (starts[its[i]] - ends[its[i-1]])/1e6) for i in range(1, len(its))]
big = [g for g in gaps if g[1] > 50]
print(f"\n== inter-iteration host gaps (dev0): median {st.median(g for _, g in gaps):.3f} ms, p99 {sorted(g for _, g in gaps)[int(0.99*len(gaps))]:.3f} ms, >50 ms: {len(big)} -> {big[:20]}")
print("iteration index gaps (missing indices):", [(its[i-1], its[i]) for i in range(1, len(its)) if its[i] != its[i-1] + 1][:10])
# CUDA graph membership via sqlite for sampled iterations
con = sqlite3.connect(f"{P}/nsys/dsv41_flash_1017_gb300_tp4_diverse_nsys_run2.sqlite")
print("\n== CUDA graph membership (sqlite graphId) per stage, device 0, sampled iterations")
def graph_stats(it_list, label):
    tot = ing = 0; gids = collections.Counter(); attn_in = attn_out = 0; per_it_graphs = []
    for it_index in it_list:
        dix = next(i for i, x in enumerate(det) if x["iteration"] == it_index)
        m = (di == dix) & (dev_k == 0)
        s0, e0 = int(tab["start_ns"].to_numpy()[m].min()), int(tab["end_ns"].to_numpy()[m].max())
        rows = con.execute("select graphId, shortName from CUPTI_ACTIVITY_KIND_KERNEL k where deviceId=0 and start>=? and start<=?", (s0, e0)).fetchall()
        g = set()
        for gid, sn in rows:
            tot += 1
            if gid: ing += 1; g.add(gid)
        per_it_graphs.append(len(g))
        sids = {r[0]: r[1] for r in con.execute("select id, value from StringIds where id in (%s)" % ",".join(str(r[1]) for r in rows[:0]))} if False else None
    print(f"  {label}: kernels {tot}, in graphs {ing} ({100*ing/max(tot,1):.1f}%), distinct graphIds per iteration: {sorted(set(per_it_graphs))}")
dec_its = [x["iteration"] for x in det if x["stage"] == "decode"]
graph_stats(dec_its[200:205] + dec_its[2000:2003], "decode (8 sampled)")
graph_stats([x["iteration"] for x in det if x["stage"] != "decode"][:12], "mixed/prefill (first 12)")
graph_stats([x["iteration"] for x in det if x["stage"] != "decode"][12:], "mixed/prefill (rest)")
# attention kernels in/out of graph by stage
print("\n== attention/indexer kernels inside vs outside graphs (device 0, whole capture)")
for pat in ("fmhaSm100f", "mqa_logits", "_dsv4_topk", "twoshotAllreduce", "bmm_", "routing"):
    for stg, (lo, hi) in (("decode", (starts[dec_its[200]], ends[dec_its[2000]])), ("mixed", (starts[det[0]["iteration"]], ends[[x for x in det if x["stage"]!="decode"][-1]["iteration"]]))):
        pass
    r = con.execute("""select sum(k.graphId is not null and k.graphId!=0), count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.deviceId=0 and s.value like ?""", (f"%{pat}%",)).fetchone()
    print(f"  {pat:18s} in-graph {r[0]} / {r[1]}")
# Memcpy H2D / D2H within decode iterations (Engram offload evidence)
print("\n== CUDA memcpy / memset / sync in sampled decode iterations (device 0)")
cols = [r[1] for r in con.execute("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")]
print("  memcpy cols:", cols[:14])
for it_index in dec_its[200:203] + dec_its[2000:2001]:
    s0, e0 = starts[it_index], ends[it_index]
    rows = con.execute("select copyKind, count(*), sum(bytes), sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=0 and start>=? and start<=? group by copyKind", (s0, e0)).fetchall()
    sync = con.execute("select count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where deviceId=0 and start>=? and start<=?", (s0, e0)).fetchone()
    print(f"  it {it_index}: memcpy by copyKind (kind,count,bytes,ns) {rows}; sync events {sync}")
