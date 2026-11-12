# Decode kernel path vs batch size: per decode iteration (rank 0), counts of path-discriminating families, grouped by decode_requests.
import json, sys, collections, numpy as np, pyarrow.parquet as pq
PATS = [("twoshot", "twoshotAllreduce"), ("oneshot", "oneshotAllreduceFusion"), ("ar_mhc", "all_reduce_mhc"), ("nccl_ar", "ncclDevKernel_AllReduce"),
        ("mega_mhc", "sm100_mega_mhc"), ("mhc_pre_tl", "mhc_pre_big_fuse_with_norm_tilelang"), ("hc_prenorm", "tf32_hc_prenorm"),
        ("dense_persist", "Sm100BlockScaledPersistentDenseGemm"), ("dense_splitk", "Sm100BlockScaledSplitKGemm"), ("ll_bf16_splitk", "cute_dsl_ll_bf16_splitk"),
        ("nvjet", "nvjet_sm103"), ("wo_a_cute", "kernel_cutlass_device_kernel"), ("fp8_fp4_gemm", "sm100_fp8_fp4_gemm"),
        ("route_cluster", "routingIndicesClusterKernel"), ("route_dyn", "routingIndicesDynBlockKernel"), ("finalize", "finalizeKernel"), ("fmha", "fmhaSm100f"), ("mqa_logits", "mqa_logits")]
for P in sys.argv[1:]:
    d = json.load(open(f"{P}/parsed.json")); det = d["iteration_details"]; names = {int(k): v for k, v in d["kernel_names"].items()}
    tab = pq.read_table(f"{P}/parsed.kernels.parquet", columns=["detail_index", "range_index", "name_id", "start_ns", "end_ns"])
    di = tab["detail_index"].to_numpy(); ri = tab["range_index"].to_numpy(); nid = tab["name_id"].to_numpy(); dur = (tab["end_ns"].to_numpy() - tab["start_ns"].to_numpy()).astype(np.int64)
    rd = np.zeros(len(det) * 64, dtype=np.int8) - 1
    for a, x in enumerate(det):
        for b, r in enumerate(x["ranges"]): rd[a * 64 + b] = r["device_id"]
    dev0 = rd[di.astype(np.int64) * 64 + ri] == 0
    ids = {lab: np.array([k for k, v in names.items() if pat in v]) for lab, pat in PATS}
    nreq_of = np.array([x["metrics"].get("decode_requests", 0) if x["stage"] == "decode" else -1 for x in det])
    nreq_k = nreq_of[di]
    print(f"\n== {P.split('/')[-2]}: decode iterations (rank 0) — median calls per iteration by decode_requests; busy = median GPU busy ms (rank 0)")
    busy_by = collections.defaultdict(list)
    for a, x in enumerate(det):
        if x["stage"] == "decode": busy_by[x["metrics"].get("decode_requests", 0)].append(sum(r["kernel_busy_union_ns"] for r in x["ranges"] if r["device_id"] == 0) / 1e6)
    print(f"{'nreq':>4} {'iters':>5} {'busy':>6} | " + " ".join(f"{lab:>13}" for lab, _ in PATS))
    for n in sorted(busy_by):
        if len(busy_by[n]) < 3: continue
        sel = dev0 & (nreq_k == n)
        cnt = {lab: np.isin(nid[sel], ids[lab]).sum() / len(busy_by[n]) for lab, _ in PATS}
        print(f"{n:>4} {len(busy_by[n]):>5} {np.median(busy_by[n]):6.2f} | " + " ".join(f"{cnt[lab]:13.1f}" for lab, _ in PATS))
