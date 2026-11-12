import json, re, sys, statistics as st, collections
R = sys.argv[1]
d = json.load(open(f"{R}/parsed.json"))
det = d["iteration_details"]
print("== iterations", len(det), "index range", det[0]["iteration"], det[-1]["iteration"])
print("step_alignment", d["step_alignment"])
print("stages", collections.Counter(x["stage"] for x in det))
print("types", collections.Counter(x["iteration_type"] for x in det))
print("phases observed", d["phases"])
# phases per iteration (device 0)
ph_sets = collections.Counter()
rng_per_iter = collections.Counter()
for x in det:
    r0 = [r for r in x["ranges"] if r["device_id"] == 0]
    ph_sets[tuple(sorted({r["phase"] for r in r0}))] += 1
    rng_per_iter[len(x["ranges"])] += 1
print("phase sets per iteration (dev0)", dict(ph_sets))
print("nvtx ranges per iteration (all devices)", dict(rng_per_iter))
print("devices_absent nonempty:", sum(1 for x in det if x["devices_absent"]))
print("worker names:", sorted({r["worker"]["name"] for x in det[:5] for r in x["ranges"]}))
# JIT
jit = [x for x in det if x["jit_module_loads"]]
print("\n== JIT module loads: total", d["jit_module_loads"], "iterations affected", len(jit), "of", len(det))
tot_stall = sum(x["jit_stall_ns"] for x in jit) / 1e6
print("total jit stall ms", round(tot_stall, 2))
def busy(x, dev=0):
    return sum(r["kernel_busy_union_ns"] for r in x["ranges"] if r["device_id"] == dev) / 1e6
def span(x, dev=0):
    r = [r for r in x["ranges"] if r["device_id"] == dev]
    return (max(q["end_ns"] for q in r) - min(q["start_ns"] for q in r)) / 1e6
for x in jit[:40]:
    print(f"  it {x['iteration']:5d} {x['stage']:7s} loads={x['jit_module_loads']:3d} stall_ms={x['jit_stall_ns']/1e6:9.2f} busy_ms(dev0)={busy(x):8.2f} span_ms(dev0)={span(x):8.2f}")
if len(jit) > 40: print("  ...", len(jit) - 40, "more")
# per-stage span stats
print("\n== per-iteration dev0 stats by stage (ms)")
for stg in d["stages"]:
    xs = [x for x in det if x["stage"] == stg]
    sp = sorted(span(x) for x in xs); bz = sorted(busy(x) for x in xs)
    kc = sorted(sum(r["kernel_count"] for r in x["ranges"] if r["device_id"] == 0) for x in xs)
    q = lambda a, p: a[min(len(a)-1, int(p*len(a)))]
    print(f"  {stg:7s} n={len(xs):5d} span med={st.median(sp):.3f} p90={q(sp,.9):.3f} p99={q(sp,.99):.3f} max={sp[-1]:.3f} | busy med={st.median(bz):.3f} | kernels/iter med={st.median(kc):.0f} min={kc[0]} max={kc[-1]}")
dec = [x for x in det if x["stage"] == "decode"]
n = len(dec)
for lbl, sl in (("first 300", dec[:300]), ("middle", dec[n//2-150:n//2+150]), ("last 300", dec[-300:])):
    print(f"  decode {lbl}: span med {st.median(span(x) for x in sl):.3f} busy med {st.median(busy(x) for x in sl):.3f} reqs med {st.median(x['metrics'].get('decode_requests',0) for x in sl)}")
# kernel ranking
print("\n== category ms/iter (dev0, all):", {k: round(v, 3) for k, v in d["all"]["0"]["category_ms_per_iter"].items()})
for stg in d["stages"]:
    s = d["by_stage"][stg]["0"]
    print(f"\n== top kernels {stg} (dev0, n_iters={s['n_iters']}, busy_ms_mean={s['busy_ms_mean']:.3f}, span_ms_mean={s['kernel_span_ms_mean']:.3f})")
    print("   categories:", {k: round(v, 3) for k, v in s["category_ms_per_iter"].items()})
    for k in s["top_kernels"][:25]:
        print(f"   {k['ms_per_iter']:8.4f} ms  {k['calls_per_iter']:7.1f}/it  {k['name'][:120]}")
# kernel name classes
names = list(d["kernel_names"].values())
print("\n== unique kernel names", len(names))
cls = collections.OrderedDict([
 ("ep_dispatch_combine", r"dispatch|combine|all_to_all|alltoall|deep_ep|deepep|a2a"),
 ("allreduce_comm", r"all_reduce|allreduce|cross_device_reduce|ncclDevKernel|symm|oneshot|twoshot|trtllm.*reduce|reduce_scatter|allgather|all_gather"),
 ("moe", r"moe|expert|rout|topk|top_k|gating|Mxfp4|mxfp4"),
 ("sparse_mla_indexer", r"sparse|indexer|mla|Mla|MLA|mqa|logits|lightning|topk_|paged"),
 ("attention_other", r"attention|Attention|flash|Flash|BatchPrefill|BatchDecode|prefill|decode"),
 ("gemm", r"gemm|Gemm|GEMM|cutlass|Cutlass|cublas|Mma|sm100|sm_100|fp8|Fp8|mxfp8|bmm|matmul"),
 ("engram", r"engram|Engram|hash|embedding"),
 ("norm_quant_misc", r"norm|Norm|quant|Quant|rope|Rope|silu|gelu|act_"),
])
seen = set()
for c, pat in cls.items():
    hits = [nm for nm in names if nm not in seen and re.search(pat, nm)]
    seen.update(hits)
    print(f"\n  [{c}] {len(hits)}")
    for nm in hits[:40]:
        print("     ", nm[:150])
rest = [nm for nm in names if nm not in seen]
print(f"\n  [unclassified] {len(rest)}")
for nm in rest[:60]: print("     ", nm[:150])
