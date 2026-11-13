import sys, json, numpy as np
sys.path.insert(0, ".")
from costmodel.data import PassData
from costmodel.partition import load
from costmodel.predict import Predictor
from costmodel.mixed import host_decomposition
T = sys.argv[1]; part = load("partition"); pred = Predictor.load("out/coefficients.json")
print("launch model", pred.mixed.launch); print("c_host_us", np.round(pred.mixed.c_host_ns/1e3,1), "c_api_us", round(pred.mixed.c_api_ns/1e3,1), "tail_ms", round(pred.mixed.tail_ns/1e6,2), "c_kv_us", round(pred.mixed.c_kv_ns/1e3,1), pred.mixed.c_kv_evidence)
print("prefill ranges", pred.prefill.ranges); print("coll", {k: (round(v['a_us'],1), round(v['c_us_per_token'],4), v['n']) for k, v in pred.coll.eager.items()}, pred.coll.counts)
for g, c in pred.prefill.coef.items(): print(f"  {g:12s} rank0 coef (ms): const {c[0][0]/1e6:8.3f} T {c[0][1]/1e6*1000:8.4f}/1k Q {c[0][2]/1e6:8.4f} KVW {c[0][3]/1e6:8.4f}/1k b {c[0][4]/1e6:8.4f}  rmse {np.array(pred.prefill.rmse[g])/1e6}")
for lab in ("quadrant_nsys", "diverse_nsys_run2"):
    pd = PassData.load(lab, f"{T}/{lab}", part)
    print(f"\n== {lab}")
    for f in pd.feats:
        if f.stage != "mixed" or f.excluded or f.mixed_kind == "graph" or (f.is_chunk16k and f.kv0 > 0): continue
        p = pred.predict(f); h = host_decomposition(pd, f.iteration)
        print(f"{f.iteration:6d} {f.role[:4]} {f.mixed_kind:5s} P={f.P:5d} b={f.b:2d} ch={str(f.chunks)[:30]:30s} wall={f.wall_ns/1e6:6.1f} pred={p['wall_ns']/1e6:6.1f} host={p['host_path_ns']/1e6:6.1f} gpu={p['gpu_path_ns']/1e6:6.1f} bound={p['bound']} | meas fwd={np.round(h['fwd_window_ns']/1e6,0)} L={int(h['launches'][0])} predL={p['launches']:.0f} cu={np.round(h['gpu_compute_ns']/1e6,0)} predcu={np.round(p['rank_compute_union_ns']/1e6,0)} comm_pred={p['comm_intrinsic_ns']/1e6:.1f} idle={np.round(p['rank_idle_ns']/1e6,1)} st={p['status']}/{p['prefill_status']}")
