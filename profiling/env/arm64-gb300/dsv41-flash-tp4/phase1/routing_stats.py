# Phase 1 planning: MoE routing distribution statistics from the frozen token corpus (pass 5), read-only.
# Per request: tokens, per-layer expert-load imbalance (max expert load / mean load over the 384 experts, decode tokens of that request),
# normalized entropy of the expert histogram; global per-layer statistics; candidate held-out requests by routing distribution.
import json, sys, numpy as np
D = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
man = json.load(open(f"{D}/manifest.json")); prov = json.load(open(f"{D}/provenance.json"))
T, L, K, E = man["num_tokens"], man["num_layers"], man["top_k"], man["num_experts"]
r = np.fromfile(f"{D}/{man['data_file']}", dtype="<u2"); assert r.size == T * L * K, (r.size, T * L * K)
r = r.reshape(T, L, K); assert r.max() < E
print(f"corpus: {T} tokens x {L} layers x top-{K}, {E} experts; max expert id {r.max()}")
# global per-layer expert popularity
hist = np.zeros((L, E), np.int64)
for l in range(L):
    hist[l] = np.bincount(r[:, l, :].ravel(), minlength=E)
mean = T * K / E
print(f"\n== per-layer expert load over the whole corpus (mean load {mean:.0f} assignments per expert)")
print(f"{'layer':>5} {'max/mean':>8} {'p99/mean':>8} {'min/mean':>8} {'experts<10%mean':>15} {'top-16 share':>12} {'H/Hmax':>7}")
for l in range(L):
    h = hist[l]; p = h / h.sum(); H = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(E)
    print(f"{l:>5} {h.max()/mean:8.2f} {np.percentile(h,99)/mean:8.2f} {h.min()/mean:8.2f} {(h < 0.1*mean).sum():>15} {np.sort(h)[-16:].sum()/h.sum():12.3f} {H:7.3f}")
# per request
rows = []
for seg in prov["request_segments"]:
    a, n = seg["start"], seg["length"]
    if n == 0: continue
    x = r[a:a + n]
    imb = []; ent = []
    for l in range(L):
        h = np.bincount(x[:, l, :].ravel(), minlength=E); m = n * K / E
        imb.append(h.max() / m); p = h / h.sum(); ent.append(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(min(E, n * K)))
    # per-token "batch" view: with b tokens in a decode step, expected #distinct experts touched (sampled from this request's tokens)
    rows.append(dict(request=seg["request"], tokens=n, imb_mean=float(np.mean(imb)), imb_max=float(np.max(imb)), ent_mean=float(np.mean(ent)),
                     distinct_experts_mean=float(np.mean([len(np.unique(x[:, l, :])) for l in range(L)]))))
rows.sort(key=lambda z: -z["tokens"])
print(f"\n== per request (decode tokens of the request): expert-load imbalance (max/mean over experts, averaged over layers), entropy, distinct experts per layer")
print(f"{'request':>16} {'tokens':>6} {'imb_mean':>8} {'imb_max':>8} {'ent':>6} {'distinct/layer':>14}")
for z in rows: print(f"{z['request']:>16} {z['tokens']:>6} {z['imb_mean']:8.2f} {z['imb_max']:8.2f} {z['ent_mean']:6.3f} {z['distinct_experts_mean']:14.1f}")
# distinct experts touched as a function of batch size (random token batches from the whole corpus): drives MoE GEMM group count
rng = np.random.default_rng(0)
print(f"\n== distinct experts touched per layer vs #tokens in a decode step (random token batches, 200 draws), and max tokens on one expert")
print(f"{'tokens':>6} {'distinct/384 (mean)':>19} {'p90':>6} {'max tokens/expert mean':>22}")
for b in (1, 2, 4, 8, 11, 16, 24, 32, 48, 64, 128, 256, 1024, 4096, 16384):
    dis = []; mx = []
    for _ in range(200 if b <= 256 else 40):
        idx = rng.integers(0, T, b); x = r[idx]
        dd = [len(np.unique(x[:, l, :])) for l in range(0, L, 5)]; dis.append(np.mean(dd))
        mx.append(np.mean([np.bincount(x[:, l, :].ravel(), minlength=E).max() for l in range(0, L, 5)]))
    print(f"{b:>6} {np.mean(dis):19.1f} {np.percentile(dis,90):6.1f} {np.mean(mx):22.2f}")
if OUT: json.dump(dict(per_request=rows, layer_max_over_mean=[float(hist[l].max() / mean) for l in range(L)]), open(OUT, "w"), indent=1)
