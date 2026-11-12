import json, sys, collections, re
R = sys.argv[1]
rows = [json.loads(l) for l in open(f"{R}/profile/vllm/dsv41_flash_1017_gb300_tp4_diverse_nsys_metrics.jsonl")]
print("iterations", len(rows), "index", rows[0]["iteration_index"], rows[-1]["iteration_index"])
t0 = rows[0]["observed_start_monotonic_ns"]
b = collections.defaultdict(list)
for r in rows: b[int((r["observed_start_monotonic_ns"] - t0) / 60e9)].append(r)
print(f"{'min':>3} {'iters':>5} {'med_ms':>7} {'p90_ms':>7} {'max_ms':>8} {'sum_s':>6} {'mixed':>5} {'pre_tok_sum':>11} {'max_kv':>7} {'dec_req_med':>11} {'iters>1s':>8}")
import statistics as st
for k in sorted(b):
    v = b[k]; el = [r["observed_elapsed_ms"] for r in v]
    mixed = [r for r in v if r["prefill_tokens"] > 0]
    mk = max([max(r["decode_kv_lens"]) if r["decode_kv_lens"] else 0 for r in v] + [max((p[0] + p[1]) for r in v for p in r["prefill_chunk_pairs"]) if any(r["prefill_chunk_pairs"] for r in v) else 0])
    print(f"{k:>3} {len(v):>5} {st.median(el):7.1f} {sorted(el)[int(0.9*len(el))-1]:7.1f} {max(el):8.1f} {sum(el)/1e3:6.1f} {len(mixed):>5} {sum(r['prefill_tokens'] for r in v):>11} {mk:>7} {st.median([r['decode_requests'] for r in v]):11.0f} {sum(1 for e in el if e > 1000):>8}")
slow = sorted(rows, key=lambda r: -r["observed_elapsed_ms"])[:12]
print("\nslowest iterations: (idx, t_s, elapsed_ms, prefill_tokens, chunks, decode_reqs, max_kv)")
for r in slow:
    print(" ", r["iteration_index"], round((r["observed_start_monotonic_ns"] - t0) / 1e9, 1), round(r["observed_elapsed_ms"], 1), r["prefill_tokens"], str(r["prefill_chunk_pairs"])[:60], r["decode_requests"], max(r["decode_kv_lens"]) if r["decode_kv_lens"] else 0)
# mixed iteration elapsed vs chunk size, binned by max kv (context position)
print("\nmixed iterations: elapsed by prefill-token and kv-position bins (median ms, n)")
bins = collections.defaultdict(list)
for r in rows:
    if r["prefill_tokens"] == 0: continue
    kvpos = max(p[0] + p[1] for p in r["prefill_chunk_pairs"])
    bins[(min(r["prefill_tokens"] // 4096 * 4096, 16384), kvpos // 32768 * 32768)].append(r["observed_elapsed_ms"])
for k in sorted(bins): print(f"  pre_tok>={k[0]:>5} kvpos>={k[1]:>6}: med {st.median(bins[k]):8.1f} ms  max {max(bins[k]):8.1f}  n={len(bins[k])}")
rep = [json.loads(l) for l in open(f"{R}/profile/load_generator/replay.jsonl")]
print("\nrequests: completed vs truncated")
unf = []; fin = []
for x in rep:
    s = x["source"]["data"]; o = x["outcome"]
    (fin if o["output_len_actual"] == s["output_len_target"] else unf).append((s["id"], s["input_len"], s["output_len_target"], o["output_len_actual"], o.get("finish_reason"), round(s["arrival_time_ms"] / 1e3, 1), o.get("first_token_id_ms")))
print(" complete:", len(fin), " incomplete:", len(unf))
print(" incomplete (id, in, out_target, out_actual, finish, arrival_s, ttft_ms):")
for u in sorted(unf, key=lambda z: z[5]): print("  ", u)
print(" completed with TTFT > 20 s:", [(f[0], f[1], f[5], round(f[6] / 1e3, 1)) for f in fin if f[6] and f[6] > 20000])
