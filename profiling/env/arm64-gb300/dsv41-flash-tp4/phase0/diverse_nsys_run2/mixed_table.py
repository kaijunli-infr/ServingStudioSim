import json, sys
P = sys.argv[1]
d = json.load(open(f"{P}/parsed.json")); det = d["iteration_details"]
print(f"{'it':>5} {'stage':>7} {'pre_tok':>7} {'dec_req':>7} {'stall_s':>7} {'loads':>5} {'host_span_s':>11} {'busy_ms_r0':>10}  chunks[(kv_start,len)]")
for x in det:
    if x["stage"] == "decode": continue
    r0 = [r for r in x["ranges"] if r["device_id"] == 0]
    span = (max(r["end_ns"] for r in r0) - min(r["start_ns"] for r in r0)) / 1e9
    busy = sum(r["kernel_busy_union_ns"] for r in r0) / 1e6
    ch = x["metrics"].get("prefill_chunk_pairs")
    print(f"{x['iteration']:>5} {x['stage']:>7} {x['metrics'].get('prefill_tokens',0):>7} {x['metrics'].get('decode_requests',0):>7} {x['jit_stall_ns']/1e9:7.2f} {x['jit_module_loads']:>5} {span:11.3f} {busy:10.1f}  {str(ch)[:100]}")
prev = None; big = []
for x in det:
    r0 = [r for r in x["ranges"] if r["device_id"] == 0]; h0 = min(r["start_ns"] for r in r0); h1 = max(r["end_ns"] for r in r0)
    if prev and (h0 - prev[1]) / 1e6 > 50: big.append((prev[0], x["iteration"], round((h0 - prev[1]) / 1e6, 1)))
    prev = (x["iteration"], h1)
print("host gaps >50 ms (prev_it, it, ms):", big)
dec = [x for x in det if x["stage"] == "decode"]
import collections
print("decode_requests histogram:", sorted(collections.Counter(x["metrics"].get("decode_requests") for x in dec).items(), key=lambda kv: -kv[1])[:6])
kv = [k for x in dec for k in x["metrics"].get("decode_kv_lens", [])]
print("decode kv_len min/max:", min(kv), max(kv))
