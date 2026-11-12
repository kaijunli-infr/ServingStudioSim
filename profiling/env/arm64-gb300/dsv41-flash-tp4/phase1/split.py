# Phase 1 planning: deterministic calibration / held-out split over the clean iterations of the frozen nsys passes (metadata only).
# Usage: split.py <out.json> <pass_label>=<profile_dir>=<excl_csv> ...
import json, sys, collections
out = sys.argv[1]; passes = [a.split("=") for a in sys.argv[2:]]
HOLD_NREQ = {"diverse_nsys_run2": {5, 10, 14}, "quadrant_nsys": {26, 30, 36}, "diverse_nsys_salvaged": set()}
HOLD_CHAIN_KV_LO, HOLD_CHAIN_KV_HI = 47000, 52000     # interpolation hold-out band for 16k chunks inside calibration chains
def h(it): return (it * 2654435761) % 100            # deterministic 80/20
res = {}; summary = []
for label, P, excl in passes:
    EX = set(int(x) for x in excl.split(",") if x)
    d = json.load(open(f"{P}/parsed.json")); det = d["iteration_details"]
    cls = collections.defaultdict(lambda: {"calibration": [], "heldout": [], "excluded": []})
    # chains for single 16k chunks
    rows = []
    for x in det:
        ch = x["metrics"].get("prefill_chunk_pairs", [])
        if len(ch) == 1 and ch[0][1] >= 16000: rows.append((x["iteration"], ch[0][0], ch[0][1]))
    rows.sort(); chain = {}; cid = 0; last = None
    for it, kv0, ln in rows:
        if last and kv0 == last[1] + last[2] and it - last[0] <= 3: chain[it] = cid
        else: cid += 1; chain[it] = cid
        last = (it, kv0, ln)
    # hold out whole chains: every 3rd chain (by id) that has >= 5 chunks, plus the deepest chain's duplicates
    chain_len = collections.Counter(chain.values()); hold_chain = {c for c in chain_len if chain_len[c] >= 5 and c % 3 == 0}
    for x in det:
        it = x["iteration"]; m = x["metrics"]; bad = it in EX or x["jit_module_loads"] > 0
        if x["stage"] == "decode":
            n = m.get("decode_requests", 0); kv = m.get("decode_kv_lens", []); mk = max(kv) if kv else 0
            kvb = "kv<32k" if mk < 32768 else ("kv32-96k" if mk < 98304 else "kv>=96k")
            c = f"decode nreq={n} {kvb}"
            held = n in HOLD_NREQ.get(label, set()) or (mk >= 120000 and h(it) < 50) or h(it) < 20
        else:
            ch = m.get("prefill_chunk_pairs", []); pt = m.get("prefill_tokens", 0)
            if it in chain:
                kv0 = ch[0][0]; c = f"prefill 16k chunk kv0={round(kv0/16384)*16}k chain={chain[it]}"
                held = chain[it] in hold_chain or (HOLD_CHAIN_KV_LO <= kv0 < HOLD_CHAIN_KV_HI)
            else:
                c = "mixed >=10k tok multi-chunk" if pt >= 10000 else ("mixed 1k-10k tok" if pt >= 1000 else "mixed <1k tok")
                held = h(it) < 20
        cls[c]["excluded" if bad else ("heldout" if held else "calibration")].append(it)
    res[label] = {"profile_dir": P, "exclusions": sorted(EX), "held_out_batch_sizes": sorted(HOLD_NREQ.get(label, set())), "held_out_chains": sorted(hold_chain),
                  "held_out_kv0_band": [HOLD_CHAIN_KV_LO, HOLD_CHAIN_KV_HI], "classes": {k: v for k, v in sorted(cls.items())}}
    for k, v in sorted(cls.items()): summary.append((label, k, len(v["calibration"]), len(v["heldout"]), len(v["excluded"])))
json.dump(res, open(out, "w"))
print(f"{'pass':>22} {'class':44s} {'calib':>6} {'held':>5} {'excl':>4}")
tot = collections.Counter()
for lab, k, a, b, c in summary:
    print(f"{lab:>22} {k:44s} {a:>6} {b:>5} {c:>4}"); tot[lab] += a; tot[lab + "_h"] += b
print({k: v for k, v in tot.items()})
