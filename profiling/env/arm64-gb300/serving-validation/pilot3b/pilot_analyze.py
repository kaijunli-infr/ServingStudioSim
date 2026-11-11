#!/usr/bin/env python3
"""CPU-only analysis of one pilot run against the TP1 serving trace.

    pilot_analyze.py RUN_DIR PREP_DIR OUT_JSON

Inputs: RUN_DIR/timing/timing.json, RUN_DIR/identity/identity.sqlite (Nsight,
NVTX ranges `<case>:eager` / `<case>:graph`), RUN_DIR/pilot_triton.json (from
triton_sources.py over RUN_DIR/cache/inductor), PREP_DIR/trace_reference.json
and PREP_DIR/served_triton.json.

Per case: the launch sequence of one eager call and one graph replay (name,
grid, block, registers, dynamic shared memory, cluster) beside the trace role's
launch configurations; whether each matches by name / family / geometry; the
Inductor source match; and the timing table (cold eager, warm eager, graph
replay, the trace's per-kernel duration and exclusive time, and the trace's
per-operation union / exclusive time).
"""
import difflib, json, re, sqlite3, sys
from collections import defaultdict
from pathlib import Path

run, prep, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
timing = json.loads((run / "timing/timing.json").read_text())
trace = json.loads((prep / "trace_reference.json").read_text())["classes"]
served = json.loads((prep / "served_triton.json").read_text())
pilot_src = json.loads((run / "pilot_triton.json").read_text()) if (run / "pilot_triton.json").exists() else []

# Served Inductor kernel per trace role (variants told apart by their outputs, see the report).
SERVED = {
    "embed_input_norm": ("triton_red_fused__to_copy_embedding_rms_norm_0", None),
    "post_norm": ("triton_red_fused_fused_add_rms_norm_0", None),
    "input_norm": ("triton_red_fused_fused_add_rms_norm_2", "22d580403e0b0e8b"),
    "final_norm": ("triton_red_fused_fused_add_rms_norm_2", "92ea4242313e6119"),
    "activation": ("triton_poi_fused_mul_silu_slice_1", None),
    "rope": ("triton_poi_fused_3", None),
}


def family(name):
    return re.sub(r"_\d+$", "", name.split("(")[0].split("<")[0].strip())


def norm_body(e):
    return e["body"].replace(e["name"], "KERNEL")


def identity():
    path = run / "identity/identity.sqlite"
    if not path.exists():
        return {}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    strings = dict(con.execute("select id, value from StringIds"))
    cols = [r[1] for r in con.execute("pragma table_info(NVTX_EVENTS)")]
    text_expr = "coalesce(text, (select value from StringIds where id = textId))" if "textId" in cols else "text"
    ranges = con.execute(f"select start, end, globalTid, {text_expr} from NVTX_EVENTS "
                         "where end is not null").fetchall()
    rt = con.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    kern = defaultdict(list)
    for row in con.execute("select correlationId, start, end, demangledName, gridX, gridY, gridZ, blockX, "
                           "blockY, blockZ, registersPerThread, dynamicSharedMemory, clusterX, clusterY, "
                           "clusterZ, graphId from CUPTI_ACTIVITY_KIND_KERNEL"):
        kern[row[0]].append(row)
    res = {}
    for s, e, tid, text in ranges:
        if not text or ":" not in text:
            continue
        corr = {c for st_, t, c in rt if t == tid and s <= st_ <= e}
        ks = sorted((k for c in corr for k in kern.get(c, [])), key=lambda k: k[1])
        res[text] = [{"name": strings.get(k[3], str(k[3])), "grid": list(k[4:7]), "block": list(k[7:10]),
                      "regs": k[10], "smem_dynamic": k[11], "cluster": list(k[12:15]),
                      "in_graph": k[15] not in (None, 0), "dur_us": (k[2] - k[1]) / 1e3} for k in ks]
    return res


def split3(ks):
    n = len(ks) // 3 if len(ks) % 3 == 0 and ks else None
    return [ks[i * n:(i + 1) * n] for i in range(3)] if n else [ks]


def trace_configs(cls, role):
    r = trace.get(cls, {}).get("roles", {}).get(role)
    if not r:
        return []
    return [dict(json.loads(k), count=v) for k, v in r["launch_configs"].items()]


def compare(launch, configs):
    out = []
    for k in launch:
        best = None
        for c in configs:
            score = {
                "name_exact": k["name"][:160] == c["name"][:160] or k["name"].startswith(c["name"][:120]),
                "family": family(k["name"]) == family(c["name"]),
                "grid": k["grid"] == c["grid"], "block": k["block"] == c["block"],
                "regs": k["regs"] == c["regs"], "smem_dynamic": k["smem_dynamic"] == c["smem_dynamic"],
                "cluster": k["cluster"] == c["cluster"],
            }
            if best is None or sum(score.values()) > sum(best[0].values()):
                best = (score, c)
        out.append({"pilot": k, "trace_best": best[1] if best else None, "match": best[0] if best else None})
    return out


def source_match(role, pilot_names):
    if role not in SERVED:
        return None
    sname, sbody = SERVED[role]
    cands = [e for e in served if e["name"] == sname and (sbody is None or e["body_sha"] == sbody)]
    if not cands:
        return {"served": sname, "error": "served source not found"}
    s = cands[0]
    target = norm_body(s)
    pool = [e for e in pilot_src if e["name"] in pilot_names] or pilot_src
    best = max(pool, key=lambda e: difflib.SequenceMatcher(None, norm_body(e), target).ratio(), default=None)
    if best is None:
        return {"served": sname, "error": "no pilot source"}
    ratio = difflib.SequenceMatcher(None, norm_body(best), target).ratio()
    diff = list(difflib.unified_diff(target.splitlines(), norm_body(best).splitlines(), "served", "pilot",
                                     lineterm="", n=0))
    meta_diff = list(difflib.unified_diff(s["meta"].replace(sname, "K").splitlines(),
                                          best["meta"].replace(best["name"], "K").splitlines(),
                                          "served", "pilot", lineterm="", n=0))
    return {"served": sname, "served_body_sha": s["body_sha"], "pilot": best["name"],
            "pilot_body_sha": best["body_sha"], "body_identical": ratio == 1.0, "similarity": round(ratio, 4),
            "body_diff": diff[:80], "meta_diff": [l for l in meta_diff if "backend_hash" not in l][:40]}


def us(x):
    return None if x is None else round(x * 1e3, 3)


ident = identity()
report = {"run": str(run), "status": timing.get("status"), "elapsed_s": timing.get("elapsed_s"),
          "peak_reserved_bytes": timing.get("peak_reserved_bytes"), "env": timing.get("env"), "cases": {}}
for name, c in timing["cases"].items():
    t = c.get("trace", {})
    tokens = c.get("meta", {}).get("tokens")
    cls = t.get("class") or ("decode8" if tokens == 8 else "prefill256" if tokens == 256 else None)
    role = t.get("role")
    roles = [role] + (["attention_aux_fill"] if role == "attention" else [])
    configs = [x for r in roles for x in trace_configs(cls, r)]
    eager = split3(ident.get(f"{name}:eager", []))
    graph = split3(ident.get(f"{name}:graph", []))
    entry = {"part": c.get("part"), "status": c.get("status"), "error": c.get("error"), "trace_class": cls,
             "trace_role": role, "meta": c.get("meta"), "oracle": c.get("oracle"),
             "oracle_after_first_replay": c.get("oracle_after_first_replay"),
             "oracle_after_timed_replays": c.get("oracle_after_timed_replays"),
             "eager_launch_consistent": len({json.dumps([(k["name"], k["grid"], k["block"]) for k in l]) for l in eager}) == 1 if eager and eager[0] else None,
             "eager_launch": compare(eager[0], configs) if eager and eager[0] else [],
             "graph_launch": compare(graph[0], configs) if graph and graph[0] else []}
    names = {k["name"] for l in eager for k in l}
    entry["inductor_source"] = source_match(role, names) if role in SERVED else None
    tm = {}
    for mode in ("cold_eager", "warm_eager", "graph_warm"):
        b = c.get(mode, {}) or {}
        tm[mode] = {"timer_us": us(c.get(f"{mode}_ms")),
                    "per_kernel_median_us": {k[:70]: round(v["median"], 3) for k, v in (b.get("per_kernel_us") or {}).items()},
                    "per_kernel_cv": {k[:70]: round(v["cv"], 4) for k, v in (b.get("per_kernel_us") or {}).items() if v["cv"] is not None},
                    "op_sum_median_us": (b.get("launch_sum_us") or {}).get("median"),
                    "op_union_median_us": (b.get("launch_union_us") or {}).get("median"),
                    "op_span_median_us": (b.get("launch_span_us") or {}).get("median"),
                    "op_union_cv": (b.get("launch_union_us") or {}).get("cv"),
                    "sequences": b.get("sequences"), "launches": b.get("launches"),
                    "irregular": b.get("irregular_record_count")}
    tm["graph_event_us_per_replay"] = us(c.get("graph_event_ms_per_replay"))
    tm["graph_cupti_error"] = c.get("graph_cupti_error")
    tr = trace.get(cls, {}).get("roles", {})
    tm["trace"] = {r: {"per_kernel_median_us": {k[:70]: round(v["duration_us"]["median"], 3) for k, v in tr[r]["by_name"].items()},
                       "per_kernel_exclusive_median_us": {k[:70]: round(v["exclusive_us"]["median"], 3) for k, v in tr[r]["by_name"].items()},
                       "op_union_median_us": round(tr[r]["operation"]["union_us"]["median"], 3),
                       "op_exclusive_median_us": round(tr[r]["operation"]["exclusive_us"]["median"], 3),
                       "op_span_median_us": round(tr[r]["operation"]["span_us"]["median"], 3),
                       "calls": tr[r]["operation"]["union_us"]["n"]}
                   for r in roles if r in tr}
    entry["timing"] = tm
    report["cases"][name] = entry
out.write_text(json.dumps(report, indent=1, default=str) + "\n")

# Short console summary
for name, e in report["cases"].items():
    el = e["eager_launch"]
    geo = "; ".join(f"{k['pilot']['name'][:48]} g{k['pilot']['grid']} b{k['pilot']['block']} r{k['pilot']['regs']} "
                    f"[{'='.join(x for x, v in (k['match'] or {}).items() if v)}]" for k in el)
    tm = e["timing"]
    tr = "; ".join(f"{r}: union {v['op_union_median_us']} excl {v['op_exclusive_median_us']}" for r, v in tm["trace"].items())
    src = e["inductor_source"]
    print(f"{name:28s} {e['status']:5s} cold {tm['cold_eager']['timer_us']} warm {tm['warm_eager']['timer_us']} "
          f"graph {tm['graph_warm']['timer_us']} | trace {tr}")
    print(f"   eager: {geo}")
    if src:
        print(f"   source: served {src.get('served')} vs pilot {src.get('pilot')} identical={src.get('body_identical')} sim={src.get('similarity')}")
