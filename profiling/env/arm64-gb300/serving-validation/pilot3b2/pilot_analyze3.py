#!/usr/bin/env python3
"""CPU-only analysis of the P5-P10 continuation run against the TP1 serving trace.

    pilot_analyze3.py RUN_DIR TRACE_REFERENCE2_JSON OUT_JSON

Reads the run's per-pass summaries (timing/timing.json, control{2048,8192}/control.json,
identity/identity.json), the Nsight export identity/identity.sqlite, the control
passes' own Inductor and Triton caches, and trace_reference2.json. Writes:

- ``timing``: per case the oracles, cold / warm / graph medians (Timer.cupti) and
  burst statistics, against the trace role's per-kernel median (P7, P9);
- ``p8``: P16 vs P32 per shape (page-size effect at the runner's split policy)
  and the batch-1 split cases (split effect at P16);
- ``p9_graph``: per kind and size the three independent graph-backend calls
  (times, batch medians, kernel sequences) and the between-call / within-call spread;
- ``p10``: per traced attention shape, like-for-like operation quantities:
  cold-eager fill, FMHA, union; graph-replay fill, FMHA, union, span, gap,
  overlap; the trace's fill, FMHA, union, span, gap, overlap;
- ``controls``: per tuning size the Inductor sources (size hints, loads, stores,
  body hash), autotune decisions (threads), compiled Triton kernels and compile
  wall time, and whether the two sizes share any source or kernel;
- ``identity``: per identity case the eager and graph launch sequence (name,
  grid, block, registers, shared memory) from Nsight, beside the trace role's
  launch configurations.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

run, trace_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
trace = json.loads(trace_path.read_text())["classes"]


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


timing = load(run / "timing/timing.json")
controls = {t: load(run / f"control{t}/control.json") for t in (2048, 8192)}
identity_doc = load(run / "identity/identity.json")


def us(ms):
    return None if ms is None else round(ms * 1e3, 3)


def stat(b, key, field="median"):
    v = (b or {}).get(key)
    return round(v[field], 3) if v and v.get(field) is not None else None


def per_kernel(b, field="median"):
    return {
        k[:70]: round(v[field], 3)
        for k, v in ((b or {}).get("per_kernel_us") or {}).items()
        if v.get(field) is not None
    }


def split_fill(pk: dict) -> tuple:
    """(fill, fmha) medians from a per-kernel dict: the fill is the non-FMHA kernel."""
    fmha = [v for k, v in pk.items() if k.startswith("fmha")]
    fill = [v for k, v in pk.items() if not k.startswith("fmha")]
    return (fill[0] if fill else None), (fmha[0] if fmha else None)


def trace_role(cls, role):
    r = trace.get(cls, {}).get("roles", {}).get(role)
    if not r:
        return None
    return {
        "kernel_median_us": round(r["duration_us"]["median"], 3),
        "kernel_p10_p90_us": [round(r["duration_us"]["p10"], 3), round(r["duration_us"]["p90"], 3)],
        "calls": r["duration_us"]["n"],
        "launch_configs": [dict(json.loads(k), count=n) for k, n in r["launch_configs"].items()][
            :3
        ],
    }


def methods(c):
    return {
        "timer_us": {m: us(c.get(f"{m}_ms")) for m in ("cold_eager", "warm_eager", "graph_warm")},
        "per_kernel_median_us": {
            m: per_kernel(c.get(m)) for m in ("cold_eager", "warm_eager", "graph_warm")
        },
        "union_median_us": {
            m: stat(c.get(m), "launch_union_us") for m in ("cold_eager", "warm_eager", "graph_warm")
        },
        "union_cv": {
            m: stat(c.get(m), "launch_union_us", "cv")
            for m in ("cold_eager", "warm_eager", "graph_warm")
        },
        "span_median_us": {
            m: stat(c.get(m), "launch_span_us") for m in ("cold_eager", "warm_eager", "graph_warm")
        },
        "gap_median_us": {
            m: stat(c.get(m), "launch_gap_us") for m in ("cold_eager", "warm_eager", "graph_warm")
        },
        "overlap_median_us": {
            m: stat(c.get(m), "launch_overlap_us")
            for m in ("cold_eager", "warm_eager", "graph_warm")
        },
        "sequences": {m: (c.get(m) or {}).get("sequences") for m in ("cold_eager", "graph_warm")},
    }


report: dict = {"run": str(run)}
cases = (timing or {}).get("cases", {})

# ---- timing (P7, P9 method comparison; P8 kept for the P8/P10 sections)
report["timing"] = {}
for name, c in cases.items():
    t = c.get("trace") or {}
    report["timing"][name] = {
        "oracle": c.get("oracle"),
        "oracle_after_first_replay": c.get("oracle_after_first_replay"),
        "oracle_after_timed_replays": c.get("oracle_after_timed_replays"),
        "meta": c.get("meta"),
        "build_s": c.get("build_s"),
        **methods(c),
        "trace": trace_role(t.get("class"), t.get("role")) if t.get("class") else None,
        "flashinfer_rebuild_check": c.get("flashinfer_rebuild_check"),
    }

# ---- P8: page size (P16 vs P32 at the runner's split) and split policy (P16)
p8 = {"pairs": {}, "split": {}}
for name in cases:
    if name.startswith("p8_") and name.endswith("_P16"):
        short = name[3:-4]
        a, b = cases[name], cases.get(f"p8_{short}_P32")
        if not b:
            continue
        row = {}
        for label, c in (("P16", a), ("P32", b)):
            fill_c, fmha_c = split_fill(per_kernel(c.get("cold_eager")))
            fill_g, fmha_g = split_fill(per_kernel(c.get("graph_warm")))
            row[label] = {
                "cold_timer_us": us(c.get("cold_eager_ms")),
                "graph_timer_us": us(c.get("graph_warm_ms")),
                "cold_fill_us": fill_c,
                "cold_fmha_us": fmha_c,
                "cold_union_us": stat(c.get("cold_eager"), "launch_union_us"),
                "graph_fmha_us": fmha_g,
                "fmha_names": [k for k in per_kernel(c.get("cold_eager")) if k.startswith("fmha")],
                "max_seq_len": c["meta"].get("max_seq_len"),
            }
        row["p16_vs_p32_out_max_abs"] = a["meta"].get("p16_vs_p32_out_max_abs")
        if row["P16"]["cold_fmha_us"] and row["P32"]["cold_fmha_us"]:
            row["fmha_cold_p16_over_p32"] = round(
                row["P16"]["cold_fmha_us"] / row["P32"]["cold_fmha_us"] - 1, 4
            )
            row["union_cold_p16_over_p32"] = round(
                row["P16"]["cold_union_us"] / row["P32"]["cold_union_us"] - 1, 4
            )
        p8["pairs"][short] = row
    elif name.startswith("p8_dec_b1_") and "_msl" in name:
        c = cases[name]
        fill_c, fmha_c = split_fill(per_kernel(c.get("cold_eager")))
        p8["split"][name[3:]] = {
            "max_seq_len": c["meta"]["max_seq_len"],
            "cold_fmha_us": fmha_c,
            "cold_fill_us": fill_c,
            "cold_union_us": stat(c.get("cold_eager"), "launch_union_us"),
            "graph_fmha_us": split_fill(per_kernel(c.get("graph_warm")))[1],
        }
report["p8"] = p8

# ---- P9: graph backend stability
p9 = {}
for name, c in cases.items():
    if name.startswith("p9_") and c.get("extra"):
        e = c["extra"]
        p9[name] = {
            "times_us": [round(x["time_ms"] * 1e3, 3) for x in e["calls"]],
            "batch_us": [[round(b * 1e3, 3) for b in x["batch_ms"]] for x in e["calls"]],
            "between_call_range_rel": round(e["between_call_range_rel"], 4),
            "within_call_batch_range_rel_max": round(e["within_call_batch_range_rel_max"], 4),
            "kernel_sequences": e["kernel_sequences"],
            "wall_s": [round(x["wall_s"], 2) for x in e["calls"]],
        }
report["p9_graph"] = p9

# ---- P10: attention operation accounting at the traced shapes (P16 cases)
p10 = {}
for name, c in cases.items():
    t = c.get("trace") or {}
    if not name.startswith("p8_") or not t.get("class") or name.endswith("_P32"):
        continue
    kind = t.get("attention_kind")
    tr = trace.get(t["class"], {}).get("attention_ops", {}).get(kind)
    fill_c, fmha_c = split_fill(per_kernel(c.get("cold_eager")))
    fill_g, fmha_g = split_fill(per_kernel(c.get("graph_warm")))
    p10[name] = {
        "trace_class": t["class"],
        "kind": kind,
        "cold": {
            "fill_us": fill_c,
            "fmha_us": fmha_c,
            "union_us": stat(c.get("cold_eager"), "launch_union_us"),
            "span_us": stat(c.get("cold_eager"), "launch_span_us"),
            "gap_us": stat(c.get("cold_eager"), "launch_gap_us"),
            "overlap_us": stat(c.get("cold_eager"), "launch_overlap_us"),
            "timer_us": us(c.get("cold_eager_ms")),
        },
        "graph": {
            "fill_us": fill_g,
            "fmha_us": fmha_g,
            "union_us": stat(c.get("graph_warm"), "launch_union_us"),
            "span_us": stat(c.get("graph_warm"), "launch_span_us"),
            "gap_us": stat(c.get("graph_warm"), "launch_gap_us"),
            "overlap_us": stat(c.get("graph_warm"), "launch_overlap_us"),
        },
        "trace": None
        if not tr
        else {
            k: round(tr[k]["median"], 3)
            for k in (
                "fill_us",
                "fmha_us",
                "union_us",
                "span_us",
                "gap_us",
                "overlap_us",
                "exclusive_us",
            )
        },
        "trace_pairs": tr and tr["pairs"],
    }
report["p10"] = p10


# ---- controls
def cache_inventory(inductor: Path, triton: Path) -> dict:
    kernels = {}
    for f in glob.glob(str(triton / "*/*.json")):
        if os.path.basename(f).startswith("__grp__"):
            continue
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if "num_warps" in d and d.get("name", "").startswith("triton_"):
            kernels[Path(f).parent.name] = {
                "name": d["name"],
                "threads": 32 * d["num_warps"],
                "smem": d.get("shared"),
            }
    sources = []
    for f in glob.glob(str(inductor / "**/*.py"), recursive=True):
        s = Path(f).read_text(errors="replace")
        for m in re.finditer(r"async_compile\.triton\('(triton_\w+)', '''(.*?)'''", s, re.S):
            body = m.group(2)
            hints = re.search(r"size_hints=(\{[^}]*\})", body)
            norm = re.sub(r"triton_\w+_\d+", "K", body)
            sources.append(
                {
                    "name": m.group(1),
                    "size_hints": hints.group(1) if hints else None,
                    "in_ptr_loads": len(re.findall(r"tl\.load\(in_ptr", body)),
                    "stores": len(re.findall(r"tl\.store\(", body)),
                    "body_sha": hashlib.sha256(norm.encode()).hexdigest()[:16],
                }
            )
    best = []
    for f in glob.glob(str(inductor / "**/*.best_config"), recursive=True):
        v = json.loads(Path(f).read_text())
        k = kernels.get(v.get("triton_cache_hash"), {})
        best.append(
            {
                "R0_BLOCK": v.get("R0_BLOCK"),
                "num_warps": v.get("num_warps"),
                "threads": 32 * v.get("num_warps", 0),
                "selected_kernel": k.get("name"),
                "triton_hash": v.get("triton_cache_hash", "")[:16],
            }
        )
    return {"sources": sources, "autotune": best, "triton_kernels": kernels}


report["controls"] = {}
for t, doc in controls.items():
    inv = cache_inventory(run / f"cache/inductor-tune{t}", run / f"cache/triton-tune{t}")
    report["controls"][t] = {
        "status": (doc or {}).get("status"),
        "cases": {
            n: {
                "build_s": round(c.get("build_s", 0), 3),
                "oracle": c.get("oracle"),
                "timer_us": {
                    m: us(c.get(f"{m}_ms")) for m in ("cold_eager", "warm_eager", "graph_warm")
                },
                "trace": trace_role(
                    (c.get("trace") or {}).get("class"), (c.get("trace") or {}).get("role")
                ),
            }
            for n, c in ((doc or {}).get("cases") or {}).items()
        },
        **inv,
    }
if all(report["controls"].get(t) for t in (2048, 8192)):
    a, b = report["controls"][2048], report["controls"][8192]
    report["controls"]["shared_source_bodies"] = sorted(
        {s["body_sha"] for s in a["sources"]} & {s["body_sha"] for s in b["sources"]}
    )
    report["controls"]["shared_triton_kernels"] = sorted(
        set(a["triton_kernels"]) & set(b["triton_kernels"])
    )


# ---- identity (Nsight)
def nsight_identity(path: Path) -> dict:
    if not path.exists():
        return {}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    strings = dict(con.execute("select id, value from StringIds"))
    cols = [r[1] for r in con.execute("pragma table_info(NVTX_EVENTS)")]
    text_expr = (
        "coalesce(text, (select value from StringIds where id = textId))"
        if "textId" in cols
        else "text"
    )
    ranges = con.execute(
        f"select start, end, globalTid, {text_expr} from NVTX_EVENTS where end is not null"
    ).fetchall()
    rt = defaultdict(list)
    for s, tid, corr in con.execute(
        "select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        rt[tid].append((s, corr))
    kern = defaultdict(list)
    for row in con.execute(
        "select correlationId, start, end, demangledName, gridX, gridY, gridZ, "
        "blockX, blockY, blockZ, registersPerThread, dynamicSharedMemory, "
        "staticSharedMemory, graphId from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        kern[row[0]].append(row)
    res = {}
    for s, e, tid, text in ranges:
        if not text or ":" not in text:
            continue
        corr = {c for st_, c in rt[tid] if s <= st_ <= e}
        ks = sorted((k for c in corr for k in kern.get(c, [])), key=lambda k: k[1])
        launches = [
            {
                "name": strings.get(k[3], str(k[3]))[:140],
                "grid": list(k[4:7]),
                "block": list(k[7:10]),
                "regs": k[10],
                "smem_dynamic": k[11],
                "smem_static": k[12],
                "in_graph": k[13] not in (None, 0),
            }
            for k in ks
        ]
        uniq = []
        for x in launches:
            key = {k: v for k, v in x.items() if k != "in_graph"}
            if key not in uniq:
                uniq.append(key)
        res[text] = {"launches": len(launches), "distinct": uniq}
    return res


ident = nsight_identity(run / "identity/identity.sqlite")
report["identity"] = {}
for name, c in ((identity_doc or {}).get("cases") or {}).items():
    t = c.get("trace") or {}
    roles = [t.get("role")] + (["attention_aux_fill"] if t.get("role") == "attention" else [])
    report["identity"][name] = {
        "eager": ident.get(f"{name}:eager"),
        "graph": ident.get(f"{name}:graph"),
        "trace": {r: trace_role(t.get("class"), r) for r in roles} if t.get("class") else None,
        "oracle_after_replay": c.get("oracle_after_replay"),
    }

out.write_text(json.dumps(report, indent=1, default=str) + "\n")
print(
    f"timing cases {len(report['timing'])}, p8 pairs {len(p8['pairs'])}, split {len(p8['split'])}, "
    f"p9 {len(p9)}, p10 {len(p10)}, identity {len(report['identity'])} (nsight ranges {len(ident)})"
)
