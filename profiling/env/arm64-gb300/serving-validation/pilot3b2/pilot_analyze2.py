#!/usr/bin/env python3
"""CPU-only analysis of a P5-P10 pilot run (complete or stopped) against the TP1 trace.

    pilot_analyze2.py RUN_DIR TRACE_REFERENCE2_JSON OUT_JSON

Reads RUN_DIR/timing/timing.json (whatever cases it holds), the Triton cache
metadata (num_warps, shared memory) and the Inductor autotune records
(``*.best_config``) of RUN_DIR/cache, and trace_reference2.json. Writes one
JSON with, per case: the oracle results, cold / warm / graph medians and
per-launch variability (CV), the trace's per-kernel duration for the case's
role and class, and for P6 the graph-placement diagnostic. Also lists every
compiled Triton reduction kernel with its launch width and every autotune
decision with the Inductor source it belongs to (loads, stores, size hints).
Prints a short table.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

run, trace_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
timing = json.loads((run / "timing/timing.json").read_text())
trace = json.loads(trace_path.read_text())["classes"]


def us(ms):
    return None if ms is None else round(ms * 1e3, 3)


def med(b, key="launch_union_us"):
    return round(b[key]["median"], 3) if b and b.get(key) else None


def cv(b, key="launch_union_us"):
    return round(b[key]["cv"], 4) if b and b.get(key) and b[key]["cv"] is not None else None


def per_kernel(b):
    if not b or "per_kernel_us" not in b:
        return {}
    return {k[:60]: round(v["median"], 3) for k, v in b["per_kernel_us"].items()}


def trace_role(cls, role):
    r = trace.get(cls, {}).get("roles", {}).get(role)
    if not r:
        return None
    return {
        "kernel_median_us": round(r["duration_us"]["median"], 3),
        "kernel_p10_p90_us": [round(r["duration_us"]["p10"], 3), round(r["duration_us"]["p90"], 3)],
        "calls": r["duration_us"]["n"],
        "launch_configs": [dict(json.loads(k), count=n) for k, n in r["launch_configs"].items()][
            :2
        ],
    }


def triton_kernels():
    rows = []
    for f in glob.glob(str(run / "cache/triton/*/*.json")):
        if os.path.basename(f).startswith("__grp__"):
            continue
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if "num_warps" in d and d.get("name", "").startswith("triton_"):
            rows.append(
                {
                    "name": d["name"],
                    "threads": 32 * d["num_warps"],
                    "smem": d.get("shared"),
                    "triton_hash": Path(f).parent.name,
                    "mtime": os.path.getmtime(f),
                }
            )
    return sorted(rows, key=lambda r: r["mtime"])


def autotune_decisions():
    best = {
        Path(f).name[: -len(".best_config")]: json.loads(Path(f).read_text())
        for f in glob.glob(str(run / "cache/inductor/**/*.best_config"), recursive=True)
    }
    sources = []
    for f in glob.glob(str(run / "cache/inductor/**/*.py"), recursive=True):
        s = Path(f).read_text(errors="replace")
        for m in re.finditer(r"async_compile\.triton\('(triton_\w+)', '''(.*?)'''", s, re.S):
            body = m.group(2)
            hints = re.search(r"size_hints=(\{[^}]*\})", body)
            sources.append(
                {
                    "name": m.group(1),
                    "size_hints": hints.group(1) if hints else None,
                    "in_ptr_loads": len(re.findall(r"tl\.load\(in_ptr", body)),
                    "stores": len(re.findall(r"tl\.store\(", body)),
                    "file": Path(f).name,
                    "mtime": os.path.getmtime(f),
                }
            )
    kernels = {k["triton_hash"]: k for k in triton_kernels()}
    decisions = [
        {
            "best_config": k,
            "R0_BLOCK": v.get("R0_BLOCK"),
            "XBLOCK": v.get("XBLOCK"),
            "num_warps": v.get("num_warps"),
            "threads": 32 * v.get("num_warps", 0),
            "selected_kernel": kernels.get(v.get("triton_cache_hash"), {}).get("name"),
            "selected_mtime": kernels.get(v.get("triton_cache_hash"), {}).get("mtime"),
        }
        for k, v in best.items()
    ]
    return sorted(sources, key=lambda r: r["mtime"]), sorted(
        decisions, key=lambda r: r["selected_mtime"] or 0
    )


report = {
    "run": str(run),
    "status": timing.get("status"),
    "elapsed_s": timing.get("elapsed_s"),
    "env": timing.get("env"),
    "cases": {},
}
for name, c in timing["cases"].items():
    t = c.get("trace") or {}
    entry = {
        "status": c.get("status"),
        "part": c.get("part"),
        "meta": c.get("meta"),
        "oracle": c.get("oracle"),
        "oracle_after_first_replay": c.get("oracle_after_first_replay"),
        "oracle_after_timed_replays": c.get("oracle_after_timed_replays"),
        "build_s": c.get("build_s"),
        "timer_us": {m: us(c.get(f"{m}_ms")) for m in ("cold_eager", "warm_eager", "graph_warm")},
        "burst_union_median_us": {
            m: med(c.get(m)) for m in ("cold_eager", "warm_eager", "graph_warm")
        },
        "burst_union_cv": {m: cv(c.get(m)) for m in ("cold_eager", "warm_eager", "graph_warm")},
        "per_kernel_median_us": {
            m: per_kernel(c.get(m)) for m in ("cold_eager", "warm_eager", "graph_warm")
        },
        "sequences": {m: (c.get(m) or {}).get("sequences") for m in ("cold_eager", "graph_warm")},
        "trace": trace_role(t.get("class"), t.get("role")) if t.get("class") else None,
    }
    extra = c.get("extra")
    if extra and "first" in extra:
        entry["placement"] = {
            label: {
                cond: {
                    "per_kernel_median_us": per_kernel(extra[label][cond]),
                    "per_kernel_cv": {
                        k[:60]: round(v["cv"], 4)
                        for k, v in extra[label][cond]["per_kernel_us"].items()
                    },
                }
                for cond in ("warm", "cold")
            }
            for label in ("first", "later", "first_again")
        }
        entry["placement"]["oracles"] = {
            k: extra[k]["oracle"] for k in ("first", "later", "first_again")
        }
    report["cases"][name] = entry
report["triton_kernels"] = triton_kernels()
report["inductor_sources"], report["autotune_decisions"] = autotune_decisions()
out.write_text(json.dumps(report, indent=1, default=str) + "\n")

for name, e in report["cases"].items():
    tr = e["trace"]["kernel_median_us"] if e["trace"] else None
    tm = e["timer_us"]

    def d(x):
        return f"{x:7.2f} ({100 * (x / tr - 1):+4.0f}%)" if (x and tr) else f"{x}"

    print(
        f"{name:26s} trace {tr}  cold {d(tm['cold_eager'])}  warm {d(tm['warm_eager'])}  "
        f"graph {d(tm['graph_warm'])}  cv(cold/warm/graph) {list(e['burst_union_cv'].values())}"
    )
    if "placement" in e:
        for label in ("first", "later", "first_again"):
            p = e["placement"][label]
            warm, cold = p["warm"]["per_kernel_median_us"], p["cold"]["per_kernel_median_us"]
            print(f"    placement {label:12s} warm {warm}  cold {cold}")
for s in report["inductor_sources"]:
    print(
        "source", s["name"][:50], s["size_hints"], "loads", s["in_ptr_loads"], "stores", s["stores"]
    )
for dcs in report["autotune_decisions"]:
    print("autotune", dcs["selected_kernel"], dcs["threads"], "threads, R0_BLOCK", dcs["R0_BLOCK"])
