#!/usr/bin/env python3
"""CPU-only: per-role kernel identity and timing of the TP1 smoke trace, for pilot P5-P10.

    trace_reference2.py --run-dir RUN --audit-dir AUDIT --out trace_reference2.json

Phase 3B pilot's trace_reference.py plus more iteration classes and the
attention operation as the simulator's leaf sees it (counter fill + FMHA).

Iteration classes: decode batch 8 (183, the audit's set), decode 7, decode 4,
prefill 256 and 568 (one request, no prefix), and the two mixed classes of
2,048 / 1,933 scheduled tokens (2,047 prefill + 1 decode; 1,929 + 4).
`mixed_decode1` (the batch-1 decode of the 2,047 + 1 class) is kept under its
earlier name.
Per role (the audit's `assign_forward_roles`, `lm_head` = sample-phase GEMM):
demangled names, grid, block, registers, shared memory, graph membership, and
per-call duration and exclusive time (the part of the kernel's interval no other
kernel of the iteration covers), in microseconds.

`attention_ops`: every forward attention kernel paired with the counter-fill
kernel launched just before it in the same layer, split into prefill (the
`Context` kernels) and decode. Per pair: fill and FMHA durations, their union,
span (fill start to FMHA end), gap (FMHA start minus fill end, 0 when they
overlap), overlap, and the pair's exclusive time.
"""

import argparse
import json
import sqlite3
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def stats(v):
    if not v:
        return None
    v = sorted(v)
    q = lambda f: v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))]  # noqa: E731
    return {
        "n": len(v),
        "median": st.median(v),
        "mean": st.fmean(v),
        "p10": q(0.1),
        "p90": q(0.9),
        "min": v[0],
        "max": v[-1],
    }


def exclusive(intervals):
    """Per interval, the length not covered by any other interval."""
    pts = sorted(set([s for s, _ in intervals] + [e for _, e in intervals]))
    out = [0] * len(intervals)
    for a, b in zip(pts, pts[1:]):
        cover = [i for i, (s, e) in enumerate(intervals) if s <= a and e >= b]
        if len(cover) == 1:
            out[cover[0]] += b - a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.audit_dir))
    import pyarrow.parquet as pq
    from tp1_trace_audit import assign_forward_roles, is_gemm

    parsed = json.loads((a.run_dir / "parsed.json").read_text())
    det = parsed["iteration_details"]
    kernels = defaultdict(list)
    for k in pq.read_table(a.run_dir / "parsed.kernels.parquet").to_pylist():
        kernels[k["detail_index"]].append(k)
    con = sqlite3.connect(f"file:{next((a.run_dir / 'nsys').glob('*.sqlite'))}?mode=ro", uri=True)
    strings = dict(con.execute("select id, value from StringIds"))
    meta = {}
    for row in con.execute(
        "select correlationId, start, demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ, "
        "registersPerThread, staticSharedMemory, dynamicSharedMemory, graphId, graphNodeId, "
        "clusterX, clusterY, clusterZ from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        meta[(row[0], row[1])] = row
    decode8 = json.loads((a.audit_dir / "decode8_iterations.json").read_text())

    def m(i):
        return det[i]["metrics"]

    classes = {
        "decode8": decode8,
        "decode4": [
            i
            for i in range(len(det))
            if m(i)["prefill_tokens"] == 0 and m(i)["decode_requests"] == 4
        ],
        "prefill256": [
            i
            for i in range(len(det))
            if m(i)["prefill_tokens"] == 256 and m(i)["decode_requests"] == 0
        ],
        "mixed_decode1": [
            i
            for i in range(len(det))
            if m(i)["prefill_tokens"] > 0 and m(i)["decode_requests"] == 1
        ],
        "decode7": [
            i
            for i in range(len(det))
            if m(i)["prefill_tokens"] == 0 and m(i)["decode_requests"] == 7
        ],
        "prefill568": [
            i
            for i in range(len(det))
            if m(i)["prefill_tokens"] == 568 and m(i)["decode_requests"] == 0
        ],
        "mixed2048": [
            i
            for i in range(len(det))
            if m(i)["prefill_tokens"] == 2047 and m(i)["decode_requests"] == 1
        ],
        "mixed1933": [
            i
            for i in range(len(det))
            if m(i)["prefill_tokens"] == 1929 and m(i)["decode_requests"] == 4
        ],
    }
    report = {"run_dir": str(a.run_dir), "classes": {}}
    for cname, idxs in classes.items():
        roles_acc = defaultdict(
            lambda: {
                "names": defaultdict(int),
                "launch": defaultdict(int),
                "dur_us": [],
                "excl_us": [],
                "per_layer_calls": [],
                "by_name": defaultdict(lambda: {"dur_us": [], "excl_us": []}),
                "op_sum_us": [],
                "op_union_us": [],
                "op_span_us": [],
                "op_excl_us": [],
            }
        )
        batches = []
        attn = defaultdict(lambda: defaultdict(list))
        for idx in idxs:
            x, ks = det[idx], kernels[idx]
            batches.append(
                {
                    "iteration": idx,
                    "prefill_chunk_pairs": m(idx)["prefill_chunk_pairs"],
                    "decode_kv_lens": m(idx)["decode_kv_lens"],
                }
            )
            phase = lambda k: x["ranges"][k["range_index"]]["phase"]  # noqa: B023,E731
            every = sorted(ks, key=lambda k: k["start_ns"])
            excl = exclusive([(k["start_ns"], k["end_ns"]) for k in every])
            ex_of = {id(k): e for k, e in zip(every, excl)}
            fwd = [k for k in every if phase(k) == "forward"]
            names = [parsed["kernel_names"][str(k["name_id"])] for k in fwd]
            roles = assign_forward_roles(names, 32)
            tagged = list(zip(fwd, roles))
            tagged += [
                (k, "lm_head")
                for k in every
                if phase(k) == "sample" and is_gemm(parsed["kernel_names"][str(k["name_id"])])
            ]
            # One operation instance = a run of consecutive kernels with the same role.
            instances, prev = [], None
            for k, role in tagged:
                if role != prev or not instances:
                    instances.append((role, []))
                instances[-1][1].append(k)
                prev = role
            for role, ks_ in instances:
                iv = [(k["start_ns"], k["end_ns"]) for k in ks_]
                acc = roles_acc[role]
                acc["op_sum_us"].append(sum(e - s for s, e in iv) / 1e3)
                u, cur = 0, None
                for s, e in sorted(iv):
                    if cur is None or s > cur[1]:
                        u += (cur[1] - cur[0]) if cur else 0
                        cur = [s, e]
                    else:
                        cur[1] = max(cur[1], e)
                acc["op_union_us"].append((u + (cur[1] - cur[0])) / 1e3)
                acc["op_span_us"].append((max(e for _, e in iv) - min(s for s, _ in iv)) / 1e3)
                acc["op_excl_us"].append(sum(ex_of[id(k)] for k in ks_) / 1e3)
            # Attention operation: fill + the FMHA kernel right after it.
            for j, (k, role) in enumerate(tagged):
                if role != "attention" or j == 0 or tagged[j - 1][1] != "attention_aux_fill":
                    continue
                f = tagged[j - 1][0]
                name = parsed["kernel_names"][str(k["name_id"])]
                which = "prefill" if "Context" in name else "decode"
                fd, ad = f["end_ns"] - f["start_ns"], k["end_ns"] - k["start_ns"]
                ov = max(0, min(f["end_ns"], k["end_ns"]) - max(f["start_ns"], k["start_ns"]))
                acc = attn[which]
                acc["fill_us"].append(fd / 1e3)
                acc["fmha_us"].append(ad / 1e3)
                acc["union_us"].append((fd + ad - ov) / 1e3)
                acc["span_us"].append((max(f["end_ns"], k["end_ns"]) - f["start_ns"]) / 1e3)
                acc["gap_us"].append(max(0, k["start_ns"] - f["end_ns"]) / 1e3)
                acc["overlap_us"].append(ov / 1e3)
                acc["exclusive_us"].append((ex_of[id(f)] + ex_of[id(k)]) / 1e3)
                acc["names"].append(name[:120])
            for k, role in tagged:
                r = meta.get((k["correlation_id"], k["start_ns"]))
                acc = roles_acc[role]
                if r is None:
                    acc["names"]["<no sqlite row>"] += 1
                    continue
                name = strings.get(r[2], str(r[2]))
                acc["names"][name] += 1
                acc["launch"][
                    json.dumps(
                        {
                            "name": name[:160],
                            "grid": r[3:6],
                            "block": r[6:9],
                            "regs": r[9],
                            "smem_static": r[10],
                            "smem_dynamic": r[11],
                            "in_graph": r[12] not in (None, 0),
                            "cluster": r[14:17],
                        }
                    )
                ] += 1
                acc["dur_us"].append((k["end_ns"] - k["start_ns"]) / 1e3)
                acc["excl_us"].append(ex_of[id(k)] / 1e3)
                acc["by_name"][name]["dur_us"].append((k["end_ns"] - k["start_ns"]) / 1e3)
                acc["by_name"][name]["excl_us"].append(ex_of[id(k)] / 1e3)
        report["classes"][cname] = {
            "iterations": len(idxs),
            "batches_first_last": [batches[0], batches[-1]] if batches else [],
            "roles": {
                role: {
                    "names": dict(v["names"]),
                    "launch_configs": {
                        k: n for k, n in sorted(v["launch"].items(), key=lambda t: -t[1])
                    },
                    "duration_us": stats(v["dur_us"]),
                    "exclusive_us": stats(v["excl_us"]),
                    "by_name": {
                        n: {"duration_us": stats(x["dur_us"]), "exclusive_us": stats(x["excl_us"])}
                        for n, x in v["by_name"].items()
                    },
                    "operation": {
                        "sum_us": stats(v["op_sum_us"]),
                        "union_us": stats(v["op_union_us"]),
                        "span_us": stats(v["op_span_us"]),
                        "exclusive_us": stats(v["op_excl_us"]),
                    },
                }
                for role, v in roles_acc.items()
            },
            "attention_ops": {
                which: {
                    "pairs": len(v["fill_us"]),
                    "fmha_names": sorted(set(v["names"])),
                    **{k: stats(x) for k, x in v.items() if k != "names"},
                }
                for which, v in attn.items()
            },
        }
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                c: {r: (v["duration_us"] or {}).get("median") for r, v in d["roles"].items()}
                for c, d in report["classes"].items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
