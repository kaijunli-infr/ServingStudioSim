"""TP1 trace-accounting audit of a vLLM serving capture against the simulator (CPU-only).

    python tp1_trace_audit.py --run-dir RUN --breakdown iter_breakdown.json \
        --iterations decode8_iterations.json --out audit.json
    python tp1_trace_audit.py --self-test

Inputs are the alignment pipeline's artifacts of one capture (`parsed.json`,
`parsed.kernels.parquet`, the Nsight Systems sqlite) and the simulator's
`timing-predict` breakdown for the same iterations, one case per iteration in
the order of `--iterations`.

Stage A, accounting per iteration (GPU clock, ns):

- `fwd_sum`: sum of the durations of the kernels the pipeline attached to the
  iteration's `forward` range (by CUDA correlation id).
- `fwd_union`: the time at least one of them runs; `fwd_sum - fwd_union` is
  overlap between kernels; `fwd_span - fwd_union` is idle time inside the
  forward pass.
- the same for all phases of the iteration (preprocess, forward, sample,
  bookkeep); `iter_span` is first kernel to last kernel of the iteration.
- `cycle`: this iteration's first kernel to the next iteration's first kernel;
  `cycle - iter_span` is GPU idle between iterations.
- `unattributed`: kernels of the capture that start inside the cycle but are
  attached to no range.
- host: duration of the iteration's NVTX ranges on the engine thread.

Stage B, roles: forward kernels are assigned to model operations by a fixed
per-layer state machine over their order (vLLM Llama decoder layer under
torch.compile), and the GEMM of the `sample` phase to `lm_head`.
Stage C, categories and coverage, and per-role exploratory differences against
the simulator leaves.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

# role -> (simulator leaf label prefix or None, category)
ROLES = {
    "embed_input_norm": ("embedding+input_norm", "fused"),
    "input_norm": ("input_norm", "fused"),
    "qkv_proj": ("qkv_proj", "direct"),
    "rope": (None, "unmodeled"),
    "kv_cache_append": ("kv_cache_append", "direct"),
    "attention_aux_fill": ("decode", "implementation"),
    "attention": ("decode", "implementation"),
    "o_proj": ("o_proj", "direct"),
    "post_norm": ("post_norm", "fused"),
    "up_gate_proj": ("up_gate_proj", "direct"),
    "activation": ("activation", "implementation"),
    "down_proj": ("down_proj", "direct"),
    "final_norm": ("final_norm", "fused"),
    "lm_head": ("lm_head", "direct"),
    "runtime": (None, "runtime"),
    "unknown": (None, "unknown"),
}
GEMM_ORDER = ["qkv_proj", "o_proj", "up_gate_proj", "down_proj"]


def is_gemm(name: str) -> bool:
    return name.startswith("nvjet_") or ("cublasLt::" in name and "splitKreduce" not in name)


def assign_forward_roles(names: list[str], num_layers: int) -> list[str]:
    """Roles for one forward pass's kernels, in launch order.

    Per decoder layer vLLM (torch.compile, TRT-LLM attention) issues: a fused
    residual add + RMSNorm (`fused_add_rms_norm`, or embedding + RMSNorm in
    layer 0), the QKV GEMM, a RoPE kernel (`triton_poi_fused_<n>`), the KV
    write, a fill and the attention kernel, the O GEMM (+ split-K reduce), a
    fused add + RMSNorm, the gate/up GEMM, SiLU-and-mul, the down GEMM
    (+ split-K reduce). The last fused add + RMSNorm is the final norm.
    """
    roles: list[str] = []
    layer = -1
    gemm_index = 0
    after_attention = False
    norms_in_layer = 0
    last_gemm = "unknown"
    for name in names:
        if "embedding_rms_norm" in name:
            layer, gemm_index, norms_in_layer, after_attention = 0, 0, 1, False
            roles.append("embed_input_norm")
        elif "rms_norm" in name and name.startswith("triton_"):
            if norms_in_layer == 1 and after_attention:
                norms_in_layer = 2
                roles.append("post_norm")
            else:
                layer += 1
                gemm_index, norms_in_layer, after_attention = 0, 1, False
                roles.append("final_norm" if layer == num_layers else "input_norm")
        elif "splitKreduce" in name:
            roles.append(last_gemm)
        elif is_gemm(name):
            last_gemm = GEMM_ORDER[gemm_index] if gemm_index < len(GEMM_ORDER) else "unknown"
            gemm_index += 1
            roles.append(last_gemm)
        elif "mul_silu" in name:
            roles.append("activation")
        elif name.startswith("triton_poi_fused"):
            roles.append("rope")
        elif "reshape_and_cache" in name:
            roles.append("kv_cache_append")
        elif "FillFunctor<unsigned char>" in name:
            roles.append("attention_aux_fill")
        elif "fmha" in name or "Fmha" in name:
            after_attention = True
            roles.append("attention")
        else:
            roles.append("unknown")
    return roles


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def dist(values: list[float]) -> dict:
    s = sorted(values)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    return {
        "n": len(s),
        "median": st.median(s),
        "p10": q(0.10),
        "p90": q(0.90),
        "min": s[0],
        "max": s[-1],
        "cv_pct": st.pstdev(s) / st.fmean(s) * 100 if st.fmean(s) else None,
    }


def sim_leaves(iteration: dict) -> dict[str, float]:
    """Simulator leaf totals (ms, scaled by layer count) of one breakdown iteration."""
    out: dict[str, float] = defaultdict(float)
    for node in iteration["nodes"]:
        if node["kind"] == "leaf":
            out[node["label"].split(" (")[0]] += node["total_ms"]
    out["total"] = iteration["nodes"][0]["total_ms"]
    return dict(out)


def audit(run_dir: Path, breakdown: Path, iterations_file: Path, num_layers: int = 32) -> dict:
    import pyarrow.parquet as pq

    parsed = json.loads((run_dir / "parsed.json").read_text())
    det = parsed["iteration_details"]
    names = {int(k): v for k, v in parsed["kernel_names"].items()}
    wanted = json.loads(iterations_file.read_text())
    sim = json.loads(breakdown.read_text())["iterations"]
    if len(sim) != len(wanted):
        raise SystemExit(f"{len(sim)} simulated iterations for {len(wanted)} measured")

    kernels: dict[int, list[dict]] = defaultdict(list)
    for k in pq.read_table(run_dir / "parsed.kernels.parquet").to_pylist():
        kernels[k["detail_index"]].append(k)
    sqlite = next((run_dir / "nsys").glob("*.sqlite"))
    con = sqlite3.connect(f"file:{sqlite}?mode=ro", uri=True)
    all_k = con.execute(
        "select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    ).fetchall()
    attributed = {(k["correlation_id"], k["start_ns"]) for ks in kernels.values() for k in ks}
    first_start = {i: min(k["start_ns"] for k in ks) for i, ks in kernels.items() if ks}
    order = sorted(first_start)

    rows, role_ns, role_counts, sequences = [], defaultdict(list), defaultdict(list), Counter()
    for pos, idx in enumerate(wanted):
        x, ks = det[idx], kernels[idx]
        phase_of = lambda k: x["ranges"][k["range_index"]]["phase"]  # noqa: B023,E731
        fwd = sorted((k for k in ks if phase_of(k) == "forward"), key=lambda k: k["start_ns"])
        every = sorted(ks, key=lambda k: k["start_ns"])
        nxt = order[order.index(idx) + 1] if order.index(idx) + 1 < len(order) else None
        cycle_end = first_start[nxt] if nxt is not None else None
        fwd_iv = [(k["start_ns"], k["end_ns"]) for k in fwd]
        all_iv = [(k["start_ns"], k["end_ns"]) for k in every]
        row = {
            "iteration": idx,
            "fwd_kernels": len(fwd),
            "fwd_sum": sum(e - s for s, e in fwd_iv),
            "fwd_union": union_ns(fwd_iv),
            "fwd_span": fwd_iv[-1][1] - fwd_iv[0][0] if fwd_iv else 0,
            "iter_kernels": len(every),
            "iter_sum": sum(e - s for s, e in all_iv),
            "iter_union": union_ns(all_iv),
            "iter_span": max(e for _, e in all_iv) - all_iv[0][0],
            "cycle": (cycle_end - all_iv[0][0]) if cycle_end else None,
            "host_forward_range": sum(
                r["duration_ns"] for r in x["ranges"] if r["phase"] == "forward"
            ),
            "host_iteration_extent": max(r["end_ns"] for r in x["ranges"])
            - min(r["start_ns"] for r in x["ranges"]),
        }
        by_phase = defaultdict(list)
        for k in every:
            by_phase[phase_of(k)].append((k["start_ns"], k["end_ns"]))
        row["phase_sum"] = {p: sum(e - s for s, e in v) for p, v in by_phase.items()}
        if cycle_end:
            stray = [
                (s, e)
                for s, e, c in all_k
                if all_iv[0][0] <= s < cycle_end and (c, s) not in attributed
            ]
            row["unattributed_kernels"] = len(stray)
            row["unattributed_ns"] = sum(e - s for s, e in stray)
        # roles
        fnames = [names[k["name_id"]] for k in fwd]
        roles = assign_forward_roles(fnames, num_layers)
        sequences[tuple(r for r in roles[:13])] += 1
        per_role = defaultdict(int)
        per_role_n = Counter()
        for k, r in zip(fwd, roles):
            per_role[r] += k["end_ns"] - k["start_ns"]
            per_role_n[r] += 1
        for k in every:
            if phase_of(k) == "forward":
                continue
            r = "lm_head" if is_gemm(names[k["name_id"]]) and phase_of(k) == "sample" else "runtime"
            per_role[r] += k["end_ns"] - k["start_ns"]
            per_role_n[r] += 1
        for r in ROLES:
            role_ns[r].append(per_role.get(r, 0))
            role_counts[r].append(per_role_n.get(r, 0))
        row["sim"] = sim_leaves(sim[pos])
        rows.append(row)

    total_ns = [r["iter_sum"] for r in rows]
    categories = defaultdict(list)
    for i in range(len(rows)):
        acc = Counter()
        for r, (_, cat) in ROLES.items():
            acc[cat] += role_ns[r][i]
        for cat, v in acc.items():
            categories[cat].append(v / total_ns[i])

    # exploratory per-role comparison against simulator leaves
    pairs = {
        "qkv_proj": ["qkv_proj"],
        "o_proj": ["o_proj"],
        "up_gate_proj": ["up_gate_proj"],
        "down_proj": ["down_proj"],
        "lm_head": ["lm_head"],
        "kv_cache_append": ["kv_cache_append"],
        "decode": ["attention", "attention_aux_fill"],
        "activation": ["activation"],
        "input_norm+post_norm+final_norm+embedding": [
            "input_norm",
            "post_norm",
            "final_norm",
            "embed_input_norm",
        ],
    }
    leaf_names = {
        "input_norm+post_norm+final_norm+embedding": [
            "input_norm",
            "post_norm",
            "final_norm",
            "embedding",
        ],
    }
    explore = {}
    for leaf, roles in pairs.items():
        meas = [sum(role_ns[r][i] for r in roles) / 1e6 for i in range(len(rows))]
        simv = [
            sum(rows[i]["sim"].get(n, 0.0) for n in leaf_names.get(leaf, [leaf]))
            for i in range(len(rows))
        ]
        explore[leaf] = {
            "measured_ms_median": st.median(meas),
            "simulated_ms_median": st.median(simv),
            "sim_over_measured_median": st.median(s / m for s, m in zip(simv, meas) if m),
        }

    summary = {
        "iterations": len(rows),
        "accounting_us": {
            key: dist([r[key] / 1e3 for r in rows if r.get(key) is not None])
            for key in (
                "fwd_sum",
                "fwd_union",
                "fwd_span",
                "iter_sum",
                "iter_union",
                "iter_span",
                "cycle",
                "host_forward_range",
                "host_iteration_extent",
            )
        },
        "derived_us": {
            "fwd_overlap (sum-union)": dist([(r["fwd_sum"] - r["fwd_union"]) / 1e3 for r in rows]),
            "fwd_internal_idle (span-union)": dist(
                [(r["fwd_span"] - r["fwd_union"]) / 1e3 for r in rows]
            ),
            "iter_internal_idle (span-union)": dist(
                [(r["iter_span"] - r["iter_union"]) / 1e3 for r in rows]
            ),
            "inter_iteration_gap (cycle-span)": dist(
                [(r["cycle"] - r["iter_span"]) / 1e3 for r in rows if r["cycle"]]
            ),
            "unattributed": dist([r.get("unattributed_ns", 0) / 1e3 for r in rows]),
        },
        "phase_sum_us": {
            p: dist([r["phase_sum"].get(p, 0) / 1e3 for r in rows])
            for p in sorted({p for r in rows for p in r["phase_sum"]})
        },
        "kernel_counts": {
            "forward_per_iteration": dist([r["fwd_kernels"] for r in rows]),
            "all_phases_per_iteration": dist([r["iter_kernels"] for r in rows]),
            "per_role_per_iteration": {r: st.median(v) for r, v in role_counts.items()},
        },
        "role_time_us": {r: dist([v / 1e3 for v in role_ns[r]]) for r in ROLES if any(role_ns[r])},
        "category_share_of_kernel_time": {c: dist(v) for c, v in categories.items()},
        "first_layer_role_sequences": {" > ".join(k): v for k, v in sequences.items()},
        "simulator_total_ms": dist([r["sim"]["total"] for r in rows]),
        "exploratory_role_vs_leaf": explore,
    }
    return {"summary": summary, "rows": rows}


def self_test() -> int:
    layer = [
        "triton_red_fused_fused_add_rms_norm_2",
        "nvjet_a",
        "triton_poi_fused_3",
        "void vllm::reshape_and_cache_flash_kernel<...>",
        "void at::native::vectorized_elementwise_kernel<(int)8, "
        "at::native::FillFunctor<unsigned char>, x>",
        "fmhaSm100fKernel_x",
        "nvjet_b_splitK",
        "void cublasLt::splitKreduce_kernel<...>",
        "triton_red_fused_fused_add_rms_norm_0",
        "nvjet_c",
        "triton_poi_fused_mul_silu_slice_1",
        "nvjet_d_splitK",
        "void cublasLt::splitKreduce_kernel<...>",
    ]
    first = ["triton_red_fused__to_copy_embedding_rms_norm_0"] + layer[1:]
    seq = first + layer * 2 + ["triton_red_fused_fused_add_rms_norm_2"]
    roles = assign_forward_roles(seq, num_layers=3)
    want_layer = [
        "input_norm",
        "qkv_proj",
        "rope",
        "kv_cache_append",
        "attention_aux_fill",
        "attention",
        "o_proj",
        "o_proj",
        "post_norm",
        "up_gate_proj",
        "activation",
        "down_proj",
        "down_proj",
    ]
    want = ["embed_input_norm"] + want_layer[1:] + want_layer * 2 + ["final_norm"]
    ok = roles == want
    print("self-test:", "pass" if ok else f"FAIL\n{roles}\n{want}")
    assert union_ns([(0, 10), (5, 12), (20, 25)]) == 17
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--breakdown", type=Path)
    parser.add_argument("--iterations", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    result = audit(args.run_dir, args.breakdown, args.iterations)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
