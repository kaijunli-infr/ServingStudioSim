"""Phase 2F: stability and eager-against-graph report (CPU-only).

    allreduce_graph_report.py --db PRIVATE_DB --stage 2=DIR --stage 4=DIR --out report.json

For every (num_gpus, message size, method): the group estimate is the median
of the group's three batch times (the slowest-rank figure for graph methods,
rank 0's event time for the eager method); the estimate is the median of the
five group estimates. Stable = at least four of five group estimates within
10 % of that median. No observation is discarded. The database is read-only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics as st
from pathlib import Path

GROUPS = 5
TOLERANCE = 0.10
SPIKE = 1.25


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--stage", action="append", required=True, help="NUM_GPUS=DIR")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--candidates",
        choices=("small", "all"),
        default="small",
        help="which ten-collective graph rows count as production candidates",
    )
    args = parser.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = []
    for item in args.stage:
        num_gpus, directory = item.split("=", 1)
        num_gpus = int(num_gpus)
        stored = dict(
            con.execute(
                "select message_size_bytes, time_ms from all_reduce where gpu_name = 'NVIDIA GB300'"
                " and backend = 'nccl' and fabric = 'nvlink' and num_gpus = ?",
                (num_gpus,),
            )
        )
        data: dict = {}  # (size, method) -> group -> rank -> method record
        mismatches: dict = {}
        provenance = None
        rank_uuids: dict[int, str] = {}
        for g in range(1, GROUPS + 1):
            for rank in range(num_gpus):
                path = Path(directory, f"group{g}-rank{rank}.json")
                if not path.exists():
                    continue
                report = json.loads(path.read_text())
                provenance = provenance or report["provenance"] | {
                    "torch": report["torch"],
                    "nccl": report["nccl"],
                }
                rank_uuids[rank] = report["uuid"]
                for s in report["sizes"]:
                    size = s["message_size_bytes"]
                    for name, method in s["methods"].items():
                        data.setdefault((size, name), {}).setdefault(g, {})[rank] = method
                        bad = s["eager_sum_mismatches"] + method.get("replay_sum_mismatches", 0)
                        mismatches[(size, name)] = mismatches.get((size, name), 0) + bad
        for (size, name), groups in sorted(data.items()):
            complete = len(groups) == GROUPS and all(len(r) == num_gpus for r in groups.values())
            batch_ms = {g: [b["time_ms"] for b in groups[g][0]["batches"]] for g in sorted(groups)}
            estimates = [st.median(v) for v in batch_ms.values()]
            final = st.median(estimates)
            within = sum(abs(e / final - 1) <= TOLERANCE for e in estimates)
            everything = [t for v in batch_ms.values() for t in v]
            row = {
                "num_gpus": num_gpus,
                "message_size_bytes": size,
                "method": name,
                "production_candidate": name.endswith("ops10")
                and (args.candidates == "all" or size < 16 << 20),
                "complete": complete,
                "sum_mismatches": mismatches[(size, name)],
                "stored_eager_us": stored[size] * 1e3,
                "estimate_us": final * 1e3,
                "stored_over_estimate": stored[size] / final,
                "group_estimates_us": [e * 1e3 for e in estimates],
                "min_us": min(estimates) * 1e3,
                "max_us": max(estimates) * 1e3,
                "mad_us": st.median(abs(e - final) for e in estimates) * 1e3,
                "cv_pct": st.stdev(estimates) / st.fmean(estimates) * 100,
                "groups_within_10pct": within,
                "stable": complete and within >= 4,
                "batch_us": {str(g): [t * 1e3 for t in v] for g, v in batch_ms.items()},
                "batch_min_us": min(everything) * 1e3,
                "batch_max_us": max(everything) * 1e3,
                "spike_batches": sum(t > SPIKE * final for t in everything),
            }
            if "cuda_graph" in name:
                # Second clock (CUDA events) and rank spread of the same batches.
                event = [
                    st.median(b["local_event_time_ms"] for b in groups[g][0]["batches"])
                    for g in groups
                ]
                row["event_clock_estimate_us"] = st.median(event) * 1e3
                row["rank_local_max_over_min"] = max(
                    max(groups[g][r]["batches"][i]["local_time_ms"] for r in groups[g])
                    / min(groups[g][r]["batches"][i]["local_time_ms"] for r in groups[g])
                    for g in groups
                    for i in range(len(groups[g][0]["batches"]))
                )
            rows.append(row)
        rows[-1]["provenance"] = provenance | {
            "rank_uuids": [rank_uuids[r] for r in sorted(rank_uuids)],
            "raw_dir": directory,
        }
    report = {"groups": GROUPS, "tolerance": TOLERANCE, "spike_threshold": SPIKE, "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    for r in rows:
        groups = [round(e, 1) for e in r["group_estimates_us"]]
        print(
            f"tp{r['num_gpus']} {r['message_size_bytes']:>9} {r['method'].split('.')[-1]:<12} "
            f"est {r['estimate_us']:7.1f} stored {r['stored_eager_us']:7.1f} groups {groups} "
            f"cv {r['cv_pct']:.1f}% in10 {r['groups_within_10pct']}/5 "
            f"{'PASS' if r['stable'] else 'FAIL'} spikes {r['spike_batches']} "
            f"mism {r['sum_mismatches']}"
        )
    candidates = [r for r in rows if r["production_candidate"]]
    print(f"candidates stable {sum(r['stable'] for r in candidates)}/{len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
