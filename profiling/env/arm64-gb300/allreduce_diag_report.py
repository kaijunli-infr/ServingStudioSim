"""Phase 2D: stability report over the diagnostic rank groups (CPU-only).

    allreduce_diag_report.py --db PRIVATE_DB --stage 2=DIR --stage 4=DIR --out report.json

Per configuration: the group estimate is the median of a group's three batch
times (rank 0's CUDA-event time, the quantity a stored row holds); the
diagnostic estimate is the median of the five group estimates. Stable = at
least four of five group estimates within 10 % of that median. No observation
is discarded. The database is opened read-only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics as st
from pathlib import Path

GROUPS = 5
TOLERANCE = 0.10
SPIKE = 1.25  # a batch more than 25 % above the diagnostic estimate is listed as a spike


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--stage", action="append", required=True, help="NUM_GPUS=DIR")
    parser.add_argument("--out", required=True)
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
        data = {}  # size -> group -> rank -> batches
        for g in range(1, GROUPS + 1):
            for rank in range(num_gpus):
                path = Path(directory, f"group{g}-rank{rank}.json")
                if not path.exists():
                    continue
                for s in json.loads(path.read_text())["sizes"]:
                    data.setdefault(s["message_size_bytes"], {}).setdefault(g, {})[rank] = s
        for size in sorted(data):
            groups = data[size]
            complete = len(groups) == GROUPS and all(len(r) == num_gpus for r in groups.values())
            batch_ms = {g: [b["time_ms"] for b in groups[g][0]["batches"]] for g in sorted(groups)}
            estimates = [st.median(v) for v in batch_ms.values()]
            final = st.median(estimates)
            within = sum(abs(e / final - 1) <= TOLERANCE for e in estimates)
            all_batches = [t for v in batch_ms.values() for t in v]
            spikes = [
                {"group": g, "batch": i + 1, "time_us": t * 1e3, "over_estimate": t / final}
                for g, v in batch_ms.items()
                for i, t in enumerate(v)
                if t > SPIKE * final
            ]
            # Rank-level and host-side views of the same batches.
            rank_spread, enqueue_share, max_gap_us = [], [], []
            for g in groups:
                for i in range(len(groups[g][0]["batches"])):
                    per_rank = [groups[g][r]["batches"][i]["time_ms"] for r in groups[g]]
                    rank_spread.append(max(per_rank) / min(per_rank))
                    b0 = groups[g][0]["batches"][i]
                    enqueue_share.append(b0["host_enqueue_ms"] / b0["time_ms"])
                    max_gap_us.append(
                        max(max(groups[g][r]["batches"][i]["call_gaps_ns"]) for r in groups[g])
                        / 1e3
                    )
            rows.append(
                {
                    "num_gpus": num_gpus,
                    "message_size_bytes": size,
                    "complete": complete,
                    "stored_us": stored[size] * 1e3,
                    "estimate_us": final * 1e3,
                    "change_pct": (final / stored[size] - 1) * 100,
                    "group_estimates_us": [e * 1e3 for e in estimates],
                    "min_us": min(estimates) * 1e3,
                    "max_us": max(estimates) * 1e3,
                    "mad_us": st.median(abs(e - final) for e in estimates) * 1e3,
                    "cv_pct": st.stdev(estimates) / st.fmean(estimates) * 100
                    if len(estimates) > 1
                    else None,
                    "groups_within_10pct": within,
                    "stable": complete and within >= 4,
                    "batch_us": {str(g): [t * 1e3 for t in v] for g, v in batch_ms.items()},
                    "batch_min_us": min(all_batches) * 1e3,
                    "batch_max_us": max(all_batches) * 1e3,
                    "spikes": spikes,
                    "sum_mismatches": sum(
                        groups[g][r]["sum_mismatches"] for g in groups for r in groups[g]
                    ),
                    "rank_time_max_over_min": max(rank_spread),
                    "host_enqueue_share_median": st.median(enqueue_share),
                    "max_call_gap_us": max(max_gap_us),
                }
            )
    report = {"groups": GROUPS, "tolerance": TOLERANCE, "spike_threshold": SPIKE, "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    for r in rows:
        groups = [round(e, 1) for e in r["group_estimates_us"]]
        print(
            f"tp{r['num_gpus']} {r['message_size_bytes']:>8} stored {r['stored_us']:6.1f} "
            f"new {r['estimate_us']:6.1f} ({r['change_pct']:+5.1f}%) groups {groups} "
            f"cv {r['cv_pct']:.1f}% in10 {r['groups_within_10pct']}/5 "
            f"{'PASS' if r['stable'] else 'FAIL'} spikes {len(r['spikes'])} "
            f"enq {r['host_enqueue_share_median']:.2f} gap {r['max_call_gap_us']:.0f}us"
        )
    print(f"stable {sum(r['stable'] for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
