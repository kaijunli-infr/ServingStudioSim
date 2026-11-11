"""Write the fixed-shape request traces of the serving-validation cases (CPU-only).

    python make_traces.py --out-dir DIR

One `independent` trace per case for the alignment load generator
(`id,input_len,output_len,arrival_time`, token-in/token-out). All requests of a
case arrive at time 0; the profile config sets `arrival_mode: saturated` and
`max_concurrency` to the case's request count so they run as one batch.

Decode cases use 568 prompt tokens and 64 output tokens, so the cached length
during the decode steps runs from 568 to 631 and averages 600, the value the
timing predictions use. Prefill cases generate one token.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CASES = {
    # name: (requests, input_len, output_len)
    "prefill_256": (1, 256, 1),
    "prefill_2048": (1, 2048, 1),
    "decode_1": (1, 568, 64),
    "decode_8": (8, 568, 64),
    "decode_64": (64, 568, 64),
    "decode_256": (256, 568, 64),
}
REPEATS = 5  # each case's batch is sent this many times, one batch after another


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (requests, input_len, output_len) in CASES.items():
        rows = [
            (f"{name}-r{rep}-{i}", input_len, output_len, float(rep * 1000))
            for rep in range(REPEATS)
            for i in range(requests)
        ]
        with (out / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "input_len", "output_len", "arrival_time"])
            writer.writerows(rows)
        summary[name] = {
            "requests_per_batch": requests,
            "batches": REPEATS,
            "input_len": input_len,
            "output_len": output_len,
            "hidden_bytes_per_all_reduce": requests * 4096 * 2
            if name.startswith("decode")
            else input_len * 4096 * 2,
        }
    (out / "cases.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
