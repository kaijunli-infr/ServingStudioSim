#!/usr/bin/env python
"""Verify a profile DB merge: `before` (the tracked DB) against `after` (merged).

    merge_check.py --before profile.db.bak --after profile-merged.db \
        --gpu "NVIDIA GB300" --expect-new 1236 --out report.json

Standard library only, read-only on both files. Checks, per measurement table:
every row of `before` is present in `after` with identical content (all
columns except the surrogate `id`), every row `after` adds belongs to `--gpu`,
no semantic key (the table's unique `(gpu_name, backend, …)` index) repeats,
every added row has a finite positive `time_ms` and a run record, and the
`_profile_run` rows `after` adds are exactly the ones the added rows cite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path

RUN_TABLE = "_profile_run"


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tables(con: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            "select name from sqlite_master where type='table' and name not like 'sqlite_%' "
            "order by name"
        )
    ]


def columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r["name"] for r in con.execute(f"pragma table_info({q(table)})")]


def semantic_key(con: sqlite3.Connection, table: str) -> list[str] | None:
    for idx in con.execute(f"pragma index_list({q(table)})").fetchall():
        if not idx["unique"] or idx["partial"]:
            continue
        cols = [r["name"] for r in con.execute(f"pragma index_info({q(idx['name'])})")]
        if cols[:2] == ["gpu_name", "backend"]:
            return cols
    return None


def content_rows(con: sqlite3.Connection, table: str, cols: list[str]) -> Counter:
    sel = ", ".join(q(c) for c in cols)
    return Counter(tuple(r) for r in con.execute(f"select {sel} from {q(table)}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--gpu", default="NVIDIA GB300")
    ap.add_argument("--expect-new", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    before, after = connect(a.before), connect(a.after)
    problems: list[str] = []
    report: dict = {
        "before": {"path": str(a.before), "sha256": sha256(a.before)},
        "after": {"path": str(a.after), "sha256": sha256(a.after)},
        "gpu": a.gpu,
        "tables": {},
    }
    for name, con in (("before", before), ("after", after)):
        report[name]["integrity_check"] = con.execute("pragma integrity_check").fetchone()[0]
        if report[name]["integrity_check"] != "ok":
            problems.append(f"{name}: integrity_check {report[name]['integrity_check']}")

    tb, ta = tables(before), tables(after)
    if tb != ta:
        problems.append(f"table sets differ: {sorted(set(tb) ^ set(ta))}")

    per_gpu_before: Counter = Counter()
    per_gpu_after: Counter = Counter()
    added_total = 0
    added_run_keys: set = set()
    for table in ta:
        cols = [c for c in columns(after, table) if c != "id"]
        if table not in tb:
            continue
        if cols != [c for c in columns(before, table) if c != "id"]:
            problems.append(f"{table}: column lists differ")
            continue
        rb, ra = content_rows(before, table, cols), content_rows(after, table, cols)
        lost = rb - ra
        added = ra - rb
        entry = {
            "before": sum(rb.values()),
            "after": sum(ra.values()),
            "lost_or_changed": sum(lost.values()),
            "added": sum(added.values()),
        }
        if lost:
            problems.append(f"{table}: {sum(lost.values())} rows of `before` missing or changed")
        if table == RUN_TABLE:
            entry["added_run_keys"] = sorted(row[cols.index("run_key")] for row in added)
            report["tables"][table] = entry
            continue
        if "gpu_name" not in cols:
            if added:
                problems.append(f"{table}: {sum(added.values())} rows added to a non-GPU table")
            if added or lost:
                report["tables"][table] = entry
            continue

        gi = cols.index("gpu_name")
        for row, n in rb.items():
            per_gpu_before[row[gi]] += n
        for row, n in ra.items():
            per_gpu_after[row[gi]] += n
        foreign = sum(n for row, n in added.items() if row[gi] != a.gpu)
        if foreign:
            problems.append(f"{table}: {foreign} added rows are not {a.gpu}")

        key = semantic_key(after, table)
        if key is None:
            problems.append(f"{table}: no unique (gpu_name, backend, …) key")
        else:
            ksel = ", ".join(q(c) for c in key)
            dup = after.execute(
                f"select count(*) from (select 1 from {q(table)} group by {ksel} "
                "having count(*) > 1)"
            ).fetchone()[0]
            entry["duplicate_keys"] = dup
            if dup:
                problems.append(f"{table}: {dup} duplicate semantic keys")

        if added:
            ti = cols.index("time_ms")
            ri = cols.index("run_key")
            bi = cols.index("backend")
            bad = [
                row
                for row in added
                if not isinstance(row[ti], (int, float))
                or not math.isfinite(row[ti])
                or row[ti] <= 0
            ]
            entry["added_invalid_time_ms"] = len(bad)
            if bad:
                problems.append(f"{table}: {len(bad)} added rows with invalid time_ms")
            times = sorted(row[ti] for row in added)
            entry["added_time_ms_min"] = times[0]
            entry["added_time_ms_max"] = times[-1]
            entry["added_by_backend"] = dict(Counter(row[bi] for row in added))
            if "is_outlier" in cols:
                oi = cols.index("is_outlier")
                entry["added_outliers"] = sum(1 for row in added if row[oi])
            null_run = sum(1 for row in added if row[ri] is None)
            if null_run:
                problems.append(f"{table}: {null_run} added rows without a run_key")
            added_run_keys |= {row[ri] for row in added}
            added_total += sum(added.values())
        if added or lost or entry.get("duplicate_keys"):
            report["tables"][table] = entry

    run_entry = report["tables"].get(RUN_TABLE, {})
    known_runs = {r[0] for r in after.execute(f"select run_key from {RUN_TABLE}")}
    orphans = added_run_keys - known_runs
    if orphans:
        problems.append(f"{len(orphans)} added rows cite a run_key with no {RUN_TABLE} row")
    if set(run_entry.get("added_run_keys", [])) != added_run_keys:
        problems.append("added _profile_run rows are not exactly the runs the added rows cite")
    report["added_runs"] = [
        dict(r)
        for r in after.execute(
            f"select * from {RUN_TABLE} where run_key in "
            f"({','.join('?' * len(added_run_keys))}) order by id",
            sorted(added_run_keys),
        )
    ]

    report["rows_per_gpu"] = {
        g: {"before": per_gpu_before.get(g, 0), "after": per_gpu_after.get(g, 0)}
        for g in sorted(set(per_gpu_before) | set(per_gpu_after))
    }
    for g, c in report["rows_per_gpu"].items():
        if g != a.gpu and c["before"] != c["after"]:
            problems.append(f"{g}: row count changed {c['before']} -> {c['after']}")
    report["added_rows"] = added_total
    if a.expect_new is not None and added_total != a.expect_new:
        problems.append(f"added {added_total} rows, expected {a.expect_new}")
    report["problems"] = problems
    report["verdict"] = "PASS" if not problems else "FAIL"

    text = json.dumps(report, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
