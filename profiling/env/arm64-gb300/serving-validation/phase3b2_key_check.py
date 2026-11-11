#!/usr/bin/env python3
"""Phase 3B-2 preservation gate (CPU only, read-only): every stored row keeps its key.

    python phase3b2_key_check.py PROFILE_DB [--out report.json]

For every registered kind's table in PROFILE_DB, rebuild each row's args from
its stored columns through the current schema (``coerce_args``, which also
canonicalises optional args), recompute ``args_hash`` with the current
``Table`` and compare it with the stored one. A schema change that moved any
existing key (a new required field, an optional field whose absent value does
not match what the rows were measured with) shows up as a mismatch or a
coercion error. Also lists every (kind, backend) the registry holds per table,
so two timing methods sharing one backend name would show. Opens the DB
read-only; writes nothing to it.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import fields

from profiling.db.args import field_types, optional_fields
from profiling.db.batch import coerce_args
from profiling.db.registry import iter_kernel_profiler_specs
from profiling.db.table import Table


def _value(raw, annotation):
    origin = getattr(annotation, "__origin__", None)
    if origin in (tuple, list) and isinstance(raw, str):
        return json.loads(raw)
    return raw


def main() -> int:
    db = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    by_table = defaultdict(list)
    for spec in iter_kernel_profiler_specs():
        by_table[spec.table_name].append(spec)
    report = {"db": db, "tables": {}, "backends_per_table": {}}
    failures = 0
    for name, specs in sorted(by_table.items()):
        report["backends_per_table"][name] = sorted(s.backend for s in specs)
        if name not in tables:
            continue
        spec = specs[0]
        table = Table(spec, db)
        types = field_types(spec.args_schema)
        optional = optional_fields(spec.args_schema)
        present = {r[1] for r in conn.execute(f"pragma table_info({name})")}
        columns = [f.name for f in fields(spec.args_schema) if f.name in present]
        rows = conn.execute(
            f"select backend, gpu_name, args_hash, {', '.join(columns)} from {name}"
        )
        stats = {"rows": 0, "match": 0, "mismatch": 0, "error": 0, "examples": []}
        for row in rows:
            stats["rows"] += 1
            values = {
                c: _value(row[c], types[c])
                for c in columns
                if not (c in optional and row[c] is None)
            }
            try:
                args = coerce_args(spec.args_schema, values)
                ok = table.args_hash(table.db_key(args)) == bytes(row["args_hash"])
            except Exception as exc:  # noqa: BLE001 - reported per row
                stats["error"] += 1
                if len(stats["examples"]) < 3:
                    stats["examples"].append(f"{row['backend']}: {exc}")
                continue
            stats["match" if ok else "mismatch"] += 1
            if not ok and len(stats["examples"]) < 3:
                stats["examples"].append(f"{row['backend']}/{row['gpu_name']}: {values}")
        failures += stats["mismatch"] + stats["error"]
        report["tables"][name] = stats
    report["total_rows"] = sum(t["rows"] for t in report["tables"].values())
    report["failures"] = failures
    text = json.dumps(report, indent=1)
    if out:
        with open(out, "w") as fh:
            fh.write(text + "\n")
    print(f"tables {len(report['tables'])} rows {report['total_rows']} failures {failures}")
    for name in (
        "multi_add_rms_norm",
        "kv_cache_append",
        "paged_attn_batch_decode",
        "paged_attn_batch_prefill",
        "silu_and_mul",
        "gqa_rope_neox",
        "embedding_rms_norm",
    ):
        t = report["tables"].get(name)
        print(
            f"  {name}: {t if t is None else {k: v for k, v in t.items() if k != 'examples'}}"
            f" backends={report['backends_per_table'].get(name)}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
