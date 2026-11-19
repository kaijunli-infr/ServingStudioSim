#!/usr/bin/env python3
"""Persistent storage of a run including its archive, against the registered ceiling.

    python3 -I checks/e1_archive_size.py --run <run root> --archive <archive dir> --ceiling-gb 38

Sums file sizes (each inode once) under both directories, in GB (1e9 B). Exit 0 if the sum is at or below the ceiling,
1 if above, 3 if a directory is missing. Stdlib only; read-only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import os
import sys
from pathlib import Path


def size(root: Path) -> int:
    seen, total = set(), 0
    for d, _dirs, names in os.walk(root):
        for n in names:
            st = os.stat(os.path.join(d, n), follow_symlinks=False)
            if (st.st_dev, st.st_ino) not in seen:
                seen.add((st.st_dev, st.st_ino))
                total += st.st_size
    return total


def check(run: Path, archive: Path, ceiling_gb: float) -> tuple[int, str]:
    if not run.is_dir() or not archive.is_dir():
        return 3, f"INCONCLUSIVE run root or archive directory missing ({run}, {archive})"
    r, a = size(run), size(archive)
    total = (r + a) / 1e9
    return (0 if total <= ceiling_gb else 1), f"INFO  run root {r / 1e9:.3f} GB + archive {a / 1e9:.3f} GB = {total:.3f} GB (ceiling {ceiling_gb:g})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--ceiling-gb", required=True, type=float)
    args = ap.parse_args(argv)
    rc, text = check(args.run, args.archive, args.ceiling_gb)
    print(text)
    return rc


if __name__ == "__main__":
    sys.exit(main())
