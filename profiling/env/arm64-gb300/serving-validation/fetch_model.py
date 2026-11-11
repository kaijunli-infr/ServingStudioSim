"""Download and verify the pinned Llama 3.1 8B Instruct snapshot (CPU-only).

    python fetch_model.py --cache DIR/hub --out report.json

Downloads only what serving needs from revision `REVISION`: the top-level
safetensors shards and their index, `config.json`, `generation_config.json`
and the tokenizer files. The `original/` checkpoint is skipped. The
credential is whatever `huggingface_hub` finds by itself; it is never read,
printed or stored here. Every downloaded file is then checked against the
hub's metadata for the same revision: size for all files, the LFS sha256 for
the weights, and the git blob id for the config and tokenizer files. The exit
status is non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
PATTERNS = [
    "model-*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        print(f"revision resolves to {info.sha}, not {REVISION}")
        return 2
    path = Path(
        snapshot_download(REPO, revision=REVISION, cache_dir=args.cache, allow_patterns=PATTERNS)
    )
    expected = {s.rfilename: s for s in info.siblings}
    files, problems = {}, []
    for local in sorted(p for p in path.iterdir() if p.is_file()):
        meta = expected.get(local.name)
        if meta is None:
            problems.append(f"{local.name}: not in the revision")
            continue
        entry = {"bytes": local.stat().st_size, "expected_bytes": meta.size}
        if meta.lfs is not None:
            entry["sha256"] = sha256(local)
            entry["expected_sha256"] = meta.lfs.sha256
            if entry["sha256"] != entry["expected_sha256"]:
                problems.append(f"{local.name}: sha256 mismatch")
        else:
            # Plain git files (config, tokenizer): the hub records their git blob id.
            data = local.read_bytes()
            entry["git_blob_sha1"] = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            entry["expected_git_blob_sha1"] = meta.blob_id
            if entry["git_blob_sha1"] != entry["expected_git_blob_sha1"]:
                problems.append(f"{local.name}: git blob mismatch")
        if entry["bytes"] != entry["expected_bytes"]:
            problems.append(f"{local.name}: size mismatch")
        files[local.name] = entry
    wanted = [n for n in expected if "/" not in n and any(Path(n).match(p) for p in PATTERNS)]
    problems += [f"{n}: not downloaded" for n in wanted if n not in files]
    report = {
        "repo": REPO,
        "revision": REVISION,
        "snapshot_dir": str(path),
        "snapshot_dir_revision": path.name,
        "files": files,
        "weight_bytes": sum(v["bytes"] for k, v in files.items() if k.endswith(".safetensors")),
        "problems": problems,
        "pass": not problems and path.name == REVISION,
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("snapshot_dir_revision", "weight_bytes", "problems", "pass")}
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
