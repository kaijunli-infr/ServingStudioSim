"""Check access to the pinned Llama 3.1 8B snapshot without downloading weights (CPU-only).

    python hf_access_check.py --out hf-access.json

Reads repository metadata from the Hugging Face Hub and asks whether the
current credentials may download from the gated repository. It never reads,
prints or stores a token: authentication is whatever `huggingface_hub` finds
on its own, and only the outcome is reported. No file is downloaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import huggingface_hub
    from huggingface_hub import HfApi

    api = HfApi()
    report: dict = {"repo": REPO, "pinned_revision": REVISION, "hub": huggingface_hub.__version__}
    try:
        api.whoami()
        report["authenticated"] = True
    except Exception as exc:  # noqa: BLE001 - only the outcome class is recorded
        report["authenticated"] = False
        report["authentication_error"] = type(exc).__name__
    info = api.model_info(REPO, revision=REVISION, files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings}
    weights = {name: size for name, size in files.items() if name.endswith(".safetensors")}
    report |= {
        "resolved_revision": info.sha,
        "gated": info.gated,
        "weight_files": weights,
        "weight_bytes": sum(size or 0 for size in weights.values()),
        "tokenizer_files": sorted(n for n in files if "token" in n and "/" not in n),
        "config_files": sorted(n for n in files if n.endswith("config.json")),
    }
    try:
        api.auth_check(REPO)
        report["download_access"] = True
    except Exception as exc:  # noqa: BLE001
        report["download_access"] = False
        report["download_access_error"] = type(exc).__name__
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "weight_files"}))
    return 0 if report["download_access"] and report["resolved_revision"] == REVISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
