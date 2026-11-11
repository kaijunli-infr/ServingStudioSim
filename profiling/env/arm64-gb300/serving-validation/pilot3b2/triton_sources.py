#!/usr/bin/env python3
"""Extract Inductor-generated Triton kernels from a cache directory (CPU-only).

    triton_sources.py CACHE_DIR OUT_JSON [name-substring ...]

For each `NAME = async_compile.triton('NAME', '''SRC''', ...)` in the cache's
generated .py files: the kernel source with the name normalized (`body_sha`
hashes the `def` and body only, `meta` is the decorator block: size_hints,
triton_meta signature, inductor_meta), and the `NAME.run(...)` call lines of the
same file (arguments show buffers and xnumel). Deduplicated by (name, body_sha, meta).
"""
import hashlib, json, re, sys
from pathlib import Path

root, out, wanted = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
DEF = re.compile(r"(\w+) = async_compile\.triton\('(\w+)', '''(.*?)'''", re.S)
VOLATILE = re.compile(r"'(backend_hash|kernel_num_gb|kernel_flop|tiling_scores)': [^,}]*(\{[^}]*\})?,? ?")
seen = {}
for f in sorted(root.rglob("*.py")):
    text = f.read_text(errors="replace")
    for m in DEF.finditer(text):
        name, src = m.group(2), m.group(3)
        if wanted and not any(w in name for w in wanted):
            continue
        cut = src.find("\ndef ")
        meta, body = src[:cut], src[cut + 1:]
        body_n = body.replace(name, "KERNEL")
        meta_n = VOLATILE.sub("", meta.replace(name, "KERNEL"))
        runs = [l.strip() for l in text.splitlines() if f"{name}.run(" in l]
        key = (name, hashlib.sha256(body_n.encode()).hexdigest()[:16],
               hashlib.sha256(meta_n.encode()).hexdigest()[:16])
        e = seen.setdefault(key, {"name": name, "body_sha": key[1], "meta_sha": key[2],
                                  "meta": meta, "body": body, "runs": set(), "files": []})
        e["runs"].update(runs)
        e["files"].append(str(f.relative_to(root)))
res = []
for e in sorted(seen.values(), key=lambda d: (d["name"], d["body_sha"], d["meta_sha"])):
    e["runs"] = sorted(e["runs"])
    res.append(e)
out.write_text(json.dumps(res, indent=1) + "\n")
for d in res:
    print(d["name"], d["body_sha"], d["meta_sha"], len(d["files"]), d["runs"][:2])
