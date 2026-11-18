#!/usr/bin/env python3
import json, sys, time
rows = [json.loads(l) for l in open(sys.argv[1])]; t0 = rows[0]["at"] if rows else 0
seen = {}
for r in rows:
    for p in r["procs"]:
        k = (p["pid"], p["starttime"]); s = seen.setdefault(k, {"first": r["at"], "p0": p, "samples": []}); s["last"] = r["at"]; s["pl"] = p
        s["samples"].append((r["at"], p["ticks"], (p.get("io") or {}).get("rchar"), (p.get("io") or {}).get("wchar"), p["state"]))
print(f"{len(rows)} samples over {rows[-1]['at']-t0:.1f} s; t0 = {time.strftime('%H:%M:%SZ', time.gmtime(t0))}")
for (pid, st), s in sorted(seen.items(), key=lambda kv: kv[1]["first"]):
    p0, pl = s["p0"], s["pl"]; io0, io1 = p0.get("io") or {}, pl.get("io") or {}
    print(f"pid {pid} comm={p0['comm']} ppid={p0['ppid']}→{pl['ppid']} pgid={p0['pgid']} sid={p0['sid']} seen +{s['first']-t0:.2f}..+{s['last']-t0:.2f} s ticks {p0['ticks']}→{pl['ticks']} rchar {io0.get('rchar')}→{io1.get('rchar')} wchar {io0.get('wchar')}→{io1.get('wchar')}")
    print(f"    exe={p0['exe']}\n    cmd={pl.get('cmdline','')[:330]}\n    fds={pl.get('fds')}")
if len(sys.argv) > 2:
    for (pid, st), s in seen.items():
        if sys.argv[2] in s["p0"]["comm"]:
            ss = s["samples"]; step = max(1, len(ss)//14)
            for at, tk, rc, wc, state in ss[::step] + [ss[-1]]: print(f"    +{at-t0:6.2f} s state {state} ticks {tk} rchar {rc} wchar {wc}")
