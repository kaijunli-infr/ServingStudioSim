#!/usr/bin/env python3
"""Sample /proc for Nsight processes of this uid (exe under the Nsight install root, or comm matching) every --interval s.
Writes one JSON line per sample: time, and per process pid/ppid/pgid/sid/comm/exe/cmdline/state/ticks/io/fds(of interest)."""
import argparse, json, os, re, time
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--interval", type=float, default=0.1)
ap.add_argument("--max-s", type=float, default=120); ap.add_argument("--stop-file"); ap.add_argument("--extra-pid-file")
a = ap.parse_args(); uid = os.getuid(); t_end = time.time() + a.max_s
PAT = re.compile(r"nsys|Qdstrm|Importer|nsight", re.I)
def one(pid):
    try:
        if os.stat(f"/proc/{pid}").st_uid != uid: return None
        st = open(f"/proc/{pid}/stat").read(); comm = st[st.index("(") + 1:st.rindex(")")]; f = st[st.rindex(")") + 2:].split()
        try: exe = os.readlink(f"/proc/{pid}/exe")
        except OSError: exe = None
        return pid, comm, exe, f
    except (OSError, ValueError): return None
with open(a.out, "w") as out:
    while time.time() < t_end and not (a.stop_file and os.path.exists(a.stop_file)):
        now = time.time(); procs = []; extra = set()
        if a.extra_pid_file and os.path.exists(a.extra_pid_file):
            extra = {int(x) for x in open(a.extra_pid_file).read().split() if x.isdigit()}
        for d in os.listdir("/proc"):
            if not d.isdigit(): continue
            r = one(int(d))
            if not r: continue
            pid, comm, exe, f = r
            if not (PAT.search(comm) or (exe and "nsight" in exe) or pid in extra): continue
            rec = {"pid": pid, "comm": comm, "exe": exe, "state": f[0], "ppid": int(f[1]), "pgid": int(f[2]), "sid": int(f[3]),
                   "ticks": int(f[11]) + int(f[12]), "starttime": int(f[19])}
            try: rec["cmdline"] = open(f"/proc/{pid}/cmdline").read().replace("\0", " ").strip()[:400]
            except OSError: pass
            try: rec["io"] = {k.strip(): int(v) for k, v in (l.split(":") for l in open(f"/proc/{pid}/io"))}
            except (OSError, ValueError): rec["io"] = None
            try:
                rec["fds"] = sorted({os.readlink(f"/proc/{pid}/fd/{x}") for x in os.listdir(f"/proc/{pid}/fd")} & set()) or sorted(
                    t for t in {os.readlink(f"/proc/{pid}/fd/{x}") for x in os.listdir(f"/proc/{pid}/fd")} if re.search(r"qdstrm|nsys-rep|sqlite", t))
            except OSError: rec["fds"] = None
            try: rec["mem_kb"] = {l.split(":")[0]: int(l.split()[1]) for l in open(f"/proc/{pid}/status") if l.startswith(("VmRSS", "VmHWM"))}
            except (OSError, ValueError, IndexError): rec["mem_kb"] = None
            procs.append(rec)
        out.write(json.dumps({"at": now, "procs": procs}) + "\n"); out.flush()
        time.sleep(a.interval)
