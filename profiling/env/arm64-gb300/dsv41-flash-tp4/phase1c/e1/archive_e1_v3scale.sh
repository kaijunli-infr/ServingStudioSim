#!/usr/bin/env bash
# Archive and archive verification after the V3 post-run, ON the capture tray as dc:
#   bash archive_e1_v3scale.sh <operator log dir> [archive dir]
# Copies the run root, the tray's preserved stream with its retention record and the operator's logs; writes SHA256SUMS over every
# file; verifies the copy of the run root against the source file by file; re-reads the archived stream and compares its complete
# SHA-256 with the controller's final record; makes the archive read-only; measures run root + archive against the 16 GB ceiling.
# Refuses to overwrite an existing archive. Removes nothing. Exit 1 on any FAIL.
set -u
export HOME=/mnt/shared/home/dc; S=$HOME/workspace/serving-studio
RUN=${E1_V3S_RUN:-$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v3scale}; PASS=e1_v3_scale
OP=${1:?operator log directory}; A=${2:-$S/artifacts/phase1c-e1-v3scale-dsv41-flash-gb300-tp4-$(date -u +%Y%m%d)}
TP=$(python3 -I -c "import json,os,sys; print(os.path.dirname((json.load(open(sys.argv[1])).get('stream') or {}).get('preserved_path') or ''))" $RUN/$PASS/profile/nsys/lifecycle.json 2>/dev/null)   # the directory the controller preserved the stream in
fail=0; ok() { echo "PASS  $*"; }; bad() { echo "FAIL  $*"; fail=1; }
echo "# archive $PASS on $(hostname) at $(date -u +%FT%TZ) -> $A"
[ -e "$A" ] && { bad "archive directory exists: not touching it"; exit 1; }
[ -d "$RUN/$PASS" ] || { bad "no run directory $RUN/$PASS"; exit 1; }
mkdir -p $A/run $A/preserved $A/operator
cp -a $RUN/. $A/run/ && ok "run root copied" || bad "copy of the run root"
[ -n "$TP" ] && cp -p $TP/* $A/preserved/ 2>/dev/null && ok "tray-preserved stream and retention record copied from $TP" || bad "tray-preserved stream not found (lifecycle preserved_path directory: '${TP}')"
cp -p $OP/* $A/operator/ 2>/dev/null; echo "INFO  operator logs: $(ls $A/operator | wc -l) file(s)"
( cd $A && find run preserved operator -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
( cd $RUN && find . -type f -print0 | sort -z | xargs -0 sha256sum ) | sed 's#  \./#  run/#' > $A/.src.sha
grep "  run/" $A/SHA256SUMS | diff -q - $A/.src.sha >/dev/null && ok "archive copy of the run root equals the source, file by file ($(grep -c '  run/' $A/SHA256SUMS) files)" || bad "archive copy differs from the source"
rm -f $A/.src.sha
want=$(python3 -I -c "import json,sys; print(((json.load(open(sys.argv[1])).get('stream') or {}).get('final') or {}).get('sha256'))" $RUN/$PASS/profile/nsys/lifecycle.json 2>/dev/null)
n=0; for f in $A/preserved/*.qdstrm $A/run/$PASS/preserved/*.qdstrm; do [ -f "$f" ] || continue; n=$((n+1)); h=$(sha256sum $f | cut -d' ' -f1); [ "$h" = "$want" ] && ok "archived stream ${f#$A/}: sha256 $h = the controller's final record" || bad "archived stream ${f#$A/}: sha256 $h != final record $want"; done
[ $n -ge 2 ] || bad "expected the stream twice in the archive (tray original and durable copy), found $n"
( cd $A && sha256sum -c SHA256SUMS --quiet ) && ok "SHA256SUMS verifies ($(wc -l < $A/SHA256SUMS) files; its own sha256 $(sha256sum $A/SHA256SUMS | cut -d' ' -f1))" || bad "SHA256SUMS does not verify"
chmod -R a-w $A
python3 -I - "$RUN" "$A" <<'PY' && ok "persistent storage including the archive within 16 GB" || bad "persistent storage including the archive exceeds 16 GB"
import os, sys
def size(root):
    seen, total = set(), 0
    for d, _dirs, names in os.walk(root):
        for n in names:
            st = os.stat(os.path.join(d, n), follow_symlinks=False)
            if (st.st_dev, st.st_ino) not in seen:
                seen.add((st.st_dev, st.st_ino)); total += st.st_size
    return total
run, arc = size(sys.argv[1]), size(sys.argv[2])
print(f"INFO  run root {run / 1e9:.3f} GB + archive {arc / 1e9:.3f} GB = {(run + arc) / 1e9:.3f} GB (ceiling 16)")
sys.exit(0 if (run + arc) / 1e9 <= 16 else 1)
PY
echo "INFO  nothing was removed; the tray original stays in $TP"
echo "# archive result: $([ $fail = 0 ] && echo VERIFIED || echo NOT VERIFIED)"
exit $fail
