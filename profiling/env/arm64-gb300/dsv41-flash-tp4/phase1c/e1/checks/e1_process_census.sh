#!/usr/bin/env bash
# Process DISCOVERY on THIS tray (read-only; nothing is killed, nothing is classified as owned). Exit 1 if any candidate is alive.
#   bash checks/e1_process_census.sh            # PASS/FOUND lines + the candidate processes (pid, start, cmd)
# A match here is a *candidate* for review: command-line or cwd matches never establish ownership. Ownership comes only from
# the registry written by the driver / `e1_pids.py spawn` (pid + start time + boot id + uid), and termination is done only by
# `e1_pids.py cleanup --execute`, which re-verifies each identity immediately before signalling (see E1-B8-RECOVERY-PACKET.md §5).
# With E1_PID_REGISTRY set, the registry plan (owned / stale / refused / lookalike) is printed after the discovery.
# Covers what the Stage 1.5 closeout missed (three `tail -F` log watchers survived passes 1-3): not only the GPU-holding
# processes (vllm, nsys, session_runner) but every shell job the experiment spawns or an operator leaves behind: tmux
# sessions/servers, driver and launcher shells, log watchers (tail/less), nvidia-smi loops, sleep-based watchers, python
# monitors/checkers, and any process whose command line or cwd references the experiment directories. Patterns are written
# so this script's own command line never matches ([x] trick); its own pid, ancestors, own pipeline siblings (e.g. the invoking
# driver's `tee`) and the tmux session it runs in are excluded as well (2026-11-15 GO attempt 1: the driver's own tee and tmux
# session were flagged and the driver refused to start).
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
RUN_ROOT_RE='dsv41_flash_gb300_tp4_e1|phase1c/e1|20261114_dsv41_flash_gb300_tp4_e1'
me=$$; anc=""; p=$me; while [ "$p" != 1 ] && [ -n "$p" ]; do anc="$anc $p"; p=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' '); [ -z "$p" ] && break; done
# own pipeline: processes whose parent is this shell or one of its ancestors (e.g. the `tee preflight.log` of the driver that
# invoked us, or an operator's `| tee`); they belong to the invocation itself and are excluded from candidates, listed as INFO
mysid=$(sed 's/^.*) //' /proc/$me/stat | awk '{print $4}')   # session id: a tmux pane / ssh login shell is a session leader
# own invocation pipeline = processes whose parent is this process or one of its ancestors AND that share our session id
# (driver -> preflight -> `| tee`); never other panes of the same tmux server, never processes adopted by pid 1
sib=""; for d in /proc/[0-9]*; do x=${d#/proc/}; pp=$(awk '/^PPid:/{print $2}' $d/status 2>/dev/null) || continue; [ "$pp" = 1 ] && continue; case " $me $anc " in *" $pp "*) [ "$x" != "$me" ] && [ "$(sed 's/^.*) //' $d/stat 2>/dev/null | awk '{print $4}')" = "$mysid" ] && sib="$sib $x";; esac; done
[ -n "${sib// /}" ] && echo "INFO  own invocation pipeline (excluded from candidates):$(for x in $sib; do printf ' %s[%s]' $x "$(ps -o comm= -p $x 2>/dev/null)"; done)"
fail=0; found=""
report() { local label=$1 pids=$2; pids=$(echo "$pids" | tr ' ' '\n' | grep -v '^$' | sort -un | while read -r x; do case " $anc $sib " in *" $x "*) ;; *) echo $x;; esac; done | tr '\n' ' ')
  if [ -n "${pids// /}" ]; then echo "FOUND $label: pids $pids (candidate; ownership only via the registry)"; for x in $pids; do echo "      $(ps -o pid=,user=,lstart=,args= -p $x 2>/dev/null | cut -c1-200)"; done; fail=1; found="$found $pids"; else echo "PASS  $label: none"; fi; }
report "vllm server processes"            "$(pgrep -f 'vllm.entrypoints.cli.mai[n]|vllm.entrypoints.openai.api_serve[r]|VLLM::EngineCor[e]')"
report "nsys profiler sessions"           "$(pgrep -f 'nsys profil[e]|nsys expor[t]|nsys stat[s]')"
report "req-frontend session_runner"      "$(pgrep -f 'session_runne[r]')"
report "launcher (alignment profile)"     "$(pgrep -f 'launcher alignment profil[e]')"
report "E1 driver / preflight / postrun shells" "$(pgrep -f 'run_e1[_a-z0-9]*\.s[h]|preflight_e1[_a-z0-9]*\.s[h]|postrun_e1[_a-z0-9]*\.s[h]|postpass_e1[_a-z0-9]*\.s[h]|collect_e1[_a-z0-9]*\.s[h]')"
report "log watchers (tail -F/-f, less +F) on experiment logs" "$(pgrep -f "tail .*(-F|-f|--follow).*($RUN_ROOT_RE)|less \+[F].*($RUN_ROOT_RE)")"
report "nvidia-smi loops / watch"         "$(pgrep -f 'nvidia-smi.* (-l|--loop|-lms|--loop-ms)[= ]|watc[h] .*nvidia-smi|watc[h] .*e1_')"
report "python monitors/checkers of the experiment" "$(pgrep -f "pytho[n][0-9.]* .*($RUN_ROOT_RE)")"
report "other processes whose cmdline references the experiment dirs" "$(for d in /proc/[0-9]*; do x=${d#/proc/}; [ "$x" = "$me" ] && continue; c=$(tr '\0' ' ' < $d/cmdline 2>/dev/null) || continue; case "$c" in *e1_process_census*) continue;; esac; echo "$c" | grep -Eq "$RUN_ROOT_RE" && echo $x; done | tr '\n' ' ')"
report "processes whose cwd is inside the experiment dirs" "$(for d in /proc/[0-9]*; do x=${d#/proc/}; w=$(readlink $d/cwd 2>/dev/null) || continue; echo "$w" | grep -Eq "$RUN_ROOT_RE" && echo $x; done | tr '\n' ' ')"
# tmux: sessions on any server socket of this user
# tmux: experiment-named sessions (e1-*, e15-*, e1b8-*) are a FAIL; other users' sessions on a shared tray are reported only
# the tmux session this very invocation runs in (its pane pid is one of our ancestors, e.g. the driver's own session) is ours
own_tm=$(tmux list-panes -a -F '#{session_name} #{pane_pid}' 2>/dev/null | while read -r sn pp; do case " $anc " in *" $pp "*) echo $sn;; esac; done | sort -u | tr '\n' ' ')
[ -n "${own_tm// /}" ] && echo "INFO  own tmux session (excluded): $own_tm"
tm=$(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' '); tme=$(echo "$tm" | tr ' ' '\n' | grep -E '^(e1|e15|e1b8)[-_]' | while read -r sn; do case " $own_tm " in *" $sn "*) ;; *) echo $sn;; esac; done | tr '\n' ' ')
if [ -n "${tme// /}" ]; then echo "FOUND experiment tmux sessions alive: $tme (closed by terminating the registered driver tree; never killed by name)"; fail=1; else echo "PASS  experiment tmux sessions (e1-*/e15-*/e1b8-*) other than our own: none"; fi
[ -n "${tm// /}" ] && echo "INFO  other tmux sessions on this tray (not experiment-owned): $tm"
# GPUs
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$apps" ]; then echo "FOUND GPU compute apps:"; echo "$apps" | sed 's/^/      /'; fail=1; else echo "PASS  GPU compute apps: none"; fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
echo "INFO  GPU memory used MiB per GPU: $used"
if [ -n "${E1_PID_REGISTRY:-}" ]; then echo "## registry plan ($E1_PID_REGISTRY)"; python3 -I "$(dirname "${BASH_SOURCE[0]}")/e1_pids.py" plan --registry "$E1_PID_REGISTRY" | sed 's/^/  /'; fi
echo "# process discovery on $(hostname) at $(date -u +%FT%TZ): $([ $fail = 0 ] && echo CLEAN || echo "CANDIDATES FOUND (pids:$found) -> review against the registry; nothing was signalled")"
exit $fail
