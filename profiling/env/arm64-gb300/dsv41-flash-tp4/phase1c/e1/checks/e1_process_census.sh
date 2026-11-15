#!/usr/bin/env bash
# Experiment-owned process census on THIS tray (read-only; nothing is killed). Exit 1 if anything experiment-owned is alive.
#   bash checks/e1_process_census.sh            # PASS/FAIL lines + the offending processes (pid, start, cmd)
# Covers what the Stage 1.5 closeout missed (three `tail -F` log watchers survived passes 1-3): not only the GPU-holding
# processes (vllm, nsys, session_runner) but every shell job the experiment spawns or an operator leaves behind: tmux
# sessions/servers, driver and launcher shells, log watchers (tail/less), nvidia-smi loops, sleep-based watchers, python
# monitors/checkers, and any process whose command line or cwd references the experiment directories. Patterns are written
# so this script's own command line never matches ([x] trick); its own pid and ancestors are excluded as well.
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
RUN_ROOT_RE='dsv41_flash_gb300_tp4_e1|phase1c/e1|20261114_dsv41_flash_gb300_tp4_e1'
me=$$; anc=""; p=$me; while [ "$p" != 1 ] && [ -n "$p" ]; do anc="$anc $p"; p=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' '); [ -z "$p" ] && break; done
fail=0; found=""
report() { local label=$1 pids=$2; pids=$(echo "$pids" | tr ' ' '\n' | grep -v '^$' | sort -un | while read -r x; do case " $anc " in *" $x "*) ;; *) echo $x;; esac; done | tr '\n' ' ')
  if [ -n "${pids// /}" ]; then echo "FAIL  $label: pids $pids"; for x in $pids; do echo "      $(ps -o pid=,user=,lstart=,args= -p $x 2>/dev/null | cut -c1-200)"; done; fail=1; found="$found $pids"; else echo "PASS  $label: none"; fi; }
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
tm=$(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' '); tme=$(echo "$tm" | tr ' ' '\n' | grep -E '^(e1|e15|e1b8)[-_]' | tr '\n' ' ')
if [ -n "${tme// /}" ]; then echo "FAIL  experiment tmux sessions alive: $tme"; fail=1; else echo "PASS  experiment tmux sessions (e1-*/e15-*/e1b8-*): none"; fi
[ -n "${tm// /}" ] && echo "INFO  other tmux sessions on this tray (not experiment-owned): $tm"
# GPUs
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$apps" ]; then echo "FAIL  GPU compute apps:"; echo "$apps" | sed 's/^/      /'; fail=1; else echo "PASS  GPU compute apps: none"; fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
echo "INFO  GPU memory used MiB per GPU: $used"
echo "# process census on $(hostname) at $(date -u +%FT%TZ): $([ $fail = 0 ] && echo CLEAN || echo "NOT CLEAN (pids:$found)")"
exit $fail
