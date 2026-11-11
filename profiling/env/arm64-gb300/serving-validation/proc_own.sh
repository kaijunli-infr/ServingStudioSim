# Ownership of processes for run_serving_profile.sh (sourced, not executed).
#
# A process belongs to the run when it descends from the wrapper's own shell
# (OWN_ROOT), whatever session or process group it chose: the launcher starts
# Nsight Systems and vLLM in sessions of their own. Every descendant seen is
# recorded with its kernel start time in OWN_FILE, so it stays identified after
# its parent dies and it is re-parented, and a reused PID is never mistaken
# for it. Nsight Systems' agent detaches at once; it is recognised by the
# session name its `nsys profile` parent gives it.

own_tree() { # print the PIDs descending from OWN_ROOT
  ps -eo pid=,ppid= | awk -v root="$OWN_ROOT" '
    { parent[$1] = $2 }
    END { for (p in parent) { q = p; while (q > 1) { if (parent[q] == root) { print p; break } q = parent[q] } } }'
}

start_time() { # start_time PID -> kernel start time in clock ticks, empty if gone
  [ -r "/proc/$1/stat" ] || return 0
  sed 's/^.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'
}

record_own() {
  local pid st
  for pid in $(own_tree); do
    st="$(start_time "$pid")"
    [ -n "$st" ] && echo "$pid $st"
  done >>"$OWN_FILE"
  sort -u -o "$OWN_FILE" "$OWN_FILE"
}

is_own() { # is_own PID -> true if recorded with the same start time, or a live descendant
  local pid="$1" st
  st="$(start_time "$pid")"
  [ -n "$st" ] && grep -qx "$pid $st" "$OWN_FILE" 2>/dev/null && return 0
  own_tree | grep -qx "$pid"
}

live_own() { # print recorded PIDs still alive with their recorded start time, plus nsys agents
  local pid st nsys
  while read -r pid st; do
    [ "$(start_time "$pid")" = "$st" ] && echo "$pid"
  done <"$OWN_FILE"
  for nsys in $(while read -r pid st; do
                  cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' | grep -q "nsys profile" && echo "$pid"
                done <"$OWN_FILE"); do
    pgrep -u "$(id -u)" -f "[-]-session-name profile-$nsys( |$)" 2>/dev/null
  done
}
