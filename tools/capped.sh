#!/bin/bash
# Run a command under a resident-memory watchdog: poll the RSS of the whole process
# tree every 0.2 s and kill it (SIGKILL) once it exceeds CAP_MB (default 8000).
# Optional TIME_S: wall-clock limit. (ulimit -v is unsuitable for the GPU path:
# the NVIDIA OpenCL driver reserves huge virtual ranges while real use is small.)
# usage: CAP_MB=8000 TIME_S=1800 tools/capped.sh <command...>
CAP_MB=${CAP_MB:-8000}
TIME_S=${TIME_S:-0}
"$@" &
pid=$!
peak=0
t0=$(date +%s)
tree_rss() {   # MB over the process and all descendants
    local pids=$1 kids=$1
    while [ -n "$kids" ]; do
        kids=$(pgrep -d, -P "$kids" 2>/dev/null)
        [ -n "$kids" ] && pids="$pids,$kids"
    done
    ps -o rss= -p "$pids" 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}'
}
while kill -0 $pid 2>/dev/null; do
    rss=$(tree_rss $pid)
    [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
    if [ -n "$rss" ] && [ "$rss" -gt "$CAP_MB" ]; then
        echo "[capped] RSS ${rss} MB > cap ${CAP_MB} MB -> killing $pid" >&2
        pkill -9 -P $pid 2>/dev/null; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; exit 137
    fi
    if [ "$TIME_S" -gt 0 ] && [ $(( $(date +%s) - t0 )) -gt "$TIME_S" ]; then
        echo "[capped] wall time > ${TIME_S} s -> killing $pid" >&2
        pkill -9 -P $pid 2>/dev/null; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; exit 124
    fi
    sleep 0.2
done
wait $pid; rc=$?
echo "[capped] peak RSS ${peak} MB" >&2
exit $rc
