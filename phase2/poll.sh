#!/usr/bin/env bash
# Laptop-side progress poll for the extended batches. Every 15 minutes it
# appends one status line to data/live/progress.log: instance hours since
# this start, runs exported, impact counts, last line of the batch log.
# It exits (so whoever started it gets woken) when a batch completes, an
# export fails, everything is done, the log stops moving for 40 minutes, or
# the instance cannot be reached for two hours (24 tries, 5 minutes apart).
#   ./phase2/poll.sh START_UNIX
cd "$(dirname "$0")/.."
START=${1:?instance start time, unix seconds}
LOG=data/live/progress.log
mkdir -p data/live
# grep -c prints 0 AND exits 1 when nothing matches, so "|| echo 0" would give
# two lines; keep only the first number, default 0.
done_before=$(./phase2/ssh2.sh 'grep -c "BATCH .* COMPLETE" phase2/batches.log' 2>/dev/null | head -1)
done_before=${done_before:-0}
fails=0
while true; do
  out=$(./phase2/ssh2.sh 'cd phase2; n=$(ls batches/*/*/*/traces.csv 2>/dev/null | wc -l); t=$(grep -c "impact=true" batches.log); f=$(grep -c "impact=false" batches.log); z=$(grep -c "impact=null" batches.log); e=$(grep -c "EXPORT FAILED" batches.log); c=$(grep -c "BATCH .* COMPLETE" batches.log); a=$(grep -c "ALL BATCHES DONE" batches.log); age=$(( $(date +%s) - $(stat -c %Y batches.log) )); echo "$n $t $f $z $e $c $a $age | $(grep "run \|BATCH" batches.log | tail -1)"' 2>/dev/null)
  if [ -z "$out" ]; then
    fails=$((fails + 1)); echo "[$(date -u +%FT%TZ)] could not reach the instance ($fails)" >> "$LOG"
    [ "$fails" -ge 24 ] && { echo "UNREACHABLE"; exit 1; }
    sleep 300; continue
  fi
  fails=0
  read -r n t f z e c a age rest <<< "$out"
  hours=$(python3 -c "import time;print(round((time.time()-$START)/3600,2))")
  echo "[$(date -u +%FT%TZ)] instance hours this start $hours | runs exported $n | impact true $t false $f none $z | export failures $e | batches complete $c $rest" >> "$LOG"
  [ "$a" -ge 1 ] && { echo "ALL DONE"; tail -1 "$LOG"; exit 0; }
  [ "$e" -ge 1 ] && { echo "EXPORT FAILED"; tail -1 "$LOG"; exit 2; }
  [ "$c" -gt "$done_before" ] && { echo "BATCH COMPLETE"; tail -1 "$LOG"; exit 0; }
  [ "$age" -gt 2400 ] && { echo "STALLED"; tail -1 "$LOG"; exit 3; }
  sleep 900
done
