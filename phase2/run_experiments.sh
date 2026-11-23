#!/usr/bin/env bash
# Runs a batch of fault injections. Each run: normal operation, then a fault
# for FAULT_S seconds at a RANDOM offset, then recovery. After every run the
# telemetry is exported and the impact check is run before the next run
# starts, so nothing depends on Jaeger keeping old traces in memory.
#
# Usage:  ./run_experiments.sh runs/                 # default matrix
#         SERVICES="cartservice" FAULTS="cpu loss" ./run_experiments.sh runs/
#         ./run_experiments.sh runs/ --none 20       # 20 no-fault runs instead
# A run whose folder already has traces.csv is skipped, so a batch that
# stopped half way can be started again with the same command.
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}

OUT=${1:-runs}
NONE_N=0
if [ "${2:-}" = "--none" ]; then NONE_N=${3:?--none needs a count}; fi
SERVICES=${SERVICES:-"checkoutservice currencyservice emailservice productcatalogservice recommendationservice cartservice paymentservice shippingservice frontend"}
FAULTS=${FAULTS:-"cpu mem disk socket delay loss"}
REPS=${REPS:-1}
RUN_S=${RUN_S:-900}        # total run length (15 min)
FAULT_S=${FAULT_S:-300}    # fault duration (5 min)
MIN_PRE=${MIN_PRE:-420}    # fault starts no earlier than this (420 = 120 s history + one 300 s window before the fault)
MIN_POST=${MIN_POST:-120}  # and leaves at least this much time after it ends
SETTLE_S=${SETTLE_S:-60}   # pause between runs
HERE=$(cd "$(dirname "$0")" && pwd)
STATE="$HERE/state.json"   # written by set_monitoring.sh / set_traffic.sh / set_limits.sh
mkdir -p "$OUT"

# rate() window for the export: at least 4 scrapes, never below 30s.
SCRAPE_S=$(python3 -c "import json,sys;print(int(json.load(open('$STATE')).get('scrape','15s').rstrip('s')))" 2>/dev/null || echo 15)
export RATE_WINDOW="$(( SCRAPE_S * 4 > 30 ? SCRAPE_S * 4 : 30 ))s"
# Trace sampling ratio, applied at export (see export_telemetry.py for why).
export TRACE_SAMPLE=$(python3 -c "import json;print(json.load(open('$STATE')).get('sample','1.0'))" 2>/dev/null || echo 1.0)

python3 - "$OUT/settings.json" "$STATE" <<PY
import json, os, sys, time
state = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {}
json.dump({"written_unix": int(time.time()), "SERVICES": "$SERVICES", "FAULTS": "$FAULTS",
           "REPS": $REPS, "RUN_S": $RUN_S, "FAULT_S": $FAULT_S, "MIN_PRE": $MIN_PRE,
           "MIN_POST": $MIN_POST, "SETTLE_S": $SETTLE_S, "NONE_N": $NONE_N,
           "RATE_WINDOW": "$RATE_WINDOW", "TRACE_SAMPLE_applied_at_export": "$TRACE_SAMPLE", "cluster_state": state}, open(sys.argv[1], "w"), indent=1)
PY

# DNS fault: names the target service looks up for its downstream calls.
dns_patterns() {
  case "$1" in
    frontend) echo '"adservice.*", "cartservice.*", "checkoutservice.*", "currencyservice.*", "productcatalogservice.*", "recommendationservice.*", "shippingservice.*"' ;;
    checkoutservice) echo '"cartservice.*", "currencyservice.*", "emailservice.*", "paymentservice.*", "productcatalogservice.*", "shippingservice.*"' ;;
    recommendationservice) echo '"productcatalogservice.*"' ;;
    cartservice) echo '"redis-cart.*"' ;;
    *) echo '"*"' ;;   # leaf service: no downstream calls, so every name
  esac
}

# The export reads Prometheus, Jaeger and Loki through local port-forwards;
# restart any that died during a long batch.
ensure_pf() {
  curl -sf -o /dev/null localhost:9090/-/ready || { nohup kubectl port-forward -n monitoring svc/kps-kube-prometheus-stack-prometheus 9090:9090 >/dev/null 2>&1 & sleep 3; }
  curl -sf -o /dev/null localhost:16686/ || { nohup kubectl port-forward -n monitoring svc/jaeger-query 16686:16686 >/dev/null 2>&1 & sleep 3; }
  curl -sf -o /dev/null localhost:3100/ready || { nohup kubectl port-forward -n monitoring svc/loki 3100:3100 >/dev/null 2>&1 & sleep 3; }
}

export_and_check() {
  ensure_pf
  if python3 "$HERE/export_telemetry.py" "$1"; then
    echo "[$(date +%T)] $(python3 "$HERE/impact_check.py" "$1")"
  else
    echo "[$(date +%T)] EXPORT FAILED $1"
  fi
}

if [ "$NONE_N" -gt 0 ]; then
  for i in $(seq 1 "$NONE_N"); do
    dir="$OUT/none_none/$i"
    [ -s "$dir/traces.csv" ] && { echo "skip $dir (already exported)"; continue; }
    mkdir -p "$dir"
    run_start=$(date +%s)
    echo "[$(date +%T)] run none-$i: no fault (run ${RUN_S}s)"
    sleep "$RUN_S"
    echo 0 > "$dir/inject_time.txt"
    echo "{\"service\": \"none\", \"fault\": \"none\", \"rep\": $i, \"run_start\": $run_start, \"inject_time\": 0, \"fault_end\": 0, \"run_end\": $(date +%s)}" > "$dir/run.json"
    export_and_check "$dir"
    sleep "$SETTLE_S"
  done
  echo "all runs done -> $OUT"
  exit 0
fi

for rep in $(seq 1 "$REPS"); do
for svc in $SERVICES; do
for fault in $FAULTS; do
  name="${svc}-${fault}-r${rep}-$(date +%s)"
  dir="$OUT/${svc}_${fault}/${rep}"
  [ -s "$dir/traces.csv" ] && { echo "skip $dir (already exported)"; continue; }
  mkdir -p "$dir"
  max_pre=$(( RUN_S - FAULT_S - MIN_POST ))
  pre=$(( MIN_PRE + RANDOM % (max_pre - MIN_PRE + 1) ))
  run_start=$(date +%s)
  echo "[$(date +%T)] run $name: fault at +${pre}s for ${FAULT_S}s (run ${RUN_S}s)"

  sleep "$pre"
  inject=$(date +%s)
  NAME=$name SERVICE=$svc DURATION="${FAULT_S}s" DNS_PATTERNS=$(dns_patterns "$svc") \
    envsubst '${NAME} ${SERVICE} ${DURATION} ${DNS_PATTERNS}' < "$HERE/chaos/${fault}.yaml" | kubectl apply -f -

  sleep $(( RUN_S - pre ))
  kubectl delete -n boutique "$(grep -m1 '^kind:' "$HERE/chaos/${fault}.yaml" | awk '{print tolower($2)}')" "$name" --ignore-not-found >/dev/null
  run_end=$(date +%s)

  echo "$inject" > "$dir/inject_time.txt"
  cat > "$dir/run.json" <<JSON
{"service": "$svc", "fault": "$fault", "rep": $rep, "chaos_name": "$name",
 "run_start": $run_start, "inject_time": $inject, "fault_end": $(( inject + FAULT_S )), "run_end": $run_end}
JSON
  export_and_check "$dir"
  # Give the services a moment to settle before the next run.
  sleep "$SETTLE_S"
done; done; done
echo "all runs done -> $OUT"
