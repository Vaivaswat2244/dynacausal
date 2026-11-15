#!/usr/bin/env bash
# Runs a batch of fault injections and writes one label file per run.
# Each run: NORMAL_S seconds of normal operation, then a fault for FAULT_S
# seconds, then RECOVER_S seconds of recovery. The fault start offset inside
# the run is randomised, so the model cannot learn a fixed fault time.
#
# Usage:  ./run_experiments.sh runs/                 # default matrix
#         SERVICES="cartservice" FAULTS="cpu loss" ./run_experiments.sh runs/
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}

OUT=${1:-runs}
SERVICES=${SERVICES:-"checkoutservice currencyservice emailservice productcatalogservice recommendationservice cartservice paymentservice shippingservice adservice"}
FAULTS=${FAULTS:-"cpu mem delay loss podkill"}
REPS=${REPS:-1}
RUN_S=${RUN_S:-600}        # total run length (10 min)
FAULT_S=${FAULT_S:-180}    # fault duration (3 min)
MIN_PRE=${MIN_PRE:-180}    # fault starts no earlier than this
MIN_POST=${MIN_POST:-120}  # and leaves at least this much time after it ends
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

for rep in $(seq 1 "$REPS"); do
for svc in $SERVICES; do
for fault in $FAULTS; do
  name="${svc}-${fault}-r${rep}-$(date +%s)"
  dir="$OUT/${svc}_${fault}/${rep}"
  mkdir -p "$dir"
  max_pre=$(( RUN_S - FAULT_S - MIN_POST ))
  pre=$(( MIN_PRE + RANDOM % (max_pre - MIN_PRE + 1) ))
  run_start=$(date +%s)
  echo "[$(date +%T)] run $name: fault at +${pre}s for ${FAULT_S}s (run ${RUN_S}s)"

  sleep "$pre"
  inject=$(date +%s)
  NAME=$name SERVICE=$svc DURATION="${FAULT_S}s" envsubst < "$HERE/chaos/${fault}.yaml" | kubectl apply -f -

  sleep $(( RUN_S - pre ))
  kubectl delete -n boutique "$(grep -m1 '^kind:' "$HERE/chaos/${fault}.yaml" | awk '{print tolower($2)}')" "$name" --ignore-not-found >/dev/null
  run_end=$(date +%s)

  echo "$inject" > "$dir/inject_time.txt"
  cat > "$dir/run.json" <<EOF
{"service": "$svc", "fault": "$fault", "rep": $rep, "chaos_name": "$name",
 "run_start": $run_start, "inject_time": $inject, "fault_end": $(( inject + FAULT_S )), "run_end": $run_end}
EOF
  # Give the services a moment to settle before the next run.
  sleep 60
done; done; done
echo "all runs done -> $OUT"
