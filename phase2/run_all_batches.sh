#!/usr/bin/env bash
# Runs the extended batches one after another on the instance, each into its
# own folder under batches/. Start with:
#   nohup ./run_all_batches.sh A A0 B C D > batches.log 2>&1 &
# Give only some names to run only those. Every run is exported right after
# it is recorded (run_experiments.sh), and finished runs are skipped, so the
# same command resumes a batch that was interrupted.
set -uo pipefail
cd "$(dirname "$0")"
D1="checkoutservice currencyservice emailservice productcatalogservice recommendationservice"
ROOT=batches
mkdir -p "$ROOT"
say() { echo "[$(date -u +%FT%TZ)] $*"; }

baseline() { ./set_monitoring.sh 15s 1.0; ./set_traffic.sh 1; }

for b in "$@"; do
  say "BATCH $b START"
  case "$b" in
    A)  # main: 9 services x 6 faults x 3 reps
      baseline
      REPS=3 ./run_experiments.sh "$ROOT/A" ;;
    A0) # no-fault runs
      baseline
      ./run_experiments.sh "$ROOT/A0" --none 20 ;;
    B)  # monitoring detail: scrape interval, then trace sampling
      for scrape in 5s 30s 60s; do
        ./set_monitoring.sh "$scrape" 1.0
        SERVICES="$D1" FAULTS="cpu mem delay loss" ./run_experiments.sh "$ROOT/B_scrape${scrape}"
      done
      for sample in 0.1 0.01; do
        ./set_monitoring.sh 15s "$sample"
        SERVICES="$D1" FAULTS="cpu mem delay loss" ./run_experiments.sh "$ROOT/B_sample${sample}"
      done
      baseline ;;
    C)  # fault types the model never saw
      baseline
      SERVICES="$D1" FAULTS="dns time restart podkill" REPS=2 ./run_experiments.sh "$ROOT/C" ;;
    D)  # traffic level
      for mult in 0.5 2 4; do
        ./set_monitoring.sh 15s 1.0; ./set_traffic.sh "$mult"
        SERVICES="$D1" FAULTS="cpu mem delay loss" ./run_experiments.sh "$ROOT/D_users${mult}x"
      done
      baseline ;;
    *) say "unknown batch $b" ;;
  esac
  say "BATCH $b COMPLETE $(du -sh "$ROOT" | cut -f1) total, $(ls "$ROOT"/*/*/*/traces.csv 2>/dev/null | wc -l) runs exported so far"
done
say "ALL BATCHES DONE"
