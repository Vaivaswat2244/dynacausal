#!/usr/bin/env bash
# Sets how detailed the monitoring is.
#   ./set_monitoring.sh SCRAPE SAMPLE     SCRAPE = 5s|15s|30s|60s   SAMPLE = 1.0|0.1|0.01
# SCRAPE: Prometheus scrape interval (global, and the kubelet/cAdvisor
#   targets that carry the container metrics), via helm upgrade of the
#   kube-prometheus-stack release at its CURRENT chart version.
# SAMPLE: share of traces kept. Sets the standard OpenTelemetry sampler env
#   vars on every boutique deployment and waits for the rollout.
# The export grid stays 1 s and the feature grid 15 s whatever the scrape
# interval is, so a coarser scrape shows up as repeated values.
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
HERE=$(cd "$(dirname "$0")" && pwd)
SCRAPE=${1:?usage: set_monitoring.sh SCRAPE SAMPLE}
SAMPLE=${2:?usage: set_monitoring.sh SCRAPE SAMPLE}
case "$SCRAPE" in 5s|15s|30s|60s) ;; *) echo "SCRAPE must be 5s, 15s, 30s or 60s"; exit 1 ;; esac
case "$SAMPLE" in 1.0|0.1|0.01) ;; *) echo "SAMPLE must be 1.0, 0.1 or 0.01"; exit 1 ;; esac

VER=$(helm list -n monitoring -o json | python3 -c "import json,sys;print([r['chart'] for r in json.load(sys.stdin) if r['name']=='kps'][0].rsplit('-',1)[1])")
helm upgrade kps prometheus-community/kube-prometheus-stack -n monitoring --version "$VER" --reuse-values \
  --set prometheus.prometheusSpec.scrapeInterval="$SCRAPE" \
  --set kubelet.serviceMonitor.interval="$SCRAPE" \
  --set kubelet.serviceMonitor.cAdvisorInterval="$SCRAPE" \
  --wait --timeout 10m >/dev/null
python3 "$HERE/state_set.py" scrape "$SCRAPE"

for d in $(kubectl get deploy -n boutique -o name); do
  kubectl set env -n boutique "$d" OTEL_TRACES_SAMPLER=parentbased_traceidratio \
    OTEL_TRACES_SAMPLER_ARG="$SAMPLE" >/dev/null
done
kubectl rollout status -n boutique deploy --timeout=10m >/dev/null
python3 "$HERE/state_set.py" sample "$SAMPLE"
echo "monitoring set: scrape $SCRAPE, trace sampling $SAMPLE"
