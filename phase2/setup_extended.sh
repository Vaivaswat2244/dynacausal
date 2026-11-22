#!/usr/bin/env bash
# One-off additions to the cluster for the extended batches. Safe to re-run.
#   - Chaos Mesh DNS server, needed by chaos/dns.yaml (DNSChaos).
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
VER=$(helm list -n chaos-mesh -o json | python3 -c "import json,sys;print([r['chart'] for r in json.load(sys.stdin) if r['name']=='chaos-mesh'][0].rsplit('-',1)[1])")
helm upgrade chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh --version "$VER" --reuse-values \
  --set dnsServer.create=true --wait --timeout 5m >/dev/null
kubectl get pods -n chaos-mesh
