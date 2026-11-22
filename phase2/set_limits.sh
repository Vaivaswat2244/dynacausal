#!/usr/bin/env bash
# Sets CPU and memory LIMITS on every boutique container equal to the
# REQUESTS already in its manifest, so a stressed pod cannot borrow spare
# node capacity. On for all batches from now on.
#   ./set_limits.sh
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
HERE=$(cd "$(dirname "$0")" && pwd)
for d in $(kubectl get deploy -n boutique -o name); do
  patch=$(kubectl get -n boutique "$d" -o json | python3 -c "
import json, sys
dep = json.load(sys.stdin)
cs = [{'name': c['name'], 'resources': {'requests': c['resources']['requests'], 'limits': dict(c['resources']['requests'])}}
      for c in dep['spec']['template']['spec']['containers'] if c.get('resources', {}).get('requests')]
print(json.dumps({'spec': {'template': {'spec': {'containers': cs}}}}))")
  kubectl patch -n boutique "$d" --type strategic -p "$patch" >/dev/null
done
kubectl rollout status -n boutique deploy --timeout=10m
python3 "$HERE/state_set.py" limits "requests_as_limits"
kubectl get deploy -n boutique -o custom-columns='NAME:.metadata.name,CPU:.spec.template.spec.containers[0].resources.limits.cpu,MEM:.spec.template.spec.containers[0].resources.limits.memory'
