#!/usr/bin/env bash
# Sets the load generator's number of simulated users.
#   ./set_traffic.sh MULT      MULT = 0.5 | 1 | 2 | 4, times the manifest default
# The manifest default (USERS=10 in the Online Boutique release manifest) is
# read from the deployment the first time and remembered in state.json.
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
HERE=$(cd "$(dirname "$0")" && pwd)
MULT=${1:?usage: set_traffic.sh 0.5|1|2|4}
case "$MULT" in 0.5|1|2|4) ;; *) echo "MULT must be 0.5, 1, 2 or 4"; exit 1 ;; esac
BASE=$(python3 -c "import json;print(json.load(open('$HERE/state.json')).get('users_default',''))" 2>/dev/null || true)
if [ -z "$BASE" ]; then
  BASE=$(kubectl get deploy loadgenerator -n boutique -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="USERS")].value}')
  python3 "$HERE/state_set.py" users_default "$BASE"
fi
USERS=$(python3 -c "print(max(1, int($BASE * $MULT)))")
kubectl set env -n boutique deploy/loadgenerator USERS="$USERS" >/dev/null
kubectl rollout status -n boutique deploy/loadgenerator --timeout=5m >/dev/null
python3 "$HERE/state_set.py" users "$USERS"
python3 "$HERE/state_set.py" users_mult "$MULT"
echo "traffic set: USERS=$USERS ($MULT x default $BASE)"
