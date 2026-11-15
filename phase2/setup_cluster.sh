#!/usr/bin/env bash
# Phase 2 cluster setup. Run once on a fresh Ubuntu 24.04 EC2 instance.
# Installs k3s, Helm, Online Boutique, Prometheus, Jaeger, Loki and Chaos Mesh.
# Takes about 10 to 15 minutes. Re-running is safe.
set -euo pipefail

echo "== 1. k3s =="
if ! command -v k3s >/dev/null; then
  curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
fi
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
grep -q KUBECONFIG ~/.bashrc || echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >> ~/.bashrc
kubectl wait --for=condition=Ready node --all --timeout=180s

echo "== 2. Helm =="
if ! command -v helm >/dev/null; then
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
helm repo add grafana https://grafana.github.io/helm-charts >/dev/null
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts >/dev/null
helm repo add chaos-mesh https://charts.chaos-mesh.org >/dev/null
helm repo update >/dev/null

echo "== 3. Monitoring: Prometheus + Grafana =="
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install kps prometheus-community/kube-prometheus-stack -n monitoring \
  --set prometheus.prometheusSpec.scrapeInterval=15s \
  --set prometheus.prometheusSpec.retention=7d \
  --set alertmanager.enabled=false \
  --wait --timeout 10m

echo "== 4. Jaeger (all-in-one, OTLP enabled) =="
helm upgrade --install jaeger jaegertracing/jaeger -n monitoring \
  --set provisionDataStore.cassandra=false \
  --set allInOne.enabled=true \
  --set storage.type=memory \
  --set agent.enabled=false --set collector.enabled=false --set query.enabled=false \
  --set allInOne.extraEnv[0].name=COLLECTOR_OTLP_ENABLED \
  --set-string allInOne.extraEnv[0].value=true \
  --set allInOne.extraEnv[1].name=MEMORY_MAX_TRACES \
  --set-string allInOne.extraEnv[1].value=2000000 \
  --wait --timeout 5m

echo "== 5. Loki + Alloy (logs) =="
helm upgrade --install loki grafana/loki -n monitoring \
  --set deploymentMode=SingleBinary \
  --set loki.commonConfig.replication_factor=1 \
  --set loki.storage.type=filesystem \
  --set loki.auth_enabled=false \
  --set singleBinary.replicas=1 \
  --set backend.replicas=0 --set read.replicas=0 --set write.replicas=0 \
  --set loki.schemaConfig.configs[0].from=2024-01-01 \
  --set loki.schemaConfig.configs[0].store=tsdb \
  --set loki.schemaConfig.configs[0].object_store=filesystem \
  --set loki.schemaConfig.configs[0].schema=v13 \
  --set loki.schemaConfig.configs[0].index.prefix=index_ \
  --set loki.schemaConfig.configs[0].index.period=24h \
  --set chunksCache.enabled=false --set resultsCache.enabled=false \
  --set lokiCanary.enabled=false --set test.enabled=false --set gateway.enabled=false \
  --wait --timeout 5m
helm upgrade --install alloy grafana/k8s-monitoring -n monitoring \
  --set cluster.name=phase2 \
  --set destinations[0].name=loki --set destinations[0].type=loki \
  --set destinations[0].url=http://loki.monitoring.svc:3100/loki/api/v1/push \
  --set podLogs.enabled=true --set clusterMetrics.enabled=false \
  --set alloy-logs.enabled=true --set alloy-metrics.enabled=false \
  --set alloy-singleton.enabled=false \
  --wait --timeout 5m || echo "alloy install failed, logs can be collected with kubectl logs instead"

echo "== 6. Chaos Mesh =="
kubectl create namespace chaos-mesh --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/k3s/containerd/containerd.sock \
  --set dashboard.create=true \
  --wait --timeout 5m

echo "== 7. Online Boutique with tracing to Jaeger =="
kubectl create namespace boutique --dry-run=client -o yaml | kubectl apply -f -
OB=https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/release/kubernetes-manifests.yaml
curl -sfL "$OB" -o /tmp/ob.yaml
kubectl apply -n boutique -f /tmp/ob.yaml
# Enable OpenTelemetry tracing in every service and point it at Jaeger's OTLP port.
for d in $(kubectl get deploy -n boutique -o name); do
  kubectl set env -n boutique "$d" ENABLE_TRACING=1 \
    COLLECTOR_SERVICE_ADDR=jaeger-collector.monitoring.svc:4317 \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger-collector.monitoring.svc:4317 \
    OTEL_EXPORTER_OTLP_INSECURE=true >/dev/null
done
kubectl rollout status -n boutique deploy --timeout=10m

echo "== done =="
kubectl get pods -A | grep -vE "Running|Completed" || true
echo
echo "Port-forwards (run each in its own terminal, or use ssh -L from your laptop):"
echo "  kubectl port-forward -n monitoring svc/kps-kube-prometheus-stack-prometheus 9090:9090"
echo "  kubectl port-forward -n monitoring svc/jaeger-query 16686:16686"
echo "  kubectl port-forward -n monitoring svc/loki 3100:3100"
echo "  kubectl port-forward -n boutique svc/frontend 8080:80"
