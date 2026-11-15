# Phase 2 runbook: live testing on AWS

Goal for today (minimum viable Phase 2): a k3s cluster running Online
Boutique with Prometheus, Jaeger, Loki and Chaos Mesh; a batch of labelled
fault runs recorded; the Phase 1 model run on those recordings WITHOUT the
fault time; first accuracy and time-to-diagnosis numbers.

The five condition experiments (fault time, environment, monitoring detail,
traffic, unseen faults) and the streaming service are the next days' work.

Rough timeline: A+B 45 min, C 20 min, D about 3 hours of unattended
recording, E 1 to 2 hours (Claude Code), F 30 min.

---

## A. Launch the instance (AWS console, 10 min)

1. EC2 > Launch instance.
   - Name: dynacausal-phase2
   - AMI: Ubuntu Server 24.04 LTS (x86_64)
   - Instance type: **m5.2xlarge** (8 vCPU, 32 GB). t3.xlarge (16 GB) also
     works for Online Boutique but leaves little headroom.
   - Key pair: create or pick one, download the .pem.
   - Storage: **100 GB gp3**.
   - Security group: allow SSH (22) from "My IP" only. Nothing else; every
     UI is reached through SSH port-forwarding.
2. Note the public IP.
3. From your laptop:
   ```bash
   chmod 600 ~/Downloads/your-key.pem
   ssh -i ~/Downloads/your-key.pem ubuntu@<PUBLIC_IP>
   ```

Cost: m5.2xlarge is about $0.38/hour (around $9/day). **Stop the instance
when you are not recording.** A stopped instance only costs the disk.

## B. Set up the cluster (on the instance, 15 min)

```bash
sudo apt update && sudo apt install -y gettext-base python3-pip
mkdir -p ~/phase2 && cd ~/phase2
# copy the phase2 folder here first:  scp -i key.pem -r phase2 ubuntu@<IP>:~/
chmod +x setup_cluster.sh run_experiments.sh
./setup_cluster.sh
```

Check:
```bash
kubectl get pods -A          # everything Running
kubectl top pods -n boutique # loadgenerator should be generating traffic
```

Chaos Mesh dashboard (optional, from your laptop):
```bash
ssh -i key.pem -L 2333:localhost:2333 ubuntu@<IP> \
  kubectl port-forward -n chaos-mesh svc/chaos-dashboard 2333:2333
```
then open http://localhost:2333 and create a token as its page explains.

## C. Smoke test one fault (20 min)

```bash
cd ~/phase2
SERVICES="productcatalogservice" FAULTS="loss" RUN_S=300 FAULT_S=90 ./run_experiments.sh runs_smoke
```
While it runs, in another SSH session:
```bash
kubectl get networkchaos -n boutique         # the fault object appears
kubectl logs -n boutique deploy/frontend --tail=20   # errors should show
```
Then export it (port-forwards in 3 background terminals, or use `&`):
```bash
kubectl port-forward -n monitoring svc/kps-kube-prometheus-stack-prometheus 9090:9090 &
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686 &
kubectl port-forward -n monitoring svc/loki 3100:3100 &
pip install requests pandas
python3 export_telemetry.py runs_smoke --all
head -c 600 runs_smoke/productcatalogservice_loss/1/traces.csv
```
You should see spans with serviceName values like `frontend` and
operationName values like `hipstershop.ProductCatalogService/GetProduct`.
If traces.csv is empty, tracing is not reaching Jaeger. See "If things
break" below before running the full batch.

## D. Record the batch (about 3 hours unattended)

Default matrix: 9 services x 5 faults x 1 repetition = 45 runs of 10 min,
plus 1 min between runs, about 8 hours. For today, use the 5 D1 services
and 4 faults = 20 runs, about 3.7 hours:

```bash
cd ~/phase2
SERVICES="checkoutservice currencyservice emailservice productcatalogservice recommendationservice" \
FAULTS="cpu mem delay loss" \
nohup ./run_experiments.sh runs > runs.log 2>&1 &
tail -f runs.log
```

Also record 3 runs with NO fault (needed for the false-alarm measurement):
```bash
for i in 1 2 3; do d=runs/none_none/$i; mkdir -p $d; s=$(date +%s); sleep 600; \
  echo "{\"service\":\"none\",\"fault\":\"none\",\"rep\":$i,\"run_start\":$s,\"inject_time\":0,\"fault_end\":0,\"run_end\":$(date +%s)}" > $d/run.json; echo 0 > $d/inject_time.txt; done
```

Export everything once done (port-forwards still running):
```bash
python3 export_telemetry.py runs --all
du -sh runs
```

Copy to your laptop:
```bash
scp -i key.pem -r ubuntu@<IP>:~/phase2/runs ~/Documents/projects/MTP/data/live/
```
Then **stop the EC2 instance** in the console.

## E. Evaluate with the Phase 1 model (laptop, Claude Code)

Put `CLAUDE_CODE_TASK_phase2_eval.md` in the repo root and tell Claude Code:
"Do the task in CLAUDE_CODE_TASK_phase2_eval.md". It will:
1. Check the trained Phase 1 model is saved (if not, retrain on D1 first;
   D1 must be re-downloaded since data/raw was deleted).
2. Load the live runs through the existing loaders and feature extractor,
   fixing column-name differences.
3. Slide windows over each whole run (no fault time given) and record the
   model's ranking per window.
4. Report accuracy, time to diagnosis and false alarms.

## F. Record the numbers

Whatever comes out goes into the results table as is, labelled "live
cluster, N runs, fault time not given". A drop from the D1 numbers is the
expected finding and is what Phase 2 exists to measure.

---

## If things break

- **Pods Pending / OOM:** instance too small. `kubectl describe pod <name> -n boutique`.
- **No traces in Jaeger:** check `kubectl logs -n boutique deploy/frontend | grep -i otel`.
  The manifest version may use different env names. Try
  `OTEL_SERVICE_NAME`, `OTEL_TRACES_EXPORTER=otlp` and
  `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger-collector.monitoring.svc:4318`
  (HTTP port) instead of 4317. Confirm the service name with
  `kubectl get svc -n monitoring | grep jaeger`.
- **Chaos has no effect:** `kubectl describe networkchaos <name> -n boutique`
  shows an event. If it says the daemon cannot find the container, check
  `chaosDaemon.socketPath` matches `ls /run/k3s/containerd/`.
- **Loki export empty:** fall back to `kubectl logs --since-time` per pod;
  logs matter least for the model.
- **Prometheus has no container metrics:** `kubectl get pods -n monitoring`
  should show a kube-state-metrics and a node-exporter pod; cAdvisor metrics
  come through the kubelet target, visible at Prometheus > Status > Targets.
