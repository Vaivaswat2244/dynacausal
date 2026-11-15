# Task for Claude Code: run Phase 2 end to end on AWS

Run from the repo root (the dynacausal repo). Read OBJECTIVE.Md,
PROJECT_CONTEXT.md and phase2/PHASE2_RUNBOOK.md first. The phase2/ folder
holds the scripts; the runbook explains them. Rules: plain-English
explanations, never adjust results, never backdate commits, no em dashes.

Ask the user before anything that costs money or cannot be undone (launching
the instance, deleting anything on AWS). Everything else, go ahead.

## Prerequisites (check, then ask the user for what is missing)

- `aws --version` works and `aws sts get-caller-identity` returns an account.
  If not: `pip install awscli` or the installer, then the user runs
  `aws configure` with their access key, secret, region (ap-south-1) and
  output json.
- A key pair: `aws ec2 describe-key-pairs`. If none, create one:
  `aws ec2 create-key-pair --key-name dynacausal --query KeyMaterial --output text > ~/.ssh/dynacausal.pem && chmod 600 ~/.ssh/dynacausal.pem`
- Trained Phase 1 checkpoint present in results/ (see the eval task). If
  missing, start the D1 re-download and retrain in the background NOW so it
  runs while the cluster records.

## Step 1: launch the instance (ask first; about $0.38/hour)

```bash
MYIP=$(curl -s https://checkip.amazonaws.com)
SG=$(aws ec2 create-security-group --group-name dynacausal-p2 --description "phase2 ssh" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 --cidr $MYIP/32
AMI=$(aws ssm get-parameters --names /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id --query 'Parameters[0].Value' --output text)
ID=$(aws ec2 run-instances --image-id $AMI --instance-type m5.2xlarge --key-name dynacausal \
  --security-group-ids $SG --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=dynacausal-phase2}]' \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --instance-ids $ID
IP=$(aws ec2 describe-instances --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$ID $IP" > phase2/instance.txt
```
If m5.2xlarge is refused (quota), fall back to t3.xlarge and tell the user.
Wait until `ssh -o StrictHostKeyChecking=accept-new -i ~/.ssh/dynacausal.pem ubuntu@$IP true` succeeds.

## Step 2: set up the cluster

```bash
scp -i ~/.ssh/dynacausal.pem -r phase2 ubuntu@$IP:~/
ssh -i ~/.ssh/dynacausal.pem ubuntu@$IP 'sudo apt-get update -qq && sudo apt-get install -y -qq gettext-base python3-pip python3-pandas python3-requests && cd phase2 && chmod +x *.sh && ./setup_cluster.sh' 2>&1 | tee phase2/setup.log
ssh -i ~/.ssh/dynacausal.pem ubuntu@$IP 'kubectl get pods -A'
```
Every pod must be Running or Completed. Fix failures using the runbook's
"If things break" section before going on. Report to the user what you had
to change.

## Step 3: smoke test

Run the section C commands over ssh (one fault, RUN_S=300, FAULT_S=90),
then start the three port-forwards in the background on the instance and
run `python3 export_telemetry.py runs_smoke --all`. Check:
- traces.csv has rows, serviceName values like `frontend`, operationName
  like `hipstershop.ProductCatalogService/GetProduct`, and some non-zero
  statusCode after inject_time;
- metrics.csv has non-constant columns for the target service;
- logs.csv has rows (least important; skip if Loki is the only failure).
If traces are empty, fix tracing first (runbook list). Do not start the
batch until traces work. Show the user the `head` of traces.csv.

## Step 4: record the batch (about 4 hours, unattended)

On the instance, in nohup as in runbook section D: 5 D1 services x 4
faults (cpu mem delay loss), REPS=1, then the 3 no-fault runs. Poll
`runs.log` every 15 minutes and tell the user progress. Meanwhile, do the
loader and feature mapping work of the eval task (Step 1 there) against
the smoke-test run, so it is ready when the batch finishes.

## Step 5: export, copy back, stop the instance

```bash
ssh ... 'cd phase2 && python3 export_telemetry.py runs --all'
scp -i ~/.ssh/dynacausal.pem -r ubuntu@$IP:~/phase2/runs data/live/
aws ec2 stop-instances --instance-ids $ID      # stop, not terminate; ask the user before terminating
```
Keep the instance stopped, not terminated, so the condition experiments can
reuse it later.

## Step 6: evaluate

Do `phase2/CLAUDE_CODE_TASK_phase2_eval.md` in full and give its report.

## Step 7: final report to the user

Instance id and state, hours used, runs recorded (per service and fault),
everything the eval task reports, and every deviation from the runbook.
