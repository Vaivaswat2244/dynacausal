#!/usr/bin/env bash
# SSH to the phase2 instance, first making sure the security group allows the
# laptop's CURRENT public IP (it rotates behind carrier NAT). Adds a /32 rule
# for the current IP if missing; never widens beyond /32.
# Usage: ./ssh2.sh [ssh args and remote command...]
set -euo pipefail
IP=$(cut -d' ' -f2 "$(dirname "$0")/instance.txt")
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-name dynacausal-p2 \
  --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true
exec ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
  -i ~/.ssh/dynacausal.pem ubuntu@"$IP" "$@"
