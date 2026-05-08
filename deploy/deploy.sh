#!/usr/bin/env bash
# One-shot deploy / redeploy on a DigitalOcean Ubuntu droplet.
#
# Usage on a fresh droplet (as root):
#   apt update && apt install -y python3-pip python3-venv git
#   cd /root && git clone https://github.com/DanielPearl/tennis-forecast.git
#   cd tennis-forecast && bash deploy/deploy.sh

set -euo pipefail

REPO_DIR="/root/tennis-forecast"
cd "$REPO_DIR"

# Pull latest if this is a redeploy.
if [ -d .git ]; then
  echo "[deploy] git pull"
  git pull --ff-only
fi

# Virtualenv (idempotent).
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# .env starter (only if missing — never overwrites existing creds).
if [ ! -f .env ]; then
  cp .env.example .env
  echo "[deploy] wrote .env from .env.example — edit before starting"
fi

# Initial train + watchlist build (so the dashboard has something to show
# before the live-monitor unit takes over).
echo "[deploy] training pre-match model — first run takes a few minutes"
python scripts/run_daily_prematch.py

# systemd units.
cp deploy/baseline-break-dashboard.service /etc/systemd/system/
cp deploy/baseline-break-monitor.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now baseline-break-dashboard
systemctl enable --now baseline-break-monitor

echo
echo "[deploy] up — dashboard on :8090, live monitor running"
echo "[deploy] logs:  journalctl -u baseline-break-dashboard -f"
echo "[deploy]        journalctl -u baseline-break-monitor -f"
