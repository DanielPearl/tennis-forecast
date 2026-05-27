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

# systemd units. The dashboard is served by the central trading-dashboard
# service; this bot only runs the live-monitor loop that refreshes the
# watchlist JSON the trading dashboard reads. The daily retrain timer
# re-fits the prematch model + last_match_date dict so diff_days_rest
# (the top feature by permutation importance) doesn't drift stale.
cp deploy/baseline-break-monitor.service /etc/systemd/system/
cp deploy/tennis-forecast-train.service /etc/systemd/system/
cp deploy/tennis-forecast-train.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now baseline-break-monitor
systemctl enable --now tennis-forecast-train.timer

echo
echo "[deploy] up — live monitor running; dashboard served by trading-dashboard"
echo "[deploy] retrain timer armed (next fire: \$(systemctl show -p NextElapseUSecRealtime --value tennis-forecast-train.timer))"
echo "[deploy] logs:"
echo "  journalctl -u baseline-break-monitor -f"
echo "  journalctl -u tennis-forecast-train -f"
