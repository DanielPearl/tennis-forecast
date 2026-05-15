# Baseline Break — Tennis Forecast

A pre-match + live-adjustment forecast system for tour-level tennis
singles, built for event/contract-style markets (Kalshi-style).
The point isn't to predict winners. The point is to compare a
calibrated model probability against the market's implied probability
and surface the spots where the two materially disagree.

```
baseline-break/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                         # Sackmann CSVs + live-state fixture
│   ├── processed/                   # cleaned panels + model artifacts
│   └── outputs/                     # watchlist.csv / watchlist.json / backtest_results.csv
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_backtest.ipynb
├── src/
│   ├── data/
│   │   ├── fetch_matches.py         # Sackmann tour-year CSV puller
│   │   ├── fetch_odds.py            # The Odds API (optional)
│   │   └── fetch_live_scores.py     # provider OR fixture
│   ├── features/
│   │   ├── elo.py                   # overall + surface Elo
│   │   ├── build_prematch_features.py
│   │   └── build_live_features.py
│   ├── models/
│   │   ├── train_prematch_model.py  # logistic + GBT/XGB ensemble + sigmoid calibration
│   │   ├── live_adjustment_model.py # transparent rules layer
│   │   └── predict.py               # inference (loads bundle + Elo state)
│   ├── trading/
│   │   ├── ev.py                    # edge / EV / breakeven math
│   │   ├── signals.py               # WATCH / SMALL_EDGE / STRONG_EDGE / …
│   │   └── backtest.py              # holdout replay + ROI / drawdown / cohorts
│   ├── dashboard/
│   │   └── export_watchlist.py      # writes watchlist.{csv,json}
│   └── utils/
│       ├── config.py
│       └── logging_setup.py
├── scripts/
│   ├── run_daily_prematch.py        # train + export
│   ├── run_live_monitor.py          # loop: refresh watchlist every refresh_seconds
│   └── run_backtest.py
└── deploy/
    ├── baseline-break-monitor.service
    └── deploy.sh
```

## What the model does

**Pre-match** — gives a baseline win probability before the match.
Inputs: overall Elo, surface-specific Elo, ranking, recent-form
windows, rolling serve / return / break-point stats, head-to-head,
days of rest, tournament level + round. We blend a logistic
regression on Elo features with a calibrated boosted ensemble (XGBoost
when available, sklearn HGB otherwise). Sigmoid-calibrated on a
holdout slice.

**Live adjustment** — a transparent rules layer (phase-1) that
nudges the pre-match probability using the in-match score, serve %,
momentum, tiebreak / decider / medical-timeout flags, and the most
recent market move. Every nudge is logged into a `reason` string the
dashboard surfaces alongside the signal.

**Signals** — never fire on winner probability alone. They fire only
when the model's view differs from the market by more than a
configured edge floor, with separate gates for high volatility and
injury risk:

```
INJURY_RISK ▸ AVOID_VOLATILE ▸ MARKET_OVERREACTION
            ▸ STRONG_EDGE ▸ SMALL_EDGE ▸ WATCH ▸ NO_TRADE
```

`MARKET_OVERREACTION` fires when the market moves substantially while
the model's adjusted probability barely changed — the system's call
that the price has run past the underlying state.

## Running locally (VS Code)

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 1) train the pre-match model (pulls Sackmann CSVs the first time;
#    cached under data/raw/ thereafter).
python scripts/run_daily_prematch.py

# 2) start the live-monitor loop — refreshes data/outputs/watchlist.json
#    on a fixed interval. The central trading-dashboard reads that file.
python scripts/run_live_monitor.py
```

VS Code: open the repo folder, select the `.venv` interpreter from the
status bar, then use the "Run and Debug" panel to run any of the
scripts. `notebooks/` are pre-configured for the same interpreter.

## Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit — Baseline Break tennis forecast"
gh repo create tennis-forecast --public --source=. --remote=origin --push
```

If you don't use the `gh` CLI:

```bash
git remote add origin git@github.com:<you>/tennis-forecast.git
git branch -M main
git push -u origin main
```

## Deploying to a DigitalOcean Ubuntu droplet

```bash
# one-time, on a fresh droplet (as root)
apt update && apt install -y python3-pip python3-venv git
cd /root && git clone https://github.com/<you>/tennis-forecast.git
cd tennis-forecast
bash deploy/deploy.sh
```

The deploy script:

1. Creates a venv in `.venv/` and installs `requirements.txt`.
2. Copies `.env.example` → `.env` if not already present (edit the
   live values before letting it run for real).
3. Runs `scripts/run_daily_prematch.py` once so the dashboard has a
   model + watchlist on first boot.
4. Installs and starts the live-monitor systemd unit:
   - `baseline-break-monitor.service` — the live-monitor loop that
     refreshes `data/outputs/watchlist.json` (the central
     trading-dashboard reads this file directly).

```bash
systemctl status baseline-break-monitor
journalctl -u baseline-break-monitor -f
```

To redeploy on top of an existing checkout: `bash deploy/deploy.sh`.

### Running under tmux instead

If you don't want systemd:

```bash
tmux new -s tennis-monitor
source .venv/bin/activate
python scripts/run_live_monitor.py
```

## Exporting dashboard data

The watchlist is always written to:

- `data/outputs/watchlist.csv` — Sheets / Notion / spreadsheets
- `data/outputs/watchlist.json` — also served at `/api/watchlist.json`

`run_backtest.py` writes `data/outputs/backtest_results.csv` with one
row per (test) match and per-cohort metrics (surface, level,
favorite-vs-dog, edge bucket).

## Backtest

```bash
python scripts/run_backtest.py
```

Reports:

- Accuracy / log-loss / Brier on the held-out window
- Win rate, average EV, simulated ROI under a unit-stake policy
- Max drawdown
- Splits by surface / tournament level / favorite-vs-dog / edge bucket

By default the simulated market is a noisy version of the Elo-only
probability — useful as a sanity check, but **not** a real ROI number.
For real numbers, drop your captured closing prices into the test
panel as a `closing_market_prob` column; the backtest will use them
verbatim.

## What's deferred to phase 2

- ML-based live adjustment model (need closed-bet feedback first).
- Point-by-point feature ingestion — currently we use whatever
  per-match aggregates the live provider exposes.
- Real Kalshi market integration. The `trading/` layer is
  market-agnostic; plug in whatever feed you license.
- Walk-forward feature selection (the NBA bot's permutation-importance
  selector) — useful once we have many more pre-match features than
  we currently do.
