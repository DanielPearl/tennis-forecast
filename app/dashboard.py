"""Baseline Break — read-only dashboard server.

Stdlib http.server only. Same pattern as the trading-dashboard project
on the droplet so we can deploy with the same systemd unit shape and
no Flask dependency. Two pages:

  /            — home page with a Tennis Forecast card.
                 The card mirrors the visual style of the multi-bot
                 trading-dashboard "bot card" but lives standalone here.
  /watchlist   — full watchlist table with every match, signal, edge.

Data source: the watchlist JSON written by ``src/dashboard/export_watchlist.py``.
The server *does not run the model* — that's by design. The model
is run by the cron job (or live-monitor systemd unit) and the
dashboard reads the cached output. This keeps the page snappy and
means the site stays up even if the next refresh job fails.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Allow `python app/dashboard.py` and `python -m app.dashboard` to both work.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import setup_logging

log = setup_logging("dashboard.server")


_LABEL_COLORS = {
    "STRONG_EDGE":   "#3fb950",
    "SMALL_EDGE":    "#56d364",
    "MARKET_OVERREACTION": "#e3b341",
    "WATCH":         "#58a6ff",
    "AVOID_VOLATILE":"#d29922",
    "INJURY_RISK":   "#f85149",
    "NO_TRADE":      "#8b949e",
}


def _load_watchlist() -> dict:
    cfg = load_config()
    fp = resolve_path(cfg["paths"]["watchlist_json"])
    if not fp.exists():
        return {"generated_at": None, "rows": []}
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _summary_stats(rows: list[dict]) -> dict:
    """Headline numbers for the home-page card. Built off whatever
    the latest watchlist export contains — no extra computation here."""
    if not rows:
        return {
            "total_matches": 0, "live_matches": 0, "actionable": 0,
            "avg_confidence": 0.0, "max_edge_pp": 0.0,
            "best_match_label": "—",
        }
    total = len(rows)
    actionable = sum(1 for r in rows if r["recommended_action"] in
                     ("STRONG_EDGE", "SMALL_EDGE", "MARKET_OVERREACTION"))
    live = sum(1 for r in rows
               if (r.get("current_score") or "0-0") not in ("0-0", "—"))
    confs = [float(r.get("confidence_score") or 0) for r in rows]
    edges = [abs(float(r.get("edge_a") or 0)) for r in rows]
    best = max(rows, key=lambda r: abs(float(r.get("edge_a") or 0)))
    return {
        "total_matches": total,
        "live_matches": live,
        "actionable": actionable,
        "avg_confidence": round(sum(confs) / max(1, len(confs)), 3),
        "max_edge_pp": round(max(edges) * 100, 1) if edges else 0.0,
        "best_match_label": f"{best['player_a']} vs {best['player_b']}",
    }


# ---------------------------------------------------------------------
# HTML rendering. Inline styles only — keeping the dashboard stdlib-only
# means we don't ship static asset routing.
# ---------------------------------------------------------------------
_BASE_CSS = """
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
       Roboto, Helvetica, Arial, sans-serif; background: #0d1117; color: #c9d1d9; }
header { background: #161b22; border-bottom: 1px solid #30363d;
         padding: 14px 24px; display: flex; align-items: center;
         justify-content: space-between; }
header h1 { font-size: 18px; margin: 0; font-weight: 600; color: #f0f6fc; }
header nav a { color: #8b949e; text-decoration: none; margin-right: 18px;
               font-size: 13px; }
header nav a.active, header nav a:hover { color: #f0f6fc; }
main { max-width: 1200px; margin: 0 auto; padding: 24px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 20px 22px; margin-bottom: 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.4); }
.card h2 { margin: 0 0 12px 0; font-size: 16px; color: #f0f6fc; }
.card .small { color: #8b949e; font-size: 12px; }
.row { display: flex; gap: 14px; flex-wrap: wrap; }
.stat { flex: 1 1 0; min-width: 140px; background: #1d232c;
        border: 1px solid #30363d; border-radius: 6px; padding: 12px 14px; }
.stat .label { font-size: 10px; text-transform: uppercase; color: #8b949e;
               letter-spacing: 0.06em; margin-bottom: 4px; }
.stat .value { font-size: 20px; font-weight: 600; color: #f0f6fc; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: #8b949e; font-weight: 500; font-size: 11px;
     text-transform: uppercase; letter-spacing: 0.05em;
     border-bottom: 1px solid #30363d; padding: 10px 8px; position: sticky; top: 0;
     background: #161b22; }
td { padding: 10px 8px; border-bottom: 1px solid #21262d; vertical-align: middle; }
tr:hover td { background: #1c222b; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 12px;
        font-size: 11px; font-weight: 600; }
.bot-card { display: block; background: #161b22; border: 1px solid #30363d;
            border-radius: 10px; padding: 18px 22px; text-decoration: none;
            color: inherit; transition: border-color 0.15s, transform 0.15s; }
.bot-card:hover { border-color: #58a6ff; transform: translateY(-1px); }
.bot-card-head { display: flex; justify-content: space-between;
                 align-items: baseline; margin-bottom: 14px; }
.bot-card-head .name { font-size: 16px; font-weight: 700; color: #f0f6fc; }
.bot-card-head .meta { font-size: 11px; color: #8b949e; }
.bot-card dl { display: grid; grid-template-columns: 1fr auto;
               gap: 6px 14px; margin: 0; font-size: 13px; }
.bot-card dt { color: #8b949e; }
.bot-card dd { margin: 0; color: #c9d1d9; font-weight: 500; }
.bot-card-foot { margin-top: 14px; font-size: 12px; color: #8b949e;
                 display: flex; justify-content: space-between; }
.green { color: #3fb950; } .red { color: #f85149; } .gray { color: #8b949e; }
.kbd { display: inline-block; padding: 1px 5px; font-size: 11px;
       background: #1d232c; border: 1px solid #30363d; border-radius: 4px;
       color: #c9d1d9; font-family: monospace; }
"""


def _header(active: str) -> str:
    cfg = load_config()
    title = cfg["dashboard"]["title"]
    home_cls = "active" if active == "home" else ""
    wl_cls = "active" if active == "watchlist" else ""
    return f"""
<header>
  <h1>🎾 {title}</h1>
  <nav>
    <a href="/" class="{home_cls}">Home</a>
    <a href="/watchlist" class="{wl_cls}">Watchlist</a>
    <a href="/api/watchlist.json">JSON</a>
  </nav>
</header>
"""


def _bot_card(stats: dict, generated_at: str | None) -> str:
    age_str = "—"
    if generated_at:
        try:
            ts = datetime.fromisoformat(generated_at)
            delta = (datetime.now(timezone.utc) - ts).total_seconds()
            age_str = f"{int(delta//60)}m {int(delta%60)}s ago"
        except Exception:
            pass
    return f"""
<a class="bot-card" href="/watchlist">
  <div class="bot-card-head">
    <span class="name">Tennis Forecast</span>
    <span class="meta">Baseline Break · MVP</span>
  </div>
  <dl>
    <dt>Matches tracked</dt><dd>{stats['total_matches']}</dd>
    <dt>Live right now</dt><dd>{stats['live_matches']}</dd>
    <dt>Actionable signals</dt><dd class="green">{stats['actionable']}</dd>
    <dt>Avg confidence</dt><dd>{stats['avg_confidence']:.0%}</dd>
    <dt>Largest edge</dt><dd>{stats['max_edge_pp']:.1f}pp</dd>
  </dl>
  <div class="bot-card-foot">
    <span>Last updated: {age_str}</span>
    <span>Open watchlist →</span>
  </div>
</a>
"""


def render_home() -> str:
    data = _load_watchlist()
    stats = _summary_stats(data["rows"])
    card = _bot_card(stats, data.get("generated_at"))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Baseline Break</title>
<style>{_BASE_CSS}</style></head>
<body>
{_header('home')}
<main>
  <div class="card">
    <h2>Forecast bots</h2>
    <p class="small">Each card links into its own watchlist. The numbers
       are the headline reads from the latest export — pre-match probability,
       live-adjusted probability, and the recommended-action label per match.</p>
    <div class="row">
      {card}
    </div>
  </div>

  <div class="card">
    <h2>How Baseline Break works</h2>
    <p class="small">The pre-match model gives a baseline win probability
       from Elo + surface-Elo + form + serve/return rolling stats. The
       live model adjusts that probability during the match using score
       state, serve %, and momentum. Signals only fire when the
       model's view differs from the market's by more than the
       configured edge floor — never on winner probability alone.</p>
  </div>
</main>
</body></html>"""


def _label_pill(label: str) -> str:
    color = _LABEL_COLORS.get(label, "#8b949e")
    return f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}55">{label}</span>'


def _pct(v) -> str:
    if v is None: return "—"
    try: return f"{float(v)*100:.1f}%"
    except Exception: return "—"


def _signed_pp(v) -> str:
    if v is None: return "—"
    try: return f"{float(v)*100:+.1f}pp"
    except Exception: return "—"


def render_watchlist() -> str:
    data = _load_watchlist()
    rows = sorted(
        data["rows"],
        key=lambda r: (
            -1 if r["recommended_action"] in ("STRONG_EDGE", "MARKET_OVERREACTION") else
            -0.5 if r["recommended_action"] == "SMALL_EDGE" else 0,
            -abs(float(r.get("edge_a") or 0)),
        ),
    )
    body_rows = []
    for r in rows:
        body_rows.append(f"""
<tr>
  <td><strong>{r['player_a']}</strong> vs {r['player_b']}<br>
      <span class="small gray">{r.get('round_label','')}</span></td>
  <td>{r['tournament']}</td>
  <td>{r['surface']}</td>
  <td>{r.get('current_score') or '—'}</td>
  <td>{_pct(r.get('market_prob_a'))}</td>
  <td>{_pct(r.get('pre_match_prob_a'))}</td>
  <td>{_pct(r.get('live_prob_a'))}</td>
  <td class="{'green' if (r.get('edge_a') or 0)>0 else ('red' if (r.get('edge_a') or 0)<0 else 'gray')}">{_signed_pp(r.get('edge_a'))}</td>
  <td>{r.get('ev_a') if r.get('ev_a') is None else f"{float(r['ev_a']):+.3f}"}</td>
  <td>{_pct(r.get('confidence_score'))}</td>
  <td>{_pct(r.get('volatility_score'))}</td>
  <td>{'<span class="red">⚠ injury</span>' if r.get('injury_news_flag') else '—'}</td>
  <td>{_label_pill(r['recommended_action'])}<br>
      <span class="small gray">{r.get('reason_for_signal','')}</span></td>
  <td class="small gray">{r.get('last_updated','')[:19]}</td>
</tr>""")
    table_html = (
        f"<table><thead><tr>"
        f"<th>Match</th><th>Tournament</th><th>Surface</th><th>Score</th>"
        f"<th>Market</th><th>Pre-match</th><th>Live</th>"
        f"<th>Edge</th><th>EV</th><th>Conf</th><th>Vol</th>"
        f"<th>Injury</th><th>Signal</th><th>Updated</th>"
        f"</tr></thead><tbody>{''.join(body_rows) or '<tr><td colspan=14 class=small>No matches yet — run scripts/run_daily_prematch.py</td></tr>'}</tbody></table>"
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Watchlist · Baseline Break</title>
<style>{_BASE_CSS}</style></head>
<body>
{_header('watchlist')}
<main>
  <div class="card">
    <h2>Watchlist · {len(rows)} matches</h2>
    <p class="small">Generated at {data.get('generated_at') or '—'}.
       Sorted by signal priority then absolute edge. <span class="kbd">Edge</span>
       is from player_a's perspective; positive = model thinks A is undervalued
       by the market.</p>
    {table_html}
  </div>
</main>
</body></html>"""


# ---------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "BaselineBreak/0.1"

    def _write(self, status: int, body: str | bytes, ctype: str = "text/html"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        try:
            if self.path in ("/", "/index", "/index.html"):
                return self._write(200, render_home())
            if self.path.startswith("/watchlist"):
                return self._write(200, render_watchlist())
            if self.path.startswith("/api/watchlist.json"):
                return self._write(200, json.dumps(_load_watchlist(), default=str),
                                   ctype="application/json")
            if self.path.startswith("/healthz"):
                return self._write(200, "ok", ctype="text/plain")
            return self._write(404, "<h1>404 not found</h1>")
        except Exception as exc:
            log.exception("handler error: %s", exc)
            return self._write(500, f"<pre>{exc}</pre>")

    def log_message(self, fmt, *args):
        log.info("%s — %s", self.address_string(), fmt % args)


def main() -> None:
    cfg = load_config()
    host = os.environ.get("HOST") or cfg["dashboard"]["host"]
    port = int(os.environ.get("PORT") or cfg["dashboard"]["port"])
    server = ThreadingHTTPServer((host, port), Handler)
    log.info("Baseline Break dashboard listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
