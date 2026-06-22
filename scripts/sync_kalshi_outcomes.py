"""Sync closed Kalshi tennis bets into ``training_history.db``.

For every settled tennis-ticker the account has touched, store one
row in the ``kalshi_outcomes`` table:

  * which side we bought (the side_player) and the opposing player
  * entry price (from /portfolio/fills — the actual yes_price the bot
    paid), settle price (from /portfolio/settlements — the canonical
    Kalshi-credited value)
  * whether our side won, realized P&L, fee paid
  * timestamps

The script is idempotent — re-running it on the same day just
re-writes the same rows. Designed to be wired into the daily
training-related housekeeping (or run on demand to backfill).

Run::

    python -m scripts.sync_kalshi_outcomes
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.data.training_db import upsert_kalshi_outcomes  # noqa: E402


log = logging.getLogger("scripts.sync_kalshi_outcomes")


def _paginate(client, path: str, key: str) -> list[dict]:
    """Walk the cursor pages on a Kalshi list endpoint and return
    everything. Used for /portfolio/fills + /portfolio/settlements
    which can each carry thousands of records on long-lived accounts.
    """
    out: list[dict] = []
    cursor = None
    while True:
        params: dict = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = client._request("GET", path, params=params)
        out.extend(r.get(key, []))
        cursor = r.get("cursor")
        if not cursor:
            break
    return out


def build_records(fills: list[dict], settlements: list[dict]) -> list[dict]:
    """Join fills against settlements and produce the dict shape
    ``training_db.upsert_kalshi_outcomes`` consumes.

    Only the tennis prefixes (``KXATPMATCH-`` / ``KXWTAMATCH-``) are
    kept — other Kalshi markets (NBA, jobless claims, etc.) aren't
    part of this database.
    """
    is_tennis = lambda t: (t or "").startswith(("KXATPMATCH-", "KXWTAMATCH-"))

    # Aggregate fills per ticker: first buy price + total fee. A given
    # ticker can have multiple fills (the close-via-offset trade is
    # also recorded as a fill); we use action='buy' for entry price
    # and sum all fees.
    entries: dict[str, dict] = {}
    for f in fills:
        tkr = f.get("ticker")
        if not is_tennis(tkr):
            continue
        e = entries.setdefault(tkr, {
            "entry_price": None,
            "fee_cost": 0.0,
            "opened_at": f.get("created_time"),
        })
        action = (f.get("action") or "").lower()
        if action == "buy" and e["entry_price"] is None:
            try:
                e["entry_price"] = float(f.get("yes_price_dollars") or 0)
            except (TypeError, ValueError):
                e["entry_price"] = None
            ct = f.get("created_time")
            if ct and (e["opened_at"] is None or ct < e["opened_at"]):
                e["opened_at"] = ct
        try:
            e["fee_cost"] += float(f.get("fee_cost") or 0)
        except (TypeError, ValueError):
            pass

    records: list[dict] = []
    for s in settlements:
        tkr = s.get("ticker")
        if not is_tennis(tkr):
            continue
        event_ticker = s.get("event_ticker")
        # Derive players from the ticker: KXATPMATCH-26JUN09STRGAL-GAL
        # -> event=KXATPMATCH-26JUN09STRGAL, side player tri=GAL,
        # other player tri = the prefix's last 6 (3+3 chars).
        parts = (tkr or "").rsplit("-", 1)
        side_tri = parts[1] if len(parts) == 2 else None
        # Players tri-codes encoded in the event suffix: last 6 chars
        # of the event ticker are the two 3-letter player codes.
        suffix = ""
        if event_ticker:
            suffix = event_ticker.split("-")[-1][-6:]
        other_tri = None
        if side_tri and suffix and side_tri in suffix:
            other_tri = suffix.replace(side_tri, "", 1) or None

        market_result = (s.get("market_result") or "").lower()
        # ``yes_count_fp`` > 0 means we held YES. Our convention is
        # always YES-side, so won = (market_result == "yes").
        try:
            yes_held = float(s.get("yes_count_fp") or 0) > 0
        except (TypeError, ValueError):
            yes_held = False
        if market_result in ("yes", "no") and yes_held:
            won = 1 if market_result == "yes" else 0
        elif market_result and market_result not in ("yes", "no"):
            won = None  # void / scalar / refund
        else:
            won = 0
        # Settle price in dollars (settle_value is cents 0..100).
        try:
            settle_value = int(s.get("value"))
        except (TypeError, ValueError):
            settle_value = None
        settle_price = (settle_value / 100.0
                         if settle_value is not None else None)

        e = entries.get(tkr, {})
        entry_price = e.get("entry_price")
        realized = None
        if entry_price is not None and settle_price is not None:
            realized = settle_price - entry_price - (e.get("fee_cost") or 0)

        records.append({
            "ticker": tkr,
            "event_ticker": event_ticker,
            "side_player": side_tri,
            "other_player": other_tri,
            "surface": None,  # not in settlement payload; left for future
            "market_result": market_result or None,
            "settle_value": settle_value,
            "won": won,
            "entry_price": entry_price,
            "settle_price": settle_price,
            "realized_pnl": realized,
            "fee_cost": e.get("fee_cost"),
            "opened_at": e.get("opened_at"),
            "closed_at": s.get("settled_time"),
        })
    return records


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        from kalshi_sdk import KalshiClient
    except ImportError:
        log.error("kalshi_sdk not installed in this venv")
        return 1

    api_key_id = os.environ.get("KALSHI_API_KEY_ID")
    priv_key = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if not api_key_id or not priv_key:
        log.error("KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH not set")
        return 1
    client = KalshiClient(api_key_id=api_key_id, private_key_path=priv_key)

    log.info("fetching fills…")
    fills = _paginate(client, "/portfolio/fills", "fills")
    log.info("fetching settlements…")
    settlements = _paginate(client, "/portfolio/settlements", "settlements")
    records = build_records(fills, settlements)
    log.info("building %d outcome records from %d fills / %d settlements",
              len(records), len(fills), len(settlements))

    db_path = ROOT / "data" / "training_history.db"
    n = upsert_kalshi_outcomes(db_path, records)
    log.info("upserted %d rows into %s", n, db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
