"""Close tennis positions that violate the new
``max_hours_to_close_for_open`` gate.

After deploying the 12-hour time-to-close gate, we still hold
positions opened pre-deploy on matches that close >12h away. Each
one (a) was placed by the OLD code with no time-gate filter and
(b) burns a slot in the new max_open=5 cap. This script enumerates
them and places IOC limit sell orders at the current best bid so
they exit at market.

Pass ``--commit`` to actually fire the sells. Default is dry-run:
prints what it would do without touching real money.

Run with the tennis-forecast venv (needs kalshi_sdk):
  source /root/tennis-forecast/.env
  /root/tennis-forecast/.venv/bin/python scripts/close_violating_positions.py --commit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.utils.logging_setup import setup_logging

log = setup_logging("scripts.close_violating")


# Same default as the executor's hard cap. Positions whose Kalshi
# market expiration is more than this many hours away are considered
# violators of the new gate.
DEFAULT_MAX_HOURS = 12.0


def _client():
    from kalshi_sdk import KalshiClient
    return KalshiClient(
        api_key_id=os.environ["KALSHI_API_KEY_ID"],
        private_key_path=os.environ["KALSHI_PRIVATE_KEY_PATH"],
    )


def _close_order_id(ticker: str, side: str, price_cents: int) -> str:
    """Distinct idempotency key from any buy-side order — prefixes with
    ``close-`` so we don't collide with the executor's own
    client_order_id."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    seed = f"close|{ticker}|{side}|{price_cents}|{today}"
    return "close-" + hashlib.sha256(seed.encode()).hexdigest()[:28]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--commit", action="store_true",
                   help="Actually place the sell orders. Default: dry-run.")
    p.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS,
                   help=f"Threshold in hours. Default {DEFAULT_MAX_HOURS}.")
    p.add_argument("--state-path", type=str,
                   default=str(_REPO / "data" / "outputs-live"
                                / "sim_state.json"),
                   help="Path to sim_state.json (default: live's).")
    args = p.parse_args()

    state_path = Path(args.state_path)
    if not state_path.exists():
        log.error("sim_state %s not found", state_path)
        return 2
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    opens = state.get("open_positions", []) or []
    if not opens:
        log.info("no open positions; nothing to close")
        return 0

    c = _client()
    now_ts = datetime.now(timezone.utc).timestamp()

    violators: list[dict] = []
    for op in opens:
        ticker = op.get("ticker") or ""
        if not ticker:
            continue
        try:
            mkt_resp = c.get_market(ticker)
        except Exception as exc:  # noqa: BLE001
            log.warning("get_market(%s) failed: %s", ticker, exc)
            continue
        mkt = (mkt_resp or {}).get("market") or {}
        exp_str = mkt.get("expected_expiration_time") or ""
        try:
            exp_ts = datetime.fromisoformat(
                exp_str.replace("Z", "+00:00")
            ).timestamp()
        except (TypeError, ValueError):
            log.warning("can't parse expiration %s for %s",
                         exp_str, ticker)
            continue
        hours_to_close = (exp_ts - now_ts) / 3600.0
        if hours_to_close <= args.max_hours:
            log.info("ok: %s closes in %.1fh (<= %.1f threshold) — keep",
                     ticker, hours_to_close, args.max_hours)
            continue
        # Pricing for the sell. We hold YES; selling YES at the best
        # available bid is the cleanest exit. yes_bid is the price
        # someone is willing to pay us right now.
        yes_bid_d = mkt.get("yes_bid_dollars")
        if yes_bid_d is None:
            yb = mkt.get("yes_bid")
            yes_bid_cents = int(yb) if yb is not None else None
        else:
            yes_bid_cents = int(round(float(yes_bid_d) * 100))
        if yes_bid_cents is None or yes_bid_cents <= 0:
            log.warning("skip %s: no yes_bid available (book too thin)",
                         ticker)
            continue
        entry = int(round(float(op.get("entry_market_prob") or 0) * 100))
        pnl_per = yes_bid_cents - entry
        violators.append({
            "ticker": ticker,
            "side_player": op.get("side_player") or "",
            "match_id": op.get("match_id") or "",
            "entry_cents": entry,
            "current_yes_bid_cents": yes_bid_cents,
            "hours_to_close": hours_to_close,
            "pnl_per_contract_cents": pnl_per,
        })

    if not violators:
        log.info("no positions violating the %.1fh threshold; done",
                 args.max_hours)
        return 0

    print()
    print(f"Found {len(violators)} positions violating the "
          f"{args.max_hours:.1f}h gate:")
    print(f"  {'ticker':45} {'player':28} {'closes_in':>10} "
          f"{'entry':>6} {'bid':>6} {'pnl/ct':>8}")
    total_pnl = 0
    for v in violators:
        print(f"  {v['ticker']:45} {v['side_player']:28} "
              f"{v['hours_to_close']:9.1f}h "
              f"{v['entry_cents']:5d}¢ {v['current_yes_bid_cents']:5d}¢ "
              f"{v['pnl_per_contract_cents']:+7d}¢")
        total_pnl += v["pnl_per_contract_cents"]
    print(f"  {'total':>45} {'':28} {'':>10} {'':>6} {'':>6} "
          f"{total_pnl:+7d}¢")
    print()

    if not args.commit:
        print("DRY-RUN. Rerun with --commit to actually place the sells.")
        return 0

    print("placing IOC sell orders...")
    placed = 0
    failed = 0
    for v in violators:
        ticker = v["ticker"]
        bid = v["current_yes_bid_cents"]
        coid = _close_order_id(ticker, "yes-sell", bid)
        try:
            # Sell YES at the best bid (or 1¢ below for taker certainty).
            # IOC: fill what we can immediately, cancel the rest.
            resp = c.place_order(
                ticker=ticker,
                side="yes",
                action="sell",
                count=1,
                order_type="limit",
                yes_price=bid,
                time_in_force="immediate_or_cancel",
                client_order_id=coid,
            )
            order = (resp or {}).get("order") or {}
            order_id = order.get("order_id") or "?"
            status = order.get("status") or "?"
            print(f"  CLOSED {ticker} at {bid}¢ — order_id={order_id} "
                  f"status={status}")
            placed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {ticker} at {bid}¢ — {exc!r}")
            failed += 1

    print()
    print(f"placed {placed} sells, {failed} failures. Reconcile will "
          f"pick up the executions on the next bot tick.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
