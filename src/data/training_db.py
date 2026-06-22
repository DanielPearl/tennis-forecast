"""SQLite-backed store of the model's training panel + the Kalshi
bet outcomes the live trader has produced.

Two tables:

* ``training_matches`` — one row per (match, player-A-orientation)
  observation the trainer feeds into the model. Carries every
  ``PREMATCH_FEATURES`` value the trainer used, the binary label
  (1 = player_a won), and which time-slice this row landed in
  (``train`` / ``val`` / ``test``). Populated at training time by
  :func:`upsert_training_panel` so the dashboard's Training Data
  page can render the exact rows the model saw — not a re-derived
  approximation.

* ``kalshi_outcomes`` — one row per ticker the bot has actually
  filled, with the entry price, the settlement price, the realized
  P&L, and the binary "did our side win" label. Populated by
  :func:`upsert_kalshi_outcomes` (called by the daily Kalshi sync
  script) from ``/portfolio/fills`` + ``/portfolio/settlements``.

Schema is conservative: integer surrogate primary key, natural-key
uniqueness via ``UNIQUE`` constraints so re-runs of the trainer or
Kalshi sync are idempotent. The DB lives at
``data/training_history.db`` relative to the repo root.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pandas as pd


_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_matches (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    tourney_date                TEXT    NOT NULL,
    tourney_name                TEXT,
    tour                        TEXT,
    surface                     TEXT,
    level                       TEXT,
    round                       TEXT,
    player_a                    TEXT    NOT NULL,
    player_b                    TEXT    NOT NULL,
    label                       INTEGER NOT NULL,
    used_in_split               TEXT,
    diff_elo_pre                REAL,
    diff_surface_elo_pre        REAL,
    diff_form_last5             REAL,
    diff_form_last10            REAL,
    diff_avg_serve_pts_won_10   REAL,
    diff_avg_return_pts_won_10  REAL,
    diff_avg_bp_saved_10        REAL,
    diff_days_rest              REAL,
    h2h_a_wins_minus_b_wins     REAL,
    rank_diff                   REAL,
    level_rank                  REAL,
    round_rank                  REAL,
    trained_at                  TEXT    NOT NULL,
    UNIQUE (tourney_date, player_a, player_b, round)
);

CREATE INDEX IF NOT EXISTS idx_training_matches_date
    ON training_matches (tourney_date);
CREATE INDEX IF NOT EXISTS idx_training_matches_tour
    ON training_matches (tour);
CREATE INDEX IF NOT EXISTS idx_training_matches_split
    ON training_matches (used_in_split);

CREATE TABLE IF NOT EXISTS kalshi_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    UNIQUE NOT NULL,
    event_ticker    TEXT,
    side_player     TEXT,
    other_player    TEXT,
    surface         TEXT,
    market_result   TEXT,
    settle_value    INTEGER,
    won             INTEGER,
    entry_price     REAL,
    settle_price    REAL,
    realized_pnl    REAL,
    fee_cost        REAL,
    opened_at       TEXT,
    closed_at       TEXT,
    synced_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kalshi_outcomes_closed
    ON kalshi_outcomes (closed_at);
"""


_FEATURE_COLS = (
    "diff_elo_pre", "diff_surface_elo_pre",
    "diff_form_last5", "diff_form_last10",
    "diff_avg_serve_pts_won_10", "diff_avg_return_pts_won_10",
    "diff_avg_bp_saved_10", "diff_days_rest",
    "h2h_a_wins_minus_b_wins", "rank_diff",
    "level_rank", "round_rank",
)


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a connection with WAL on (so the dashboard can read while
    the trainer writes) and the schema applied. ``check_same_thread``
    is off because the dashboard's request handler may dispatch to a
    different thread than the one that opened the connection."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path), check_same_thread=False, isolation_level=None,
    )
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA)
        yield conn
    finally:
        conn.close()


def upsert_training_panel(
    db_path: str | Path,
    oriented: pd.DataFrame,
    *,
    split_cutoff_train: pd.Timestamp,
    split_cutoff_val: pd.Timestamp,
) -> int:
    """Write every row from the ``oriented`` (player-A orientation)
    panel to ``training_matches``. Idempotent — rerunning the trainer
    overwrites prior records via ``ON CONFLICT REPLACE`` on the
    natural key ``(tourney_date, player_a, player_b, round)``.

    Each row is tagged with which time-window it fell into:
      * ``train`` if its date is < ``split_cutoff_train``
      * ``val``   if  train ≤ date < ``split_cutoff_val``
      * ``test``  if  date ≥ ``split_cutoff_val``

    Returns the number of rows written.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for _, r in oriented.iterrows():
        date = r.get("tourney_date")
        if pd.isna(date):
            continue
        if isinstance(date, pd.Timestamp):
            date_str = date.strftime("%Y-%m-%d")
            ts = date
        else:
            ts = pd.Timestamp(date)
            date_str = ts.strftime("%Y-%m-%d")
        if ts < split_cutoff_train:
            split = "train"
        elif ts < split_cutoff_val:
            split = "val"
        else:
            split = "test"
        rows.append((
            date_str,
            _safe_str(r.get("tourney_name")),
            _safe_str(r.get("tour")),
            _safe_str(r.get("surface")),
            _safe_str(r.get("level")) or _safe_str(r.get("tourney_level")),
            _safe_str(r.get("round")),
            _safe_str(r.get("player_a")) or "?",
            _safe_str(r.get("player_b")) or "?",
            int(r.get("y") or 0),
            split,
            *[_safe_float(r.get(c)) for c in _FEATURE_COLS],
            now,
        ))
    if not rows:
        return 0
    placeholders = ", ".join("?" * (10 + len(_FEATURE_COLS) + 1))
    cols = (
        "tourney_date, tourney_name, tour, surface, level, round, "
        "player_a, player_b, label, used_in_split, "
        + ", ".join(_FEATURE_COLS)
        + ", trained_at"
    )
    sql = (
        f"INSERT INTO training_matches ({cols}) VALUES ({placeholders}) "
        "ON CONFLICT (tourney_date, player_a, player_b, round) "
        "DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in _FEATURE_COLS)
        + ", label=excluded.label, used_in_split=excluded.used_in_split,"
        " trained_at=excluded.trained_at"
    )
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        conn.executemany(sql, rows)
        conn.execute("COMMIT")
    return len(rows)


def upsert_kalshi_outcomes(
    db_path: str | Path,
    records: Iterable[dict],
) -> int:
    """Persist closed Kalshi bets. Idempotent on ``ticker``. Returns
    the number of rows touched (insert + update).

    Each input record is the dict shape we already store in
    sim_state.json's ``closed_positions`` — see
    :func:`scripts/sync_kalshi_outcomes.build_records` for the
    field-by-field mapping from /portfolio/fills + /portfolio/settlements.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for rec in records:
        rows.append((
            rec.get("ticker"),
            rec.get("event_ticker"),
            rec.get("side_player"),
            rec.get("other_player"),
            rec.get("surface"),
            rec.get("market_result"),
            _safe_int(rec.get("settle_value")),
            _safe_int(rec.get("won")),
            _safe_float(rec.get("entry_price")),
            _safe_float(rec.get("settle_price")),
            _safe_float(rec.get("realized_pnl")),
            _safe_float(rec.get("fee_cost")),
            rec.get("opened_at"),
            rec.get("closed_at"),
            now,
        ))
    if not rows:
        return 0
    sql = (
        "INSERT INTO kalshi_outcomes ("
        "ticker, event_ticker, side_player, other_player, surface, "
        "market_result, settle_value, won, entry_price, settle_price, "
        "realized_pnl, fee_cost, opened_at, closed_at, synced_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(ticker) DO UPDATE SET "
        "market_result=excluded.market_result, "
        "settle_value=excluded.settle_value, won=excluded.won, "
        "settle_price=excluded.settle_price, "
        "realized_pnl=excluded.realized_pnl, "
        "fee_cost=excluded.fee_cost, "
        "closed_at=excluded.closed_at, synced_at=excluded.synced_at"
    )
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        conn.executemany(sql, rows)
        conn.execute("COMMIT")
    return len(rows)


def count_training_matches(db_path: str | Path,
                              tour: str | None = None,
                              split: str | None = None) -> int:
    """Cheap row count for the dashboard pagination."""
    sql = "SELECT COUNT(*) FROM training_matches WHERE 1=1"
    params: list[Any] = []
    if tour:
        sql += " AND tour = ?"
        params.append(tour)
    if split:
        sql += " AND used_in_split = ?"
        params.append(split)
    with connect(db_path) as conn:
        return int(conn.execute(sql, params).fetchone()[0])


def fetch_training_matches(
    db_path: str | Path,
    *,
    page: int = 1,
    page_size: int = 50,
    tour: str | None = None,
    split: str | None = None,
) -> list[dict]:
    """Pull a page of training rows in newest-date-first order."""
    offset = max(0, (page - 1) * page_size)
    sql = "SELECT * FROM training_matches WHERE 1=1"
    params: list[Any] = []
    if tour:
        sql += " AND tour = ?"
        params.append(tour)
    if split:
        sql += " AND used_in_split = ?"
        params.append(split)
    sql += " ORDER BY tourney_date DESC, id DESC LIMIT ? OFFSET ?"
    params.extend([page_size, offset])
    with connect(db_path) as conn:
        cur = conn.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _safe_str(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


def _safe_float(v: Any) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
