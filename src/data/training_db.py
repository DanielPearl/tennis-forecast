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
    draw_size                   INTEGER,
    best_of                     INTEGER,
    player_a                    TEXT    NOT NULL,
    player_b                    TEXT    NOT NULL,
    label                       INTEGER NOT NULL,
    used_in_split               TEXT,
    -- Player A raw attributes (snapshot at match time)
    a_age                       REAL,
    a_height_cm                 REAL,
    a_hand                      TEXT,
    a_country                   TEXT,
    a_rank                      INTEGER,
    a_rank_points               INTEGER,
    a_seed                      INTEGER,
    a_entry                     TEXT,
    -- Player B raw attributes
    b_age                       REAL,
    b_height_cm                 REAL,
    b_hand                      TEXT,
    b_country                   TEXT,
    b_rank                      INTEGER,
    b_rank_points               INTEGER,
    b_seed                      INTEGER,
    b_entry                     TEXT,
    -- Engineered features (the 12 PREMATCH_FEATURES the model uses)
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
    -- Extra candidate features (computed but currently NOT in the
    -- model's selected list — included so the dashboard can show
    -- "every feature considered before pruning")
    age_diff                    REAL,
    height_diff_cm              REAL,
    rank_points_diff            INTEGER,
    seed_diff                   INTEGER,
    hand_match                  INTEGER,  -- 1 same, 0 diff, NULL unknown
    same_country                INTEGER,
    trained_at                  TEXT    NOT NULL,
    UNIQUE (tourney_date, player_a, player_b, round)
);

CREATE INDEX IF NOT EXISTS idx_training_matches_date
    ON training_matches (tourney_date);
CREATE INDEX IF NOT EXISTS idx_training_matches_tour
    ON training_matches (tour);
CREATE INDEX IF NOT EXISTS idx_training_matches_split
    ON training_matches (used_in_split);

-- Self-evolving schema: add columns if an older DB lives on disk.
-- Avoids requiring an explicit migration step.


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

# Raw + derived columns the dashboard surfaces in addition to the
# 12 modeled features. These are stored regardless of whether the
# trainer ends up selecting them — the user wants the page to show
# every feature considered before pruning.
_EXTRA_COLS = (
    "draw_size", "best_of",
    "a_age", "a_height_cm", "a_hand", "a_country",
    "a_rank", "a_rank_points", "a_seed", "a_entry",
    "b_age", "b_height_cm", "b_hand", "b_country",
    "b_rank", "b_rank_points", "b_seed", "b_entry",
    "age_diff", "height_diff_cm",
    "rank_points_diff", "seed_diff",
    "hand_match", "same_country",
)


# Columns introduced after the first cut of the schema. Each entry is
# (column-name, sqlite-type) and gets ALTER TABLE-added on connect if
# missing — keeps existing DBs forward-compatible without a separate
# migration script.
_NEW_COLUMNS_TRAINING: tuple[tuple[str, str], ...] = (
    ("draw_size", "INTEGER"), ("best_of", "INTEGER"),
    ("a_age", "REAL"), ("a_height_cm", "REAL"),
    ("a_hand", "TEXT"), ("a_country", "TEXT"),
    ("a_rank", "INTEGER"), ("a_rank_points", "INTEGER"),
    ("a_seed", "INTEGER"), ("a_entry", "TEXT"),
    ("b_age", "REAL"), ("b_height_cm", "REAL"),
    ("b_hand", "TEXT"), ("b_country", "TEXT"),
    ("b_rank", "INTEGER"), ("b_rank_points", "INTEGER"),
    ("b_seed", "INTEGER"), ("b_entry", "TEXT"),
    ("age_diff", "REAL"), ("height_diff_cm", "REAL"),
    ("rank_points_diff", "INTEGER"), ("seed_diff", "INTEGER"),
    ("hand_match", "INTEGER"), ("same_country", "INTEGER"),
)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add any column from ``_NEW_COLUMNS_TRAINING`` that's missing from
    the live ``training_matches`` table. Idempotent — ``PRAGMA
    table_info`` is checked first so re-running is a no-op."""
    existing = {r[1] for r in conn.execute(
        "PRAGMA table_info(training_matches)").fetchall()}
    for name, sql_type in _NEW_COLUMNS_TRAINING:
        if name not in existing:
            conn.execute(
                f"ALTER TABLE training_matches ADD COLUMN {name} {sql_type}"
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
        _ensure_columns(conn)
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


def backfill_extra_attrs(
    db_path: str | Path,
    matches_csv_path: str | Path,
) -> int:
    """Backfill the raw player attributes + derived diffs by joining
    ``training_matches`` against ``matches_clean.csv`` on
    (tourney_date, winner_name, loser_name). The CSV is the only
    source for height/hand/country/rank etc.; the trainer doesn't
    carry those through ``build_player_a_panel``.

    Idempotent. Called by both the trainer (after each fit) and the
    manual backfill script.
    """
    src = pd.read_csv(matches_csv_path, usecols=[
        "tourney_date", "winner_name", "loser_name",
        "draw_size", "best_of",
        "winner_age", "winner_ht", "winner_hand", "winner_ioc",
        "winner_rank", "winner_rank_points",
        "winner_seed", "winner_entry",
        "loser_age", "loser_ht", "loser_hand", "loser_ioc",
        "loser_rank", "loser_rank_points",
        "loser_seed", "loser_entry",
    ])
    src["tourney_date"] = pd.to_datetime(
        src["tourney_date"], errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    # Lookup keyed on the orientation we'll see in training_matches:
    # both (date, winner, loser) and (date, loser, winner) point at
    # the same source row, with a flag that says which side is A.
    by_key: dict[tuple[str, str, str], tuple[pd.Series, bool]] = {}
    for r in src.itertuples():
        date = r.tourney_date
        if not date or pd.isna(date):
            continue
        w_name = r.winner_name
        l_name = r.loser_name
        by_key[(date, w_name, l_name)] = (r, True)   # A = winner
        by_key[(date, l_name, w_name)] = (r, False)  # A = loser

    update_sql = (
        "UPDATE training_matches SET "
        + ", ".join(f"{c} = ?" for c in _EXTRA_COLS)
        + " WHERE id = ?"
    )

    updated = 0
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, tourney_date, player_a, player_b "
            "FROM training_matches"
        ).fetchall()
        batch: list[tuple] = []
        for rid, date, pa, pb in rows:
            hit = by_key.get((date, pa, pb))
            if not hit:
                continue
            r, a_is_winner = hit
            if a_is_winner:
                a_age, a_ht, a_hand, a_ioc = (r.winner_age, r.winner_ht,
                                                 r.winner_hand, r.winner_ioc)
                a_rank, a_rp = r.winner_rank, r.winner_rank_points
                a_seed, a_entry = r.winner_seed, r.winner_entry
                b_age, b_ht, b_hand, b_ioc = (r.loser_age, r.loser_ht,
                                                 r.loser_hand, r.loser_ioc)
                b_rank, b_rp = r.loser_rank, r.loser_rank_points
                b_seed, b_entry = r.loser_seed, r.loser_entry
            else:
                a_age, a_ht, a_hand, a_ioc = (r.loser_age, r.loser_ht,
                                                 r.loser_hand, r.loser_ioc)
                a_rank, a_rp = r.loser_rank, r.loser_rank_points
                a_seed, a_entry = r.loser_seed, r.loser_entry
                b_age, b_ht, b_hand, b_ioc = (r.winner_age, r.winner_ht,
                                                 r.winner_hand, r.winner_ioc)
                b_rank, b_rp = r.winner_rank, r.winner_rank_points
                b_seed, b_entry = r.winner_seed, r.winner_entry

            age_diff = _diff(a_age, b_age)
            height_diff = _diff(a_ht, b_ht)
            rp_diff = _diff(a_rp, b_rp)
            seed_diff = _diff(a_seed, b_seed)
            hand_match = _hand_match(a_hand, b_hand)
            same_country = (1 if (a_ioc and b_ioc and a_ioc == b_ioc)
                              else 0)

            batch.append((
                _safe_int(r.draw_size), _safe_int(r.best_of),
                _safe_float(a_age), _safe_float(a_ht),
                _safe_str(a_hand), _safe_str(a_ioc),
                _safe_int(a_rank), _safe_int(a_rp),
                _safe_int(a_seed), _safe_str(a_entry),
                _safe_float(b_age), _safe_float(b_ht),
                _safe_str(b_hand), _safe_str(b_ioc),
                _safe_int(b_rank), _safe_int(b_rp),
                _safe_int(b_seed), _safe_str(b_entry),
                age_diff, height_diff, rp_diff, seed_diff,
                hand_match, same_country,
                rid,
            ))
            if len(batch) >= 5000:
                conn.execute("BEGIN")
                conn.executemany(update_sql, batch)
                conn.execute("COMMIT")
                updated += len(batch)
                batch = []
        if batch:
            conn.execute("BEGIN")
            conn.executemany(update_sql, batch)
            conn.execute("COMMIT")
            updated += len(batch)
    return updated


def _diff(a, b) -> Optional[float]:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    try:
        return float(a) - float(b)
    except (TypeError, ValueError):
        return None


def _hand_match(a, b) -> Optional[int]:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    if not str(a).strip() or not str(b).strip():
        return None
    return 1 if str(a) == str(b) else 0


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
    offset: int | None = None,
    limit: int | None = None,
    tour: str | None = None,
    split: str | None = None,
) -> list[dict]:
    """Pull a slice of training rows in newest-date-first order.

    Callers that paginate uniformly pass ``page`` + ``page_size``.
    Callers that need an arbitrary offset (e.g. a combined union with
    a Kalshi-only prefix where the historical slice may start at a
    non-multiple-of-page-size offset) pass ``offset`` + ``limit``
    directly — those take precedence when set.
    """
    if offset is not None or limit is not None:
        used_offset = max(0, offset or 0)
        used_limit = max(0, limit if limit is not None else page_size)
    else:
        used_offset = max(0, (page - 1) * page_size)
        used_limit = page_size
    sql = "SELECT * FROM training_matches WHERE 1=1"
    params: list[Any] = []
    if tour:
        sql += " AND tour = ?"
        params.append(tour)
    if split:
        sql += " AND used_in_split = ?"
        params.append(split)
    sql += " ORDER BY tourney_date DESC, id DESC LIMIT ? OFFSET ?"
    params.extend([used_limit, used_offset])
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
