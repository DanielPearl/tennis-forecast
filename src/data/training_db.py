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
-- ── Normalised relational schema (the v2 design) ───────────────────
-- 6 tables FK-connected. ``training_matches`` and ``kalshi_outcomes``
-- below are the legacy flat tables — kept around because the trainer
-- writes to ``training_matches`` and the dashboard's legacy queries
-- still read it. The new tables below mirror the same data in a
-- normalised shape for queries that JOIN across stats / outcomes /
-- bets.

CREATE TABLE IF NOT EXISTS players (
    player_id       TEXT    PRIMARY KEY,
    name            TEXT    NOT NULL,
    tour            TEXT,
    country         TEXT,
    hand            TEXT,
    height_cm       INTEGER,
    first_seen_date TEXT,
    last_seen_date  TEXT
);
CREATE INDEX IF NOT EXISTS idx_players_name ON players (name);
CREATE INDEX IF NOT EXISTS idx_players_tour ON players (tour);

CREATE TABLE IF NOT EXISTS tournaments (
    tourney_id      TEXT    PRIMARY KEY,
    name            TEXT,
    tour            TEXT,
    surface         TEXT,
    level           TEXT,
    draw_size       INTEGER,
    year            INTEGER,
    start_date      TEXT
);
CREATE INDEX IF NOT EXISTS idx_tournaments_year ON tournaments (year);

CREATE TABLE IF NOT EXISTS matches (
    match_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tourney_id          TEXT    REFERENCES tournaments (tourney_id),
    match_date          TEXT    NOT NULL,
    round               TEXT,
    match_num           INTEGER,
    best_of             INTEGER,
    minutes             INTEGER,
    score               TEXT,
    winner_id           TEXT    REFERENCES players (player_id),
    loser_id            TEXT    REFERENCES players (player_id),
    winner_seed         INTEGER,
    loser_seed          INTEGER,
    winner_entry        TEXT,
    loser_entry         TEXT,
    winner_age          REAL,
    loser_age           REAL,
    winner_rank         INTEGER,
    loser_rank          INTEGER,
    winner_rank_points  INTEGER,
    loser_rank_points   INTEGER,
    UNIQUE (tourney_id, match_num, round, winner_id, loser_id)
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (match_date);
CREATE INDEX IF NOT EXISTS idx_matches_tourney ON matches (tourney_id);
CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches (winner_id);
CREATE INDEX IF NOT EXISTS idx_matches_loser ON matches (loser_id);

CREATE TABLE IF NOT EXISTS match_stats (
    match_id             INTEGER NOT NULL REFERENCES matches (match_id),
    player_id            TEXT    NOT NULL REFERENCES players (player_id),
    is_winner            INTEGER NOT NULL,
    aces                 INTEGER,
    double_faults        INTEGER,
    serve_points         INTEGER,
    first_serves_in      INTEGER,
    first_serves_won     INTEGER,
    second_serves_won    INTEGER,
    service_games        INTEGER,
    break_points_saved   INTEGER,
    break_points_faced   INTEGER,
    PRIMARY KEY (match_id, player_id)
);

CREATE TABLE IF NOT EXISTS match_features (
    feature_row_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id                  INTEGER NOT NULL REFERENCES matches (match_id),
    orientation               TEXT    NOT NULL,  -- 'a_winner' or 'a_loser'
    player_a_id               TEXT    REFERENCES players (player_id),
    player_b_id               TEXT    REFERENCES players (player_id),
    label                     INTEGER NOT NULL,
    used_in_split             TEXT,
    diff_elo_pre              REAL,
    diff_surface_elo_pre      REAL,
    diff_form_last5           REAL,
    diff_form_last10          REAL,
    diff_avg_serve_pts_won_10 REAL,
    diff_avg_return_pts_won_10 REAL,
    diff_avg_bp_saved_10      REAL,
    diff_days_rest            REAL,
    h2h_a_wins_minus_b_wins   REAL,
    rank_diff                 REAL,
    level_rank                REAL,
    round_rank                REAL,
    UNIQUE (match_id, orientation)
);
CREATE INDEX IF NOT EXISTS idx_match_features_match
    ON match_features (match_id);

CREATE TABLE IF NOT EXISTS kalshi_bets (
    bet_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    UNIQUE NOT NULL,
    event_ticker    TEXT,
    match_id        INTEGER REFERENCES matches (match_id),
    side_player_id  TEXT    REFERENCES players (player_id),
    side_tricode    TEXT,
    other_tricode   TEXT,
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
CREATE INDEX IF NOT EXISTS idx_kalshi_bets_match
    ON kalshi_bets (match_id);
CREATE INDEX IF NOT EXISTS idx_kalshi_bets_closed
    ON kalshi_bets (closed_at);

-- ── Legacy flat tables (still written by the trainer + sync) ──────
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


def populate_normalized_tables(
    db_path: str | Path,
    matches_csv_path: str | Path,
) -> Dict[str, int]:
    """One-shot populate (or refresh) the v2 normalised tables from
    ``matches_clean.csv``: players, tournaments, matches, match_stats.

    Idempotent — uses ``INSERT OR REPLACE`` on every table's natural
    key so re-running just refreshes any value that changed.

    Returns a dict with the row count written per table.
    """
    src = pd.read_csv(matches_csv_path, low_memory=False)
    src["tourney_date_str"] = pd.to_datetime(
        src["tourney_date"], errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    src["year"] = pd.to_datetime(
        src["tourney_date"], errors="coerce",
    ).dt.year
    # tour comes in as ``atp``/``wta`` — normalise to uppercase to
    # match what the dashboard's filters use.
    src["tour"] = src["tour"].fillna("").str.upper()

    # ── Players ──────────────────────────────────────────────────────
    # Concatenate winner + loser rosters, then dedupe by player_id
    # keeping the most-recent record so name spelling / height
    # corrections propagate.
    w = src.rename(columns={
        "winner_id": "player_id", "winner_name": "name",
        "winner_hand": "hand", "winner_ht": "height_cm",
        "winner_ioc": "country",
    })[["player_id", "name", "tour", "country", "hand", "height_cm",
         "tourney_date_str"]]
    l = src.rename(columns={
        "loser_id": "player_id", "loser_name": "name",
        "loser_hand": "hand", "loser_ht": "height_cm",
        "loser_ioc": "country",
    })[["player_id", "name", "tour", "country", "hand", "height_cm",
         "tourney_date_str"]]
    roster = pd.concat([w, l], ignore_index=True)
    roster = roster.dropna(subset=["player_id"])
    roster["player_id"] = roster["player_id"].astype(str)
    roster = roster.sort_values("tourney_date_str")
    first_seen = roster.groupby("player_id")["tourney_date_str"].first()
    last_seen = roster.groupby("player_id")["tourney_date_str"].last()
    latest = roster.drop_duplicates("player_id", keep="last")
    latest = latest.merge(first_seen.rename("first_seen_date"),
                            left_on="player_id", right_index=True)
    latest = latest.merge(last_seen.rename("last_seen_date"),
                            left_on="player_id", right_index=True)

    # ── Tournaments ──────────────────────────────────────────────────
    tdf = src.drop_duplicates("tourney_id")[[
        "tourney_id", "tourney_name", "tour", "surface", "tourney_level",
        "draw_size", "year", "tourney_date_str",
    ]].copy()
    tdf = tdf.dropna(subset=["tourney_id"])

    # ── Matches ──────────────────────────────────────────────────────
    mdf = src.copy()
    mdf["match_natural_key"] = (
        mdf["tourney_id"].astype(str) + "|"
        + mdf["match_num"].astype(str) + "|"
        + mdf["round"].fillna("").astype(str) + "|"
        + mdf["winner_id"].astype(str) + "|"
        + mdf["loser_id"].astype(str)
    )

    n_players = 0
    n_tourneys = 0
    n_matches = 0
    n_stats = 0
    with connect(db_path) as conn:
        # players
        conn.execute("BEGIN")
        for r in latest.itertuples(index=False):
            conn.execute(
                "INSERT OR REPLACE INTO players "
                "(player_id, name, tour, country, hand, height_cm, "
                "first_seen_date, last_seen_date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(r.player_id),
                 _safe_str(r.name) or "?",
                 _safe_str(r.tour),
                 _safe_str(r.country),
                 _safe_str(r.hand),
                 _safe_int(r.height_cm),
                 _safe_str(r.first_seen_date),
                 _safe_str(r.last_seen_date)),
            )
            n_players += 1
        conn.execute("COMMIT")

        # tournaments
        conn.execute("BEGIN")
        for r in tdf.itertuples(index=False):
            conn.execute(
                "INSERT OR REPLACE INTO tournaments "
                "(tourney_id, name, tour, surface, level, draw_size, "
                "year, start_date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(r.tourney_id),
                 _safe_str(r.tourney_name),
                 _safe_str(r.tour),
                 _safe_str(r.surface),
                 _safe_str(r.tourney_level),
                 _safe_int(r.draw_size),
                 _safe_int(r.year),
                 _safe_str(r.tourney_date_str)),
            )
            n_tourneys += 1
        conn.execute("COMMIT")

        # matches: INSERT OR IGNORE then UPDATE so we get back a stable
        # match_id we can use for match_stats below.
        conn.execute("BEGIN")
        match_ids: dict[str, int] = {}
        for r in mdf.itertuples(index=False):
            key = (str(r.tourney_id), _safe_int(r.match_num),
                   _safe_str(r.round),
                   str(r.winner_id), str(r.loser_id))
            cur = conn.execute(
                "INSERT OR IGNORE INTO matches "
                "(tourney_id, match_date, round, match_num, best_of, "
                "minutes, score, winner_id, loser_id, "
                "winner_seed, loser_seed, winner_entry, loser_entry, "
                "winner_age, loser_age, "
                "winner_rank, loser_rank, "
                "winner_rank_points, loser_rank_points) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?)",
                (key[0], _safe_str(r.tourney_date_str), key[2], key[1],
                 _safe_int(r.best_of), _safe_int(getattr(r, "minutes", None)),
                 _safe_str(getattr(r, "score", None)),
                 key[3], key[4],
                 _safe_int(r.winner_seed), _safe_int(r.loser_seed),
                 _safe_str(r.winner_entry), _safe_str(r.loser_entry),
                 _safe_float(r.winner_age), _safe_float(r.loser_age),
                 _safe_int(r.winner_rank), _safe_int(r.loser_rank),
                 _safe_int(r.winner_rank_points),
                 _safe_int(r.loser_rank_points)),
            )
            mid = conn.execute(
                "SELECT match_id FROM matches WHERE tourney_id = ? "
                "AND match_num IS ? AND round IS ? AND winner_id = ? "
                "AND loser_id = ?",
                key,
            ).fetchone()
            if mid:
                match_ids[r.match_natural_key] = mid[0]
                n_matches += 1
        conn.execute("COMMIT")

        # match_stats: 2 rows per match, one for winner one for loser
        conn.execute("BEGIN")
        for r in mdf.itertuples(index=False):
            match_id = match_ids.get(r.match_natural_key)
            if match_id is None:
                continue
            for is_winner, prefix, pid in (
                (1, "w_", str(r.winner_id)),
                (0, "l_", str(r.loser_id)),
            ):
                conn.execute(
                    "INSERT OR REPLACE INTO match_stats "
                    "(match_id, player_id, is_winner, "
                    "aces, double_faults, serve_points, "
                    "first_serves_in, first_serves_won, "
                    "second_serves_won, service_games, "
                    "break_points_saved, break_points_faced) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (match_id, pid, is_winner,
                     _safe_int(getattr(r, f"{prefix}ace", None)),
                     _safe_int(getattr(r, f"{prefix}df", None)),
                     _safe_int(getattr(r, f"{prefix}svpt", None)),
                     _safe_int(getattr(r, f"{prefix}1stIn", None)),
                     _safe_int(getattr(r, f"{prefix}1stWon", None)),
                     _safe_int(getattr(r, f"{prefix}2ndWon", None)),
                     _safe_int(getattr(r, f"{prefix}SvGms", None)),
                     _safe_int(getattr(r, f"{prefix}bpSaved", None)),
                     _safe_int(getattr(r, f"{prefix}bpFaced", None))),
                )
                n_stats += 1
        conn.execute("COMMIT")

    return {"players": n_players, "tournaments": n_tourneys,
            "matches": n_matches, "match_stats": n_stats}


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


def backfill_match_features(db_path: str | Path) -> int:
    """Populate the v2 ``match_features`` table from the legacy
    ``training_matches`` rows. Join is on (date, winner_name,
    loser_name) -> matches.match_id. Both orientations (a = winner
    and a = loser) get a row in match_features, matching the
    balanced shape the trainer expects.

    Returns the number of feature rows written.
    """
    n = 0
    sql_select = (
        "SELECT tm.tourney_date, tm.player_a, tm.player_b, tm.label, "
        "tm.used_in_split, "
        "tm.diff_elo_pre, tm.diff_surface_elo_pre, "
        "tm.diff_form_last5, tm.diff_form_last10, "
        "tm.diff_avg_serve_pts_won_10, tm.diff_avg_return_pts_won_10, "
        "tm.diff_avg_bp_saved_10, tm.diff_days_rest, "
        "tm.h2h_a_wins_minus_b_wins, tm.rank_diff, "
        "tm.level_rank, tm.round_rank, "
        "m.match_id, m.winner_id, m.loser_id, pw.name AS wname, "
        "pl.name AS lname "
        "FROM training_matches tm "
        "JOIN matches m ON m.match_date = tm.tourney_date "
        "JOIN players pw ON pw.player_id = m.winner_id "
        "JOIN players pl ON pl.player_id = m.loser_id "
        "WHERE (tm.player_a = pw.name AND tm.player_b = pl.name) "
        "   OR (tm.player_a = pl.name AND tm.player_b = pw.name)"
    )
    insert_sql = (
        "INSERT OR REPLACE INTO match_features "
        "(match_id, orientation, player_a_id, player_b_id, label, "
        "used_in_split, diff_elo_pre, diff_surface_elo_pre, "
        "diff_form_last5, diff_form_last10, "
        "diff_avg_serve_pts_won_10, diff_avg_return_pts_won_10, "
        "diff_avg_bp_saved_10, diff_days_rest, "
        "h2h_a_wins_minus_b_wins, rank_diff, level_rank, round_rank) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    with connect(db_path) as conn:
        cur = conn.execute(sql_select)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        conn.execute("BEGIN")
        for raw in rows:
            r = dict(zip(cols, raw))
            # Determine the orientation by matching player_a's name to
            # winner / loser. Then pick player_a_id / player_b_id.
            if r["player_a"] == r["wname"]:
                orientation = "a_winner"
                a_id = r["winner_id"]
                b_id = r["loser_id"]
            else:
                orientation = "a_loser"
                a_id = r["loser_id"]
                b_id = r["winner_id"]
            conn.execute(insert_sql, (
                r["match_id"], orientation, a_id, b_id,
                r["label"], r["used_in_split"],
                r["diff_elo_pre"], r["diff_surface_elo_pre"],
                r["diff_form_last5"], r["diff_form_last10"],
                r["diff_avg_serve_pts_won_10"],
                r["diff_avg_return_pts_won_10"],
                r["diff_avg_bp_saved_10"], r["diff_days_rest"],
                r["h2h_a_wins_minus_b_wins"], r["rank_diff"],
                r["level_rank"], r["round_rank"],
            ))
            n += 1
        conn.execute("COMMIT")
    return n


def backfill_kalshi_bets(db_path: str | Path) -> int:
    """Populate the v2 ``kalshi_bets`` table from the legacy
    ``kalshi_outcomes`` table. Attempt to link each ticker to a row
    in ``matches`` by joining on (closed_at date, side player tricode
    matching one of winner/loser last-name initials).

    Returns the number of rows written.
    """
    n = 0
    with connect(db_path) as conn:
        outcomes = list(conn.execute(
            "SELECT ticker, event_ticker, side_player, other_player, "
            "market_result, settle_value, won, entry_price, "
            "settle_price, realized_pnl, fee_cost, opened_at, "
            "closed_at, synced_at FROM kalshi_outcomes"
        ).fetchall())
        ocols = ["ticker","event_ticker","side_player","other_player",
                 "market_result","settle_value","won","entry_price",
                 "settle_price","realized_pnl","fee_cost","opened_at",
                 "closed_at","synced_at"]
        # Precompute (date, frozenset[tricodes]) -> match_id for every
        # match whose winner/loser last-name initials might match a
        # Kalshi ticker. Cheap since matches has ~110k rows but the
        # date join filters most away.
        import re
        from datetime import datetime
        conn.execute("BEGIN")
        for raw in outcomes:
            ko = dict(zip(ocols, raw))
            ev = ko.get("event_ticker") or ""
            sp = ko.get("side_player") or ""
            op = ko.get("other_player") or ""
            match_id = None
            side_player_id = None
            if "-" in ev and len(ev.split("-", 1)[1]) >= 7:
                date_part = ev.split("-", 1)[1][:7]
                try:
                    dt = datetime.strptime(date_part, "%y%b%d")
                    date_iso = dt.strftime("%Y-%m-%d")
                    # Look up matches on that date where last-name
                    # initials match the (sp, op) pair.
                    cands = conn.execute(
                        "SELECT m.match_id, m.winner_id, m.loser_id, "
                        "pw.name AS wname, pl.name AS lname "
                        "FROM matches m "
                        "JOIN players pw ON pw.player_id = m.winner_id "
                        "JOIN players pl ON pl.player_id = m.loser_id "
                        "WHERE m.match_date = ?",
                        (date_iso,),
                    ).fetchall()
                    for cmid, wid, lid, wname, lname in cands:
                        wtri = (wname.split()[-1][:3].upper()
                                 if wname else "")
                        ltri = (lname.split()[-1][:3].upper()
                                 if lname else "")
                        pair = frozenset({wtri, ltri})
                        if pair == frozenset({sp, op}):
                            match_id = cmid
                            if sp == wtri:
                                side_player_id = wid
                            else:
                                side_player_id = lid
                            break
                except ValueError:
                    pass
            conn.execute(
                "INSERT OR REPLACE INTO kalshi_bets "
                "(ticker, event_ticker, match_id, side_player_id, "
                "side_tricode, other_tricode, market_result, "
                "settle_value, won, entry_price, settle_price, "
                "realized_pnl, fee_cost, opened_at, closed_at, "
                "synced_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ko["ticker"], ko["event_ticker"], match_id,
                 side_player_id, sp, op,
                 ko["market_result"], ko["settle_value"],
                 ko["won"], ko["entry_price"], ko["settle_price"],
                 ko["realized_pnl"], ko["fee_cost"],
                 ko["opened_at"], ko["closed_at"],
                 ko["synced_at"]),
            )
            n += 1
        conn.execute("COMMIT")
    return n


def fetch_combined_matches(
    db_path: str | Path,
    *,
    offset: int = 0,
    limit: int = 20,
    tour: str | None = None,
) -> list[dict]:
    """JOIN across the v2 normalised tables and return a flat dict per
    match with everything visible at once: tournament + both players'
    attrs + outcome + per-player serving stats + the model's
    engineered features + any Kalshi bet linked to it.

    Newest-first by ``matches.match_date``. Pagination by absolute
    offset / limit since the dashboard's union pagination needs that.
    """
    sql = """
    SELECT
        m.match_id, m.match_date, m.round, m.match_num, m.best_of,
        m.minutes, m.score, m.winner_id, m.loser_id,
        m.winner_seed, m.loser_seed, m.winner_entry, m.loser_entry,
        m.winner_age, m.loser_age,
        m.winner_rank, m.loser_rank,
        m.winner_rank_points, m.loser_rank_points,
        t.tourney_id, t.name AS tourney_name, t.tour, t.surface,
        t.level, t.draw_size, t.year,
        pw.name AS winner_name, pw.country AS winner_country,
        pw.hand AS winner_hand, pw.height_cm AS winner_height_cm,
        pl.name AS loser_name, pl.country AS loser_country,
        pl.hand AS loser_hand, pl.height_cm AS loser_height_cm,
        sw.aces AS w_aces, sw.double_faults AS w_df,
        sw.serve_points AS w_svpt, sw.first_serves_in AS w_1stIn,
        sw.first_serves_won AS w_1stWon,
        sw.second_serves_won AS w_2ndWon,
        sw.service_games AS w_SvGms,
        sw.break_points_saved AS w_bpSaved,
        sw.break_points_faced AS w_bpFaced,
        sl.aces AS l_aces, sl.double_faults AS l_df,
        sl.serve_points AS l_svpt, sl.first_serves_in AS l_1stIn,
        sl.first_serves_won AS l_1stWon,
        sl.second_serves_won AS l_2ndWon,
        sl.service_games AS l_SvGms,
        sl.break_points_saved AS l_bpSaved,
        sl.break_points_faced AS l_bpFaced,
        mf.diff_elo_pre, mf.diff_surface_elo_pre,
        mf.diff_form_last5, mf.diff_form_last10,
        mf.diff_avg_serve_pts_won_10, mf.diff_avg_return_pts_won_10,
        mf.diff_avg_bp_saved_10, mf.diff_days_rest,
        mf.h2h_a_wins_minus_b_wins, mf.rank_diff,
        mf.level_rank, mf.round_rank,
        kb.ticker AS bet_ticker, kb.side_tricode AS bet_side,
        kb.entry_price AS bet_entry, kb.realized_pnl AS bet_pnl,
        kb.won AS bet_won, kb.market_result AS bet_market_result
    FROM matches m
    LEFT JOIN tournaments t ON t.tourney_id = m.tourney_id
    LEFT JOIN players pw   ON pw.player_id = m.winner_id
    LEFT JOIN players pl   ON pl.player_id = m.loser_id
    LEFT JOIN match_stats sw
        ON sw.match_id = m.match_id AND sw.player_id = m.winner_id
    LEFT JOIN match_stats sl
        ON sl.match_id = m.match_id AND sl.player_id = m.loser_id
    LEFT JOIN match_features mf
        ON mf.match_id = m.match_id AND mf.orientation = 'a_winner'
    LEFT JOIN kalshi_bets kb
        ON kb.match_id = m.match_id
    """
    params: list[Any] = []
    where: list[str] = []
    if tour:
        where.append("t.tour = ?")
        params.append(tour)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += (" ORDER BY m.match_date DESC, m.match_id DESC "
            "LIMIT ? OFFSET ?")
    params.extend([limit, offset])
    with connect(db_path) as conn:
        cur = conn.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def count_combined_matches(db_path: str | Path,
                              tour: str | None = None) -> int:
    sql = "SELECT COUNT(*) FROM matches m"
    params: list[Any] = []
    if tour:
        sql += (" LEFT JOIN tournaments t ON t.tourney_id = m.tourney_id"
                " WHERE t.tour = ?")
        params.append(tour)
    with connect(db_path) as conn:
        return int(conn.execute(sql, params).fetchone()[0])


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
