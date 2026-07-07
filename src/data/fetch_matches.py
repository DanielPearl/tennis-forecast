"""Pull tour-level singles match logs.

The historical baseline was Jeff Sackmann's ``tennis_atp`` / ``tennis_wta``
GitHub repos (surface + round tagged, serve/return aggregates per match,
back to the open era). Those repos were taken down in mid-2026; we now
walk an ordered chain of community mirrors that copy the same Sackmann
schema (identical column names + tour-year filename convention).

Order matters — the first mirror that returns 200 for a given (tour,
year) wins and is cached to disk. Adding a new mirror doesn't
invalidate the on-disk cache because we key by (tour, year), not by
mirror. ATP is well-covered by mirrors; WTA has no per-year mirror we
trust yet, so pre-cached WTA files continue to be used and missing
years silently drop out.

Why we don't scrape live ATP/WTA pages: the official sites rate-limit
aggressively and the markup turns over every season. Sackmann's CSVs
(and the mirrors that copy them) share a stable schema.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("data.fetch_matches")

# Sackmann uses one CSV per tour-year for tour-level main draw matches.
_FILE_TEMPLATE = {
    "atp": "atp_matches_{year}.csv",
    "wta": "wta_matches_{year}.csv",
}


def _mirror_urls(tour: str, year: int, base_atp: str, base_wta: str
                  ) -> list[str]:
    """Return the ordered list of URLs to try for one (tour, year).

    ``base_atp`` / ``base_wta`` from config stay first so that if the
    upstream ever comes back it wins automatically. The subsequent
    fallbacks are community mirrors that copy the Sackmann schema —
    each covers a different year range (see docstring). ATP mirrors
    carry the full 49-column serve/return panel; the WTA mirror
    (LuckyLoser91/TennisCourtLog) carries the stripped 13-column
    schema (no serve stats, but tourney_date / surface / level /
    round / rank / rank_points / best_of are all present), which is
    enough for the Elo bootstrap and the rank/surface features. The
    matches_clean concat handles the missing columns as NaN and the
    HistGradientBoosting trainer is NaN-native.
    """
    fname = _FILE_TEMPLATE[tour].format(year=year)
    if tour == "atp":
        # Coverage: config-configured base (usually Sackmann master) →
        # stakah/tennis_atp (1968-2019) → jegqwll/tennis_atp_2000_2025
        # (2000-2025). The chain covers every year 1968-2025 with at
        # least one mirror, and years present in multiple mirrors take
        # whichever wins first (config first for freshness).
        return [
            f"{base_atp}/{fname}",
            f"https://raw.githubusercontent.com/stakah/tennis_atp/master/{fname}",
            f"https://raw.githubusercontent.com/jegqwll/tennis_atp_2000_2025/main/{fname}",
        ]
    # WTA: config base first (upstream if it comes back), then the
    # LuckyLoser91 mirror which covers 1968-2026 with the stripped
    # 13-column schema. The 2015+ years we've already cached to disk
    # use the full 49-column schema (from the pre-takedown Sackmann
    # snapshot) — the on-disk cache is what fetch_year returns for
    # those years, so the schema downgrade only affects years we
    # newly pull from the mirror.
    return [
        f"{base_wta}/{fname}",
        f"https://raw.githubusercontent.com/LuckyLoser91/TennisCourtLog/main/tennis_wta/{fname}",
    ]


def _local_path(raw_dir: Path, tour: str, year: int) -> Path:
    return raw_dir / tour / _FILE_TEMPLATE[tour].format(year=year)


def fetch_year(tour: str, year: int, raw_dir: Path,
               base_atp: str, base_wta: str,
               force: bool = False) -> pd.DataFrame | None:
    """Fetch one (tour, year) CSV. Returns None when no mirror publishes
    the year (mid-season for the current year is fine; future years 404
    across every mirror)."""
    out = _local_path(raw_dir, tour, year)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return pd.read_csv(out, low_memory=False)
    for url in _mirror_urls(tour, year, base_atp, base_wta):
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            log.warning("fetch_year %s %s from %s failed: %s",
                         tour, year, url, exc)
            continue
        if r.status_code == 404:
            continue
        try:
            r.raise_for_status()
        except requests.HTTPError as exc:
            log.warning("fetch_year %s %s from %s failed: %s",
                         tour, year, url, exc)
            continue
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        df.to_csv(out, index=False)
        log.info("fetched %s %d (%d rows, %s)",
                  tour, year, len(df), url.rsplit("/", 3)[-3])
        return df
    return None


def fetch_all(force_current_year: bool = True) -> pd.DataFrame:
    """Fetch all configured years for all configured tours and return
    one stacked dataframe with a ``tour`` column."""
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    years: Iterable[int] = range(
        int(cfg["data"]["history_start_year"]),
        int(cfg["data"]["history_end_year"]) + 1,
    )
    frames: list[pd.DataFrame] = []
    last_year = max(years)
    for tour in cfg["data"]["tours"]:
        for year in years:
            # Re-pull the active season every run; cached years are
            # immutable once the season closes.
            force = force_current_year and year == last_year
            df = fetch_year(
                tour, year, raw_dir,
                cfg["data"]["atp_repo_base"],
                cfg["data"]["wta_repo_base"],
                force=force,
            )
            if df is None or df.empty:
                continue
            df = df.copy()
            df["tour"] = tour
            frames.append(df)
    if not frames:
        raise RuntimeError(
            "no Sackmann data fetched — check network access or adjust "
            "config.data.history_start_year"
        )
    matches = pd.concat(frames, ignore_index=True)
    # tourney_date arrives in two formats depending on the mirror:
    # Sackmann-schema ATP files (and cached WTA 2015+) use %Y%m%d
    # (``20240521``); the LuckyLoser91 WTA mirror ships ISO strings
    # (``2000-04-30``). Try %Y%m%d first (majority of rows), then
    # fall back to a permissive parser for whatever %Y%m%d couldn't
    # coerce. Doing it this way is much faster than a single flexible
    # pd.to_datetime call across ~200k rows.
    dt = pd.to_datetime(
        matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    unresolved = dt.isna()
    if unresolved.any():
        dt.loc[unresolved] = pd.to_datetime(
            matches.loc[unresolved, "tourney_date"].astype(str),
            errors="coerce",
        )
    matches["tourney_date"] = dt
    matches = matches.dropna(subset=["tourney_date", "winner_name", "loser_name"])
    matches = matches.sort_values("tourney_date").reset_index(drop=True)
    return matches


def save_clean(matches: pd.DataFrame) -> Path:
    # CSV (not parquet) — avoids pulling in pyarrow as a hard dep on
    # the droplet. The clean panel is written once per training run
    # and is small enough that CSV decode is not a bottleneck.
    cfg = load_config()
    out = resolve_path(cfg["paths"]["processed_dir"]) / "matches_clean.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(matches))
    return out


if __name__ == "__main__":
    df = fetch_all()
    save_clean(df)
