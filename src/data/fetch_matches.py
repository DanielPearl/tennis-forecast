"""Pull tour-level singles match logs.

The Sackmann GitHub mirrors (``tennis_atp`` / ``tennis_wta``) are the
free baseline for tennis match data — surface tagged, round tagged,
serve/return aggregates per match, back to the open era. We pull a
configurable rolling window. Files are cached under ``data/raw/`` so
re-train is a no-op for years that haven't changed.

Why we don't scrape live ATP/WTA pages: the official sites rate-limit
aggressively and the markup turns over every season. The Sackmann CSVs
are the same data, kept current, mirrored to git.
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


def _file_url(tour: str, year: int, base_atp: str, base_wta: str) -> str:
    base = base_atp if tour == "atp" else base_wta
    return f"{base}/{_FILE_TEMPLATE[tour].format(year=year)}"


def _local_path(raw_dir: Path, tour: str, year: int) -> Path:
    return raw_dir / tour / _FILE_TEMPLATE[tour].format(year=year)


def fetch_year(tour: str, year: int, raw_dir: Path,
               base_atp: str, base_wta: str,
               force: bool = False) -> pd.DataFrame | None:
    """Fetch one (tour, year) CSV. Returns None when the year isn't
    published yet (mid-season for the current year is fine; future
    years 404)."""
    out = _local_path(raw_dir, tour, year)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return pd.read_csv(out, low_memory=False)
    url = _file_url(tour, year, base_atp, base_wta)
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        log.warning("fetch_year %s %s failed: %s", tour, year, exc)
        return None
    if r.status_code == 404:
        return None
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    df.to_csv(out, index=False)
    log.info("fetched %s %d (%d rows)", tour, year, len(df))
    return df


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
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
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
