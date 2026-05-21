"""Pull Grand Slam point-by-point logs from Sackmann's
``tennis_slam_pointbypoint`` mirror.

The slam PBP CSVs are the only real, free, large-volume source of
per-point tennis state with consistent schema. Every row is one point;
fields cover set/game/point score, who served, whether the point was
an ace/winner/error/break point, momentum, and (since ~2014) shot
characteristics. We use them to train the in-match adjustment model
that replaces the rules-based layer.

Singles only (men + women). We skip doubles and mixed because
they're a different format/strategy and the live-monitor doesn't
trade them.

Files are cached under ``data/raw/slam_pbp/`` so subsequent runs are
a no-op for years that have closed.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("data.fetch_pbp")

# Sackmann publishes one file per (year, slam, kind), where kind is
# "matches" or "points". Doubles/mixed get explicit suffixes; the
# plain name is the singles main draw, which is what we want.
_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master"
_SLAMS = ("ausopen", "frenchopen", "wimbledon", "usopen")


def _url(year: int, slam: str, kind: str) -> str:
    return f"{_BASE}/{year}-{slam}-{kind}.csv"


def _local(raw_dir: Path, year: int, slam: str, kind: str) -> Path:
    return raw_dir / "slam_pbp" / f"{year}-{slam}-{kind}.csv"


def _fetch_one(year: int, slam: str, kind: str, raw_dir: Path,
               force: bool = False) -> pd.DataFrame | None:
    out = _local(raw_dir, year, slam, kind)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return pd.read_csv(out, low_memory=False)
    url = _url(year, slam, kind)
    try:
        r = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        log.warning("fetch %s-%s-%s failed: %s", year, slam, kind, exc)
        return None
    if r.status_code == 404:
        return None
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    df.to_csv(out, index=False)
    log.info("fetched %d-%s-%s (%d rows)", year, slam, kind, len(df))
    return df


def fetch_all(years: Iterable[int] | None = None,
              force_current_year: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch every (year, slam) singles PBP file we can find.

    Returns a dict with two stacked dataframes:
      - ``matches``: one row per match with player names + winner
      - ``points``:  one row per point with score state + flags

    Each dataframe has ``year`` and ``slam`` columns so the snapshot
    builder can group by tournament/year. Years that 404 on both
    files are silently skipped (mid-season or not-yet-published).
    """
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    if years is None:
        years = range(
            int(cfg["data"]["history_start_year"]),
            int(cfg["data"]["history_end_year"]) + 1,
        )
    years = list(years)
    last_year = max(years)

    matches_frames: list[pd.DataFrame] = []
    points_frames: list[pd.DataFrame] = []
    for year in years:
        for slam in _SLAMS:
            force = force_current_year and year == last_year
            m = _fetch_one(year, slam, "matches", raw_dir, force=force)
            p = _fetch_one(year, slam, "points", raw_dir, force=force)
            if m is None or p is None:
                continue
            m = m.copy()
            p = p.copy()
            m["year"] = year
            m["slam"] = slam
            p["year"] = year
            p["slam"] = slam
            matches_frames.append(m)
            points_frames.append(p)
    if not matches_frames:
        raise RuntimeError(
            "no slam PBP data fetched — check network access or the "
            "Sackmann tennis_slam_pointbypoint repo URL"
        )
    matches = pd.concat(matches_frames, ignore_index=True)
    points = pd.concat(points_frames, ignore_index=True)
    return {"matches": matches, "points": points}


if __name__ == "__main__":
    bundles = fetch_all()
    log.info(
        "fetched slam PBP: %d matches across %d tournaments / %d point rows",
        len(bundles["matches"]),
        bundles["matches"].groupby(["year", "slam"]).ngroups,
        len(bundles["points"]),
    )
