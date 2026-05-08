"""Tiny logging helper. Same shape as the NBA bot's so log files in
``data/`` look familiar across the portfolio."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def setup_logging(name: str = "tennis", log_path: str | Path | None = None,
                  level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger
