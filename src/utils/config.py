"""Config loader.

YAML → nested dict, with ``${ENV_VAR}`` interpolation for the small
number of secret-ish fields. We keep the loader stupid on purpose:
the bot only ever reads config once at startup, and every module that
needs a value imports this loader directly.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _interp_env(value: Any) -> Any:
    if isinstance(value, str):
        # Replace any ${VAR} with os.environ[VAR]; missing env vars resolve
        # to empty strings so config files render even on a fresh checkout.
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _interp_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interp_env(v) for v in value]
    return value


def load_config(path: str | Path | None = None) -> dict:
    """Load ``config/config.yaml`` (or the path given) into a nested dict."""
    if path is None:
        path = _REPO_ROOT / "config" / "config.yaml"
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _interp_env(raw)


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_path(rel: str) -> Path:
    """Resolve a config-relative path to an absolute path under the repo."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p
