"""Configuration loading.

Every tunable number in this project lives in ``config.yaml``. This module turns
it into a dotted-attribute object so call sites read as ``cfg.scoring.vivid.canvas_sigma``
rather than a chain of dict lookups.

Access to a missing key raises immediately rather than returning None, because a
silently-missing weight would produce a plausible-looking but wrong score.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import yaml

DEFAULT_CONFIG_NAME = "config.yaml"


class ConfigError(KeyError):
    """Raised when a configuration key is absent."""


class Config:
    """Recursive dotted-access wrapper around the parsed YAML."""

    __slots__ = ("_data", "_path")

    def __init__(self, data: dict, path: str = "") -> None:
        self._data = data
        self._path = path

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError:
            where = f"{self._path}.{name}" if self._path else name
            raise ConfigError(
                f"missing config key {where!r} — every tunable number must be "
                f"declared in {DEFAULT_CONFIG_NAME}"
            ) from None
        return self._wrap(value, f"{self._path}.{name}" if self._path else name)

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"Config({self._path or 'root'}: {sorted(self._data)})"

    @classmethod
    def _wrap(cls, value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return cls(value, path)
        if isinstance(value, list):
            return [cls._wrap(v, f"{path}[{i}]") for i, v in enumerate(value)]
        return value

    def get(self, name: str, default: Any = None) -> Any:
        """Dict-style access with a default, for genuinely optional keys."""
        if name not in self._data:
            return default
        return self._wrap(self._data[name], name)

    def as_dict(self) -> dict:
        """Return the raw underlying mapping (a copy, so callers cannot mutate config)."""
        import copy

        return copy.deepcopy(self._data)


def repo_root() -> Path:
    """The repository root — the directory containing this package."""
    return Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load ``config.yaml``.

    Args:
        path: explicit path; defaults to ``config.yaml`` beside the package.
    """
    cfg_path = Path(path) if path is not None else repo_root() / DEFAULT_CONFIG_NAME
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{cfg_path} did not parse to a mapping")
    return Config(data)
