"""Shared fixtures.

The package lives at the repo root, so tests import it directly. No install step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sunset_rider.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    """The real config.yaml — tests assert against shipped values, not a stub."""
    return load_config(REPO_ROOT / "config.yaml")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"
