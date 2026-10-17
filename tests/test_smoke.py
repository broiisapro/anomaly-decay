"""Smoke test: the package imports and exposes a version string."""

import re

import anomaly_decay


def test_package_imports_and_has_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", anomaly_decay.__version__)
