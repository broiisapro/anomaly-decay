"""Pre-registration guard: config/specifications.yaml must not drift after freeze.

The file is frozen by the annotated tag ``specs-frozen`` before the first
full-data estimation. This module compares the working copy with the tagged
version. While the tag does not exist the test skips cleanly (CI checkouts are
shallow and may not have tags), unless ``ANOMALY_DECAY_REQUIRE_FREEZE=1`` is
set — used once the freeze is expected to exist, where a missing tag is a
failure.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
SPECS_RELATIVE_PATH = "config/specifications.yaml"
SPECS_PATH = REPO_ROOT / SPECS_RELATIVE_PATH
FREEZE_TAG = "specs-frozen"
MISSING_TAG_REASON = "specs-frozen tag not created yet"


def frozen_specifications() -> str | None:
    """The specifications file at the freeze tag, or None when the tag is absent."""
    result = subprocess.run(
        ["git", "show", f"{FREEZE_TAG}:{SPECS_RELATIVE_PATH}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def test_specifications_file_unchanged_since_freeze_tag() -> None:
    frozen = frozen_specifications()
    if frozen is None:
        if os.environ.get("ANOMALY_DECAY_REQUIRE_FREEZE") == "1":
            pytest.fail(MISSING_TAG_REASON)
        pytest.skip(MISSING_TAG_REASON)
    assert SPECS_PATH.read_text(encoding="utf-8") == frozen


def test_freeze_guard_skips_cleanly_without_the_tag() -> None:
    if frozen_specifications() is not None:
        pytest.skip("specs-frozen tag already exists")
    with pytest.raises(pytest.skip.Exception, match=MISSING_TAG_REASON):
        test_specifications_file_unchanged_since_freeze_tag()


def test_freeze_guard_fails_when_the_tag_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if frozen_specifications() is not None:
        pytest.skip("specs-frozen tag already exists")
    monkeypatch.setenv("ANOMALY_DECAY_REQUIRE_FREEZE", "1")
    with pytest.raises(pytest.fail.Exception, match=MISSING_TAG_REASON):
        test_specifications_file_unchanged_since_freeze_tag()
