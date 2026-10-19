"""Pre-registration guard: the frozen configuration must not drift after freeze.

``config/specifications.yaml`` and ``config/phase2_preregistration.yaml`` are
frozen by the annotated tag ``specs-frozen`` before the first full-data
estimation. This module compares the working copies with the tagged versions.
While the tag does not exist the test skips cleanly (CI checkouts are shallow
and may not have tags), unless ``ANOMALY_DECAY_REQUIRE_FREEZE=1`` is set — used
once the freeze is expected to exist, where a missing tag is a failure.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[1]
FROZEN_RELATIVE_PATHS = (
    "config/specifications.yaml",
    "config/phase2_preregistration.yaml",
)
SPECS_RELATIVE_PATH = "config/specifications.yaml"
PHASE2_RELATIVE_PATH = "config/phase2_preregistration.yaml"
PHASE2_PATH = REPO_ROOT / PHASE2_RELATIVE_PATH
FREEZE_TAG = "specs-frozen"
MISSING_TAG_REASON = "specs-frozen tag not created yet"


def frozen_file(relative_path: str) -> str | None:
    """The file at the freeze tag, or None when the tag is absent."""
    result = subprocess.run(
        ["git", "show", f"{FREEZE_TAG}:{relative_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


@pytest.mark.parametrize("relative_path", FROZEN_RELATIVE_PATHS)
def test_specifications_file_unchanged_since_freeze_tag(relative_path: str) -> None:
    frozen = frozen_file(relative_path)
    if frozen is None:
        if os.environ.get("ANOMALY_DECAY_REQUIRE_FREEZE") == "1":
            pytest.fail(MISSING_TAG_REASON)
        pytest.skip(MISSING_TAG_REASON)
    assert (REPO_ROOT / relative_path).read_text(encoding="utf-8") == frozen


def test_phase2_preregistration_parses() -> None:
    data = yaml.safe_load(PHASE2_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["version"] == 1
    for section in ("bootstrap", "multiple_testing", "vintages"):
        assert isinstance(data[section], dict), section
    bootstrap = data["bootstrap"]
    assert bootstrap["seed"] == 2027
    assert bootstrap["reps"] == 2000
    assert bootstrap["block_length_months"] == 12
    assert data["multiple_testing"]["hurdle_t"] == 3.0


def test_freeze_guard_skips_cleanly_without_the_tag() -> None:
    if frozen_file(SPECS_RELATIVE_PATH) is not None:
        pytest.skip("specs-frozen tag already exists")
    with pytest.raises(pytest.skip.Exception, match=MISSING_TAG_REASON):
        test_specifications_file_unchanged_since_freeze_tag(SPECS_RELATIVE_PATH)


def test_freeze_guard_fails_when_the_tag_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if frozen_file(SPECS_RELATIVE_PATH) is not None:
        pytest.skip("specs-frozen tag already exists")
    monkeypatch.setenv("ANOMALY_DECAY_REQUIRE_FREEZE", "1")
    with pytest.raises(pytest.fail.Exception, match=MISSING_TAG_REASON):
        test_specifications_file_unchanged_since_freeze_tag(SPECS_RELATIVE_PATH)
