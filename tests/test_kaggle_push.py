"""Tests for pushing a batch kernel to Kaggle.

These tests exercise metadata rewriting and the push wrapper without any
network access: `subprocess.run` is always stubbed, and every metadata file
lives inside pytest's `tmp_path`.
"""

import json
from pathlib import Path

import pytest

from foremankit import kaggle
from foremankit.config import ConfigError

TEMPLATE_ID = "REPLACE_WITH_KAGGLE_USER/foremankit-batch"
SECRET_KEY = "super-secret-key-do-not-leak"


def write_metadata(directory: Path, metadata: dict) -> Path:
    """Write `metadata` as kernel-metadata.json inside `directory` and return its path."""
    metadata_path = directory / kaggle.METADATA_FILE
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


def test_placeholder_owner_is_replaced_by_the_user(tmp_path: Path) -> None:
    """A placeholder owner is swapped for the user while the slug is kept."""
    write_metadata(tmp_path, {"id": TEMPLATE_ID})

    metadata = kaggle.prepare_metadata(tmp_path, user="me")

    assert metadata["id"] == "me/foremankit-batch"


def test_empty_user_leaves_a_placeholder_id_untouched(tmp_path: Path) -> None:
    """With no user given, the placeholder id is never invented into something else."""
    write_metadata(tmp_path, {"id": TEMPLATE_ID})

    metadata = kaggle.prepare_metadata(tmp_path)

    assert metadata["id"] == TEMPLATE_ID


def test_datasets_are_deduplicated_and_order_is_preserved(tmp_path: Path) -> None:
    """Repeated dataset sources collapse to their first occurrence."""
    write_metadata(tmp_path, {"id": "me/foremankit-batch"})
    datasets = ["a/one", "b/two", "a/one"]

    metadata = kaggle.prepare_metadata(tmp_path, datasets=datasets)

    assert metadata["dataset_sources"] == ["a/one", "b/two"]


def test_no_datasets_leaves_existing_sources_untouched(tmp_path: Path) -> None:
    """Passing datasets=None keeps the dataset_sources list from the template."""
    write_metadata(tmp_path, {"id": "me/foremankit-batch", "dataset_sources": ["a/one"]})

    metadata = kaggle.prepare_metadata(tmp_path, datasets=None)

    assert metadata["dataset_sources"] == ["a/one"]


def test_rewritten_file_parses_back_to_the_returned_dict(tmp_path: Path) -> None:
    """The file written on disk is identical to the returned metadata."""
    metadata_path = write_metadata(tmp_path, {"id": TEMPLATE_ID})

    metadata = kaggle.prepare_metadata(tmp_path, user="me")

    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata


def test_missing_metadata_file_raises_config_error(tmp_path: Path) -> None:
    """A directory without kernel-metadata.json is reported as missing."""
    with pytest.raises(ConfigError, match="missing"):
        kaggle.prepare_metadata(tmp_path)


def test_broken_json_raises_config_error(tmp_path: Path) -> None:
    """A metadata file that is not valid JSON raises ConfigError."""
    (tmp_path / kaggle.METADATA_FILE).write_text("{not json", encoding="utf-8")

    with pytest.raises(ConfigError):
        kaggle.prepare_metadata(tmp_path)


def test_require_credentials_returns_username(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both env vars set yields the stripped username; missing key raises without leaking it."""
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setenv("KAGGLE_KEY", SECRET_KEY)
    assert kaggle.require_credentials() == "me"

    monkeypatch.delenv("KAGGLE_KEY")
    with pytest.raises(ConfigError, match="KAGGLE_KEY") as error:
        kaggle.require_credentials()
    assert SECRET_KEY not in str(error.value)


class CompletedPush:
    """Stand-in for subprocess.CompletedProcess with the fields push() reads."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_push_returns_id_and_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful push returns "id: stdout" and calls kaggle kernels push."""
    write_metadata(tmp_path, {"id": "me/foremankit-batch"})
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setenv("KAGGLE_KEY", SECRET_KEY)
    recorded_argv: list[str] = []

    def stub_run(argv, capture_output, text):
        """Record the argv and report a successful push."""
        recorded_argv.extend(argv)
        return CompletedPush(returncode=0, stdout="Kernel pushed\n")

    monkeypatch.setattr(kaggle.subprocess, "run", stub_run)

    text = kaggle.push(tmp_path)

    assert text == "me/foremankit-batch: Kernel pushed"
    assert recorded_argv == ["kaggle", "kernels", "push", "-p", str(tmp_path)]


def test_push_failure_reports_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero return code surfaces the stderr text in a ConfigError."""
    write_metadata(tmp_path, {"id": "me/foremankit-batch"})
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setenv("KAGGLE_KEY", SECRET_KEY)
    monkeypatch.setattr(
        kaggle.subprocess,
        "run",
        lambda *args, **kwargs: CompletedPush(returncode=1, stderr="403 Forbidden"),
    )

    with pytest.raises(ConfigError, match="403 Forbidden"):
        kaggle.push(tmp_path)
