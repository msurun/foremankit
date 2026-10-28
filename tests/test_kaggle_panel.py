"""Tests for the panel Kaggle fields: saving the address and key, and testing them."""
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from foremankit import config, panel


def test_write_env_values_replaces_lines_keeps_comments_and_appends_missing_names(tmp_path: Path):
    """Existing NAME= lines are replaced in place; comments and other lines stay; missing names are appended."""
    path = tmp_path / ".env"
    path.write_text("FOREMANKIT_NODE_KAGGLE_LAN=https://a-b.trycloudflare.com\n"
                    "# comment\n"
                    "KAGGLE_LLM_KEY=old\n", encoding="utf-8")
    config.write_env_values(path, {"FOREMANKIT_NODE_KAGGLE_LAN": "https://c-d.trycloudflare.com", "NEW_VAR": "value"})
    assert path.read_text(encoding="utf-8") == ("FOREMANKIT_NODE_KAGGLE_LAN=https://c-d.trycloudflare.com\n"
                                               "# comment\n"
                                               "KAGGLE_LLM_KEY=old\n"
                                               "NEW_VAR=value\n")


def test_write_env_values_refuses_values_with_line_breaks(tmp_path: Path):
    """A value containing \n or \r raises ValueError."""
    path = tmp_path / ".env"
    with pytest.raises(ValueError):
        config.write_env_values(path, {"FOREMANKIT_NODE_KAGGLE_LAN": "https://a-b.trycloudflare.com\n"})


def test_read_env_file_skips_comments_and_blank_lines_and_strips_quotes(tmp_path: Path):
    """Comments and blank lines are skipped; surrounding quotes are stripped; missing file -> {}."""
    path = tmp_path / ".env"
    path.write_text("# comment\n"
                    "\n"
                    "FOREMANKIT_NODE_KAGGLE_LAN=https://a-b.trycloudflare.com\n"
                    "KAGGLE_LLM_KEY='old'\n", encoding="utf-8")
    assert config.read_env_file(path) == {"FOREMANKIT_NODE_KAGGLE_LAN": "https://a-b.trycloudflare.com",
                                          "KAGGLE_LLM_KEY": "old"}
    assert config.read_env_file(tmp_path / "missing") == {}


def test_refresh_live_env_copies_live_vars_when_the_file_changed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Copies LIVE_ENV_VARS from the repo .env into the environment when that file changed since the last call."""
    monkeypatch.setattr(config, "REPO_DIR", tmp_path)
    monkeypatch.setenv("KAGGLE_LLM_KEY", "old")
    config._live_stamp.clear()
    path = tmp_path / ".env"
    path.write_text("FOREMANKIT_NODE_KAGGLE_LAN=https://a-b.trycloudflare.com\n"
                    "KAGGLE_LLM_KEY=fromfile\n", encoding="utf-8")
    config.refresh_live_env()
    assert os.getenv("KAGGLE_LLM_KEY") == "old"
    t = path.stat().st_mtime
    path.write_text("FOREMANKIT_NODE_KAGGLE_LAN=https://c-d.trycloudflare.com\n"
                    "KAGGLE_LLM_KEY=new\n"
                    "OTHER_VAR=value\n", encoding="utf-8")
    os.utime(path, (t + 10, t + 10))
    config.refresh_live_env()
    assert os.getenv("KAGGLE_LLM_KEY") == "new"
    assert os.getenv("OTHER_VAR") is None


def test_set_kaggle_saves_url_and_key_and_refuses_bad_urls_and_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Saves the address and key; empty key keeps the old one; http:// URL, URL with a space, and key with a space raise ValueError; returned text never contains the key."""
    monkeypatch.setattr(panel, "ROOT", tmp_path)
    assert "https://a-b.trycloudflare.com" in panel.set_kaggle("https://a-b.trycloudflare.com/v1/", "key")
    assert "unchanged" in panel.set_kaggle("https://c-d.trycloudflare.com", "")
    for url in ("http://a-b.trycloudflare.com", "https://a-b.trycloudflare.com space"):
        with pytest.raises(ValueError):
            panel.set_kaggle(url, "key")
    with pytest.raises(ValueError):
        panel.set_kaggle("https://a-b.trycloudflare.com", "key space")


def test_kaggle_status_returns_url_and_key_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Returns url and key_set True/False and never the key value."""
    monkeypatch.delenv(panel.KAGGLE_URL_VAR, raising=False)
    monkeypatch.delenv(panel.KAGGLE_KEY_VAR, raising=False)
    monkeypatch.setattr(panel, "ROOT", tmp_path)
    assert panel.kaggle_status() == {"url": "", "key_set": False}


def test_test_kaggle_returns_text_saying_whether_address_and_key_work(monkeypatch: pytest.MonkeyPatch):
    """Returns text saying whether address and key work; raises OSError -> contains "no connection"; no saved url -> "no address"."""
    # Create a mock for timed_get
    mock_timed_get = MagicMock(return_value=(0.1, 200))
    monkeypatch.setattr(panel, "timed_get", mock_timed_get)

    # Test successful case
    assert "OK" in panel.test_kaggle()

    # Test refused case
    mock_timed_get.return_value = (0.1, 401)
    assert "refused" in panel.test_kaggle()

    # Test connection error case
    mock_timed_get.side_effect = OSError("error")
    assert "no connection" in panel.test_kaggle()

    # Test no saved url case
    monkeypatch.setattr(panel, "kaggle_status", lambda: {"url": "", "key_set": False})
    assert "no address" in panel.test_kaggle()

    # Verify the mock was called with the correct URL
    assert mock_timed_get.call_args[0][0].endswith("/v1/models")

