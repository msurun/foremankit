"""Tests for the panel budget fields: writing a cap into routing.yaml and reading spend per category."""

import json
from pathlib import Path

import pytest

from foremankit import panel


ROUTING = """budget:
  # Two separate budgets
  daily_usd: 1.5        # delegated work
  monthly_usd: 5.0      # delegated work per month
  orchestrator_daily_usd: 1.5     # Claude Code on DeepSeek/Gemini
  per_call_usd: 0.10    # a single call
context_guard:
  max_tokens: 100
"""


@pytest.fixture
def routing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config dir with a routing.yaml that has comments worth keeping."""
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "state").mkdir()
    path = tmp_path / "routing.yaml"
    path.write_text(ROUTING, encoding="utf-8")
    return path


def test_a_cap_is_written_in_place_and_its_comment_survives(routing: Path):
    """Setting one number touches that line only: the inline comment and every other line stay."""
    assert "orchestrator daily limit: 2.25" in panel.set_budget("orchestrator", "daily", "2.25")
    text = routing.read_text(encoding="utf-8")
    assert "  orchestrator_daily_usd: 2.25     # Claude Code on DeepSeek/Gemini\n" in text
    assert text.replace("2.25", "1.5") == ROUTING


def test_an_empty_value_writes_no_cap_and_a_bad_value_is_refused(routing: Path):
    """Empty = no cap (yaml null); a negative value or an unknown category/period is refused without touching the file."""
    panel.set_budget("delegation", "monthly", "")
    assert "  monthly_usd:       # delegated work per month\n" in routing.read_text(encoding="utf-8")
    assert panel.budget_status()["delegation"]["monthly_cap"] is None
    for category, period, value in (("delegation", "daily", "-3"), ("delegation", "weekly", "5"), ("agents", "daily", "1")):
        with pytest.raises(ValueError):
            panel.set_budget(category, period, value)


def test_a_missing_key_is_added_under_budget(routing: Path):
    """A cap the file does not have yet is inserted right after `budget:`, keeping the block's indentation."""
    text = ROUTING.replace("  orchestrator_daily_usd: 1.5     # Claude Code on DeepSeek/Gemini\n", "")
    routing.write_text(text, encoding="utf-8")
    panel.set_budget("orchestrator", "daily", "3")
    assert "budget:\n  orchestrator_daily_usd: 3\n" in routing.read_text(encoding="utf-8")


def test_budget_status_shows_spend_and_caps_per_category(routing: Path):
    """The GUI data: today's and this month's spend next to each category's caps."""
    from foremankit import budget
    now = 1_000_000.0
    panel.StateStore().update(lambda state: budget.record(state, now, 0.4, budget.ORCHESTRATOR))
    rows = panel.budget_status(now)
    assert rows["orchestrator"] == {"day_usd": 0.4, "month_usd": 0.4, "daily_cap": 1.5, "monthly_cap": None}
    assert rows["delegation"]["day_usd"] == 0.0 and rows["delegation"]["daily_cap"] == 1.5
    assert json.dumps(rows)
