"""Week boundaries, the session ledger and hour estimates of the Kaggle free-tier budget."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from foremankit import kaggle

SATURDAY = datetime(2026, 10, 24, 13, 15, tzinfo=timezone.utc).timestamp()
MONDAY = datetime(2026, 10, 19, 0, 0, tzinfo=timezone.utc).timestamp()
LAST_WEEK = MONDAY - 2 * 86400.0


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    """Point the state directory at a scratch folder so tests never touch ~/.config."""
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    return tmp_path


def test_week_start_is_monday_midnight_utc():
    """Any moment of a week maps to that week's Monday 00:00 UTC, and the value is stable."""
    assert kaggle.week_start(SATURDAY) == MONDAY
    assert kaggle.week_start(MONDAY) == MONDAY
    assert kaggle.week_start(MONDAY + 6.9 * 86400.0) == MONDAY


def test_logged_hours_reduce_the_remaining_budget(ledger):
    """Two GPU hours logged leave the rest of the weekly quota, and the clamp stops at zero."""
    assert kaggle.remaining("gpu", SATURDAY) == kaggle.quota("gpu")
    kaggle.log_session({"gpu": 7200.0}, jobs_done=3, now=SATURDAY)
    assert kaggle.week_used("gpu", SATURDAY) == pytest.approx(7200.0)
    assert kaggle.remaining("gpu", SATURDAY) == pytest.approx(kaggle.quota("gpu") - 2.0)
    kaggle.log_session({"gpu": 100 * 3600.0}, now=SATURDAY)
    assert kaggle.remaining("gpu", SATURDAY) == 0.0


def test_tpu_and_gpu_ledgers_are_separate(ledger):
    """A TPU session does not eat the GPU quota: Kaggle counts the two accelerators apart."""
    kaggle.log_session({"tpu": 600.0}, now=SATURDAY)
    assert kaggle.week_used("tpu", SATURDAY) == pytest.approx(600.0)
    assert kaggle.week_used("gpu", SATURDAY) == 0.0


def test_sessions_from_an_older_week_are_ignored(ledger):
    """The quota resets weekly, so hours logged last week must not count."""
    kaggle.log_session({"gpu": 3600.0}, now=LAST_WEEK)
    assert kaggle.week_used("gpu", SATURDAY) == 0.0
    assert kaggle.remaining("gpu", SATURDAY) == kaggle.quota("gpu")


def test_broken_ledger_lines_are_skipped(ledger):
    """A truncated write must not break the budget report."""
    kaggle.sessions_path().write_text('{"started": 1, "gpu_seconds": 5}\nnot json\n[1, 2]\n',
                                      encoding="utf-8")
    assert [record["gpu_seconds"] for record in kaggle.read_sessions()] == [5]


def test_estimate_counts_prefill_and_decode_apart():
    """One job costs the accelerator's start-up overhead plus prefill and decode time, in hours."""
    speed = {"prompt_tps": 200.0, "gen_tps": 20.0}
    jobs = [{"prompt_tokens": 20000, "max_tokens": 400}, {"tokens_in": 1000, "tokens_out": 100}]
    expected = kaggle.OVERHEAD_HOURS["gpu"] + (21000 / 200.0 + 500 / 20.0) / 3600.0
    assert kaggle.estimate_hours(jobs, speed) == pytest.approx(expected)
    assert kaggle.estimate_hours([]) == 0.0


def test_estimate_uses_the_accelerator_rates_and_overhead():
    """TPU prices the same queue with its own speeds and a longer warm-up, never with GPU numbers."""
    jobs = [{"tokens_in": 90000, "tokens_out": 600}]
    tpu_seconds = 90000 / kaggle.DEFAULT_SPEED["tpu"]["prompt_tps"] + 600 / kaggle.DEFAULT_SPEED["tpu"]["gen_tps"]
    assert kaggle.estimate_hours(jobs, accelerator="tpu") == pytest.approx(
        kaggle.OVERHEAD_HOURS["tpu"] + tpu_seconds / 3600.0)
    # The tiers are never mixed: priced with TPU numbers it is not the GPU estimate.
    assert kaggle.estimate_hours(jobs, accelerator="tpu") != kaggle.estimate_hours(jobs, accelerator="gpu")


def test_tpu_undercuts_gpu_once_prefill_dominates():
    """The ~6x TPU prefill pays for its longer warm-up on a large prompt queue."""
    jobs = [{"tokens_in": 3_600_000, "tokens_out": 6000}]
    assert kaggle.estimate_hours(jobs, accelerator="tpu") < kaggle.estimate_hours(jobs, accelerator="gpu")


def test_estimate_rejects_an_unknown_accelerator():
    """A typo in the tier must fail loudly instead of silently billing GPU rates."""
    with pytest.raises(ValueError, match="accelerator"):
        kaggle.estimate_hours([{"tokens_in": 10}], accelerator="tpu4")


def test_sessions_needed_respects_the_nine_hour_cap():
    """Work is cut into whole sessions, so any remainder needs one more session."""
    assert kaggle.sessions_needed(0) == 0
    assert kaggle.sessions_needed(8.5) == 1
    assert kaggle.sessions_needed(9.0) == 1
    assert kaggle.sessions_needed(9.1) == 2


def test_fits_compares_an_estimate_with_the_remaining_budget(ledger):
    """A plan is accepted only while its estimate stays inside this week's hours."""
    assert kaggle.fits("gpu", 29.0, SATURDAY)
    assert not kaggle.fits("gpu", 30.5, SATURDAY)


def test_unknown_accelerator_is_rejected():
    """Only the two free tiers have a quota; anything else is a caller bug, not a silent zero."""
    with pytest.raises(ValueError, match="accelerator"):
        kaggle.quota("tpu3")
