"""Tests for foremankit.pricing.py: pure functions for pricing logic."""

from datetime import datetime, timezone
import pytest
from foremankit import pricing

@pytest.fixture
def schedule():
    return {
        "peak_utc_hours": [[1, 4], [6, 10]],
        "peak_weekdays": [0, 1, 2, 3, 4],
        "off_peak_factor": 0.5,
    }

@pytest.fixture
def provider_cfg(schedule):
    return {"price_schedule": schedule}

@pytest.fixture
def entry():
    return {
        "price": {
            "input_miss": 1.32,
            "input_hit": 0.044,
            "output": 3.96,
        },
    }

@pytest.fixture
def entry_with_price_from(entry):
    entry["price_from"] = {
        "2027-01-01": {
            "input_miss": 2.64,
            "input_hit": 0.088,
            "output": 7.92,
        },
    }
    return entry

def test_is_peak_without_schedule():
    """is_peak() returns None without a schedule."""
    assert pricing.is_peak(None, 0) is None

def test_is_peak_friday_0200_utc(schedule):
    """is_peak() returns True on Friday 02:00 UTC."""
    ts = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.is_peak(schedule, ts) is True

def test_is_peak_friday_0400_utc(schedule):
    """is_peak() returns False on Friday 04:00 UTC."""
    ts = datetime(2026, 10, 23, 4, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.is_peak(schedule, ts) is False

def test_is_peak_friday_0600_utc(schedule):
    """is_peak() returns True on Friday 06:00 UTC."""
    ts = datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.is_peak(schedule, ts) is True

def test_is_peak_friday_1000_utc(schedule):
    """is_peak() returns False on Friday 10:00 UTC."""
    ts = datetime(2026, 10, 23, 10, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.is_peak(schedule, ts) is False

def test_is_peak_saturday_0200_utc(schedule):
    """is_peak() returns False on Saturday 02:00 UTC."""
    ts = datetime(2026, 10, 24, 2, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.is_peak(schedule, ts) is False

def test_is_peak_friday_0500_utc(schedule):
    """is_peak() returns False on Friday 05:00 UTC."""
    ts = datetime(2026, 10, 23, 5, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.is_peak(schedule, ts) is False

def test_base_price_without_price_from(entry):
    """base_price() returns the entry's price without price_from."""
    ts = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.base_price(entry, ts) == entry["price"]

def test_base_price_with_price_from_before_date(entry_with_price_from):
    """base_price() returns the entry's price before the price_from date."""
    ts = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.base_price(entry_with_price_from, ts) == entry_with_price_from["price"]

def test_base_price_with_price_from_on_date(entry_with_price_from):
    """base_price() returns the new price on the price_from date."""
    ts = datetime(2027, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.base_price(entry_with_price_from, ts) == entry_with_price_from["price_from"]["2027-01-01"]

def test_base_price_with_price_from_after_date(entry_with_price_from):
    """base_price() returns the new price after the price_from date."""
    ts = datetime(2027, 1, 2, 0, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.base_price(entry_with_price_from, ts) == entry_with_price_from["price_from"]["2027-01-01"]

def test_effective_price_without_schedule(entry, provider_cfg):
    """effective_price() returns the base price without a schedule."""
    ts = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.effective_price(entry, provider_cfg, ts) == entry["price"]

def test_effective_price_peak(entry, provider_cfg):
    """effective_price() returns the base price during peak."""
    ts = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.effective_price(entry, provider_cfg, ts) == entry["price"]

def test_effective_price_off_peak(entry, provider_cfg):
    """effective_price() returns the scaled price during off-peak."""
    ts = datetime(2026, 10, 23, 4, 0, tzinfo=timezone.utc).timestamp()
    expected_price = {
        "input_miss": entry["price"]["input_miss"] * provider_cfg["price_schedule"]["off_peak_factor"],
        "input_hit": entry["price"]["input_hit"] * provider_cfg["price_schedule"]["off_peak_factor"],
        "output": entry["price"]["output"] * provider_cfg["price_schedule"]["off_peak_factor"],
    }
    assert pricing.effective_price(entry, provider_cfg, ts) == expected_price

def test_next_phase_change_without_schedule():
    """next_phase_change() returns None without a schedule."""
    ts = datetime(2026, 10, 23, 5, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.next_phase_change(None, ts) is None

def test_next_phase_change_friday_0500_to_0600(schedule):
    """next_phase_change() returns Friday 06:00 UTC from Friday 05:00 UTC."""
    ts = datetime(2026, 10, 23, 5, 0, tzinfo=timezone.utc).timestamp()
    expected_ts = datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.next_phase_change(schedule, ts) == expected_ts

def test_next_phase_change_friday_1100_to_monday_0100(schedule):
    """next_phase_change() returns Monday 01:00 UTC from Friday 11:00 UTC."""
    ts = datetime(2026, 10, 23, 11, 0, tzinfo=timezone.utc).timestamp()
    expected_ts = datetime(2026, 10, 26, 1, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.next_phase_change(schedule, ts) == expected_ts

def test_next_phase_change_saturday_to_monday_0100(schedule):
    """next_phase_change() returns Monday 01:00 UTC from Saturday."""
    ts = datetime(2026, 10, 24, 0, 0, tzinfo=timezone.utc).timestamp()
    expected_ts = datetime(2026, 10, 26, 1, 0, tzinfo=timezone.utc).timestamp()
    assert pricing.next_phase_change(schedule, ts) == expected_ts

def test_price_status_without_schedule(entry):
    """price_status() returns None phase and no next_change without a schedule."""
    ts = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc).timestamp()
    status = pricing.price_status(entry, {}, ts)
    assert status["phase"] is None
    assert status["next_change"] is None

def test_price_status_peak(entry, provider_cfg):
    """price_status() returns peak phase and next_change during peak."""
    ts = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc).timestamp()
    status = pricing.price_status(entry, provider_cfg, ts)
    assert status["phase"] == "peak"
    assert status["next_change"] == datetime(2026, 10, 23, 4, 0, tzinfo=timezone.utc).timestamp()

def test_price_status_off_peak(entry, provider_cfg):
    """price_status() returns off-peak phase and next_change during off-peak."""
    ts = datetime(2026, 10, 23, 4, 0, tzinfo=timezone.utc).timestamp()
    status = pricing.price_status(entry, provider_cfg, ts)
    assert status["phase"] == "off-peak"
    assert status["next_change"] == datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc).timestamp()

def test_price_status_with_price_from(entry_with_price_from, provider_cfg):
    """price_status() includes upcoming price_from date and price."""
    ts = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc).timestamp()
    status = pricing.price_status(entry_with_price_from, provider_cfg, ts)
    assert status["upcoming"] == {
        "from": "2027-01-01",
        "price": entry_with_price_from["price_from"]["2027-01-01"],
    }
