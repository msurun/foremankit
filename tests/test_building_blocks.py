"""Unit tests of the small pure building blocks the router is made of (chain order, estimates, cooldowns, prompts)."""
import pytest

from foremankit import prompting, quota, reasoning
from foremankit.chain import build_chain, parse_entry
from foremankit.formatting import format_review
from foremankit.records import CallContext, RouterError
from foremankit.selection import estimate_seconds, unsuited_reason

ROUTING = {"chains": {"quick": ["groq", "mistral"], "hard_code": ["deepseek"], "private": ["llm"]},
           "escalation": ["deepseek"]}


def context(kind="quick", est_in=100, max_out=500) -> CallContext:
    return CallContext(call_id="id", kind=kind, sensitivity="normal", escalate=False, est_in=est_in, max_out=max_out)


# ---------------------------------------------------------------- chain order

def test_a_private_task_only_sees_the_private_chain():
    assert build_chain(ROUTING, "quick", "private", escalate=False) == ["llm"]


def test_an_internal_task_tries_the_local_llm_first():
    assert build_chain(ROUTING, "quick", "internal", escalate=False) == ["llm", "groq", "mistral"]


def test_an_internal_hard_task_tries_the_local_llm_last():
    assert build_chain(ROUTING, "hard_code", "internal", escalate=False) == ["deepseek", "llm"]


def test_escalation_replaces_the_chain_of_the_kind():
    assert build_chain(ROUTING, "quick", "normal", escalate=True) == ["deepseek"]


def test_a_chain_entry_is_split_into_provider_alias_and_node():
    assert parse_entry("gemini:flash") == ("gemini", "flash", "")
    assert parse_entry("llm@legion") == ("llm", None, "legion")


# ---------------------------------------------------------------- estimates and suitability

def test_the_time_estimate_adds_reading_the_prompt_to_writing_the_answer():
    speed = {"prompt_tps": 100, "gen_tps": 10}
    assert estimate_seconds(speed, est_in=1000, max_out=200) == pytest.approx(10 + 20)


def test_the_estimate_counts_only_the_answer_size_a_node_usually_writes():
    speed = {"prompt_tps": 100, "gen_tps": 10, "expected_output_cap": 100}
    assert estimate_seconds(speed, est_in=0, max_out=4000) == pytest.approx(10)


def test_a_kind_missing_from_the_local_policy_is_not_suited_for_the_node():
    reason = unsuited_reason({"summarize": {}}, context(kind="quick"), {})
    assert "not set up for local models" in reason


def test_an_input_above_the_node_limit_is_not_suited_for_the_node():
    policy = {"quick": {"max_input_tokens": 50}}
    assert "input too big" in unsuited_reason(policy, context(est_in=100), {})


def test_a_task_longer_than_the_node_allows_is_not_suited_for_the_node():
    speed = {"prompt_tps": 100, "gen_tps": 10, "max_seconds": 5}
    assert "too long" in unsuited_reason(None, context(est_in=1000), speed)


# ---------------------------------------------------------------- cooldowns

def test_each_rate_limit_in_a_row_waits_longer_than_the_one_before():
    state: dict = {}
    first = quota.set_cooldown(state, quota.FailedCall("p", now=0.0, status=429), {"cooldown_429_s": [10, 50]})
    second = quota.set_cooldown(state, quota.FailedCall("p", now=1.0, status=429), {"cooldown_429_s": [10, 50]})
    assert (first, second) == (10, 50)


def test_the_server_retry_after_wins_when_it_asks_for_longer():
    delay = quota.set_cooldown({}, quota.FailedCall("p", now=0.0, status=429, retry_after=900), {})
    assert delay == 900


def test_a_server_error_uses_the_fixed_server_error_pause():
    assert quota.set_cooldown({}, quota.FailedCall("p", now=0.0, status=503), {"cooldown_5xx_s": 7}) == 7


# ---------------------------------------------------------------- reasoning and prompts

def test_the_internal_floor_raises_a_low_reasoning_level():
    routing = {"reasoning": {"kinds": {"quick": "minimal"}, "internal_floor": "medium"}}
    assert reasoning.reasoning_level(routing, "quick", "internal", escalate=False) == "medium"


def test_a_big_requested_answer_gets_a_size_warning():
    routing = {"calls": {"warn_output_tokens": 1000}}
    assert reasoning.size_notes(routing, 2000) and not reasoning.size_notes(routing, 500)


def test_a_missing_context_file_is_refused_with_its_name(tmp_path):
    with pytest.raises(RouterError, match="context file not found"):
        prompting.read_context_files({}, [str(tmp_path / "nope.py")], trusted=False)


def test_a_big_data_file_is_refused_for_an_external_model_but_not_for_a_trusted_one(tmp_path):
    data = tmp_path / "rows.csv"
    data.write_text("a,b\n" * 100, encoding="utf-8")
    routing = {"context_guard": {"data_extensions": [".csv"], "max_data_lines_non_private": 10}}
    with pytest.raises(RouterError, match="lines of data refused"):
        prompting.read_context_files(routing, [str(data)], trusted=False)
    assert prompting.read_context_files(routing, [str(data)], trusted=True)


# ---------------------------------------------------------------- formatting

def test_a_review_without_findings_says_so():
    report = {"reviewer": "llm/qwen", "findings": []}
    assert format_review(report) == ["review by llm/qwen: no findings"]
