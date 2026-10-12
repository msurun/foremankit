"""Learning a node's real speed from the calls it served (foremankit/speed.py) and using it for ranking."""
from __future__ import annotations

import pytest
from conftest import chat_json

from foremankit import speed

NOW = 1_000_000.0
DEFAULTS = speed.DEFAULT_SETTINGS
WRITTEN = {"prompt_tps": 200.0, "gen_tps": 40.0, "expected_output_cap": 1500, "max_seconds": 150}


def with_timings(predicted_per_second=2.0, predicted_n=100, prompt_per_second=None, prompt_n=0) -> dict:
    """A llama-server answer: chat_json plus its own `timings` block."""
    answer = chat_json("ok", prompt=50, completion=predicted_n)
    answer["timings"] = {"predicted_per_second": predicted_per_second, "predicted_n": predicted_n,
                         "prompt_per_second": prompt_per_second, "prompt_n": prompt_n}
    return answer


# ---------------------------------------------------------------- measure: what one call says

def test_server_timings_give_exact_generation_and_prompt_speed():
    result = {"timings": {"predicted_per_second": 33.5, "predicted_n": 120, "prompt_per_second": 240.0, "prompt_n": 900}}
    assert speed.measure(result, latency_s=9.0, assumed_prompt_tps=200) == {"gen_tps": 33.5, "prompt_tps": 240.0}


def test_a_short_answer_or_prompt_in_the_timings_says_nothing():
    result = {"timings": {"predicted_per_second": 90.0, "predicted_n": 5, "prompt_per_second": 900.0, "prompt_n": 20}}
    assert speed.measure(result, latency_s=1.0, assumed_prompt_tps=200) == {}


def test_without_timings_the_generation_speed_is_estimated_from_the_latency():
    usage = {"prompt_tokens": 400, "completion_tokens": 300}
    measured = speed.measure({"usage": usage}, latency_s=12.0, assumed_prompt_tps=200)     # 2 s reading, 10 s writing
    assert measured == {"gen_tps": pytest.approx(30.0)}


def test_a_short_answer_without_timings_is_not_used_for_the_estimate():
    assert speed.measure({"usage": {"prompt_tokens": 400, "completion_tokens": 20}}, 5.0, 200) == {}


def test_the_estimate_never_gives_the_generation_less_than_a_quarter_of_the_latency():
    usage = {"prompt_tokens": 100_000, "completion_tokens": 100}          # the prompt alone would "take" longer than the call
    assert speed.measure({"usage": usage}, latency_s=10.0, assumed_prompt_tps=100)["gen_tps"] == pytest.approx(40.0)


# ---------------------------------------------------------------- observe: the moving average

def test_the_first_measurement_is_stored_as_it_is_and_later_ones_are_averaged():
    state: dict = {}
    speed.observe(state, speed.Observation("box", {"gen_tps": 10.0}, NOW), alpha=0.5)
    speed.observe(state, speed.Observation("box", {"gen_tps": 20.0}, NOW + 1), alpha=0.5)
    entry = state["node_speed"]["box"]
    assert entry["gen_tps"] == 15.0 and entry["n_gen_tps"] == 2 and entry["ts"] == NOW + 1
    assert "prompt_tps" not in entry


def test_a_zero_measurement_is_ignored():
    state: dict = {}
    speed.observe(state, speed.Observation("box", {"gen_tps": 0.0}, NOW), alpha=0.3)
    assert "gen_tps" not in state["node_speed"]["box"]


# ---------------------------------------------------------------- effective: what the router plans with

def learned(n=3, gen=10.0):
    return {"node_speed": {"box": {"gen_tps": gen, "n_gen_tps": n, "ts": NOW}}}


def test_without_any_measurement_the_written_speed_is_used():
    assert speed.effective(speed.DeclaredSpeed(WRITTEN, "box", True), {}, DEFAULTS, NOW) == WRITTEN


def test_a_pinned_node_keeps_its_written_speed():
    assert speed.effective(speed.DeclaredSpeed(WRITTEN, "box", False), learned(), DEFAULTS, NOW) == WRITTEN


def test_learning_switched_off_keeps_the_written_speed():
    assert speed.effective(speed.DeclaredSpeed(WRITTEN, "box", True), learned(), {**DEFAULTS, "enabled": False}, NOW) == WRITTEN


def test_the_learned_value_takes_over_gradually_with_the_number_of_measurements():
    one = speed.effective(speed.DeclaredSpeed(WRITTEN, "box", True), learned(n=1), DEFAULTS, NOW)["gen_tps"]
    three = speed.effective(speed.DeclaredSpeed(WRITTEN, "box", True), learned(n=3), DEFAULTS, NOW)["gen_tps"]
    assert one == pytest.approx(40.0 * 2 / 3 + 10.0 / 3) and three == 10.0
    assert speed.effective(speed.DeclaredSpeed(WRITTEN, "box", True), learned(n=3), DEFAULTS, NOW)["prompt_tps"] == 200.0   # not measured: written


def test_an_old_measurement_is_forgotten():
    later = NOW + (DEFAULTS["max_age_h"] + 1) * 3600
    assert speed.effective(speed.DeclaredSpeed(WRITTEN, "box", True), learned(), DEFAULTS, later) == WRITTEN


def test_a_node_without_a_written_speed_uses_the_learned_one_at_once():
    assert speed.effective(speed.DeclaredSpeed(None, "box", True), learned(n=1), DEFAULTS, NOW) == {"gen_tps": 10.0}


def test_settings_overrides_are_merged_with_the_defaults():
    config = {"settings": {"speed_learning": {"alpha": 0.9}}}
    assert speed.settings(config) == {**DEFAULTS, "alpha": 0.9} and speed.settings(None) == DEFAULTS


# ---------------------------------------------------------------- the router learns and re-ranks

SLOW_CLAIM = "http://10.0.0.5:8080"          # claims 40 tok/s but delivers 2
HONEST = "http://10.0.0.6:8080"              # claims 10 tok/s and is the better choice once the claim is found out


def serve_node(mock, base, answer):
    mock.on(base, "/v1/models", {"data": [{"id": "m"}]})
    mock.on(base, "/api/tags", 404)
    mock.on(base, "/v1/chat/completions", answer)


@pytest.fixture
def two_other_nodes(cfg, mock):
    """This machine (unlisted, nothing served) plus two other nodes with equal priority."""
    cfg["nodes"] = {"nodes": {
        "claims": {"lan": ["10.0.0.5"], "speed": {"prompt_tps": 200, "gen_tps": 40, "expected_output_cap": 1500, "max_seconds": 150}},
        "honest": {"lan": ["10.0.0.6"], "speed": {"prompt_tps": 200, "gen_tps": 10, "expected_output_cap": 1500, "max_seconds": 150}}}}
    serve_node(mock, SLOW_CLAIM, with_timings(predicted_per_second=2.0))
    serve_node(mock, HONEST, with_timings(predicted_per_second=10.0))
    return cfg


def ask(router):
    return router.delegate("question", kind="quick", sensitivity="private", max_output_tokens=64)


def test_a_node_that_is_slower_than_claimed_loses_its_first_place(two_other_nodes, make_router):
    router = make_router(two_other_nodes, resolver=lambda host: ["192.168.1.20"])
    assert [ask(router)["node"] for _ in range(3)] == ["claims"] * 3        # the written speed puts it first
    assert ask(router)["node"] == "honest"                                    # three real measurements moved it back
    claims = next(n for n in router.nodes_status()["nodes"] if n["name"] == "claims")
    assert claims["speed"]["gen_tps"] == pytest.approx(2.0) and claims["speed_written"]["gen_tps"] == 40
    assert claims["speed_learned"]["n_gen_tps"] == 3 and claims["speed_pinned"] is False


def test_a_pinned_node_keeps_its_place_whatever_it_delivers(two_other_nodes, make_router):
    two_other_nodes["nodes"]["nodes"]["claims"]["learn"] = False
    router = make_router(two_other_nodes, resolver=lambda host: ["192.168.1.20"])
    assert [ask(router)["node"] for _ in range(5)] == ["claims"] * 5
    claims = next(n for n in router.nodes_status()["nodes"] if n["name"] == "claims")
    assert claims["speed"]["gen_tps"] == 40 and claims["speed_pinned"] is True and claims["speed_learned"] is None


def test_learning_can_be_switched_off_for_the_whole_lab(two_other_nodes, make_router):
    two_other_nodes["nodes"]["settings"] = {"speed_learning": {"enabled": False}}
    router = make_router(two_other_nodes, resolver=lambda host: ["192.168.1.20"])
    assert [ask(router)["node"] for _ in range(4)] == ["claims"] * 4


def test_a_dry_run_measures_nothing(two_other_nodes, make_router, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_DRY_RUN", "1")
    router = make_router(two_other_nodes, resolver=lambda host: ["192.168.1.20"])
    ask(router)
    assert not router.store.load().get("node_speed")
