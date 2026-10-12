"""Lab nodes: machines that lend their local LLM server to each other.

Covers how a machine finds itself in nodes.yaml, who may borrow from whom, how the router ranks and filters nodes
(speed, busy, data level, switches) and that private data never travels where it must not.
"""
from __future__ import annotations

import socket
import time

import pytest

from foremankit import nodes, quota
from foremankit.config import REPO_DIR
from foremankit.netguard import PrivacyError
from foremankit.router import RouterError
from conftest import chat_json, read_log, tags

SELF_SERVER = "http://127.0.0.1:8080"          # llama-server on THIS machine (providers.yaml llm.endpoints)
LEGION = "http://neuron:8080"                   # httpx lower-cases host names: what the mock transport sees
FIREFLY = "http://sgltp1738:8080"
SPEED_FIREFLY = {"prompt_tps": 190, "gen_tps": 19, "expected_output_cap": 1500, "max_seconds": 150}
SPEED_LEGION = {"prompt_tps": 240, "gen_tps": 33, "expected_output_cap": 1500, "max_seconds": 150}


def lab_config() -> dict:
    """The two laptops of the real lab, as the parsed nodes.yaml."""
    return {
        "settings": {"key_env": "FOREMANKIT_PEER_KEY", "tunnel_enabled": False,
                     "loan_to_higher_priority": {"when_idle": True, "max_seconds": 45}},
        "nodes": {
            "firefly": {"hostname": "SGLTP1738", "priority": 1, "lan": ["SGLTP1738"],
                        "tunnel": ["firefly.tail1234.ts.net"], "speed": SPEED_FIREFLY},
            "legion": {"hostname": "Neuron", "priority": 2, "lan": ["Neuron"], "speed": SPEED_LEGION}}}


def private_lan(_host):
    return ["192.168.1.20"]                     # every host sits on the home LAN


def idle(_request=None):
    return [{"id": 0, "is_processing": False}]


def busy(_request=None):
    return [{"id": 0, "is_processing": True}]


def serve(mock, base, model="m", slots=None, answer="ok"):
    """Make `base` answer like a llama-server: model list, optional /slots and chat completions."""
    mock.on(base, "/v1/models", {"data": [{"id": model}]})
    mock.on(base, "/api/tags", 404)
    mock.on(base, "/v1/chat/completions", chat_json(answer))
    if slots is not None:
        mock.on(base, "/slots", slots)


@pytest.fixture
def lab(monkeypatch, cfg):
    """Put the two-laptop lab into the router config; the test says which machine it runs on."""
    monkeypatch.setenv("FOREMANKIT_PEER_KEY", "shared-secret")
    cfg["nodes"] = lab_config()

    def run_on(machine: str) -> dict:
        monkeypatch.setenv("FOREMANKIT_SELF", machine)
        return cfg["nodes"]
    return run_on


def ask(router, sensitivity="private"):
    return router.delegate("question", kind="quick", sensitivity=sensitivity, max_output_tokens=64)


def provider_cfg(cfg) -> dict:
    return cfg["providers"]["llm"]


# ---------------------------------------------------------------- how a machine finds itself and its neighbours

def test_a_machine_finds_itself_by_hostname_ignoring_case(monkeypatch):
    monkeypatch.delenv("FOREMANKIT_SELF")
    monkeypatch.setattr(nodes.platform, "node", lambda: "sgltp1738")
    assert nodes.self_name(lab_config()) == "firefly"


def test_the_environment_can_override_the_hostname(monkeypatch):
    monkeypatch.setenv("FOREMANKIT_SELF", "legion")
    assert nodes.self_name(lab_config()) == "legion"


def test_this_machine_comes_first_and_the_more_important_machine_lends_without_limits(lab, cfg):
    config = lab("firefly")
    this, other = nodes.build_nodes(config, provider_cfg(cfg))
    assert (this.name, this.is_self) == ("firefly", True)
    assert other.name == "legion" and not other.idle_only and other.loan_max_s is None
    assert other.endpoints[0].api_key == "shared-secret"


def test_borrowing_from_a_more_important_machine_is_limited_to_idle_and_short(lab, cfg):
    config = lab("legion")
    this, other = nodes.build_nodes(config, provider_cfg(cfg))
    assert this.name == "legion"
    assert other.name == "firefly" and other.idle_only is True and other.loan_max_s == 45


def test_an_unlisted_machine_becomes_the_node_local_with_the_default_speed(lab, cfg):
    config = lab("some-other-pc")
    local, *others = nodes.build_nodes(config, provider_cfg(cfg))
    assert local.name == nodes.LOCAL_NAME and local.is_self and local.speed == provider_cfg(cfg)["speed"]
    assert {n.name for n in others} == {"firefly", "legion"} and all(n.idle_only for n in others)


def test_without_a_nodes_file_this_machine_is_the_only_node(cfg):
    only = nodes.build_nodes({}, provider_cfg(cfg))
    assert [n.name for n in only] == [nodes.LOCAL_NAME]
    assert [e.url for e in only[0].endpoints] == [SELF_SERVER, "http://localhost:11434"]


def test_a_node_that_does_not_serve_is_not_in_the_pool(lab, cfg):
    config = lab("firefly")
    config["nodes"]["legion"]["serves"] = False
    assert [n.name for n in nodes.build_nodes(config, provider_cfg(cfg))] == ["firefly"]


def test_a_machine_that_does_not_serve_has_no_node_of_its_own(lab, cfg):
    config = lab("firefly")
    config["nodes"]["firefly"]["serves"] = False          # e.g. a laptop without GPU that only uses the pool
    assert [n.name for n in nodes.build_nodes(config, provider_cfg(cfg))] == ["legion"]


def test_every_node_accepts_all_data_levels_unless_its_entry_says_otherwise(lab, cfg):
    config = lab("firefly")
    config["nodes"]["legion"]["accept"] = "internal"
    firefly, legion = nodes.build_nodes(config, provider_cfg(cfg))
    assert all(firefly.accepts(level) for level in nodes.LEVELS)
    assert legion.accepts("normal") and legion.accepts("internal") and not legion.accepts("private")


def test_the_defaults_block_applies_to_every_node(lab, cfg):
    config = lab("firefly")
    config["defaults"] = {"accept": "normal"}
    assert all(not n.accepts("internal") for n in nodes.build_nodes(config, provider_cfg(cfg)))


def test_tunnel_addresses_are_used_only_when_enabled(lab, cfg):
    config = lab("legion")
    firefly = nodes.build_nodes(config, provider_cfg(cfg))[1]
    assert [e.name for e in firefly.endpoints] == ["firefly/lan"]
    config["settings"]["tunnel_enabled"] = True
    firefly = nodes.build_nodes(config, provider_cfg(cfg))[1]
    assert [e.url for e in firefly.endpoints] == ["http://SGLTP1738:8080", "http://firefly.tail1234.ts.net:8080"]


def test_a_full_url_is_used_as_written_without_a_port(lab, cfg):
    config = lab("firefly")
    config["settings"]["tunnel_enabled"] = True
    config["nodes"]["legion"]["tunnel"] = ["https://neuron.tail1234.ts.net"]
    legion = nodes.build_nodes(config, provider_cfg(cfg))[1]
    assert legion.endpoints[-1].url == "https://neuron.tail1234.ts.net"


def test_a_changing_dhcp_address_can_be_set_from_the_environment(lab, cfg, monkeypatch):
    config = lab("firefly")
    monkeypatch.setenv("FOREMANKIT_NODE_LEGION_LAN", "192.168.1.77, 192.168.1.78")
    legion = nodes.build_nodes(config, provider_cfg(cfg))[1]
    assert [e.url for e in legion.endpoints] == ["http://192.168.1.77:8080", "http://192.168.1.78:8080"]


def test_a_host_name_is_resolved_to_ipv4_to_avoid_the_link_local_ipv6_trap(lab, cfg, monkeypatch):
    config = lab("firefly")
    monkeypatch.delenv("FOREMANKIT_PEER_RESOLVE")
    nodes._RESOLVED.clear()
    asked = []

    def fake_getaddrinfo(host, port, family=0, type=0, **kw):     # Windows answers fe80:: first and IPv4 second
        asked.append((host, family))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.4.83", 0))] if family == socket.AF_INET else []
    monkeypatch.setattr(nodes.socket, "getaddrinfo", fake_getaddrinfo)
    legion = nodes.build_nodes(config, provider_cfg(cfg))[1]
    assert legion.endpoints[0].url == "http://192.168.4.83:8080"
    nodes.build_nodes(config, provider_cfg(cfg))
    assert len(asked) == len(set(asked))                            # the second call came from the cache

    def machine_is_off(*_args, **_kwargs):
        raise OSError("asleep")
    monkeypatch.setattr(nodes.socket, "getaddrinfo", machine_is_off)
    nodes._RESOLVED.clear()
    assert nodes.build_nodes(config, provider_cfg(cfg))[1].endpoints[0].url == "http://Neuron:8080"   # keep the name


def test_nodes_off_in_the_environment_hides_the_file(monkeypatch):
    monkeypatch.setenv("FOREMANKIT_NODES", "off")
    assert nodes.load_nodes(REPO_DIR / "config") == {}


def test_the_shipped_nodes_file_parses_and_lists_the_real_lab(monkeypatch):
    monkeypatch.delenv("FOREMANKIT_NODES")
    assert set(nodes.load_nodes(REPO_DIR / "config")["nodes"]) >= {"firefly", "legion"}


def test_ensure_key_creates_it_once_and_keeps_other_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n", encoding="utf-8")
    _, created = nodes.ensure_key(env)
    first = env.read_text(encoding="utf-8")
    _, again = nodes.ensure_key(env)
    assert created and not again and env.read_text(encoding="utf-8") == first
    assert "OTHER=1" in first and len(first.split("FOREMANKIT_PEER_KEY=")[1].strip()) >= 32


# ---------------------------------------------------------------- ranking: the node expected to finish first goes first

@pytest.mark.parametrize("machine, other_server", [("firefly", LEGION), ("legion", FIREFLY)])
def test_the_faster_node_is_called_first_on_either_machine(lab, make_router, mock, machine, other_server):
    lab(machine)
    serve(mock, SELF_SERVER, "local-model", slots=idle)
    serve(mock, other_server, "other-model", slots=idle)
    assert ask(make_router(resolver=private_lan))["node"] == "legion"       # 33 tok/s beats 19 tok/s


def test_the_slower_node_answers_when_the_faster_one_is_down(lab, make_router, mock):
    lab("firefly")
    serve(mock, SELF_SERVER, "local-model")                           # the Legion is not served: switched off
    result = ask(make_router(resolver=private_lan))
    assert result["node"] == "firefly" and result["endpoint"] == "llamacpp"


def test_a_busy_node_is_ranked_after_an_idle_one(lab, make_router, mock):
    lab("firefly")
    serve(mock, SELF_SERVER, slots=idle)
    serve(mock, LEGION, slots=busy)
    assert ask(make_router(resolver=private_lan))["node"] == "firefly"


def test_an_unlisted_machine_uses_the_lab_nodes(lab, make_router, mock):
    lab("some-other-pc")
    serve(mock, LEGION, slots=idle)
    serve(mock, FIREFLY, slots=idle)
    assert ask(make_router(resolver=private_lan))["node"] == "legion"


# ---------------------------------------------------------------- the loan rule between machines of different priority

def test_the_less_important_machine_does_not_borrow_while_the_owner_is_busy(lab, make_router, mock):
    lab("legion")
    serve(mock, FIREFLY, slots=busy)
    with pytest.raises(PrivacyError, match="busy with its owner"):
        ask(make_router(resolver=private_lan))
    assert f"{FIREFLY}/v1/chat/completions" not in mock.calls


def test_the_less_important_machine_borrows_when_the_owner_is_idle_and_the_task_is_short(lab, make_router, mock):
    lab("legion")
    serve(mock, FIREFLY, slots=idle)
    assert ask(make_router(resolver=private_lan))["node"] == "firefly"


def test_a_long_task_is_never_lent_even_on_the_last_pass(lab, make_router, mock):
    lab("legion")
    serve(mock, FIREFLY, slots=idle)
    long_task = "word " * 6000                                       # far above the 45 s loan cap at 190 tok/s
    with pytest.raises(PrivacyError, match="too long for a priority loan"):
        make_router(resolver=private_lan).delegate(long_task, kind="quick", sensitivity="private",
                                                   max_output_tokens=3000)
    assert f"{FIREFLY}/v1/chat/completions" not in mock.calls


# ---------------------------------------------------------------- data levels per node

def test_a_node_that_accepts_only_normal_data_gets_no_internal_or_private_data(lab, make_router, mock):
    config = lab("firefly")
    config["nodes"]["legion"]["accept"] = "normal"
    config["nodes"]["firefly"]["serves"] = False                      # leave the Legion as the only candidate
    serve(mock, LEGION)
    for level in ("internal", "private"):
        with pytest.raises((PrivacyError, RouterError), match="accepts up to normal"):
            ask(make_router(resolver=private_lan), sensitivity=level)
    assert f"{LEGION}/v1/chat/completions" not in mock.calls


def test_a_node_that_accepts_internal_data_gets_no_private_data(lab, make_router, mock):
    config = lab("firefly")
    config["nodes"]["legion"]["accept"] = "internal"
    config["nodes"]["firefly"]["serves"] = False
    serve(mock, LEGION)
    assert ask(make_router(resolver=private_lan), sensitivity="internal")["node"] == "legion"
    with pytest.raises(PrivacyError, match="accepts up to internal"):
        ask(make_router(resolver=private_lan), sensitivity="private")


def test_private_data_goes_to_the_next_accepting_node(lab, make_router, mock):
    config = lab("firefly")
    config["nodes"]["legion"]["accept"] = "internal"
    serve(mock, LEGION)
    serve(mock, SELF_SERVER)
    assert ask(make_router(resolver=private_lan))["node"] == "firefly"


def test_private_data_is_refused_over_a_public_address(lab, make_router, mock):
    config = lab("firefly")
    config["nodes"]["firefly"]["serves"] = False
    serve(mock, LEGION)
    with pytest.raises(PrivacyError):
        ask(make_router(resolver=lambda host: ["8.8.8.8"]))
    assert f"{LEGION}/v1/chat/completions" not in mock.calls


def test_the_shared_key_is_sent_to_the_other_node(lab, make_router, mock):
    lab("firefly")
    serve(mock, LEGION)
    seen = []
    mock.routes[(LEGION, "/v1/chat/completions")] = [lambda r: (seen.append(r.headers.get("authorization")), chat_json())[1]]
    ask(make_router(resolver=private_lan))
    assert seen == ["Bearer shared-secret"]


def test_a_wrong_key_is_reported_clearly(lab, make_router, mock):
    lab("firefly")
    mock.on(LEGION, "/v1/models", 401)
    mock.on(LEGION, "/api/tags", 401)
    legion = make_router(resolver=private_lan).nodes_status()["nodes"][1]
    assert legion["reachable"] is False and any("API key" in error for error in legion["errors"])


# ---------------------------------------------------------------- a bigger lab

def test_five_nodes_each_is_judged_on_its_own(lab, make_router, mock):
    config = lab("firefly")
    config["nodes"].update({
        "dead": {"priority": 3, "lan": ["10.0.0.2"], "speed": SPEED_LEGION},
        "busy": {"priority": 3, "lan": ["10.0.0.3"], "speed": SPEED_LEGION},
        "weak": {"priority": 3, "lan": ["10.0.0.4"], "accept": "normal", "speed": SPEED_LEGION}})
    serve(mock, LEGION, slots=idle)
    serve(mock, "http://10.0.0.3:8080", slots=busy)
    serve(mock, "http://10.0.0.4:8080", slots=idle)
    result = ask(make_router(resolver=private_lan))
    assert result["node"] == "legion"
    skipped = " ".join(result["skipped"])
    assert "llm@dead(unreachable" in skipped and "llm@weak(accepts up to normal)" in skipped
    for host in ("neuron:8080", "10.0.0.2:8080", "10.0.0.3:8080"):          # all asked in the same round
        assert any(f"//{host}/" in call for call in mock.calls), host


def test_a_node_that_does_not_answer_cannot_stall_the_probe(lab, make_router, mock, cfg):
    lab("firefly")
    provider_cfg(cfg)["probe_deadline"] = 0.3
    mock.on(LEGION, "/api/tags", lambda r: (time.sleep(1.5), tags("m"))[1])      # a laptop that is asleep
    router = make_router(cfg, resolver=private_lan)
    started = time.perf_counter()
    probe = router.node_probe(router.nodes()[1], private=True)
    assert probe.endpoint is None and time.perf_counter() - started < 1.0
    assert any("no answer within" in error for error in probe.errors)


# ---------------------------------------------------------------- switches

def router_with_both_nodes_up(make_router, mock, switch):
    """A router whose two servers answer, after `switch(router)` turned something off."""
    serve(mock, SELF_SERVER)
    serve(mock, LEGION)
    router = make_router(resolver=private_lan)
    switch(router)
    return router


def test_switching_off_the_others_leaves_this_machine(lab, make_router, mock):
    lab("firefly")
    router = router_with_both_nodes_up(make_router, mock, lambda r: r.set_toggle("others", "off"))
    assert ask(router)["node"] == "firefly"


def test_switching_off_one_node_by_name(lab, make_router, mock):
    lab("firefly")
    router = router_with_both_nodes_up(make_router, mock, lambda r: r.set_toggle("legion", "off"))
    assert ask(router)["node"] == "firefly"


def test_switching_off_this_machine_leaves_the_others(lab, make_router, mock):
    lab("firefly")
    router = router_with_both_nodes_up(make_router, mock, lambda r: r.set_toggle("self", "off"))
    assert ask(router)["node"] == "legion"


def test_switching_everything_off_fails_closed_for_private_data(lab, make_router, mock):
    lab("firefly")
    router = router_with_both_nodes_up(make_router, mock, lambda r: r.set_toggle("all", "off"))
    with pytest.raises(PrivacyError, match="disabled"):
        ask(router)


def test_the_environment_can_switch_the_others_off(lab, make_router, mock, monkeypatch):
    lab("firefly")
    monkeypatch.setenv("LLM_OTHERS_ENABLED", "off")
    assert ask(router_with_both_nodes_up(make_router, mock, lambda r: None))["node"] == "firefly"


def test_a_switch_from_the_command_line_beats_the_environment(lab, make_router, mock, monkeypatch):
    lab("firefly")
    monkeypatch.setenv("LLM_OTHERS_ENABLED", "on")
    router = router_with_both_nodes_up(make_router, mock, lambda r: r.set_toggle("legion", "off"))
    assert ask(router)["node"] == "firefly"


def test_a_node_name_beats_the_group_it_belongs_to(lab, make_router, mock):
    lab("firefly")

    def others_off_but_legion_on(router):
        router.set_toggle("others", "off")
        router.set_toggle("legion", "on")
    assert ask(router_with_both_nodes_up(make_router, mock, others_off_but_legion_on))["node"] == "legion"


def test_an_unknown_selector_is_refused(lab, make_router):
    lab("firefly")
    with pytest.raises(RouterError, match="unknown node"):
        make_router().set_toggle("llm_local", "off")


# ---------------------------------------------------------------- statistics and limits are kept per node

def test_a_failing_node_gets_its_own_cooldown_and_the_others_keep_working(lab, make_router, mock):
    lab("firefly")
    serve(mock, SELF_SERVER)
    serve(mock, LEGION)
    mock.routes[(LEGION, "/v1/chat/completions")] = [500]
    router = make_router(resolver=private_lan)
    assert ask(router)["node"] == "firefly"                           # the Legion failed, this machine answered
    state = router.store.load()
    assert quota.cooldown_left(state, "llm@legion", router.clock()) > 0
    assert quota.cooldown_left(state, "llm@firefly", router.clock()) == 0


def test_the_call_log_names_the_node(lab, make_router, mock, tmp_path):
    lab("firefly")
    serve(mock, LEGION)
    ask(make_router(resolver=private_lan))
    call = [row for row in read_log(tmp_path) if row.get("event") == "call"][0]
    assert call["provider"] == "llm" and call["node"] == "legion"


def test_nodes_status_reports_busy_and_the_loan_rule(lab, make_router, mock):
    lab("legion")
    serve(mock, FIREFLY, slots=busy)
    status = make_router(resolver=private_lan).nodes_status()
    firefly = status["nodes"][1]
    assert status["self"] == "legion" and status["key_set"] is True
    assert firefly["busy"] is True and firefly["private_ok"] is True and "only when idle" in firefly["rule"]


def test_a_node_taken_by_another_session_after_the_ranking_is_skipped(lab, make_router, mock):
    lab("firefly")
    serve(mock, SELF_SERVER, slots=idle)
    questions = []

    def taken_after_the_first_look(_request=None):
        questions.append(1)
        return idle() if len(questions) == 1 else busy()

    serve(mock, LEGION, slots=taken_after_the_first_look)
    result = ask(make_router(resolver=private_lan))
    assert result["node"] == "firefly"
    assert any(text.startswith("llm@legion(busy") for text in result["skipped"])


def test_the_only_node_is_still_used_when_it_turns_out_to_be_busy(lab, make_router, mock):
    lab("firefly")
    serve(mock, SELF_SERVER, slots=busy)
    assert ask(make_router(resolver=private_lan))["node"] == "firefly"   # waits for it on the last pass
