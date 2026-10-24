from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from conftest import LAN, LOCAL, TUNNEL, chat_json, read_log, tags

from foremankit import reasoning
from foremankit.formatting import format_result
from foremankit.netguard import PrivacyError
from foremankit.router import RouterError

GEMINI = "https://generativelanguage.googleapis.com"
DEEPSEEK = "https://api.deepseek.com"
MISTRAL = "https://api.mistral.ai"
GROQ = "https://api.groq.com"


def test_routing_uses_first_provider_and_writes_file(monkeypatch, make_router, mock, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(GEMINI, "/chat/completions", chat_json("x" * 2000))
    res = make_router().delegate("write tests", kind="long_context")
    assert res["provider"] == "gemini"
    out = Path(res["path"])
    assert out.is_file() and "x" * 2000 in out.read_text(encoding="utf-8")
    assert len(res["preview"]) <= 601  # never the whole answer
    assert (tmp_path / ".agent-out" / ".gitignore").read_text() == "*\n"
    assert "x" * 2000 not in format_result(res)
    log = [row for row in read_log(tmp_path) if row["event"] == "call"]
    assert log[-1]["success"] and "prompt" not in log[-1]  # prompts not logged by default


def test_providers_without_key_are_skipped(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    res = make_router().delegate("boilerplate", kind="bulk_code")
    assert res["provider"] == "deepseek" and res["model"] == "deepseek-flash"
    assert all("generativelanguage" not in c for c in mock.calls)   # DeepSeek is first for bulk_code since 2026-10-11


def test_fallback_after_429_and_cooldown(monkeypatch, make_router, mock):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    mock.on(MISTRAL, "/chat/completions", 429)
    mock.on(GEMINI, "/chat/completions", chat_json())
    router = make_router()
    res = router.delegate("task", kind="bulk_code")
    assert res["provider"] == "gemini"  # paid Gemini fills in when the free provider is limited
    assert sum("mistral" in c for c in mock.calls) == 1  # no blind retry on 429
    res2 = router.delegate("task 2", kind="bulk_code")
    assert res2["provider"] == "gemini"
    assert any(s.startswith("mistral(cooldown") for s in res2["skipped"])


def test_single_retry_on_5xx_then_next_provider(monkeypatch, make_router, mock):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    mock.on(DEEPSEEK, "/chat/completions", 503, 503, 503)
    mock.on(GEMINI, "/chat/completions", chat_json())
    res = make_router().delegate("t", kind="long_context")
    assert res["provider"] == "gemini"
    assert sum("deepseek" in c for c in mock.calls) == 2  # 1 try + max 1 retry


def test_timeout_is_not_retried_and_falls_through_to_next_provider(monkeypatch, make_router, mock):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    mock.on(DEEPSEEK, "/chat/completions", httpx.ReadTimeout("hung"))
    mock.on(GEMINI, "/chat/completions", chat_json())
    res = make_router().delegate("t", kind="long_context")
    assert res["provider"] == "gemini"
    assert sum("deepseek" in c for c in mock.calls) == 1  # one try, no retry after a timeout
    assert any(s.startswith("deepseek:flash(timeout") for s in res["skipped"])


def test_hard_budget_stop(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    router = make_router()
    now = router.clock()
    from foremankit import budget
    router.store.update(lambda s: budget.record(s, now, router.routing["budget"]["daily_usd"]))
    with pytest.raises(RouterError, match="daily budget"):
        router.delegate("t", kind="hard_code")
    assert mock.calls == []


def test_cost_uses_cache_hit_tokens(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    mock.on(DEEPSEEK, "/chat/completions",
            chat_json(prompt=1_000_000, completion=1_000_000, prompt_cache_hit_tokens=500_000))
    router = make_router()
    res = router.delegate("t", kind="data_script")
    price = router.providers["deepseek"].cfg["models"]["flash"]["price"]
    expected = 0.5 * price["input_miss"] + 0.5 * price["input_hit"] + price["output"]
    assert res["cost_usd"] == pytest.approx(expected)
    assert router.status()["budget"]["day_usd"] == pytest.approx(expected)


def test_gemini_pro_replaces_blocked_deepseek_and_counts_toward_budget(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg["providers"]["deepseek"]["enabled"] = False  # as on the work laptop
    mock.on(GEMINI, "/chat/completions", chat_json(prompt=1_000_000, completion=100_000))
    router = make_router(cfg)
    res = router.delegate("hard problem", kind="hard_code")
    assert res["model"] == "gemini-3.1-pro-preview"
    assert any(s.startswith("deepseek:pro(disabled") for s in res["skipped"])
    assert res["cost_usd"] == pytest.approx(2.00 + 0.1 * 12.00)
    assert router.status()["budget"]["day_usd"] == pytest.approx(res["cost_usd"])


def test_connection_failure_sets_cooldown_so_blocked_host_is_not_retried(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(DEEPSEEK, "/chat/completions", httpx.ConnectTimeout("blocked"))
    mock.on(GEMINI, "/chat/completions", chat_json())
    router = make_router()
    assert router.delegate("a", kind="data_script")["provider"] == "gemini"
    n = sum("deepseek" in c for c in mock.calls)
    res = router.delegate("b", kind="data_script")
    assert sum("deepseek" in c for c in mock.calls) == n  # cooldown: no new attempts
    assert any(s.startswith("deepseek:flash(cooldown") for s in res["skipped"])


def test_hidden_thinking_tokens_are_billed_and_get_headroom(monkeypatch, make_router, mock):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    # real Gemini 3.8 reply: completion_tokens=1 but total_tokens includes ~59 hidden reasoning tokens
    mock.on(GEMINI, "/chat/completions", chat_json("OK", prompt=8, completion=1, total_tokens=68))
    router = make_router()
    res = router.delegate("t", kind="long_context", max_output_tokens=64)
    assert res["tokens_out"] == 60
    price = router.providers["gemini"].cfg["models"]["flash"]["price"]
    assert res["cost_usd"] == pytest.approx((8 * price["input_miss"] + 60 * price["output"]) / 1e6)
    # long_context -> reasoning "low": headroom 1024 so reasoning cannot eat the answer; routing.yaml output_floor lifts 64 to 2000
    assert mock.bodies[-1]["max_tokens"] == 2000 + 1024 and mock.bodies[-1]["reasoning_effort"] == "low"


@pytest.mark.parametrize("kind,sens,escalate,expected", [
    ("bulk_code", "normal", False, "minimal"), ("quick", "normal", False, "minimal"),
    ("summarize", "normal", False, "minimal"), ("long_context", "normal", False, "low"),
    ("data_script", "normal", False, "medium"), ("hard_code", "normal", False, "high"),
    ("quick", "internal", False, "low"),          # internal floor
    ("hard_code", "internal", False, "high"),     # floor never lowers
    ("bulk_code", "normal", True, "high"),        # escalation
])
def test_reasoning_level_per_kind(make_router, kind, sens, escalate, expected):
    assert reasoning.reasoning_level(make_router().routing, kind, sens, escalate) == expected


def test_reasoning_effort_only_sent_to_providers_that_support_it(monkeypatch, make_router, mock):
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    mock.on(MISTRAL, "/chat/completions", chat_json())
    make_router().delegate("t", kind="bulk_code")  # chain: gemini_free(no key) -> mistral
    body = mock.bodies[-1]
    assert "reasoning_effort" not in body and body["max_tokens"] == 4000


def test_gemini_free_bulk_code_maps_minimal_to_low_and_pro_hard_code_is_high(monkeypatch, make_router, mock, cfg):
    cfg["providers"]["gemini_free"]["enabled"] = True  # disabled in the deployed config
    cfg["routing"]["chains"]["bulk_code"] = ["gemini_free:flash", "gemini:flash"]
    monkeypatch.setenv("FREE_GEMINI_API_KEY", "f")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(GEMINI, "/chat/completions", chat_json())
    router = make_router()
    router.delegate("t", kind="bulk_code")
    # Gemini rejects "minimal" with HTTP 400, so the provider maps it to "low" (and uses low's headroom)
    assert mock.bodies[-1]["reasoning_effort"] == "low" and mock.bodies[-1]["max_tokens"] == 4000 + 1024
    cfg["providers"]["deepseek"]["enabled"] = False
    router.delegate("t", kind="hard_code")
    assert mock.bodies[-1]["model"] == "gemini-3.1-pro-preview"
    assert mock.bodies[-1]["reasoning_effort"] == "low" and mock.bodies[-1]["max_tokens"] == 4000 + 1024


def test_second_free_gemini_key_is_used_before_paid_gemini(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "paid")
    monkeypatch.setenv("FREE_GEMINI_API_KEY", "free")
    mock.on(GEMINI, "/chat/completions", chat_json())
    cfg["providers"]["gemini_free"]["enabled"] = True  # disabled in the deployed config
    cfg["routing"]["chains"]["bulk_code"] = ["gemini_free:flash", "gemini:flash"]
    router = make_router(cfg)
    res = router.delegate("t", kind="bulk_code")
    assert res["provider"] == "gemini_free" and res["cost_usd"] == 0
    assert router.status()["budget"]["day_usd"] == 0
    # free quota is tracked separately from the paid key
    q = router.store.load()["quota"]
    assert "gemini_free" in q and "gemini" not in q


def test_deepseek_auto_probe_prefers_deepseek_over_paid_gemini(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg["providers"]["deepseek"]["enabled"] = "auto"
    mock.on(DEEPSEEK, "/models", {"data": []})
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    mock.on(GEMINI, "/chat/completions", chat_json())
    res = make_router(cfg).delegate("t", kind="hard_code")
    assert res["provider"] == "deepseek" and res["model"] == "deepseek-v4-pro"


def test_deepseek_auto_probe_blocked_falls_back_and_reprobes_after_ttl(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg["providers"]["deepseek"]["enabled"] = "auto"
    now = [1_000_000.0]
    # first probe: blocked (as on the work network); later: reachable (e.g. at home)
    mock.on(DEEPSEEK, "/models", httpx.ConnectTimeout("blocked"), httpx.ConnectTimeout("blocked"), {"data": []})
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    mock.on(GEMINI, "/chat/completions", chat_json())
    router = make_router(cfg, clock=lambda: now[0])
    res = router.delegate("t", kind="hard_code")
    assert res["provider"] == "gemini"
    assert any(s.startswith("deepseek:pro(unreachable") for s in res["skipped"])
    probes = sum(c.endswith("/models") for c in mock.calls)
    router.delegate("t2", kind="hard_code")
    assert sum(c.endswith("/models") for c in mock.calls) == probes  # cached within ttl
    now[0] += 601
    assert router.delegate("t3", kind="hard_code")["provider"] == "deepseek"  # network changed


def test_probe_retries_once_and_failures_are_cached_only_briefly(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg["providers"]["deepseek"]["enabled"] = "auto"
    now = [1_000_000.0]
    # one dropped connection, then fine: the retry saves it (no failure cached at all)
    mock.on(DEEPSEEK, "/models", httpx.ConnectError("blip"), {"data": []})
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    mock.on(GEMINI, "/chat/completions", chat_json())
    router = make_router(cfg, clock=lambda: now[0])
    assert router.delegate("t", kind="hard_code")["provider"] == "deepseek"
    # a real outage is remembered for ~60 s only, not for the 10-minute success window
    router.prober.forget_clouds()
    mock.on(DEEPSEEK, "/models", httpx.ConnectError("down"))
    assert router.delegate("t", kind="hard_code")["provider"] == "gemini"
    mock.on(DEEPSEEK, "/models", {"data": []})
    now[0] += 30
    assert router.delegate("t", kind="hard_code")["provider"] == "gemini"      # still inside the 60 s failure memory
    now[0] += 40
    assert router.delegate("t", kind="hard_code")["provider"] == "deepseek"    # re-probed after 70 s


def test_deepseek_probe_never_probes_without_key(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg["providers"]["deepseek"]["enabled"] = "auto"
    mock.on(GEMINI, "/chat/completions", chat_json())
    make_router(cfg).delegate("t", kind="hard_code")
    assert not any("deepseek" in c for c in mock.calls)


def test_escalate_goes_to_pro(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    res = make_router().delegate("t", kind="bulk_code", escalate=True)
    assert res["model"] == "deepseek-v4-pro"


def test_rpd_quota_exhaustion_skips(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    cfg["providers"]["deepseek"]["limits"] = {"rpm": None, "rpd": 1, "tpd": None}
    mock.on(GEMINI, "/chat/completions", chat_json())
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    router = make_router(cfg)
    assert router.delegate("a", kind="long_context")["provider"] == "deepseek"
    res = router.delegate("b", kind="long_context")
    assert res["provider"] == "gemini" and "deepseek:flash(rpd limit reached)" in res["skipped"]


# ---------------------------------------------------------------- LLM nodes toggles / endpoints




def test_toggle_precedence_cli_over_env_over_config(monkeypatch, make_router):
    router = make_router()
    local = router.nodes()[0]
    assert router.node_status(local)["source"] == "config"
    monkeypatch.setenv("LLM_SELF_ENABLED", "off")
    status = router.node_status(local)
    assert (status["enabled"], status["source"]) == ("off", "env")
    router.set_toggle("self", "on")
    status = router.node_status(local)
    assert (status["enabled"], status["source"]) == ("on", "cli")





def test_llamacpp_server_is_detected_via_v1_models_and_preferred_over_ollama(monkeypatch, make_router, mock):
    LLAMACPP = "http://127.0.0.1:8080"
    mock.on(LLAMACPP, "/api/tags", 404)  # llama-server is up but has no Ollama API
    mock.on(LLAMACPP, "/v1/models", {"data": [{"id": "qwen3.6-35b-a3b"}]})
    mock.on(LLAMACPP, "/v1/chat/completions", chat_json("from llama.cpp"))
    mock.on(LOCAL, "/api/tags", tags("qwen2.5-coder:7b"))  # Ollama also running: llama.cpp still wins
    router = make_router()
    st = router.node_status(router.nodes()[0])
    assert st["endpoint"].startswith("llamacpp") and st["chosen_model"] == "qwen3.6-35b-a3b"
    res = router.delegate("t", kind="quick", sensitivity="private")
    assert res["provider"] == "llm" and res["endpoint"] == "llamacpp"
    assert res["model"] == "qwen3.6-35b-a3b" and not any(c.startswith(LOCAL) for c in mock.calls[-2:])


def test_down_host_is_probed_once_and_falls_through_to_ollama(make_router, mock):
    mock.on(LOCAL, "/api/tags", tags("llama3.2:3b"))  # nothing routed for :8080 -> connection error
    router = make_router()
    assert router.node_status(router.nodes()[0])["endpoint"].startswith("ollama")
    assert sum(c.startswith("http://127.0.0.1:8080") for c in mock.calls) == 1  # no second path tried


def test_fallback_model_skips_embedding_and_base(monkeypatch, make_router, mock):
    mock.on(LOCAL, "/api/tags", tags("nomic-embed-text:latest", "qwen2.5-coder:1.5b-base", "mistral:7b"))
    router = make_router()
    assert router.node_status(router.nodes()[0])["chosen_model"] == "mistral:7b"


def test_disabled_or_unreachable_llm_silently_skipped_in_normal_chain(monkeypatch, make_router, mock):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(GEMINI, "/chat/completions", chat_json())
    res = make_router().delegate("hi", kind="quick")
    assert res["provider"] == "gemini"
    assert any(s.startswith("llm@local(unreachable") for s in res["skipped"])


# ---------------------------------------------------------------- privacy

def with_box(cfg, address):
    """Add one other machine, `box`, at `address`; this machine stays the node `local`. Same priority as an unlisted
    machine, so borrowing from `box` carries no loan limits."""
    cfg["nodes"] = {"nodes": {"box": {"priority": 10**6, "lan": [address]}}}
    return cfg


def test_private_goes_to_another_machine_when_this_one_is_off(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(LOCAL, "/api/tags", tags("qwen2.5-coder:7b"))
    mock.on(TUNNEL, "/api/tags", tags("qwen3:32b"))
    mock.on(TUNNEL, "/v1/chat/completions", chat_json("private answer"))
    router = make_router(with_box(cfg, TUNNEL))
    router.set_toggle("self", "off")
    res = router.delegate("plant data", kind="data_script", sensitivity="private")
    assert res["provider"] == "llm" and res["node"] == "box" and res["endpoint"] == "box/lan"
    assert "generativelanguage.googleapis.com" not in mock.hosts()


def test_private_prefers_local(monkeypatch, make_router, mock):
    monkeypatch.setenv("OLLAMA_REMOTE_LAN_URL", LAN)
    mock.on(LOCAL, "/api/tags", tags("llama3.2:3b"))
    mock.on(LOCAL, "/v1/chat/completions", chat_json())
    res = make_router().delegate("x", kind="quick", sensitivity="private")
    assert res["provider"] == "llm" and res["model"] == "llama3.2:3b"


def test_private_refused_for_public_endpoint(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("LLM_SELF_ENABLED", "off")
    box = "http://my-box.example.com:11434"
    mock.on(box, "/api/tags", tags("qwen3:32b"))
    mock.on(box, "/v1/chat/completions", chat_json())
    router = make_router(with_box(cfg, box), resolver=lambda host: ["203.0.113.7"])
    with pytest.raises(PrivacyError, match="public address 203.0.113.7"):
        router.delegate("secret", kind="quick", sensitivity="private")
    assert not any("chat/completions" in c for c in mock.calls)


def test_private_public_endpoint_allowed_only_with_explicit_flag(monkeypatch, make_router, mock, cfg, tmp_path):
    monkeypatch.setenv("LLM_SELF_ENABLED", "off")
    box = "http://203.0.113.7:11434"
    cfg["providers"]["llm"]["allow_public_endpoint"] = True
    mock.on(box, "/api/tags", tags("qwen3:32b"))
    mock.on(box, "/v1/chat/completions", chat_json())
    res = make_router(with_box(cfg, box)).delegate("secret", kind="quick", sensitivity="private")
    assert res["provider"] == "llm"
    assert any(r.get("event") == "warning" for r in read_log(tmp_path))


def test_allow_private_false_blocks_instance(monkeypatch, make_router, mock, cfg):
    cfg["providers"]["llm"]["allow_private"] = False
    mock.on(LOCAL, "/api/tags", tags("llama3.2:3b"))
    mock.on(LOCAL, "/v1/chat/completions", chat_json())
    with pytest.raises(PrivacyError, match=r"llm@local\(not allowed for private"):
        make_router(cfg).delegate("x", kind="quick", sensitivity="private")
    assert not any("chat/completions" in c for c in mock.calls)


def test_fail_closed_without_llm_never_uses_cloud(monkeypatch, make_router, mock, cfg):
    for key in ("GEMINI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setenv(key, "k")
    # even if someone adds cloud providers to the private chain
    cfg["routing"]["chains"]["private"] = ["deepseek:flash", "gemini", "llm"]
    mock.on(GEMINI, "/chat/completions", chat_json())
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    with pytest.raises(PrivacyError, match="fail closed"):
        make_router(cfg).delegate("x", kind="hard_code", sensitivity="private", escalate=True)
    assert not any("chat/completions" in c for c in mock.calls)


def test_internal_uses_only_paid_no_train_gemini(monkeypatch, make_router, mock):
    for key in ("GEMINI_API_KEY", "FREE_GEMINI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setenv(key, "k")
    mock.on(GEMINI, "/chat/completions", chat_json())
    mock.on(GROQ, "/chat/completions", chat_json())
    mock.on(MISTRAL, "/chat/completions", chat_json())
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    res = make_router().delegate("process tags analysis", kind="bulk_code", sensitivity="internal")
    assert res["provider"] == "gemini" and res["cost_usd"] > 0  # not gemini_free/mistral/groq/deepseek
    assert any(s.startswith("mistral(not allowed for internal") for s in res["skipped"])
    assert not any(h for h in mock.hosts() if "mistral" in h or "groq" in h or "deepseek" in h)
    # hard_code would prefer DeepSeek for normal data; for internal it is excluded
    res = make_router().delegate("x", kind="hard_code", sensitivity="internal")
    assert res["provider"] == "gemini" and res["model"] == "gemini-3.1-pro-preview"


def test_internal_falls_back_to_llm_then_fails_closed(monkeypatch, make_router, mock):
    monkeypatch.setenv("FREE_GEMINI_API_KEY", "free")
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    mock.on(GEMINI, "/chat/completions", chat_json())
    mock.on(MISTRAL, "/chat/completions", chat_json())
    mock.on(LOCAL, "/api/tags", tags("llama3.2:3b"))
    mock.on(LOCAL, "/v1/chat/completions", chat_json())
    router = make_router()
    assert router.delegate("x", kind="data_script", sensitivity="internal")["provider"] == "llm"
    router.set_toggle("self", "off")
    with pytest.raises(PrivacyError, match="fail closed"):
        router.delegate("x", kind="data_script", sensitivity="internal")
    assert not any("chat/completions" in c and ("generativelanguage" in c or "mistral" in c) for c in mock.calls)


def test_deepseek_internal_only_with_per_machine_opt_in(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    cfg["providers"]["deepseek"]["trains_on_data"] = False  # hypothetical: the real config says True
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    mock.on(GEMINI, "/chat/completions", chat_json())
    # default: DeepSeek never gets internal data, the paid no-train Gemini does
    res = make_router(cfg).delegate("t", kind="hard_code", sensitivity="internal")
    assert res["provider"] == "gemini" and "deepseek:pro(not allowed for internal data)" in res["skipped"]
    assert not any("deepseek" in c for c in mock.calls)
    # explicit opt-in on this machine: DeepSeek is preferred again (it is first in the chain)
    monkeypatch.setenv("FOREMANKIT_ALLOW_INTERNAL", "DeepSeek")
    assert make_router(cfg).delegate("t", kind="hard_code", sensitivity="internal")["provider"] == "deepseek"
    # the opt-in cannot override trains_on_data, and never reaches private
    cfg["providers"]["deepseek"]["trains_on_data"] = True
    assert make_router(cfg).delegate("t", kind="hard_code", sensitivity="internal")["provider"] == "gemini"
    cfg["providers"]["deepseek"]["trains_on_data"] = False
    with pytest.raises(PrivacyError):
        make_router(cfg).delegate("t", kind="hard_code", sensitivity="private")


def test_real_config_keeps_deepseek_out_of_internal_even_with_opt_in(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("FOREMANKIT_ALLOW_INTERNAL", "deepseek")
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    mock.on(GEMINI, "/chat/completions", chat_json())
    res = make_router().delegate("t", kind="hard_code", sensitivity="internal")
    assert res["provider"] == "gemini" and not any("deepseek" in c for c in mock.calls)


def test_deepseek_internal_with_explicit_training_acceptance(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    mock.on(GEMINI, "/chat/completions", chat_json())
    assert make_router().status()["providers"][0]["internal_ok"] is False  # default: deepseek is not internal-ok
    monkeypatch.setenv("FOREMANKIT_ALLOW_INTERNAL_TRAINING", "DeepSeek")
    router = make_router()
    st = {p["name"]: p for p in router.status()["providers"]}
    assert st["deepseek"]["internal_ok"] and st["deepseek"]["trains_on_data"]  # the config still tells the truth
    res = router.delegate("t", kind="data_script", sensitivity="internal")  # deepseek:flash is first in this chain
    assert res["provider"] == "deepseek"
    assert router.delegate("t", kind="hard_code", sensitivity="internal")["model"] == "deepseek-v4-pro"
    # free tiers stay excluded, and private data never reaches DeepSeek whatever the setting
    assert not st["gemini_free"]["internal_ok"]
    with pytest.raises(PrivacyError):
        router.delegate("t", kind="quick", sensitivity="private")
    assert not any("deepseek" in c and "chat" in c for c in mock.calls[-1:])


def test_internal_requires_explicit_flag_and_no_training(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(GEMINI, "/chat/completions", chat_json())
    cfg["providers"]["gemini"]["allow_internal"] = False
    with pytest.raises(PrivacyError):
        make_router(cfg).delegate("x", kind="quick", sensitivity="internal")
    cfg["providers"]["gemini"]["allow_internal"] = True
    cfg["providers"]["gemini"]["trains_on_data"] = True  # flag alone is not enough
    with pytest.raises(PrivacyError):
        make_router(cfg).delegate("x", kind="quick", sensitivity="internal")
    assert not any("chat/completions" in c for c in mock.calls)


def test_internal_allows_large_data_files_but_private_stays_local_only(monkeypatch, make_router, mock, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(GEMINI, "/chat/completions", chat_json())
    (tmp_path / "tags.csv").write_text("a,b\n" + "1,2\n" * 500)
    router = make_router()
    res = router.delegate("analyse", kind="data_script", sensitivity="internal", context_files=["tags.csv"])
    assert res["provider"] == "gemini"
    with pytest.raises(PrivacyError):  # private never reaches Gemini, even the paid no-train key
        router.delegate("analyse", kind="data_script", sensitivity="private", context_files=["tags.csv"])


LLAMACPP = "http://127.0.0.1:8080"


def _slow_local_llm(mock, entered, release):
    """A llama-server whose chat call blocks until `release` is set (to hold its single slot)."""
    mock.on(LLAMACPP, "/api/tags", 404)
    mock.on(LLAMACPP, "/v1/models", {"data": [{"id": "qwen3.6-35b-a3b"}]})

    def slow(request):
        entered.set()
        release.wait(10)
        return chat_json("local answer")

    mock.on(LLAMACPP, "/v1/chat/completions", slow)


def test_parallel_internal_calls_are_spread_across_resources(monkeypatch, make_router, mock):
    import threading
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    entered, release = threading.Event(), threading.Event()
    _slow_local_llm(mock, entered, release)
    mock.on(GEMINI, "/chat/completions", chat_json("cloud answer"))
    router = make_router()
    results = {}
    first = threading.Thread(target=lambda: results.update(
        a=router.delegate("t1", kind="summarize", sensitivity="internal")))
    first.start()
    assert entered.wait(5)  # the first call holds the local single slot
    second = router.delegate("t2", kind="summarize", sensitivity="internal")
    assert second["provider"] == "gemini"  # local is busy: moved on at once, did not queue
    assert any(s.startswith("llm@local(busy") for s in second["skipped"])
    release.set()
    first.join(5)
    assert results["a"]["provider"] == "llm" and results["a"]["endpoint"] == "llamacpp"


def test_only_eligible_resource_busy_means_wait_not_fail(monkeypatch, make_router, mock):
    import threading
    import time
    entered, release = threading.Event(), threading.Event()
    _slow_local_llm(mock, entered, release)
    router = make_router()
    results = {}
    run = lambda key: results.update({key: router.delegate(key, kind="quick", sensitivity="private")})
    a = threading.Thread(target=run, args=("a",))
    a.start()
    assert entered.wait(5)
    b = threading.Thread(target=run, args=("b",))
    b.start()
    time.sleep(0.4)
    assert "b" not in results  # waiting for the slot, not failed closed
    release.set()
    a.join(5)
    b.join(5)
    assert results["a"]["provider"] == results["b"]["provider"] == "llm"


def test_empty_answer_after_exhausted_budget_is_a_failure_and_costs_are_kept(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    empty = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
             "usage": {"prompt_tokens": 100, "completion_tokens": 3000}}
    mock.on(DEEPSEEK, "/chat/completions", empty)
    mock.on(GEMINI, "/chat/completions", chat_json("real answer"))
    router = make_router()
    res = router.delegate("t", kind="data_script")  # deepseek:flash first
    assert res["provider"] == "gemini" and res["preview"] == "real answer"
    assert any(s.startswith("deepseek:flash(empty answer") for s in res["skipped"])
    assert router.status()["budget"]["day_usd"] > 0.003  # the wasted DeepSeek tokens are still counted


def test_deepseek_thinking_follows_the_reasoning_level(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    router = make_router()
    router.delegate("t", kind="long_context")      # level low -> thinking off, no extra headroom
    body = mock.bodies[-1]
    assert body["thinking"] == {"type": "disabled"} and "reasoning_effort" not in body and body["max_tokens"] == 4000
    router.delegate("t", kind="data_script")       # medium -> still off: paid models never think (2026-10-11)
    body = mock.bodies[-1]
    assert body["thinking"] == {"type": "disabled"} and "reasoning_effort" not in body
    assert body["max_tokens"] == 5500 and body["model"] == "deepseek-flash"   # +1500 ceiling, billed per generated token
    router.delegate("t", kind="hard_code")         # high -> off as well, on the Pro model
    body = mock.bodies[-1]
    assert body["thinking"] == {"type": "disabled"} and body["model"] == "deepseek-v4-pro" and body["max_tokens"] == 7000   # +3000


def test_local_llm_gets_thinking_off_except_for_high_level(monkeypatch, make_router, mock):
    mock.on(LLAMACPP, "/api/tags", 404)
    mock.on(LLAMACPP, "/v1/models", {"data": [{"id": "qwen3.6-35b-a3b"}]})
    mock.on(LLAMACPP, "/v1/chat/completions", chat_json("local"))
    router = make_router()
    router.delegate("t", kind="quick", sensitivity="private")  # minimal
    assert mock.bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    router.delegate("t", kind="data_script", sensitivity="private")  # medium: still off, local is slow
    assert mock.bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    router.delegate("t", kind="hard_code", sensitivity="private")  # high: let it think
    assert "chat_template_kwargs" not in mock.bodies[-1]


def test_local_model_policy_by_kind_and_size(monkeypatch, make_router, mock, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(LLAMACPP, "/api/tags", 404)
    mock.on(LLAMACPP, "/v1/models", {"data": [{"id": "qwen3.6-35b-a3b"}]})
    mock.on(LLAMACPP, "/v1/chat/completions", chat_json("local"))
    mock.on(GEMINI, "/chat/completions", chat_json("cloud"))
    (tmp_path / "big.txt").write_text("x" * 120_000)  # ~34k tokens
    router = make_router()
    # a kind that is not set up for the local model (routing.yaml llm.kinds) goes straight to the cloud
    res = router.delegate("short", kind="long_context", sensitivity="internal")
    assert res["provider"] == "gemini"
    assert any(s.startswith("llm@local(kind 'long_context' is not set up") for s in res["skipped"])
    # a suited kind is local while small ...
    assert router.delegate("short", kind="summarize", sensitivity="internal")["provider"] == "llm"
    # ... and goes to the cloud when the input is bigger than its limit
    res = router.delegate("summarise", kind="summarize", sensitivity="internal", context_files=["big.txt"])
    assert res["provider"] == "gemini"
    assert any(s.startswith("llm@local(input too big") for s in res["skipped"])
    # private data has no other allowed source: even an unsuited, huge task runs locally (no fail-closed error)
    res = router.delegate("summarise", kind="long_context", sensitivity="private", context_files=["big.txt"])
    assert res["provider"] == "llm" and not any("too big" in s or "not set up" in s for s in res["skipped"])


def test_time_estimate_is_a_second_safety_net(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(LLAMACPP, "/api/tags", 404)
    mock.on(LLAMACPP, "/v1/models", {"data": [{"id": "qwen3.6-35b-a3b"}]})
    mock.on(LLAMACPP, "/v1/chat/completions", chat_json("local"))
    mock.on(GEMINI, "/chat/completions", chat_json("cloud"))
    cfg["providers"]["llm"]["speed"]["max_seconds"] = 5  # absurdly strict
    res = make_router(cfg).delegate("short", kind="quick", sensitivity="internal")
    assert res["provider"] == "gemini" and any(s.startswith("llm@local(too long") for s in res["skipped"])


def test_machine_speed_overrides_from_env(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(LLAMACPP, "/api/tags", 404)
    mock.on(LLAMACPP, "/v1/models", {"data": [{"id": "qwen3.6-35b-a3b"}]})
    mock.on(LLAMACPP, "/v1/chat/completions", chat_json("local"))
    mock.on(GEMINI, "/chat/completions", chat_json("cloud"))
    cfg["providers"]["llm"]["speed"].update(gen_tps=1, max_seconds=100)  # static values: too slow
    slow = make_router(cfg).delegate("short", kind="quick", sensitivity="internal")
    assert slow["provider"] == "gemini"
    monkeypatch.setenv("LOCAL_LLM_GEN_TPS", "33")
    monkeypatch.setenv("LOCAL_LLM_PROMPT_TPS", "not-a-number")  # ignored
    fast = make_router(cfg).delegate("short", kind="quick", sensitivity="internal")
    assert fast["provider"] == "llm"


def test_internal_prefers_local_llm_except_hard_code_and_escalation(monkeypatch, make_router, mock):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(LLAMACPP, "/api/tags", 404)
    mock.on(LLAMACPP, "/v1/models", {"data": [{"id": "qwen3.6-35b-a3b"}]})
    mock.on(LLAMACPP, "/v1/chat/completions", chat_json("local"))
    mock.on(GEMINI, "/chat/completions", chat_json("cloud"))
    router = make_router()
    assert router.delegate("t", kind="bulk_code", sensitivity="internal")["provider"] == "llm"
    assert router.delegate("t", kind="hard_code", sensitivity="internal")["provider"] == "gemini"
    assert router.delegate("t", kind="bulk_code", sensitivity="internal", escalate=True)["provider"] == "gemini"


def test_data_file_guard(monkeypatch, make_router, mock, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(GEMINI, "/chat/completions", chat_json())
    mock.on(LOCAL, "/api/tags", tags("llama3.2:3b"))
    mock.on(LOCAL, "/v1/chat/completions", chat_json())
    (tmp_path / "big.csv").write_text("a,b\n" + "1,2\n" * 500)
    router = make_router()
    with pytest.raises(RouterError, match="schema"):
        router.delegate("analyse", kind="data_script", context_files=["big.csv"])
    res = router.delegate("analyse", kind="data_script", sensitivity="private", context_files=["big.csv"])
    assert res["provider"] == "llm"


# ---------------------------------------------------------------- profiles / dry run / stats

def test_profile_switching(make_router):
    router = make_router()
    assert router.apply_profile("work") == {"llm:self": "on", "llm:others": "auto"}
    assert router.node_status(router.nodes()[0])["enabled"] == "on"
    router.apply_profile("home")
    st = router.store.load()
    assert st["profile"] == "home" and st["toggles"] == {"llm:self": "auto", "llm:others": "auto"}
    with pytest.raises(RouterError):
        router.apply_profile("nope")


def test_dry_run_makes_no_network_calls(monkeypatch, make_router, mock):
    monkeypatch.setenv("FOREMANKIT_DRY_RUN", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    res = make_router().delegate("t", kind="hard_code")
    assert res["dry_run"] and "DRY RUN" in res["preview"] and mock.calls == []
    res = make_router().delegate("t", kind="quick", sensitivity="private")
    assert res["provider"] == "llm" and mock.calls == []


def test_stats(monkeypatch, make_router, mock):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(GEMINI, "/chat/completions", chat_json(prompt=1000, completion=1000))
    router = make_router()
    router.delegate("a", kind="long_context")
    s = router.stats()
    assert s["by_provider"]["gemini"]["calls"] == 1 and s["saved_usd"] > 0


def test_unknown_kind_and_sensitivity(make_router):
    with pytest.raises(RouterError):
        make_router().delegate("x", kind="nope")
    with pytest.raises(RouterError):
        make_router().delegate("x", kind="quick", sensitivity="secret")


def test_config_with_old_names_is_refused(make_router, cfg):
    from foremankit.config import ConfigError
    cfg["routing"]["chains"]["quick"] = ["llm_local", "llm_remote"]
    with pytest.raises(ConfigError, match="use the single provider `llm`"):
        make_router(cfg)

# ---------------------------------------------------------------- after the daily budget is spent

def spend_the_whole_daily_budget(router) -> None:
    from foremankit import budget
    router.store.update(lambda state: budget.record(state, router.clock(), router.routing["budget"]["daily_usd"]))


def local_llm_answers(mock) -> None:
    mock.on(LOCAL, "/api/tags", tags("llama3.2:3b"))
    mock.on(LOCAL, "/v1/chat/completions", chat_json("local answer"))


def test_a_spent_daily_budget_sends_an_internal_hard_task_to_the_local_llm(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    local_llm_answers(mock)
    router = make_router()
    spend_the_whole_daily_budget(router)
    result = router.delegate("t", kind="hard_code", sensitivity="internal")
    assert result["provider"] == "llm" and result["cost_usd"] == 0
    assert not any("chat/completions" in call and ("deepseek" in call or "generativelanguage" in call)
                   for call in mock.calls)


def test_a_spent_daily_budget_still_serves_a_bulk_task_from_a_free_source(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    mock.on(MISTRAL, "/chat/completions", chat_json())
    router = make_router()
    spend_the_whole_daily_budget(router)
    assert router.delegate("t", kind="bulk_code")["provider"] == "mistral"


def test_a_spent_daily_budget_never_reaches_a_paid_provider_when_nothing_free_is_left(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    router = make_router()
    spend_the_whole_daily_budget(router)
    with pytest.raises(RouterError, match="daily budget"):
        router.delegate("t", kind="bulk_code")
    assert not any("chat/completions" in call for call in mock.calls)


def test_switching_paid_delegation_off_skips_paid_sources(monkeypatch, make_router, mock):
    from foremankit.delegation import set_switch
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(LOCAL, "/api/tags", tags("llama3.2:3b"))
    mock.on(LOCAL, "/v1/chat/completions", chat_json("local answer"))
    router = make_router()
    set_switch(router.store, "paid", "off")
    res = router.delegate("x", kind="quick", sensitivity="normal")
    assert res["provider"] == "llm"
    assert "generativelanguage.googleapis.com" not in mock.hosts()


def test_delegate_with_everything_switched_off_tells_the_caller_to_do_the_work_itself(monkeypatch, mock, make_router):
    from foremankit import delegation
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    mock.on(DEEPSEEK, "/chat/completions", chat_json())
    router = make_router()
    delegation.set_switch(router.store, "all", "off")
    with pytest.raises(RouterError, match="DELEGATION IS SWITCHED OFF"):
        router.delegate("anything", kind="quick")
    assert not any(call.startswith(DEEPSEEK) for call in mock.calls)
    delegation.set_switch(router.store, "paid", "on")
    assert router.delegate("anything", kind="bulk_code")["provider"] == "deepseek"


def test_budget_categories_have_their_own_caps_and_counters():
    """A cap of one category does not block the other; the total still adds both."""
    from foremankit import budget
    state: dict = {}
    cfg = {"daily_usd": 1.5, "monthly_usd": 5.0, "orchestrator_daily_usd": 1.0}
    budget.record(state, 1_000_000.0, 1.2, budget.ORCHESTRATOR)
    assert budget.check(state, cfg, 1_000_000.0, 0.0, budget.ORCHESTRATOR) == (
        "orchestrator daily budget ($1.2000 spent + ~$0.0000 > $1.00)")
    assert budget.check(state, cfg, 1_000_000.0, 0.5) is None               # delegation counter is still empty
    budget.record(state, 1_000_000.0, 0.3)
    assert budget.spent(state, 1_000_000.0) == (1.5, 1.5)                  # total of both categories
    assert budget.spent(state, 1_000_000.0, budget.DELEGATION) == (0.3, 0.3)
    assert budget.caps({}, budget.ORCHESTRATOR) == (None, None)            # no cap until configured
