"""Regression guard for the real config/orchestrator.yaml model wiring.

The Claude Code picker keeps the Anthropic 5.5 models and adds DeepSeek (Fable slot)
and Gemini (custom entry); Kaggle is typed as fm-kaggle. DeepSeek and Gemini carry no static tier here:
the tier (auto / flash / pro; gemini also lite31) is chosen per provider in the
panel. Kaggle is a single model. Local llama.cpp is the `local_models` entry and
the default new-chat model. The 5.5 ids must stay unrouted so they keep going to
Anthropic.

This test reads the ACTUAL shipped config (not a fixture), so it fails the moment
the wiring drifts from the picker the plugin really shows.
"""
from __future__ import annotations

from pathlib import Path

from foremankit.config import _read_yaml

CONFIG = Path(__file__).resolve().parent.parent / "config" / "orchestrator.yaml"

# The old Claude ids the picker slots are wired to -> (upstream, tier). A None tier
# means the tier is dynamic (chosen per provider in the panel, not pinned here).
PROVIDER_ROUTES = {
    "fm-deepseek": ("deepseek", None),
    "fm-gemini": ("gemini", None),              # custom picker entry "Gemini"
    "fm-kaggle": ("kaggle", "main"),            # typed in /model
}

# "Local LLM" is the local llama.cpp entry: it must live in local_models, not model_routes.
LOCAL_ID = "claude-sonnet-5"

# The real 5.5 ids must stay unrouted so they keep going to Anthropic.
REAL_55_IDS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")

# Router ON: the picker shows Default plus these rows (3 Anthropic, 4 external) and hides the built-in lineup.
ANTHROPIC_ROWS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
EXTERNAL_ROWS = ("fm-deepseek", "fm-gemini", "fm-kaggle", "claude-sonnet-5")


def _config() -> dict:
    return _read_yaml(CONFIG)


def test_every_picker_slot_is_wired_to_a_provider():
    cfg = _config()
    routes = cfg.get("model_routes") or {}
    for model_id, (upstream_name, tier) in PROVIDER_ROUTES.items():
        route = routes.get(model_id)
        assert route is not None, f"{model_id} is missing from model_routes (it would fall through to the cloud)"
        assert route.get("upstream") == upstream_name, f"{model_id} should go to {upstream_name}, got {route}"
        if tier is None:
            assert route.get("tier") is None, f"{model_id} must not pin a static tier (it is chosen in the panel)"
        else:
            assert route.get("tier") == tier, f"{model_id} should pin tier {tier}, got {route}"


def test_local_llama_is_the_default_entry_not_a_route():
    cfg = _config()
    routes = cfg.get("model_routes") or {}
    local_models = cfg.get("local_models") or []
    assert LOCAL_ID in local_models, f"{LOCAL_ID} must be the local llama.cpp entry in local_models"
    assert LOCAL_ID not in routes, f"{LOCAL_ID} must not also be a model_route (it is the local entry)"


def test_real_55_ids_are_not_rewired():
    cfg = _config()
    routes = cfg.get("model_routes") or {}
    local_models = cfg.get("local_models") or []
    for model_id in REAL_55_IDS:
        assert model_id not in routes, f"{model_id} must not be a model_route (it should stay on Anthropic)"
        assert model_id not in local_models, f"{model_id} must not be in local_models"


def test_model_picker_lists_three_anthropic_and_four_external_rows():
    spec = _config().get("model_picker") or {}
    assert spec.get("replace_built_in") is True
    assert tuple(row["model"] for row in spec["options"]) == ANTHROPIC_ROWS + EXTERNAL_ROWS
    assert all(row.get("label") for row in spec["options"])


def test_old_slot_overrides_are_gone():
    assert not _config().get("picker")


def test_legacy_ids_do_not_silently_route_externally():
    routes = _config().get("model_routes") or {}
    for model_id in ("claude-sonnet-4-6", "claude-fable-5", "claude-opus-5"):
        assert model_id not in routes
