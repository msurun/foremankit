"""Tests for the provider-agnostic schema sanitizer.

These lock down the transformations that let one tool schema be sent to
providers with different JSON Schema dialects (Gemini, DeepSeek, Anthropic)
without losing information: unsupported keywords become description text,
``$ref``s are inlined, and provider-only request fields are stripped.
"""

import copy

import pytest

from foremankit import provider_schema as ps


# --------------------------------------------------------------------------- #
# profile_for
# --------------------------------------------------------------------------- #


def test_profile_for_merges_common_and_named_profile():
    merged = ps.profile_for("gemini", {})

    assert "additionalProperties" in merged["drop_keywords"]
    assert "context_management" in merged["drop_request_fields"]


def test_configured_list_replaces_profile_list_instead_of_extending():
    configured = {"drop_keywords": ["title"]}

    merged = ps.profile_for("gemini", configured)

    assert merged["drop_keywords"] == ["title"]


def test_profile_for_returns_a_new_dict_that_does_not_touch_the_source():
    configured = {"drop_keywords": ["title"]}

    merged = ps.profile_for("gemini", configured)
    merged["drop_keywords"].append("deprecated")

    assert configured["drop_keywords"] == ["title"]


# --------------------------------------------------------------------------- #
# sanitize_schema: the real failing case
# --------------------------------------------------------------------------- #


def test_pattern_on_array_items_moves_into_description_and_disappears_as_key():
    schema = {
        "type": "object",
        "properties": {
            "file_paths": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[^\\0]*$"},
            }
        },
    }
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_schema(schema, profile)

    items = clean["properties"]["file_paths"]["items"]
    assert "pattern" not in items
    assert "(pattern: ^[^\\0]*$)" in items["description"]


def test_sanitize_schema_does_not_mutate_its_input():
    schema = {
        "type": "object",
        "properties": {"file_paths": {"type": "string", "pattern": "^x$"}},
    }
    original = copy.deepcopy(schema)
    profile = ps.profile_for("deepseek", {})

    ps.sanitize_schema(schema, profile)

    assert schema == original


def test_a_property_named_pattern_is_kept_as_a_property_name():
    schema = {"type": "object", "properties": {"pattern": {"type": "string"}}}
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_schema(schema, profile)

    assert "pattern" in clean["properties"]
    assert clean["properties"]["pattern"]["type"] == "string"


def test_a_property_named_format_is_kept_as_a_property_name():
    schema = {"type": "object", "properties": {"format": {"type": "string"}}}
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_schema(schema, profile)

    assert "format" in clean["properties"]


# --------------------------------------------------------------------------- #
# sanitize_schema: refs, const, nullable
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prefix", ["#/$defs", "#/definitions"])
def test_refs_are_inlined_and_defs_removed(prefix):
    schema = {
        "type": "object",
        "properties": {"item": {"$ref": f"{prefix}/Node"}},
        "$defs": {"Node": {"type": "string"}},
        "definitions": {"Node": {"type": "string"}},
    }
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_schema(schema, profile)

    assert clean["properties"]["item"] == {"type": "string"}
    assert "$defs" not in clean
    assert "definitions" not in clean


def test_a_ref_cycle_collapses_to_an_empty_schema():
    schema = {
        "$defs": {"Node": {"$ref": "#/$defs/Node"}},
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_schema(schema, profile)

    assert clean["properties"]["root"] == {}


def test_const_becomes_a_single_value_enum():
    schema = {"const": "fixed"}
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_schema(schema, profile)

    assert clean == {"enum": ["fixed"]}


def test_nullable_anyof_collapses_to_the_non_null_schema():
    schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_schema(schema, profile)

    assert clean == {"type": "string"}


def test_nullable_oneof_collapses_to_the_non_null_schema():
    schema = {"oneOf": [{"type": "null"}, {"type": "integer"}]}
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_schema(schema, profile)

    assert clean == {"type": "integer"}


# --------------------------------------------------------------------------- #
# sanitize_schema: profile differences
# --------------------------------------------------------------------------- #


def test_gemini_drops_additional_properties():
    schema = {"type": "object", "additionalProperties": False}
    profile = ps.profile_for("gemini", {})

    clean = ps.sanitize_schema(schema, profile)

    assert "additionalProperties" not in clean


def test_deepseek_keeps_additional_properties():
    schema = {"type": "object", "additionalProperties": False}
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_schema(schema, profile)

    assert clean["additionalProperties"] is False


def test_gemini_drops_format_into_description():
    schema = {"type": "integer", "format": "int64"}
    profile = ps.profile_for("gemini", {})

    clean = ps.sanitize_schema(schema, profile)

    assert "format" not in clean
    assert "(format: int64)" in clean["description"]


def test_unrelated_schema_content_is_left_untouched():
    schema = {"type": "object", "title": "Thing", "required": ["a"]}
    profile = ps.profile_for("gemini", {})

    clean = ps.sanitize_schema(schema, profile)

    assert clean["type"] == "object"
    assert clean["title"] == "Thing"
    assert clean["required"] == ["a"]


# --------------------------------------------------------------------------- #
# sanitize_tool
# --------------------------------------------------------------------------- #


def test_sanitize_tool_drops_provider_only_tool_fields():
    tool = {
        "name": "read",
        "defer_loading": True,
        "eager_input_streaming": True,
        "cache_control": {"type": "ephemeral"},
    }
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_tool(tool, profile)

    assert "defer_loading" not in clean
    assert "eager_input_streaming" not in clean
    assert "cache_control" not in clean


def test_sanitize_tool_drops_the_custom_type():
    tool = {"name": "read", "type": "custom"}
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_tool(tool, profile)

    assert "type" not in clean


def test_sanitize_tool_sanitizes_the_input_schema():
    tool = {
        "name": "read",
        "input_schema": {"type": "string", "pattern": "^x$"},
    }
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_tool(tool, profile)

    assert "pattern" not in clean["input_schema"]


# --------------------------------------------------------------------------- #
# sanitize_for_provider
# --------------------------------------------------------------------------- #


def test_request_only_fields_are_dropped():
    request = {
        "model": "m",
        "context_management": {},
        "container": {},
        "mcp_servers": [],
        "betas": ["b"],
        "output_config": {},
    }
    profile = ps.profile_for("deepseek", {})

    clean = ps.sanitize_for_provider(request, profile)

    for field in ("context_management", "container", "mcp_servers", "betas", "output_config"):
        assert field not in clean
    assert clean["model"] == "m"


def test_thinking_blocks_are_stripped_from_assistant_messages():
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "redacted_thinking", "data": "x"},
                    {"type": "text", "text": "answer"},
                ],
            }
        ]
    }
    profile = ps.profile_for("deepseek", {"strip_thinking_history": True})

    clean = ps.sanitize_for_provider(request, profile)

    assert clean["messages"][0]["content"] == [{"type": "text", "text": "answer"}]


def test_an_assistant_message_emptied_of_thinking_is_dropped():
    request = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}]},
        ]
    }
    profile = ps.profile_for("deepseek", {"strip_thinking_history": True})

    clean = ps.sanitize_for_provider(request, profile)

    assert len(clean["messages"]) == 1
    assert clean["messages"][0]["role"] == "user"
