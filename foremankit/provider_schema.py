"""Transform an Anthropic Messages API request so a stricter third-party
Anthropic-compatible API accepts it.

This module is pure: every function receives its inputs as arguments and
returns new objects, never mutating the caller's data.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

# Default profile applied to every provider.
COMMON: dict[str, Any] = {
    "drop_request_fields": [
        "context_management",
        "container",
        "mcp_servers",
        "betas",
        "output_config",
    ],
    "drop_tool_fields": [
        "defer_loading",
        "eager_input_streaming",
        "cache_control",
    ],
    "drop_keywords": [],
    "describe_keywords": [
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "propertyNames",
    ],
    "strip_thinking_history": False,
}

# Provider-specific overrides applied on top of COMMON.
PROFILES: dict[str, dict[str, Any]] = {
    "deepseek": {"drop_keywords": ["$schema"]},
    "gemini": {
        "drop_keywords": ["$schema", "additionalProperties", "default", "examples"]
    },
}

_LOCAL_REF_PATTERN = re.compile(r"^#/(?:\$defs|definitions)/([^/]+)$")


@dataclass(frozen=True)
class _SchemaScope:
    """Shared inputs of one recursive sanitization pass.

    Groups the profile, the definitions available for ``$ref`` inlining and
    the refs currently being expanded (used to detect cycles).
    """

    profile: dict[str, Any]
    definition_views: dict[str, Any] | None
    expanded_refs: frozenset[str] = frozenset()


def profile_for(name: str, configured: dict | None = None) -> dict[str, Any]:
    """Return the merged sanitization profile for a provider.

    Layers, from lowest to highest priority: COMMON, PROFILES[name],
    `configured` (the upstream's ``schema:`` block). List values from a
    later layer replace earlier lists; scalar values replace earlier
    scalars. The input dictionaries are not modified.
    """
    merged = deepcopy(COMMON)
    provider_override = PROFILES.get(name, {})
    if provider_override:
        merged.update(deepcopy(provider_override))
    if configured:
        merged.update(deepcopy(configured))
    return merged


def sanitize_schema(schema: Any, profile: dict[str, Any]) -> Any:
    """Return a copy of a JSON schema made acceptable to the target API.

    The transformation walks all dicts and lists:

    * resolves local ``$ref`` values (``#/$defs/Name`` or
      ``#/definitions/Name``) by inlining the definition, then removes the
      ``$defs``/``definitions`` keys; cyclic refs collapse to ``{}``
    * converts ``const: value`` to ``enum: [value]``
    * collapses ``anyOf``/``oneOf`` of two choices where one is
      ``{"type": "null"}`` into the non-null choice
    * turns schema keywords listed in ``profile["describe_keywords"]`` into
      a ``"(keyword: value)"`` suffix on the schema ``description``
    * removes ``profile["drop_keywords"]``

    Property names inside ``properties`` are never treated as keywords.
    """
    scope = _SchemaScope(
        profile=profile, definition_views=_schema_definition_views(schema)
    )
    return _sanitize_schema_node(schema, scope)


def _schema_definition_views(root_schema: Any) -> dict[str, Any] | None:
    """Return a combined view of ``$defs`` and ``definitions`` if present.

    ``definitions`` is considered first so ``$defs`` takes precedence when
    both exist and a name is duplicated.
    """
    if not isinstance(root_schema, dict):
        return None
    views: dict[str, Any] = {}
    for key in ("definitions", "$defs"):
        definitions = root_schema.get(key)
        if isinstance(definitions, dict):
            views.update(definitions)
    return views


def _sanitize_schema_node(node: Any, scope: _SchemaScope) -> Any:
    """Recursively copy and sanitize one schema node."""
    if isinstance(node, list):
        return [_sanitize_schema_node(item, scope) for item in node]

    if not isinstance(node, dict):
        return deepcopy(node)

    return _sanitize_schema_dict(dict(node), scope)


def _sanitize_schema_dict(node: dict[str, Any], scope: _SchemaScope) -> dict[str, Any]:
    """Copy and sanitize a schema represented as a dictionary."""
    node = _resolve_inline_ref(node, scope)
    node = {key: value for key, value in node.items() if key not in ("$defs", "definitions")}
    node = _convert_const_to_enum(node)
    node = _collapse_null_union(node, scope)

    for key, value in list(node.items()):
        node = _sanitize_member(node, key, value, scope)
    return _apply_keyword_transforms(node, scope.profile)


def _sanitize_member(
    node: dict[str, Any], key: str, value: Any, scope: _SchemaScope
) -> dict[str, Any]:
    """Sanitize one keyword of a schema node and return the updated node."""
    profile = scope.profile
    if key == "properties" and isinstance(value, dict):
        node[key] = _sanitize_properties(value, scope)
    elif key not in profile["drop_keywords"] and key not in profile["describe_keywords"]:
        node[key] = _sanitize_schema_node(value, scope)
    elif key in profile["describe_keywords"]:
        node = _describe_keyword(node, key, value, profile)
    return node


def _sanitize_properties(
    properties: dict[str, Any], scope: _SchemaScope
) -> dict[str, Any]:
    """Sanitize each property schema while keeping property names as-is."""
    return {
        name: _sanitize_schema_node(schema, scope)
        for name, schema in properties.items()
    }


def _apply_keyword_transforms(
    node: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    """Remove dropped keywords and convert described keywords once."""
    for keyword in profile["drop_keywords"]:
        node.pop(keyword, None)
    return node


def _local_ref_name(
    node: dict[str, Any], definition_views: dict[str, Any] | None
) -> str | None:
    """Return the definition name a local ``$ref`` points to, if any."""
    ref_value = node.get("$ref")
    if not isinstance(ref_value, str) or definition_views is None:
        return None
    match = _LOCAL_REF_PATTERN.match(ref_value)
    return match.group(1) if match else None


def _resolve_inline_ref(node: dict[str, Any], scope: _SchemaScope) -> dict[str, Any]:
    """Inline a local ``$ref`` or leave the node unchanged."""
    definition_name = _local_ref_name(node, scope.definition_views)
    if definition_name is None:
        return node
    if definition_name in scope.expanded_refs:
        return {}

    definition = scope.definition_views.get(definition_name)
    if definition is None:
        return node

    inner_scope = replace(scope, expanded_refs=scope.expanded_refs | {definition_name})
    return _sanitize_schema_node(definition, inner_scope)


def _convert_const_to_enum(node: dict[str, Any]) -> dict[str, Any]:
    """Replace ``const`` with an ``enum`` containing the same value."""
    if "const" in node:
        node["enum"] = [node.pop("const")]
    return node


def _null_partner_index(union_value: Any) -> int | None:
    """Return the index of the non-null choice of a two-item union, if any."""
    if not isinstance(union_value, list) or len(union_value) != 2:
        return None
    null_indexes = [
        index
        for index, choice in enumerate(union_value)
        if isinstance(choice, dict) and choice.get("type") == "null"
    ]
    if not null_indexes:
        return None
    return 1 - null_indexes[0]


def _collapse_null_union(node: dict[str, Any], scope: _SchemaScope) -> dict[str, Any]:
    """Simplify a nullable ``anyOf``/``oneOf`` into its non-null choice."""
    for union_key in ("anyOf", "oneOf"):
        union_value = node.get(union_key)
        index = _null_partner_index(union_value)
        if index is None:
            continue

        non_null_choice = _sanitize_schema_node(union_value[index], scope)
        result = {key: value for key, value in node.items() if key != union_key}
        if isinstance(non_null_choice, dict):
            result.update(non_null_choice)
        return result

    return node


def _describe_keyword(
    node: dict[str, Any],
    keyword: str,
    value: Any,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Move a schema keyword into the description as ``(keyword: value)``."""
    rendered_value = value if isinstance(value, str) else json.dumps(value)
    suffix = f"({keyword}: {rendered_value})"
    existing_description = node.get("description")
    if isinstance(existing_description, str) and existing_description:
        node["description"] = f"{existing_description} {suffix}"
    else:
        node["description"] = suffix
    node.pop(keyword, None)
    return node


def sanitize_tool(tool: dict, profile: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a tool with provider-incompatible fields removed.

    Fields named in ``profile["drop_tool_fields"]`` are omitted. A custom
    tool's ``type`` marker is removed because the target API infers it.
    ``input_schema`` is sanitized with :func:`sanitize_schema`.
    """
    sanitized = {
        key: value
        for key, value in tool.items()
        if key not in profile["drop_tool_fields"]
    }

    if sanitized.get("type") == "custom":
        sanitized.pop("type", None)

    if isinstance(sanitized.get("input_schema"), dict):
        sanitized["input_schema"] = sanitize_schema(
            sanitized["input_schema"], profile
        )

    return sanitized


def sanitize_for_provider(data: dict, profile: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a request made acceptable to the target provider.

    Request-level fields in ``profile["drop_request_fields"]`` are removed,
    tools are passed through :func:`sanitize_tool`, and, when
    ``strip_thinking_history`` is true, thinking blocks are removed from
    assistant messages. Assistant messages left with empty content are
    dropped; all other values are copied unchanged.
    """
    sanitized = {
        key: value
        for key, value in data.items()
        if key not in profile["drop_request_fields"]
    }

    if "tools" in sanitized:
        sanitized["tools"] = [
            sanitize_tool(tool, profile) for tool in sanitized["tools"]
        ]

    if profile.get("strip_thinking_history"):
        sanitized["messages"] = _strip_thinking_history(sanitized.get("messages", []))

    return sanitized


def _strip_message(message: dict) -> dict | None:
    """Return a copy of a message without thinking blocks, or None if empty."""
    content = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, list):
        return deepcopy(message)

    kept_content = [
        deepcopy(block)
        for block in content
        if block.get("type") not in ("thinking", "redacted_thinking")
    ]
    if not kept_content:
        return None

    stripped_message = deepcopy(message)
    stripped_message["content"] = kept_content
    return stripped_message


def _strip_thinking_history(messages: list[dict]) -> list[dict]:
    """Remove thinking blocks from assistant messages and drop empties."""
    stripped_messages: list[dict] = []
    for message in messages:
        stripped = _strip_message(message)
        if stripped is not None:
            stripped_messages.append(stripped)
    return stripped_messages
