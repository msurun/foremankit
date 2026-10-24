"""Detect model ids in providers.yaml that a provider no longer offers.

Providers rename or retire models, so a configured id silently rots. This module
finds those stale ids purely from the ids a provider lists today and proposes the
closest surviving replacement, plus a surgical text edit to rewrite one id while
preserving every comment. No I/O and no network: callers pass strings and dicts.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Characters that separate id tokens: "gemini-2.5-flash" -> ["gemini", "2", "5", "flash"].
TOKEN_SEPARATOR_PATTERN = re.compile(r"[-._]")

# Tokens that name a product tier or a release channel rather than a version number.
TIER_TOKENS = frozenset({
    "pro", "flash", "mini", "lite", "ultra", "nano",
    "haiku", "sonnet", "opus", "preview", "exp",
})

# Channel tokens are hints only: a successor may drop "preview" or "exp".
OPTIONAL_TIER_TOKENS = frozenset({"preview", "exp"})

# Smaller-family markers: a candidate carrying one the old id lacks is a different product.
SMALLER_TIER_TOKENS = frozenset({"mini", "lite", "nano", "ultra"})

# Penalty subtracted from the score for each token the candidate has beyond the old id.
EXTRA_TOKEN_PENALTY = 0.1

MAX_CANDIDATES = 3

# Indentation (number of leading spaces) of each nesting level in providers.yaml.
PROVIDERS_INDENT = 0
PROVIDER_INDENT = 2
MODELS_INDENT = 4
ALIAS_INDENT = 6
ID_INDENT = 8

# Quotes an id value may be wrapped in, longest first so triple quotes win when resolving.
QUOTE_PAIRS = (('"""', '"""'), ("'''", "'''"), ('"', '"'), ("'", "'"))

# Splits an "id: value  # comment" line into (prefix, id-key, value, gap, comment).
VERSION_TOKEN = re.compile(r"^v?[0-9]+$")
VERSION_BONUS = 0.5       # a candidate that keeps a version slot beats one that dropped it
ID_LINE_PATTERN = re.compile(
    r"^(?P<prefix>[ \t]*id:[ \t]+)"
    r"(?P<value>.*?)"
    r"(?P<gap>[ \t]*)"
    r"(?P<comment>#[^\r\n]*)?$"
)


def tokenize_model_id(model_id: str) -> list[str]:
    """Split a model id into lowercase tokens on '-', '.' and '_'."""
    return [token for token in TOKEN_SEPARATOR_PATTERN.split(model_id.lower()) if token]


def _required_tier_tokens(model_id: str) -> set[str]:
    """Return the tier tokens a successor must also carry (channel tokens excluded)."""
    tokens = set(tokenize_model_id(model_id))
    return (tokens & TIER_TOKENS) - OPTIONAL_TIER_TOKENS


def _is_same_older_family(old_tokens: set[str], candidate_tokens: set[str]) -> bool:
    """Return True if the candidate is a smaller-product variant the old id is not."""
    foreign_smaller_tokens = (candidate_tokens & SMALLER_TIER_TOKENS) - old_tokens
    return bool(foreign_smaller_tokens)


def _is_successor(old_id: str, candidate_id: str) -> bool:
    """Decide whether `candidate_id` plausibly replaces `old_id`.

    Rules: the candidate differs, shares the first (family) token, contains every
    required tier token of the old id, and is not a smaller-product variant.
    """
    old_tokens = tokenize_model_id(old_id)
    candidate_tokens = tokenize_model_id(candidate_id)
    if candidate_id == old_id or not old_tokens or not candidate_tokens:
        return False
    if candidate_tokens[0] != old_tokens[0]:
        return False
    if not _required_tier_tokens(old_id) <= set(candidate_tokens):
        return False
    return not _is_same_older_family(set(old_tokens), set(candidate_tokens))


def _successor_score(old_id: str, candidate_id: str) -> float:
    """Score a candidate: shared tokens minus a penalty per extra token."""
    old_tokens = set(tokenize_model_id(old_id))
    candidate_tokens = set(tokenize_model_id(candidate_id))
    shared_count = len(old_tokens & candidate_tokens)
    extra_count = len(candidate_tokens - old_tokens)
    bonus = VERSION_BONUS if _both_versioned(old_tokens, candidate_tokens) else 0.0
    return shared_count - EXTRA_TOKEN_PENALTY * extra_count + bonus


def _both_versioned(old_tokens: set[str], candidate_tokens: set[str]) -> bool:
    """True when both ids carry a version token (v4, 3...): a successor replaces the version, it does not drop it."""
    return (any(VERSION_TOKEN.match(token) for token in old_tokens)
            and any(VERSION_TOKEN.match(token) for token in candidate_tokens))


def find_candidates(old_id: str, available: list[str]) -> list[str]:
    """Return up to 3 offered ids that look like a successor of `old_id`, best first.

    Ties are broken by id so the result is deterministic.
    """
    successors = [candidate for candidate in available if _is_successor(old_id, candidate)]
    ranked = sorted(
        successors,
        key=lambda candidate: (-_successor_score(old_id, candidate), candidate),
    )
    return ranked[:MAX_CANDIDATES]


def stale_models(providers: dict, offered: dict[str, list[str]]) -> list[dict]:
    """List configured model ids a provider no longer offers, with candidates.

    `providers` is the parsed providers.yaml 'providers' mapping; `offered` maps a
    provider name to the ids it lists now. A provider missing from `offered` or
    listing nothing is unknown rather than stale, and non-llm providers are skipped.
    Returns dicts with keys provider, alias, id, candidates, sorted by provider then alias.
    """
    stale_entries: list[dict] = []
    for provider_name, provider_config in providers.items():
        if not _is_watched_provider(provider_name, provider_config, offered):
            continue
        stale_entries.extend(_stale_entries_for_provider(provider_name, provider_config, offered))
    return sorted(stale_entries, key=lambda entry: (entry["provider"], entry["alias"]))


def _is_watched_provider(provider_name: str, provider_config: dict, offered: dict) -> bool:
    """Return True if the provider is an llm provider with a non-empty offered list."""
    if provider_config.get("type") == "llm":
        return False
    offered_ids = offered.get(provider_name)
    return bool(offered_ids)


def _stale_entries_for_provider(
    provider_name: str, provider_config: dict, offered: dict[str, list[str]]
) -> list[dict]:
    """Return one stale entry per configured alias whose id is not offered."""
    offered_ids = offered[provider_name]
    offered_set = set(offered_ids)
    configured_models = provider_config.get("models") or {}
    entries: list[dict] = []
    for alias, model_config in configured_models.items():
        old_id = (model_config or {}).get("id")
        if old_id is None or old_id in offered_set:
            continue
        entries.append({
            "provider": provider_name,
            "alias": alias,
            "id": old_id,
            "candidates": find_candidates(old_id, offered_ids),
        })
    return entries


def rewrite_model_id(yaml_text: str, provider: str, alias: str, new_id: str) -> str:
    """Return `yaml_text` with only providers.<provider>.models.<alias>.id set to `new_id`.

    Edits the single id line in place so comments, spacing, quote style, line endings
    and the final newline state stay byte-identical. Do not parse-and-dump: PyYAML
    would drop comments.
    Raises KeyError(f"{provider}.{alias}") when that location is not found.
    """
    lines = yaml_text.splitlines(keepends=True)
    id_line_index = _find_id_line_index(lines, provider, alias)
    if id_line_index is None:
        raise KeyError(f"{provider}.{alias}")
    lines[id_line_index] = _replace_id_value(lines[id_line_index], new_id)
    return "".join(lines)


def _leading_spaces(line: str) -> int:
    """Return the number of leading space characters of a line."""
    return len(line) - len(line.lstrip(" "))


def _strip_line_ending(line: str) -> str:
    """Return the line without its trailing \\r\\n or \\n."""
    return line.rstrip("\r\n")


def _line_ending_of(line: str) -> str:
    """Return the trailing \\r\\n or \\n of a line, or the empty string if none."""
    stripped = _strip_line_ending(line)
    return line[len(stripped):]


def _is_blank_or_comment(line: str) -> bool:
    """Return True for empty, whitespace-only or comment lines."""
    stripped = _strip_line_ending(line).strip()
    return stripped == "" or stripped.startswith("#")


def _is_key_line(line: str, indent: int, key: str) -> bool:
    """Return True if `line` is `<indent spaces><key>:` (value may follow, e.g. 'id: x')."""
    text = _strip_line_ending(line)
    if _leading_spaces(text) != indent:
        return False
    body = text[indent:]
    return body == f"{key}:" or body.startswith(f"{key}: ")


def _find_id_line_index(lines: list[str], provider: str, alias: str) -> int | None:
    """Locate the id line under providers.<provider>.models.<alias>, or None.

    Each level is searched inside its parent block only: the scan stops at the first non-blank, non-comment line
    indented at or below the parent's own indent (a sibling or an outer key)."""
    levels = ((PROVIDERS_INDENT, "providers", None), (PROVIDER_INDENT, provider, PROVIDERS_INDENT),
              (MODELS_INDENT, "models", MODELS_INDENT), (ALIAS_INDENT, alias, ALIAS_INDENT),
              (ID_INDENT, "id", ALIAS_INDENT))
    index = -1
    for indent, key, stop in levels:
        index = _first_index(lines, index + 1, lambda line, i=indent, k=key: _is_key_line(line, i, k), stop_indent=stop)
        if index is None:
            return None
    return index


def _first_index(lines: list[str], start: int, matches: Callable[[str], bool], stop_indent: int | None = None) -> int | None:
    """Return the first index >= start whose line matches, or None.

    When `stop_indent` is set, scanning stops at the first non-blank, non-comment line
    indented at or below it (that line has left the parent block).
    """
    for index in range(start, len(lines)):
        line = lines[index]
        if matches(line):
            return index
        if stop_indent is not None and _leaves_block(line, stop_indent):
            return None
    return None


def _leaves_block(line: str, stop_indent: int) -> bool:
    """Return True if `line` ends the block that lives deeper than `stop_indent`."""
    if _is_blank_or_comment(line):
        return False
    return _leading_spaces(_strip_line_ending(line)) <= stop_indent


def _split_quotes(value: str) -> tuple[str, str, str] | None:
    """Split a quoted value into (open_quote, inner, close_quote), or None if unquoted.

    A quote pair needs both characters to exist at distinct positions, so a single
    stray quote character in an unquoted value is not mistaken for a quoted value.
    """
    for open_quote, close_quote in QUOTE_PAIRS:
        if len(value) < len(open_quote) + len(close_quote):
            continue
        if value.startswith(open_quote) and value.endswith(close_quote):
            inner_start = len(open_quote)
            inner_end = len(value) - len(close_quote)
            if inner_end > inner_start:
                return open_quote, value[inner_start:inner_end], close_quote
    return None


def _replace_id_value(line: str, new_id: str) -> str:
    """Rewrite the value of an id line, preserving indentation, spacing, comment, quotes and the line's own ending
    (none on a last line without a newline, so the file's final-newline state is unchanged)."""
    match = ID_LINE_PATTERN.match(_strip_line_ending(line))
    if match is None:
        raise ValueError(f"Not an id line: {line!r}")
    original_value = match.group("value")
    quoted = _split_quotes(original_value)
    # Keep the original quote style so a rewrite stays diff-minimal; fall back to the
    # replacement id's own quoting when the old value was unquoted.
    replacement = f"{quoted[0]}{new_id}{quoted[2]}" if quoted else new_id
    comment = match.group("comment") or ""
    gap = match.group("gap") if comment else ""
    ending = _line_ending_of(line)
    return f"{match.group('prefix')}{replacement}{gap}{comment}{ending}"
