"""Summarise Claude Code session transcripts for the `foremankit sessions` report.

A transcript is a JSONL file in ~/.claude/projects/<project>/<session-id>.jsonl with
one JSON object per line. Lines can be malformed or incomplete, so parsing never
crashes: unreadable lines are skipped. We count the token usage of each API request
once, detect delegations and reads of delegated results, and expose the totals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

# Folder name used by Claude Code for subagent transcripts; those are not sessions.
SUBAGENT_FOLDER_NAME = "subagents"

# Marker that identifies a read of a delegated result on disk.
AGENT_OUT_MARKER = ".agent-out"

# Tool names that count as reading a delegated result.
AGENT_OUT_READ_TOOL_NAMES = frozenset({"Read", "Grep", "Glob", "Bash"})

# Tool names used to launch a subagent.
AGENT_TOOL_NAMES = frozenset({"Agent", "Task"})

# Suffix of a delegation MCP tool (e.g. mcp__foremankit__delegate).
DELEGATE_TOOL_SUFFIX = "__delegate"

# Substrings that mark a Bash command that runs the ForemanKit delegate command.
DELEGATE_COMMAND_MARKERS = ("foremankit delegate", "foremankit.cli delegate")


@dataclass
class SessionSummary:
    """Aggregated token usage and delegation activity of one session transcript."""

    session_id: str
    path: str
    started: str
    model: str
    requests: int
    input_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    output_tokens: int
    delegations: int
    agent_calls: int
    agent_out_reads: int

    def total_tokens(self) -> int:
        """Return the billed token total: input + cache creation + output.

        Cache reads are excluded because they are reported separately at a much
        lower price.

        Returns:
            The sum of the input, cache creation and output token counts.
        """
        return self.input_tokens + self.cache_creation_tokens + self.output_tokens

    def delegations_per_100_requests(self) -> float:
        """Return how many delegations happened per 100 API requests.

        Returns:
            The delegation rate, or 0.0 when the session made no requests.
        """
        if self.requests == 0:
            return 0.0
        return self.delegations * 100 / self.requests


@dataclass
class _UsageCounters:
    """Mutable tally of tokens and work for one transcript, built line by line.

    The token fields mirror SessionSummary. `requests_seen` stores the ids of the
    requests already counted, so a request spread over several lines is counted
    once; a line without a requestId is counted immediately instead.
    """

    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    delegations: int = 0
    agent_calls: int = 0
    agent_out_reads: int = 0
    started: str = ""
    model_counts: dict[str, int] = field(default_factory=dict)
    last_line_per_request: dict[str, dict[str, Any]] = field(default_factory=dict)


def summarize_transcript(path: Path) -> SessionSummary:
    """Read one transcript file and return its aggregated summary.

    The file is streamed line by line because sessions can be many megabytes.

    Args:
        path: The JSONL transcript file to read.

    Returns:
        A SessionSummary filled from the file contents. Unparseable lines and
        lines without the expected fields are ignored.

    Raises:
        OSError: If the file cannot be read.
    """
    counters = _UsageCounters()
    with path.open("r", encoding="utf-8") as transcript_file:
        for raw_line in transcript_file:
            _process_line(raw_line, counters)
    _apply_last_usage_per_request(counters)
    return _build_summary(path, counters)


def find_transcripts(days: int | None, root: Path | None = None) -> list[Path]:
    """Find transcript files modified within the last `days` days.

    Args:
        days: Only return files modified in the last this many days. None means
            no time filter.
        root: Folder to search. Defaults to ~/.claude/projects.

    Returns:
        The matching *.jsonl paths, newest first. Empty list when root is absent.
    """
    search_root = root if root is not None else _default_projects_root()
    if not search_root.is_dir():
        return []

    cutoff_timestamp = _cutoff_timestamp(days)
    found = [
        candidate
        for candidate in search_root.rglob("*.jsonl")
        if _should_include_transcript(candidate, cutoff_timestamp)
    ]
    return sorted(found, key=_modification_time, reverse=True)


def summarize_many(paths: list[Path]) -> list[SessionSummary]:
    """Summarise several transcript files, skipping the unreadable ones.

    Args:
        paths: Transcript files to summarise.

    Returns:
        One SessionSummary per readable file, in the given order.
    """
    summaries: list[SessionSummary] = []
    for path in paths:
        try:
            summaries.append(summarize_transcript(path))
        except OSError:
            continue
    return summaries


def _default_projects_root() -> Path:
    """Return the folder where Claude Code stores project transcripts."""
    return Path.home() / ".claude" / "projects"


def _cutoff_timestamp(days: int | None) -> datetime | None:
    """Return the oldest modification time to keep, or None when unfiltered."""
    if days is None:
        return None
    return datetime.now(timezone.utc) - timedelta(days=days)


def _should_include_transcript(candidate: Path, cutoff_timestamp: datetime | None) -> bool:
    """Say whether one candidate path passes the folder and time filters."""
    if SUBAGENT_FOLDER_NAME in candidate.parts:
        return False
    if cutoff_timestamp is None:
        return True
    return _modification_time(candidate) >= cutoff_timestamp


def _modification_time(path: Path) -> datetime:
    """Return the file modification time as a timezone-aware datetime."""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _process_line(raw_line: str, counters: _UsageCounters) -> None:
    """Fold one transcript line into the running totals, ignoring bad lines."""
    entry = _parse_line(raw_line)
    if entry is None:
        return
    _record_started_timestamp(entry, counters)
    message = entry.get("message")
    if isinstance(message, dict):
        _record_model(message, counters)
        _record_tool_uses(message, counters)
    _record_usage(entry, counters)


def _parse_line(raw_line: str) -> dict[str, Any] | None:
    """Decode one JSON line, returning None when it is blank or malformed."""
    stripped = raw_line.strip()
    if not stripped:
        return None
    try:
        entry = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(entry, dict):
        return None
    return entry


def _record_started_timestamp(entry: dict[str, Any], counters: _UsageCounters) -> None:
    """Keep the first timestamp seen as the session start time."""
    if counters.started:
        return
    timestamp = entry.get("timestamp")
    if isinstance(timestamp, str):
        counters.started = timestamp


def _record_model(message: dict[str, Any], counters: _UsageCounters) -> None:
    """Count one assistant model occurrence to find the most used one later."""
    model = message.get("model")
    if isinstance(model, str) and model:
        counters.model_counts[model] = counters.model_counts.get(model, 0) + 1


def _record_tool_uses(message: dict[str, Any], counters: _UsageCounters) -> None:
    """Inspect the message content blocks and tally delegations and reads."""
    content_blocks = message.get("content")
    if not isinstance(content_blocks, list):
        return
    for block in content_blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        tool_name = block.get("name")
        tool_input = block.get("input")
        if _is_delegation(tool_name, tool_input):
            counters.delegations += 1
        elif tool_name in AGENT_TOOL_NAMES:
            counters.agent_calls += 1
        if _is_agent_out_read(tool_name, tool_input):
            counters.agent_out_reads += 1


def _is_delegation(tool_name: Any, tool_input: Any) -> bool:
    """Say whether a tool_use block represents a delegation to another agent."""
    if isinstance(tool_name, str) and tool_name.endswith(DELEGATE_TOOL_SUFFIX):
        return True
    if tool_name != "Bash" or not isinstance(tool_input, dict):
        return False
    command = tool_input.get("command")
    if not isinstance(command, str):
        return False
    return any(marker in command for marker in DELEGATE_COMMAND_MARKERS)


def _is_agent_out_read(tool_name: Any, tool_input: Any) -> bool:
    """Say whether a tool_use block reads a delegated result file."""
    if tool_name not in AGENT_OUT_READ_TOOL_NAMES:
        return False
    if not isinstance(tool_input, dict):
        return False
    return AGENT_OUT_MARKER in json.dumps(tool_input)


def _record_usage(entry: dict[str, Any], counters: _UsageCounters) -> None:
    """Add the token usage of one line, counting each request only once.

    A request can span several lines (one per content block) that repeat the same
    usage, so lines sharing a requestId are buffered and only the last one is
    counted. Lines without a requestId are counted immediately.
    """
    message = entry.get("message")
    usage = message.get("usage") if isinstance(message, dict) else None
    if not isinstance(usage, dict):
        return
    request_id = entry.get("requestId")
    if isinstance(request_id, str) and request_id:
        counters.last_line_per_request[request_id] = usage
    else:
        _add_usage(usage, counters)


def _add_usage(usage: dict[str, Any], counters: _UsageCounters) -> None:
    """Add one usage record's token counts to the totals and count the request."""
    counters.input_tokens += _usage_field(usage, "input_tokens")
    counters.cache_creation_tokens += _usage_field(usage, "cache_creation_input_tokens")
    counters.cache_read_tokens += _usage_field(usage, "cache_read_input_tokens")
    counters.output_tokens += _usage_field(usage, "output_tokens")
    counters.requests += 1


def _usage_field(usage: dict[str, Any], field_name: str) -> int:
    """Return a token count from a usage record, or 0 when missing or invalid."""
    value = usage.get(field_name)
    if isinstance(value, int):
        return value
    return 0


def _apply_last_usage_per_request(counters: _UsageCounters) -> None:
    """Count the buffered usage of every requestId exactly once."""
    for usage in counters.last_line_per_request.values():
        _add_usage(usage, counters)


def _most_used_model(model_counts: dict[str, int]) -> str:
    """Return the model seen most often, ties broken by alphabetical order."""
    if not model_counts:
        return ""
    return max(sorted(model_counts), key=lambda model: model_counts[model])


def _build_summary(path: Path, counters: _UsageCounters) -> SessionSummary:
    """Turn the accumulated counters of one transcript into a SessionSummary."""
    return SessionSummary(
        session_id=path.stem,
        path=str(path),
        started=counters.started,
        model=_most_used_model(counters.model_counts),
        requests=counters.requests,
        input_tokens=counters.input_tokens,
        cache_creation_tokens=counters.cache_creation_tokens,
        cache_read_tokens=counters.cache_read_tokens,
        output_tokens=counters.output_tokens,
        delegations=counters.delegations,
        agent_calls=counters.agent_calls,
        agent_out_reads=counters.agent_out_reads,
    )
