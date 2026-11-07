"""Which worker should get one task: the lead, a Haiku subagent, opencode, agy, a split or a queue.
Pure heuristics, no I/O: the hard gates come first, then readiness, size, the small-edit case and sensitivity.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from foremankit.agent_scope import matches_any

MAX_FILES = 8                   # above this the task is split (the plan: more than 8 files)
MAX_CHANGED_LINES = 400         # above this the task is split (~400 changed lines)
MAX_TASK_CHARS = 22000          # above this the task text is split
AGY_MIN_FILES = 6               # agy gets 6-8 files (opencode stays on 1-5)
AGY_MIN_CONTEXT_LINES = 1500    # or this much context
AGY_MIN_TASK_CHARS = 8000       # or a task text above 8k chars
AGY_AFTER_FAILURES = 2          # opencode, opencode, agy: the third try escalates


@dataclass(frozen=True)
class RunnerFacts:
    """What the choice looks at: the task file, its size, its scope, its check and the runner state."""
    task_text: str = ""
    files: int = 1
    changed_lines: int = 0
    context_lines: int = 0
    sensitivity: str = "normal"
    scope: tuple[str, ...] = ()
    check: str = ""
    needs_session_context: bool = False
    touches_shared_interface: bool = False
    edits_existing_only: bool = False
    contract_cited: bool = True
    edge_cases_listed: bool = True
    opencode_failures: int = 0
    agy_enabled: bool = True
    opencode_enabled: bool = True
    codex_enabled: bool = True  # explicit plan pins only; never an automatic fallback


def decide_runner(facts: RunnerFacts, review_globs: Sequence[str] = ()) -> tuple[str, list[str]]:
    """(choice, reasons): the first matching rule decides, no rule means a fail-closed lead."""
    result = _gate(facts, review_globs) or _readiness(facts)
    return result or _by_sensitivity(facts)


def _gate(facts: RunnerFacts, review_globs: Sequence[str]) -> tuple[str, list[str]] | None:
    """The hard gates: private data, no check, live session context, a mandatory independent review."""
    if facts.sensitivity == "private":
        return "lead", ["private data never reaches a runner"]
    if not facts.check.strip():
        return "lead", ["no mechanical check command for agent-run"]
    if facts.needs_session_context:
        return "lead", ["the task needs the live context of the session"]
    review = _review_reason(facts, review_globs)
    return ("lead", [review]) if review else None


def _review_reason(facts: RunnerFacts, review_globs: Sequence[str]) -> str:
    """Why an independent review is mandatory (empty string = no review gate)."""
    if facts.touches_shared_interface:
        return "a shared interface changes: an independent review is mandatory"
    if any(matches_any(pattern, review_globs) for pattern in facts.scope):
        return "the scope matches a review path: an independent review is mandatory"
    return ""


def _readiness(facts: RunnerFacts) -> tuple[str, list[str]] | None:
    """The task-file rules: revise an unready file, split a big one, give a small edit to Haiku."""
    if not facts.contract_cited and not facts.edge_cases_listed:
        return "revise", ["the task file is not ready: cite the real source files and list the edge cases"]
    if facts.files > MAX_FILES or facts.changed_lines > MAX_CHANGED_LINES or len(facts.task_text) > MAX_TASK_CHARS:
        return "split", [_split_reason(facts)]
    if facts.edits_existing_only and facts.files <= 2 and facts.changed_lines <= 40:
        return "haiku", ["a small edit to existing code"]
    return None


def _split_reason(facts: RunnerFacts) -> str:
    """Which size limit was crossed."""
    if facts.files > MAX_FILES:
        return f"too many files ({facts.files}): split the task"
    if facts.changed_lines > MAX_CHANGED_LINES:
        return f"too many changed lines ({facts.changed_lines}): split the task"
    return f"the task text is too long ({len(facts.task_text)} chars): split the task"


def _by_sensitivity(facts: RunnerFacts) -> tuple[str, list[str]]:
    """Pick a runner by the data label; any unknown label fails closed to the lead."""
    if facts.sensitivity == "internal":
        return _internal(facts)
    if facts.sensitivity == "normal":
        return _normal(facts)
    return "lead", [f"unknown sensitivity {facts.sensitivity!r}: fail closed"]


def _internal(facts: RunnerFacts) -> tuple[str, list[str]]:
    """Internal data goes to agy only; without agy the task waits in the queue."""
    if facts.agy_enabled:
        return "agy", ["internal data may only go to agy"]
    return "queue", ["internal data may never go to opencode: the task waits for agy"]


def _normal(facts: RunnerFacts) -> tuple[str, list[str]]:
    """Normal data: agy for the big or escalated tasks, opencode as the default."""
    if _prefers_agy(facts):
        return "agy", [_agy_reason(facts)]
    if facts.opencode_enabled:
        return "opencode", ["normal data and a medium task: opencode is the default"]
    if facts.agy_enabled:
        return "agy", ["opencode is switched off: agy takes the task"]
    return "lead", ["no runner is enabled"]


def _prefers_agy(facts: RunnerFacts) -> bool:
    """A big task, a lot of context, a long text or two opencode failures go to agy."""
    return facts.agy_enabled and (facts.files >= AGY_MIN_FILES or facts.context_lines >= AGY_MIN_CONTEXT_LINES
                                  or len(facts.task_text) > AGY_MIN_TASK_CHARS
                                  or facts.opencode_failures >= AGY_AFTER_FAILURES)


def _agy_reason(facts: RunnerFacts) -> str:
    """Which agy signal fired (only called when agy is enabled)."""
    if facts.opencode_failures >= AGY_AFTER_FAILURES:
        return "opencode failed this task twice: escalate to agy"
    if facts.files >= AGY_MIN_FILES:
        return f"a big task ({facts.files} files): agy handles the context"
    if facts.context_lines >= AGY_MIN_CONTEXT_LINES:
        return "a lot of context: agy handles it"
    return "a long task text: agy handles it"
