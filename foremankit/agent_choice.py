"""Which worker should get one task: the lead, Haiku, opencode, codex, agy, a split or a queue.
Pure heuristics, no I/O: the hard gates come first, then readiness, the small-edit case and configurable priority.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from foremankit.agent_scope import matches_any

MAX_FILES = 8                   # above this the task is split (the plan: more than 8 files)
MAX_CHANGED_LINES = 400         # above this the task is split (~400 changed lines)
MAX_TASK_CHARS = 22000          # above this the task text is split
DEFAULT_ORDER_NORMAL = ("codex", "opencode", "agy")
DEFAULT_ORDER_INTERNAL = ("codex", "agy")
MAX_RUNNER_FAILURES = 2         # Failed attempts per task before fallback.


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
    agy_failures: int = 0
    agy_enabled: bool = True
    opencode_enabled: bool = True
    codex_enabled: bool = True
    codex_failures: int = 0


def decide_runner(
    facts: RunnerFacts, review_globs: Sequence[str] = (),
    order: dict[str, Sequence[str]] | None = None,
) -> tuple[str, list[str]]:
    """(choice, reasons): the first matching rule decides, no rule means a fail-closed lead."""
    result = _gate(facts, review_globs) or _readiness(facts)
    return result or _by_sensitivity(facts, order)


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


def _by_sensitivity(
    facts: RunnerFacts, order: dict[str, Sequence[str]] | None,
) -> tuple[str, list[str]]:
    """Choose the first available runner permitted for the data label."""
    defaults = {"normal": DEFAULT_ORDER_NORMAL, "internal": DEFAULT_ORDER_INTERNAL}
    if facts.sensitivity not in defaults:
        return "lead", [f"unknown sensitivity {facts.sensitivity!r}: fail closed"]
    runners = tuple((order or {}).get(facts.sensitivity, defaults[facts.sensitivity]))
    # Configuration cannot grant free-model providers access to internal data.
    runners = tuple(runner for runner in runners if runner in defaults[facts.sensitivity])
    skipped = []
    for runner in runners:
        reason = _skip_reason(facts, runner)
        if not reason:
            explanation = f"{runner} is first in the order for {facts.sensitivity} data"
            if skipped:
                explanation = f"{' and '.join(skipped)}: {runner} takes it"
            return runner, [explanation]
        skipped.append(reason)
    return _no_candidate(facts, runners)


def _skip_reason(facts: RunnerFacts, runner: str) -> str:
    """Explain why a runner cannot take this task."""
    if getattr(facts, f"{runner}_failures") >= MAX_RUNNER_FAILURES:
        return f"{runner} failed this task twice"
    if not getattr(facts, f"{runner}_enabled"):
        return f"{runner} is switched off"
    return ""


def _no_candidate(facts: RunnerFacts, runners: Sequence[str]) -> tuple[str, list[str]]:
    """Distinguish exhausted attempts from unavailable internal capacity."""
    enabled = [runner for runner in runners if getattr(facts, f"{runner}_enabled")]
    if facts.sensitivity == "internal":
        if runners and all(getattr(facts, f"{runner}_failures") >= MAX_RUNNER_FAILURES
                           for runner in runners):
            return "lead", ["escalation is exhausted"]
        return "queue", ["internal data may only go to codex or agy: the task waits"]
    if enabled:
        return "lead", ["escalation is exhausted"]
    return "lead", ["no runner is enabled"]
