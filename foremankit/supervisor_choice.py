"""Automatic choice of the supervisor for one bulk_code task: the Haiku agent or Antigravity (`foremankit supervise`).
Pure heuristics, no I/O: multi-file, test-heavy, large or broad tasks go to agy (slow, free, runs the check itself),
small single-file ones to Haiku (fast). The thresholds are guesses; routing.yaml `supervisor.auto` overrides them.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any

MIN_TARGETS = 3                  # files in one task (agy writes one file per command: 3+ files = 3+ slow agy runs)
MIN_CONTEXT_LINES = 800          # lines of context the task needs (~p75 of 563 real bulk_code calls)
MIN_TASK_CHARS = 8000            # length of the task text
BROAD_MIN_CONTEXT_LINES = 200     # a keyword alone is weak: it counts only with 2+ files or this much context
KEYWORDS = ("refactor", "migrate", "rename across", "every module", "all modules",
            "wszystkie", "refaktor", "migracj", "we wszystkich")
PYTEST_FLAGS_WITH_VALUE = frozenset({"-k", "-m", "-p", "-c", "-o", "--maxfail", "--tb", "--rootdir", "--durations"})


@dataclass(frozen=True)
class AutoSettings:
    """The thresholds of the auto supervisor (defaults above, overridable in routing.yaml supervisor.auto)."""
    min_targets: int = MIN_TARGETS
    min_context_lines: int = MIN_CONTEXT_LINES
    min_task_chars: int = MIN_TASK_CHARS
    keywords: tuple[str, ...] = KEYWORDS


def auto_settings(routing: dict[str, Any] | None) -> AutoSettings:
    """The settings from routing.yaml `supervisor.auto`; a missing or empty block gives the defaults."""
    block = ((routing or {}).get("supervisor") or {}).get("auto") or {}
    defaults = AutoSettings()
    return AutoSettings(int(block.get("min_targets", defaults.min_targets)),
                        int(block.get("min_context_lines", defaults.min_context_lines)),
                        int(block.get("min_task_chars", defaults.min_task_chars)),
                        tuple(str(word) for word in block.get("keywords", defaults.keywords)))


@dataclass(frozen=True)
class TaskFacts:
    """What the choice looks at: the task text, the files it writes, its check command and its context."""
    task_text: str
    targets: list[str]
    check: str
    context_lines: int = 0
    sensitivity: str = "normal"
    agy_enabled: bool = True


def choose_supervisor(task_text: str, targets: list[str], check: str, context_lines: int = 0,
                      sensitivity: str = "normal", agy_enabled: bool = True) -> tuple[str, str]:
    """("agy"|"haiku", short reason) with the default thresholds; decide_supervisor takes the routing.yaml settings."""
    return decide_supervisor(TaskFacts(task_text, targets, check, context_lines, sensitivity, agy_enabled))


def decide_supervisor(facts: TaskFacts, settings: AutoSettings | None = None) -> tuple[str, str]:
    """("agy"|"haiku", short reason). Private data and a switched-off agy always mean Haiku; then the first heavy rule wins."""
    if facts.sensitivity == "private":
        return "haiku", "private data never goes to agy"
    if not facts.agy_enabled:
        return "haiku", "agy is switched off"
    settings = settings or AutoSettings()
    for reason in (_multi_file(facts.targets, settings), _test_heavy(facts),
                   _large(facts.task_text, facts.context_lines, settings), _broad(facts, settings)):
        if reason:
            return "agy", reason
    return "haiku", "small single-file task"


def _multi_file(targets: list[str], settings: AutoSettings) -> str | None:
    return f"multi-file ({len(targets)} files)" if len(targets) >= settings.min_targets else None


def _test_heavy(facts: TaskFacts) -> str | None:
    """A weak signal (the check is often the same whatever the task): counts only for a task of 2+ files, and then
    only for a pytest check over a directory, over several files, or over everything (no path at all)."""
    if len(facts.targets) < 2:
        return None
    check = facts.check
    tokens = shlex.split(check.replace("\\", "/"), posix=True) if check.strip() else []
    if "pytest" not in tokens:
        return None
    paths = _pytest_paths(tokens[tokens.index("pytest") + 1:])
    if len(paths) != 1:
        return "test-heavy (pytest over several files or the whole suite)"
    single_test = paths[0].endswith(".py") or "::" in paths[0]
    return None if single_test else "test-heavy (pytest over a directory)"


def _pytest_paths(args: list[str]) -> list[str]:
    """The path arguments of a pytest command line (flags and the values of -k/-m/... left out)."""
    paths: list[str] = []
    words = iter(args)
    for arg in words:
        if arg in PYTEST_FLAGS_WITH_VALUE:
            next(words, None)
        elif not arg.startswith("-"):
            paths.append(arg)
    return paths


def _large(task_text: str, context_lines: int, settings: AutoSettings) -> str | None:
    if context_lines >= settings.min_context_lines or len(task_text) >= settings.min_task_chars:
        return "large task"
    return None


def _broad(facts: TaskFacts, settings: AutoSettings) -> str | None:
    """A broad-change keyword is a weak signal: it counts only together with 2+ files or a fair amount of context."""
    if len(facts.targets) < 2 and facts.context_lines < BROAD_MIN_CONTEXT_LINES:
        return None
    found = [word for word in settings.keywords if re.search(re.escape(word), facts.task_text, re.IGNORECASE)]
    return f"broad change ({found[0]})" if found else None
