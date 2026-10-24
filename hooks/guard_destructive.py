"""PreToolUse hook that blocks destructive shell commands before they run.

It reads one Claude Code hook event (JSON on stdin) for the Bash or PowerShell
tool, inspects every simple command, and exits 2 with a reason when the command
would force-push, wipe changes or delete files outside a scratch area.
"""

import json
import os
import sys
from pathlib import Path

# Import the sibling module by its file name: the hook runs as a script, so the
# hooks folder is not on sys.path until we add it here.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shell_words import command_name, simple_commands  # noqa: E402

ALLOW_OVERRIDE_ENV = "FOREMANKIT_ALLOW_DESTRUCTIVE"
OVERRIDE_HINT = f"Set {ALLOW_OVERRIDE_ENV}=1 only if the user explicitly asked for it."

FORCE_PUSH_FLAGS = frozenset(
    {"--force", "-f", "--force-with-lease", "--force-if-includes"}
)
SHELL_TOOLS = ("Bash", "PowerShell")
GIT_COMMAND_NAMES = frozenset({"git"})
RM_COMMAND_NAMES = frozenset({"rm", "rmdir"})
RECURSIVE_RM_FLAGS = frozenset({"-r", "-R", "-rf", "-fr", "--recursive"})
GIT_DISCARD_PATHS = frozenset({".", "--", "./"})

# Targets inside these folders are regenerable build or scratch output.
ALLOWED_PREFIXES = (
    "/tmp/",
    "/scratchpad/",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".agent-out",
)
# These targets remove the whole tree or all of the current directory.
ALWAYS_BLOCKED_TARGETS = frozenset({"/", "~", "*", ".", "..", "c:/", "~/"})

POWERSHELL_NAMES = frozenset(
    {"remove-item", "ri", "rm", "del", "erase", "rd", "rmdir"}
)
POWERSHELL_RECURSE_FLAGS = frozenset({"-recurse", "-r", "-rec"})
POWERSHELL_DISK_NAMES = frozenset({"clear-disk", "remove-partition", "format"})
# cmd builtins spell flags with a leading slash, which normalise() keeps as "//".
CMD_RECURSIVE_FLAGS = frozenset({"/s"})

SQL_CLIENTS = frozenset({"psql", "sqlite3", "mysql", "sqlcmd"})
SQL_DESTRUCTIVE_WORDS = ("drop database", "drop table", "truncate")


def normalise(token: str) -> str:
    """Return a token lowercased and with backslashes turned into slashes."""
    return token.replace("\\", "/").lower()


def is_inside_allowed_prefix(target: str) -> bool:
    """Return True when the normalised target lives in an allowed folder."""
    return any(prefix in target for prefix in ALLOWED_PREFIXES)


def is_unsafe_rm_target(raw_target: str) -> bool:
    """Return True when a recursive delete target must always be refused.

    A target is unsafe when it names the root, home, a wildcard, the current
    directory, or lies outside every allowed prefix.
    """
    target = normalise(raw_target)
    if target in ALWAYS_BLOCKED_TARGETS:
        return True
    if target.endswith(":/"):
        return True
    return not is_inside_allowed_prefix(target)


def is_recursive_short_flag(token: str) -> bool:
    """Return True for rm flags that combine -r with other letters (e.g. -fdx)."""
    if not token.startswith("-") or token.startswith("--"):
        return False
    return "r" in token[1:]


def has_recursive_rm_flag(tokens: list[str]) -> bool:
    """Return True when any rm token asks for a recursive delete."""
    for token in tokens:
        if token in RECURSIVE_RM_FLAGS or is_recursive_short_flag(token):
            return True
    return False


def rm_targets(tokens: list[str]) -> list[str]:
    """Return the non-flag arguments of an rm command."""
    return [token for token in tokens[1:] if not token.startswith("-")]


def git_reason(tokens: list[str]) -> str:
    """Return a reason when the git command is destructive, else empty."""
    if len(tokens) < 2:
        return ""
    subcommand = tokens[1]
    if subcommand == "push" and is_force_push(tokens[2:]):
        return "force push rewrites published history"
    if subcommand == "reset" and "--hard" in tokens[2:]:
        return "git reset --hard discards uncommitted work"
    if subcommand == "clean" and has_clean_flags(tokens[2:]):
        return "git clean with force deletes untracked files"
    if subcommand in ("checkout", "restore") and discards_all_changes(tokens[2:]):
        return "this git command discards all local changes"
    if subcommand == "branch" and "-D" in tokens[2:]:
        return "git branch -D deletes a branch without merging"
    if subcommand == "stash" and tokens[2:3] in (["drop"], ["clear"]):
        return "dropping a stash deletes saved work"
    return ""


def is_force_push(arguments: list[str]) -> bool:
    """Return True when push arguments contain a force flag or +refspec."""
    for argument in arguments:
        if argument in FORCE_PUSH_FLAGS:
            return True
        if argument.startswith("+") and not argument.startswith("++"):
            return True
    return False


def has_clean_flags(arguments: list[str]) -> bool:
    """Return True when git clean flags combine force with -d or -x."""
    force = False
    widen = False
    for argument in arguments:
        if argument == "--force":
            force = True
            continue
        if argument.startswith("-") and not argument.startswith("--"):
            letters = argument[1:]
            force = force or "f" in letters
            widen = widen or "d" in letters or "x" in letters
    return force and widen


def discards_all_changes(arguments: list[str]) -> bool:
    """Return True when checkout/restore targets the whole worktree."""
    if not arguments:
        return False
    if arguments[0] in ("--", "."):
        return len(arguments) <= 2 and arguments[-1] in GIT_DISCARD_PATHS
    return arguments[0] in GIT_DISCARD_PATHS


def rm_reason(tokens: list[str]) -> str:
    """Return a reason when a recursive delete leaves the allowed prefixes."""
    if not has_recursive_rm_flag(tokens):
        return ""
    targets = rm_targets(tokens)
    if not targets:
        return ""
    for target in targets:
        if is_unsafe_rm_target(target):
            return f"recursive delete of {target} is outside the allowed scratch folders"
    return ""


def powershell_reason(tokens: list[str]) -> str:
    """Return a reason for a destructive PowerShell or cmd command, else empty."""
    name = command_name(tokens)
    if name in POWERSHELL_DISK_NAMES:
        return f"{name} erases a disk or partition"
    if name not in POWERSHELL_NAMES:
        return ""
    if has_cmd_recursive_flag(tokens):
        return f"{name} /s deletes a directory tree"
    if not has_powershell_recurse(tokens):
        return ""
    targets = rm_targets(tokens) or ["."]
    for target in targets:
        if is_unsafe_rm_target(target):
            return f"{name} -Recurse on {target} is outside the allowed scratch folders"
    return ""


def has_powershell_recurse(tokens: list[str]) -> bool:
    """Return True when any token is a -Recurse flag or abbreviation."""
    for token in tokens:
        if normalise(token) in POWERSHELL_RECURSE_FLAGS:
            return True
    return False


def has_cmd_recursive_flag(tokens: list[str]) -> bool:
    """Return True when a cmd builtin is called with /s.

    cmd flags keep a leading slash, so compare the raw lowercased token.
    """
    for token in tokens[1:]:
        if token.lower() in CMD_RECURSIVE_FLAGS:
            return True
    return False


def sql_reason(tokens: list[str]) -> str:
    """Return a reason when SQL-driven deletion appears in a CLI command."""
    if command_name(tokens) not in SQL_CLIENTS:
        return ""
    for token in tokens:
        text = normalise(token)
        if any(word in text for word in SQL_DESTRUCTIVE_WORDS):
            return "this runs a DROP or TRUNCATE statement"
    return ""


def dispatch_reason(tokens: list[str]) -> str:
    """Return a reason for the command, choosing the checker by command name."""
    name = command_name(tokens)
    if name in GIT_COMMAND_NAMES:
        return git_reason(tokens)
    if name in POWERSHELL_DISK_NAMES:
        return powershell_reason(tokens)
    if name in SQL_CLIENTS:
        return sql_reason(tokens)
    return (rm_reason(tokens) if name in RM_COMMAND_NAMES else "") or powershell_reason(tokens)


def rule_for(tokens: list[str]) -> str:
    """Return "" when the command is allowed, else a short reason naming the rule."""
    if not tokens:
        return ""
    return dispatch_reason(tokens)


def first_blocking_reason(command: str) -> str:
    """Return the reason of the first simple command that is forbidden."""
    for tokens in simple_commands(command):
        reason = rule_for(tokens)
        if reason:
            return reason
    return ""


def build_message(reason: str) -> str:
    """Return the agent-facing block message for a matched rule."""
    return (
        f"Blocked by guard_destructive: {reason}. "
        "Ask the user before doing this; do not retry. "
        f"{OVERRIDE_HINT}"
    )


def decide(event: dict) -> str:
    """Return a block message for the event, or "" when it may proceed."""
    if event.get("tool_name") not in SHELL_TOOLS:
        return ""
    tool_input = event.get("tool_input") or {}
    command = tool_input.get("command") or ""
    if not command.strip():
        return ""
    reason = first_blocking_reason(command)
    return build_message(reason) if reason else ""


def read_event() -> dict:
    """Parse the hook event from stdin, tolerating a UTF-8 BOM."""
    raw = sys.stdin.buffer.read().decode("utf-8-sig")
    return json.loads(raw)


def main() -> int:
    """Run the hook: return 2 to block, 0 to allow, 0 on any unexpected error."""
    if os.environ.get(ALLOW_OVERRIDE_ENV) == "1":
        return 0
    try:
        message = decide(read_event())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 0
    if not message:
        return 0
    print(message, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
