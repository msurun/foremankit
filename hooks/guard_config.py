"""PreToolUse hook that stops an agent from loosening config instead of fixing code.

Reads one hook event from stdin, blocks writes to protected lint/CI files and warns
when an edit changes lint or test settings in pyproject.toml and friends. A hook
must never break the session, so any unexpected error ends with exit code 0.
"""

import fnmatch
import json
import os
import sys

# Exit codes mandated by the Claude Code hook protocol.
EXIT_ALLOW = 0
EXIT_BLOCK = 2

# Files whose whole purpose is to weaken checks: writing them is refused.
PROTECTED_GLOBS = [
    "*/ruff.toml",
    "*/.ruff.toml",
    "*/.flake8",
    "*/.pylintrc",
    "*/mypy.ini",
    "*/.pre-commit-config.yaml",
    "*/.claude/settings.json",
    "*/.claude/settings.local.json",
    "*/hooks/guard_*.py",
]

# Shared config files are allowed, but mention these markers in the new text
# means a setting may have been loosened, so we warn.
SHARED_CONFIG_GLOBS = ["*/pyproject.toml", "*/setup.cfg", "*/tox.ini"]
SUSPICIOUS_SETTING_MARKERS = [
    "[tool.ruff",
    "[tool.mypy",
    "[tool.black",
    "[tool.pytest",
    "[tool.pylint",
    "[flake8]",
    "[mypy]",
    "ignore =",
    "per-file-ignores",
    "max-line-length",
    "fail_under",
]

# Environment switches that let the user (not the agent) change the policy.
PROTECTED_PATHS_ENV = "FOREMANKIT_PROTECTED_PATHS"
ALLOW_CONFIG_EDIT_ENV = "FOREMANKIT_ALLOW_CONFIG_EDIT"
ENV_FLAG_ON = "1"

WARN_MESSAGE = (
    "this edit changes lint/test settings; make sure you are not loosening a "
    "check to hide a problem"
)


def normalise_path(path: str) -> str:
    """Return the path with forward slashes and lowercase, for glob matching."""
    return path.replace("\\", "/").lower()


def protected_globs() -> list[str]:
    """Return the built-in protected globs plus any from FOREMANKIT_PROTECTED_PATHS."""
    extra_raw = os.environ.get(PROTECTED_PATHS_ENV, "")
    extra_globs = [pattern.strip() for pattern in extra_raw.split(",") if pattern.strip()]
    return PROTECTED_GLOBS + extra_globs


def is_protected(path: str, globs: list[str]) -> bool:
    """Return True when the normalised path matches one of the globs."""
    normalised = normalise_path(path)
    return any(fnmatch.fnmatch(normalised, normalise_path(pattern)) for pattern in globs)


def new_text(tool_input: dict) -> str:
    """Return the text a tool is about to write, joining MultiEdit fragments.

    Write puts its text in "content", Edit in "new_string", MultiEdit in the
    "new_string" field of each entry of "edits".
    """
    if "content" in tool_input:
        return tool_input["content"]
    if "new_string" in tool_input:
        return tool_input["new_string"]
    edits = tool_input.get("edits", [])
    return "\n".join(edit["new_string"] for edit in edits)


def has_suspicious_marker(text: str) -> bool:
    """Return True when the text mentions a lint or test setting marker."""
    return any(marker in text for marker in SUSPICIOUS_SETTING_MARKERS)


def build_block_message(file_path: str) -> str:
    """Return the stderr message explaining why the write is refused."""
    return (
        f"Refusing to edit protected config file: {file_path}. "
        "Fix the code instead of loosening the check. "
        f"If this is deliberate, the user can set {ALLOW_CONFIG_EDIT_ENV}=1."
    )


def decide(event: dict) -> tuple[str, str]:
    """Return ("block" | "warn" | "allow", message) for one hook event.

    Raises KeyError, TypeError or ValueError when the event shape is wrong; the
    caller turns that into a silent allow.
    """
    if os.environ.get(ALLOW_CONFIG_EDIT_ENV) == ENV_FLAG_ON:
        return ("allow", "")

    tool_input = event["tool_input"]
    file_path = tool_input["file_path"]

    if is_protected(file_path, protected_globs()):
        return ("block", build_block_message(file_path))
    if is_protected(file_path, SHARED_CONFIG_GLOBS) and has_suspicious_marker(
        new_text(tool_input)
    ):
        return ("warn", WARN_MESSAGE)
    return ("allow", "")


def read_event() -> dict:
    """Read the JSON hook event from stdin, tolerating a leading UTF-8 BOM."""
    return json.loads(sys.stdin.buffer.read().decode("utf-8-sig"))


def emit(decision: str, message: str) -> int:
    """Print the hook output for the decision and return the process exit code."""
    if decision == "block":
        print(message, file=sys.stderr)
        return EXIT_BLOCK
    if decision == "warn":
        payload = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": message,
            }
        }
        print(json.dumps(payload))
    return EXIT_ALLOW


def main() -> int:
    """Run the hook: read the event, decide, print the output, return the exit code."""
    try:
        decision, message = decide(read_event())
    except (ValueError, KeyError, OSError, TypeError) as error:
        # A broken hook must never block the session, so fail open but visible.
        print(f"guard_config: ignored malformed event: {error}", file=sys.stderr)
        return EXIT_ALLOW
    return emit(decision, message)


if __name__ == "__main__":
    raise SystemExit(main())
