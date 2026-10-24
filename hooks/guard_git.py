"""PreToolUse hook that keeps bypasses, secrets, confidential terms and debugger
calls out of git commits. Runs only for Bash/PowerShell tool calls and blocks
the command (exit 2) when a rule fires; any unexpected error allows silently."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parent
if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))

from shell_words import command_name, simple_commands  # noqa: E402

BLOCK_EXIT_CODE = 2
ALLOW_EXIT_CODE = 0
MAX_FINDINGS_SHOWN = 5
GIT_TIMEOUT_S = 10
BYPASS_ENV_VAR = "FOREMANKIT_ALLOW_COMMIT"
DENYLIST_ENV_VAR = "FOREMANKIT_DENYLIST"
DEFAULT_DENYLIST_PATH = "~/.config/foremankit/confidential_terms.txt"

BYPASS_FLAGS = ("--no-verify",)
PYTHON_SUFFIX = ".py"
SECRET_PATTERNS = (
    ("anthropic-api-key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("private-key", re.compile(
        r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("secret-assignment", re.compile(
        r"""(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*"""
        r"""["'][^"']{12,}["']""")),
)
PLACEHOLDER_WORDS = (
    "example", "placeholder", "changeme", "your_", "xxxx", "dummy", "test",
    "fake", "${", "<",
)
DEBUGGER_PATTERN = re.compile(
    r"^\s*(?:breakpoint\(\)|import pdb|pdb\.set_trace\()")
SENSITIVE_NAME_SUFFIXES = (".env", ".pem", ".pfx", ".key")
SENSITIVE_EXACT_NAMES = ("id_rsa",)


def bypass_findings(tokens: list[str]) -> list[str]:
    """Return reasons why these git tokens bypass hooks; empty when clean."""
    if any(flag in tokens for flag in BYPASS_FLAGS):
        return ["git command disables git hooks with --no-verify"]
    if _is_commit(tokens) and "-n" in tokens:
        return ["git commit disables git hooks with -n"]
    if any(token.startswith("core.hooksPath") for token in tokens):
        return ["git command overrides core.hooksPath"]
    if contains_hooks_path_config(tokens):
        return ["git config overrides core.hooksPath"]
    return []


def contains_hooks_path_config(tokens: list[str]) -> bool:
    """Return True when 'git config' is followed by a core.hooksPath argument."""
    for index, token in enumerate(tokens):
        is_config = token == "config"
        later = tokens[index + 1:]
        if is_config and any("core.hooksPath" in next_token for next_token in later[:3]):
            return True
    return False


def _is_commit(tokens: list[str]) -> bool:
    """Return True when these git tokens represent a 'git commit' call.

    Global options such as '-C dir' may precede the subcommand, so the check
    looks for 'commit' anywhere among the tokens after the program name.
    """
    return command_name(tokens) == "git" and "commit" in tokens[1:]


def _is_placeholder(value: str) -> bool:
    """Return True when the matched secret text is clearly a placeholder."""
    lowered = value.lower()
    return any(word in lowered for word in PLACEHOLDER_WORDS)


def _line_has_secret(line: str) -> str:
    """Return the name of the first secret pattern that matches the line, or ''."""
    for name, pattern in SECRET_PATTERNS:
        match = pattern.search(line)
        if match and not _is_placeholder(match.group(0)):
            return name
    return ""


def _added_lines(diff: str) -> list[str]:
    """Return only the lines added by the diff ('+' lines, excluding '+++')."""
    added = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return added


def _added_lines_with_file(diff: str) -> list[tuple[str, str]]:
    """Return (file name, added line) pairs, tracking '+++ b/path' headers."""
    pairs = []
    current_file = ""
    for line in diff.splitlines():
        if line.startswith("+++ "):
            current_file = _file_from_header(line)
        elif line.startswith("+") and not line.startswith("+++"):
            pairs.append((current_file, line[1:]))
    return pairs


def _file_from_header(line: str) -> str:
    """Extract the file path from a '+++ b/path' diff header line."""
    raw_name = line[len("+++ "):].strip()
    if raw_name.startswith("b/"):
        return raw_name[len("b/"):]
    return raw_name


def _matches_term(text: str, terms: list[str]) -> int:
    """Return the 1-based index of the first matching term, or 0 when none match."""
    lowered = text.lower()
    for index, term in enumerate(terms, start=1):
        if term in lowered:
            return index
    return 0


def scan_added_lines(diff: str, terms: list[str]) -> list[str]:
    """Return findings for added lines: secrets, terms, and debugger calls.

    Debugger calls are only flagged inside staged '.py' files, so the diff
    header is tracked per line. Never returns the matched text itself, only a
    description, so a secret value cannot leak into hook output.
    """
    findings = []
    for file_name, line in _added_lines_with_file(diff):
        findings.extend(_scan_single_added_line(file_name, line, terms))
    return findings


def _scan_single_added_line(file_name: str, line: str, terms: list[str]) -> list[str]:
    """Return the findings contributed by one added line of the given file."""
    findings = []
    secret_name = _line_has_secret(line)
    if secret_name:
        findings.append(f"looks like a {secret_name}")
    term_index = _matches_term(line, terms)
    if term_index:
        findings.append(f"contains confidential term #{term_index}")
    if file_name.endswith(PYTHON_SUFFIX) and DEBUGGER_PATTERN.match(line):
        findings.append("contains a debugger call")
    return findings


def scan_file_names(names: list[str], terms: list[str]) -> list[str]:
    """Return findings for staged file names: sensitive names and terms."""
    findings = []
    for name in names:
        lower_name = name.lower()
        base_name = lower_name.rsplit("/", 1)[-1]
        if base_name in SENSITIVE_EXACT_NAMES or base_name.endswith(SENSITIVE_NAME_SUFFIXES):
            findings.append(f"{name}: staged sensitive file name")
        term_index = _matches_term(name, terms)
        if term_index:
            findings.append(f"{name}: name contains confidential term #{term_index}")
    return findings


def _run_git(arguments: list[str], cwd: str | None) -> str:
    """Run git with these arguments and return stdout; raise on failure."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
        check=True,
    )
    return completed.stdout


def staged_diff(cwd: str | None) -> str:
    """Return the staged diff with zero context lines, color disabled."""
    return _run_git(["diff", "--cached", "-U0", "--no-color"], cwd)


def staged_names(cwd: str | None) -> list[str]:
    """Return the names of the staged files, one per line."""
    output = _run_git(["diff", "--cached", "--name-only"], cwd)
    return [line for line in output.splitlines() if line]


def load_denylist() -> list[str]:
    """Load confidential terms from the denylist file; missing file means none."""
    raw_path = os.environ.get(DENYLIST_ENV_VAR, DEFAULT_DENYLIST_PATH)
    try:
        content = Path(raw_path).expanduser().read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    terms = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            terms.append(stripped.lower())
    return terms


def _all_findings(diff: str, names: list[str], terms: list[str]) -> list[str]:
    """Combine added-line findings and file-name findings into one list."""
    line_findings = [f"{file_name}: {finding}" for file_name, line in _added_lines_with_file(diff)
                     for finding in _scan_single_added_line(file_name, line, terms)]
    name_findings = scan_file_names(names, terms)
    prefixed = [f"{name}: {finding}" for name in names for finding in
                _findings_for_name(name, finding_of=name_findings)]
    return prefixed + line_findings


def _findings_for_name(name: str, finding_of: list[str]) -> list[str]:
    """Return the file-name findings that already reference this name."""
    return [finding.split(": ", 1)[1] for finding in finding_of
            if finding.startswith(f"{name}: ")]


def _block_message(findings: list[str]) -> str:
    """Build the block message shown to the agent, capped at MAX_FINDINGS_SHOWN."""
    shown = findings[:MAX_FINDINGS_SHOWN]
    lines = ["Blocked: this git commit would introduce risky content:"]
    lines.extend([f"  - {finding}" for finding in shown])
    extra_count = len(findings) - len(shown)
    if extra_count > 0:
        lines.append(f"  ... and {extra_count} more")
    lines.append(
        "Remove the findings and do not bypass the check. "
        f"If the user insists, they can override one commit by setting "
        f"{BYPASS_ENV_VAR}=1 in the environment.")
    return "\n".join(lines)


def _review_commit(cwd: str | None) -> str:
    """Run the commit checks for one staged state; return '' or the block message."""
    terms = load_denylist()
    findings = _all_findings(staged_diff(cwd), staged_names(cwd), terms)
    if not findings:
        return ""
    return _block_message(findings)


def _git_findings(tokens: list[str], cwd: str | None) -> str:
    """Apply all rules to one git simple command; return '' or the block message."""
    bypass = bypass_findings(tokens)
    if bypass:
        return _block_message(bypass)
    if _is_commit(tokens):
        return _review_commit(cwd)
    return ""


def decide(event: dict) -> str:
    """Return '' to allow the tool call, or the block message to deny it."""
    if os.environ.get(BYPASS_ENV_VAR) == "1":
        return ""
    command = event.get("tool_input", {}).get("command", "")
    cwd = event.get("cwd") or None
    for tokens in simple_commands(command):
        if command_name(tokens) != "git":
            continue
        message = _git_findings(tokens, cwd)
        if message:
            return message
    return ""


def _read_event() -> dict:
    """Read one JSON event from stdin, tolerating a UTF-8 BOM."""
    raw = sys.stdin.buffer.read().decode("utf-8-sig")
    return json.loads(raw)


def main() -> int:
    """Run the hook: exit 2 with reason on stderr to block, exit 0 to allow."""
    try:
        event = _read_event()
        message = decide(event)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            subprocess.SubprocessError):
        return ALLOW_EXIT_CODE
    if not message:
        return ALLOW_EXIT_CODE
    print(message, file=sys.stderr)
    return BLOCK_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
