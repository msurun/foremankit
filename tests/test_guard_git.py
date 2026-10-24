"""Tests for the guard_git pre-commit hook: pure scanning rules and the real
hook process against a throwaway git repository."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"
if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))

import guard_git  # noqa: E402

GUARD_SCRIPT = HOOKS_DIR / "guard_git.py"
BLOCK_EXIT_CODE = 2
ALLOW_EXIT_CODE = 0
ANTHROPIC_KEY = "sk-ant-" + "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5"
AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"


def staged_content_diff(file_name: str, line: str) -> str:
    """A minimal unified diff that adds exactly one line to `file_name`."""
    header = f"+++ b/{file_name}\n"
    body = f"@@ -0,0 +1 @@\n+{line}\n"
    return header + body


# ---------------------------------------------------------------------------
# Part 1: bypass detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "command",
    [
        "git commit --no-verify -m x",
        "git commit -n -m x",
        "git push --no-verify",
        "git -c core.hooksPath=/dev/null commit -m x",
        "git config core.hooksPath x",
    ],
)
def test_command_that_bypasses_hooks_is_blocked(command):
    tokens = guard_git.simple_commands(command)[0]

    findings = guard_git.bypass_findings(tokens)

    assert findings


@pytest.mark.parametrize(
    "command",
    [
        'git commit -m "mention --no-verify here"',
        "git commit -m ok",
        "git status",
    ],
)
def test_command_without_bypass_is_clean(command):
    tokens = guard_git.simple_commands(command)[0]

    findings = guard_git.bypass_findings(tokens)

    assert findings == []


# ---------------------------------------------------------------------------
# Part 1: diff scanning — secrets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "added_line",
    [
        f"token = '{ANTHROPIC_KEY}'",
        f"aws = '{AWS_KEY}'",
        "-----BEGIN RSA PRIVATE KEY-----",
        'password = "hunter2hunter2hunter2"',
    ],
)
def test_an_added_secret_line_is_flagged(added_line):
    diff = staged_content_diff("app.py", added_line)

    findings = guard_git.scan_added_lines(diff, [])

    assert findings


@pytest.mark.parametrize(
    "added_line",
    [
        'api_key = "your_api_key_here_xxxxx"',
        'api_key = "this_example_value_123456"',
        'api_key = "please_changeme_now_12345"',
    ],
)
def test_a_placeholder_value_is_not_flagged(added_line):
    diff = staged_content_diff("app.py", added_line)

    findings = guard_git.scan_added_lines(diff, [])

    assert findings == []


def test_a_removed_secret_line_is_flagged():
    removed = staged_content_diff("app.py", "x").replace(
        f"+{ANTHROPIC_KEY}", f"-{ANTHROPIC_KEY}")

    findings = guard_git.scan_added_lines(removed, [])

    assert findings == []


# ---------------------------------------------------------------------------
# Part 1: diff scanning — confidential terms and debuggers
# ---------------------------------------------------------------------------

def test_a_confidential_term_is_flagged_case_insensitively():
    diff = staged_content_diff("notes.md", "the PLANTX-CUSTOMER-NAME appears here")
    terms = ["plantx-customer-name"]

    findings = guard_git.scan_added_lines(diff, terms)

    assert findings
    assert all("PLANTX-CUSTOMER-NAME" not in finding for finding in findings)


@pytest.mark.parametrize(
    "added_line",
    ["breakpoint()", "import pdb", "pdb.set_trace()"],
)
def test_a_debugger_call_in_a_python_file_is_flagged(added_line):
    diff = staged_content_diff("app.py", added_line)

    findings = guard_git.scan_added_lines(diff, [])

    assert findings


def test_a_debugger_call_in_a_markdown_file_is_ignored():
    diff = staged_content_diff("notes.md", "breakpoint()")

    findings = guard_git.scan_added_lines(diff, [])

    assert findings == []


# ---------------------------------------------------------------------------
# Part 1: staged file names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [".env", "a/b.pem", "x.pfx", "id_rsa"])
def test_a_sensitive_staged_name_is_flagged(name):
    findings = guard_git.scan_file_names([name], [])

    assert findings


def test_a_normal_staged_name_is_not_flagged():
    findings = guard_git.scan_file_names(["main.py"], [])

    assert findings == []


# ---------------------------------------------------------------------------
# Part 2: the hook process against a real repository
# ---------------------------------------------------------------------------

def run_hook(repository: Path, event: dict, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the hook script on `event` and return the completed process."""
    environment = {
        **os.environ,
        "HOME": str(repository),
        "USERPROFILE": str(repository),
        "FOREMANKIT_DENYLIST": str(repository / "denylist.txt"),
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, str(GUARD_SCRIPT)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )


def commit_event(repository: Path) -> dict:
    """The Bash event that asks the hook to review 'git commit -m x'."""
    return {
        "tool_name": "Bash",
        "tool_input": {"command": "git commit -m x"},
        "cwd": str(repository),
    }


def write_and_stage(repository: Path, file_name: str, content: str) -> None:
    """Create `file_name` inside the repository and stage it."""
    path = repository / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", file_name], cwd=repository, check=True)


@pytest.fixture
def repository(tmp_path):
    """A fresh git repository with one empty initial commit."""
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    return tmp_path


def test_nothing_staged_allows_the_commit(repository):
    result = run_hook(repository, commit_event(repository))

    assert result.returncode == ALLOW_EXIT_CODE


def test_a_staged_secret_file_blocks_without_echoing_the_key(repository):
    write_and_stage(repository, "leak.py", f"key = '{ANTHROPIC_KEY}'\n")

    result = run_hook(repository, commit_event(repository))

    assert result.returncode == BLOCK_EXIT_CODE
    assert "leak.py" in result.stderr
    assert ANTHROPIC_KEY not in result.stderr


def test_a_staged_debugger_call_blocks(repository):
    write_and_stage(repository, "debug.py", "breakpoint()\n")

    result = run_hook(repository, commit_event(repository))

    assert result.returncode == BLOCK_EXIT_CODE


def test_a_staged_denylisted_term_blocks_without_echoing_it(repository):
    (repository / "denylist.txt").write_text("plantx-alpha\n", encoding="utf-8")
    write_and_stage(repository, "notes.md", "project PLANTX-ALPHA here\n")

    result = run_hook(repository, commit_event(repository))

    assert result.returncode == BLOCK_EXIT_CODE
    assert "PLANTX-ALPHA" not in result.stderr


def test_a_clean_staged_file_allows_the_commit(repository):
    write_and_stage(repository, "clean.py", "print('hello')\n")

    result = run_hook(repository, commit_event(repository))

    assert result.returncode == ALLOW_EXIT_CODE


def test_the_bypass_env_var_allows_a_dirty_commit(repository):
    write_and_stage(repository, "leak.py", f"key = '{ANTHROPIC_KEY}'\n")

    result = run_hook(
        repository, commit_event(repository), {"FOREMANKIT_ALLOW_COMMIT": "1"})

    assert result.returncode == ALLOW_EXIT_CODE


def test_no_verify_is_blocked_even_with_nothing_staged(repository):
    event = {
        "tool_name": "Bash",
        "tool_input": {"command": "git commit --no-verify -m x"},
        "cwd": str(repository),
    }

    result = run_hook(repository, event)

    assert result.returncode == BLOCK_EXIT_CODE


def test_a_non_git_tool_call_is_allowed(repository):
    event = {"tool_name": "Write", "tool_input": {"content": ANTHROPIC_KEY}, "cwd": str(repository)}

    result = run_hook(repository, event)

    assert result.returncode == ALLOW_EXIT_CODE


@pytest.mark.parametrize("raw_input", ["not json", ""])
def test_garbage_or_empty_stdin_allows_silently(raw_input):
    result = subprocess.run(
        [sys.executable, str(GUARD_SCRIPT)],
        input=raw_input,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == ALLOW_EXIT_CODE
    assert result.stderr == ""


def test_a_missing_denylist_file_does_not_crash(repository):
    write_and_stage(repository, "clean.py", "print('hello')\n")
    missing = repository / "does-not-exist.txt"

    result = run_hook(
        repository, commit_event(repository), {"FOREMANKIT_DENYLIST": str(missing)})

    assert result.returncode == ALLOW_EXIT_CODE
