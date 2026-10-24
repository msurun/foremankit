"""Tests for foremankit.regression: ledger parsing, prefix matching, command running,
git change detection and the end-to-end regress() flow.

These tests describe the behaviour the rest of the project relies on: a ledger maps
modules to downstream checks, only checks whose module overlaps a changed file run,
and every run ends with a human-readable report.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from foremankit import regression

GIT_AVAILABLE = shutil.which("git") is not None
requires_git = pytest.mark.skipif(not GIT_AVAILABLE, reason="git is not on PATH")

LEDGER_HEADER = "| Module | Downstream | Command |\n| --- | --- | --- |\n"


def build_ledger(*rows: str) -> str:
    """Return a ledger markdown text with a table header followed by the given rows."""
    return LEDGER_HEADER + "".join(row + "\n" for row in rows)


def python_command(source_code: str, exit_code: int) -> str:
    """Return a shell-free command string that runs python and exits with exit_code.

    The interpreter path is quoted because it can contain spaces (for example on
    Windows), and the whole thing is meant to be split without a shell.
    """
    script = f'import sys; {source_code + "; " if source_code else ""}sys.exit({exit_code})'
    return f'"{sys.executable}" -c "{script}"'


class GitRepo:
    """A throwaway git repository with one committed, then modified, file."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def run_git(self, *args: str) -> None:
        """Run a git command in the repository, failing loudly on error."""
        subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True, text=True
        )

    def create_and_modify_file(self, relative_path: str) -> None:
        """Commit a file, then change it so that `git status` reports a modification."""
        file_path = self.root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("original\n", encoding="utf-8")
        self.run_git("add", ".")
        self.run_git("commit", "-m", "add tracked file")
        file_path.write_text("changed\n", encoding="utf-8")

    def write_ledger(self, text: str, name: str = "ledger.md") -> Path:
        """Write ledger text into the repository and return its path."""
        ledger_path = self.root / name
        ledger_path.write_text(text, encoding="utf-8")
        return ledger_path


@pytest.fixture
def git_repo(tmp_path):
    """A fresh git repository with a committed-then-modified src/a/x.py file."""
    if not GIT_AVAILABLE:
        pytest.skip("git is not on PATH")
    repo = GitRepo(tmp_path)
    repo.run_git("init")
    repo.run_git("config", "user.email", "tests@example.com")
    repo.run_git("config", "user.name", "Regression Tests")
    repo.create_and_modify_file("src/a/x.py")
    return repo


class TestParseLedger:
    """Tests for one function of foremankit.regression."""

    def test_a_full_row_yields_module_downstream_and_command(self):
        text = build_ledger("| src/a/ | api | python -c pass |")

        entries = regression.parse_ledger(text)

        assert entries == [
            {"module": "src/a/", "downstream": "api", "command": "python -c pass"}
        ]

    def test_the_header_and_separator_rows_are_skipped(self):
        text = build_ledger("| src/a/ | api | python -c pass |")

        entries = regression.parse_ledger(text)

        assert len(entries) == 1

    def test_a_row_with_a_blank_cell_is_skipped(self):
        text = build_ledger("| src/a/ |  | python -c pass |")

        entries = regression.parse_ledger(text)

        assert entries == []

    def test_a_row_with_the_wrong_column_count_is_skipped(self):
        text = build_ledger("| src/a/ | api |")

        entries = regression.parse_ledger(text)

        assert entries == []

    def test_backticks_are_stripped_from_cells(self):
        text = build_ledger("| `src/a/` | `api` | `pytest` |")

        entries = regression.parse_ledger(text)

        assert entries == [
            {"module": "src/a/", "downstream": "api", "command": "pytest"}
        ]


class TestAffected:
    """Tests for one function of foremankit.regression."""

    def test_an_entry_whose_module_is_a_file_prefix_is_affected(self):
        entries = [{"module": "src/a/", "downstream": "api", "command": "c"}]

        assert regression.affected(entries, ["src/a/x.py"]) == entries

    def test_a_sibling_prefix_is_not_a_match(self):
        entries = [{"module": "src/a/", "downstream": "api", "command": "c"}]

        assert regression.affected(entries, ["src/ab/x.py"]) == []

    def test_a_leading_dot_slash_is_ignored(self):
        entries = [{"module": "src/a/", "downstream": "api", "command": "c"}]

        assert regression.affected(entries, ["./src/a/x.py"]) == entries

    def test_backslashes_in_changed_files_are_normalised(self):
        entries = [{"module": "src/a/", "downstream": "api", "command": "c"}]

        assert regression.affected(entries, ["src\\a\\x.py"]) == entries

    def test_each_entry_appears_only_once(self):
        entries = [{"module": "src/a/", "downstream": "api", "command": "c"}]

        result = regression.affected(entries, ["src/a/x.py", "src/a/y.py"])

        assert result == entries

    def test_ledger_order_is_preserved(self):
        entries = [
            {"module": "src/b/", "downstream": "b", "command": "c"},
            {"module": "src/a/", "downstream": "a", "command": "c"},
        ]

        result = regression.affected(entries, ["src/a/x.py", "src/b/x.py"])

        assert [entry["downstream"] for entry in result] == ["b", "a"]


class TestRunEntries:
    """Tests for one function of foremankit.regression."""

    def test_a_passing_command_returns_exit_code_zero(self):
        entries = [
            {"module": "m", "downstream": "d", "command": python_command("", 0)}
        ]

        results = regression.run_entries(entries, Path.cwd())

        assert results[0]["returncode"] == 0

    def test_a_failing_command_returns_its_exit_code(self):
        entries = [
            {"module": "m", "downstream": "d", "command": python_command("", 3)}
        ]

        results = regression.run_entries(entries, Path.cwd())

        assert results[0]["returncode"] == 3

    def test_a_missing_program_returns_127(self):
        entries = [
            {"module": "m", "downstream": "d", "command": "definitely-not-a-program"}
        ]

        results = regression.run_entries(entries, Path.cwd())

        assert results[0]["returncode"] == 127

    def test_a_timed_out_command_returns_124(self):
        entries = [
            {
                "module": "m",
                "downstream": "d",
                "command": python_command("import time; time.sleep(5)", 0),
            }
        ]

        results = regression.run_entries(entries, Path.cwd(), timeout=1)

        assert results[0]["returncode"] == 124

    def test_each_result_carries_the_entry_fields_and_seconds(self):
        entries = [{"module": "m", "downstream": "d", "command": python_command("", 0)}]

        results = regression.run_entries(entries, Path.cwd())

        assert results[0]["module"] == "m"
        assert results[0]["downstream"] == "d"
        assert isinstance(results[0]["seconds"], float)


class TestFormatReport:
    """Tests for one function of foremankit.regression."""

    def test_a_passing_result_is_reported_as_ok(self):
        results = [
            {
                "module": "src/a/",
                "downstream": "api",
                "command": "pytest",
                "returncode": 0,
                "seconds": 0.1,
            }
        ]

        report = regression.format_report(results)

        assert "OK api (src/a/): pytest" in report

    def test_a_failing_result_is_reported_as_fail(self):
        results = [
            {
                "module": "src/a/",
                "downstream": "api",
                "command": "pytest",
                "returncode": 1,
                "seconds": 0.1,
            }
        ]

        report = regression.format_report(results)

        assert "FAIL api (src/a/): pytest" in report

    def test_the_last_line_counts_the_failures(self):
        results = [
            {
                "module": "m1",
                "downstream": "d1",
                "command": "c",
                "returncode": 0,
                "seconds": 0.1,
            },
            {
                "module": "m2",
                "downstream": "d2",
                "command": "c",
                "returncode": 1,
                "seconds": 0.1,
            },
        ]

        report = regression.format_report(results)

        assert report.splitlines()[-1] == "1 failed of 2"

    def test_no_results_says_nothing_affected(self):
        report = regression.format_report([])

        assert "nothing affected" in report


@requires_git
class TestChangedFiles:
    """Tests for one function of foremankit.regression."""

    def test_a_modified_file_is_reported(self, git_repo):
        files = regression.changed_files(git_repo.root)

        assert "src/a/x.py" in files

    def test_a_directory_that_is_not_a_git_repo_returns_empty(self, tmp_path):
        assert regression.changed_files(tmp_path) == []


@requires_git
class TestRegress:
    """Tests for one function of foremankit.regression."""

    def test_a_matching_module_runs_its_command_and_reports_ok(self, git_repo):
        ledger = git_repo.write_ledger(
            build_ledger(f"| src/a/ | api | {python_command('', 0)} |")
        )

        exit_code, report = regression.regress(ledger, git_repo.root)

        assert exit_code == 0
        assert "OK" in report

    def test_a_failing_command_makes_regress_return_one(self, git_repo):
        ledger = git_repo.write_ledger(
            build_ledger(f"| src/a/ | api | {python_command('', 3)} |")
        )

        exit_code, report = regression.regress(ledger, git_repo.root)

        assert exit_code == 1
        assert "FAIL" in report

    def test_an_unrelated_module_says_nothing_affected(self, git_repo):
        ledger = git_repo.write_ledger(
            build_ledger(f"| src/other/ | api | {python_command('', 1)} |")
        )

        exit_code, report = regression.regress(ledger, git_repo.root)

        assert exit_code == 0
        assert "nothing affected" in report

    def test_run_false_lists_without_running_commands(self, git_repo):
        marker_path = git_repo.root / "marker.txt"
        command = python_command(
            f"from pathlib import Path; Path(r'{marker_path}').write_text('ran')", 0
        )
        ledger = git_repo.write_ledger(build_ledger(f"| src/a/ | api | {command} |"))

        exit_code, report = regression.regress(ledger, git_repo.root, run=False)

        assert exit_code == 0
        assert "api" in report
        assert not marker_path.exists()

    def test_a_missing_ledger_returns_exit_code_two(self, git_repo):
        exit_code, report = regression.regress(
            git_repo.root / "does-not-exist.md", git_repo.root
        )

        assert exit_code == 2
        assert report.startswith("no ledger at")
