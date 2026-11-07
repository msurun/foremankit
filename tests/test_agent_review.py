"""Exercise independent model review with fake runners.
Only branch snapshots and the read-only guard need a real temporary git repo.
"""
import json
from pathlib import Path

import pytest

from foremankit import agent_review as review_module
from foremankit import agent_switch
from foremankit.agent_git import git
from foremankit.agent_review import ReviewJob, choose_reviewer, parse_findings, review
from foremankit.agent_runner import RunnerResult
from foremankit.state import StateStore

FINDING = {"severity": "high", "file": "module.py", "line": 1,
           "problem": "wrong value", "evidence": "VALUE = 2 breaks the expected value 1"}


class FakeRunner:
    """Return canned answers and optionally violate the read-only instruction."""

    def __init__(self, name: str = "codex", answer: str = "[]", edits: bool = False):
        """Store fake behaviour without starting any vendor process."""
        self.name, self.answer, self.edits = name, answer, edits
        self.model = "deepseek"
        self.calls = []

    def eligible(self, sensitivity: str) -> bool:
        """Permit public data only."""
        return sensitivity == "normal"

    def run(self, prompt: str, cwd: Path, timeout: float) -> RunnerResult:
        """Remember prompts and optionally edit a tracked branch file."""
        self.calls.append(prompt)
        if self.edits:
            (cwd / "module.py").write_text("broken", encoding="utf-8")
        return RunnerResult(self.answer)


@pytest.fixture
def branch_repo(tmp_path: Path) -> tuple[Path, dict]:
    """Make an agent patch commit and its author's report entry."""
    git(["init", "-q", "-b", "main"], tmp_path)
    git(["config", "user.email", "test@example.invalid"], tmp_path)
    git(["config", "user.name", "Test"], tmp_path)
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(["add", "."], tmp_path)
    git(["commit", "-qm", "base"], tmp_path)
    base = git(["rev-parse", "HEAD"], tmp_path).strip()
    git(["checkout", "-qb", "agent-run/test"], tmp_path)
    (tmp_path / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    git(["commit", "-qam", "patch"], tmp_path)
    return tmp_path, {"branch": "agent-run/test", "runner": "agy", "base": base}


@pytest.fixture(autouse=True)
def isolated_quota(monkeypatch, tmp_path):
    """Keep machine quota pauses out of deterministic reviewer selection."""
    monkeypatch.setattr(review_module.agent_quota, "state_path", lambda: tmp_path / "quota.json")


@pytest.mark.parametrize("author,expected", [("agy", "codex"), ("codex", "agy"),
                                           ("opencode", "codex")])
def test_auto_selects_a_different_vendor(author, expected):
    """Fixed runners and OpenCode model vendors share the same selection policy."""
    runners = {"codex": FakeRunner(), "agy": FakeRunner("agy")}
    entry = {"runner": author, "model": "deepseek"}
    assert choose_reviewer(entry, "auto", {}, runners) == expected


@pytest.mark.parametrize("answer", [json.dumps([FINDING]),
                                    "```json\n" + json.dumps([FINDING]) + "\n```",
                                    "Found this: " + json.dumps([FINDING]) + " end"])
def test_json_is_parsed_inside_surrounding_text(answer):
    """Formatting wrappers do not discard valid findings."""
    assert parse_findings(answer) == [FINDING]


@pytest.mark.parametrize("answer,status", [("garbage", "unparsable"), ("[]", "ok")])
def test_answer_status_and_raw_text_are_preserved(branch_repo, answer, status):
    """An empty list succeeds while unreadable output keeps its evidence."""
    project, entry = branch_repo
    result = review(ReviewJob(entry, project), {"codex": FakeRunner(answer=answer)}, {})
    assert (result["status"], result["raw_text"], result["counts_by_severity"]["high"]) == (
        status, answer, 0)


def test_only_the_author_vendor_available_is_skipped(branch_repo):
    """Auto cannot silently fall back to a same-vendor review."""
    project, entry = branch_repo
    result = review(ReviewJob(entry, project), {"agy": FakeRunner("agy")}, {})
    assert (result["status"], result["reason"]) == (
        "skipped", "no reviewer of a different vendor than google is available")


def test_explicit_same_vendor_is_refused(branch_repo):
    """An explicit name still requires vendor independence by default."""
    project, entry = branch_repo
    result = review(ReviewJob(entry, project, "agy"), {"agy": FakeRunner("agy")}, {})
    assert result["status"] == "failed" and "different vendor" in result["reason"]


def test_owner_branches_are_refused(branch_repo):
    """Only agent-run branches may be submitted for model review."""
    project, entry = branch_repo
    entry["branch"] = "main"
    result = review(ReviewJob(entry, project), {}, {})
    assert result["status"] == "failed" and "agent-run/" in result["reason"]


def test_oversized_diff_is_refused_before_running(branch_repo):
    """The character cap avoids sending an incomplete patch."""
    project, entry = branch_repo
    config = {"routing": {"agent_run": {"review": {"max_diff_chars": 1}}}}
    result = review(ReviewJob(entry, project), {"codex": FakeRunner()}, config)
    assert result["status"] == "failed" and "split the review" in result["reason"]


def test_disabled_codex_is_skipped():
    """Runner switches apply before automatic selection."""
    agent_switch.set_switch(StateStore(), "codex", "off")
    runners = {"codex": FakeRunner(), "opencode": FakeRunner("opencode")}
    assert choose_reviewer({"runner": "agy"}, "auto", {}, runners) == "opencode"


def test_paused_codex_is_skipped(monkeypatch):
    """A quota pause sends review to the next independent runner."""
    monkeypatch.setattr(review_module.agent_quota, "load_state",
                        lambda path: {"codex": {"paused_until": 1e20}})
    runners = {"codex": FakeRunner(), "opencode": FakeRunner("opencode")}
    assert choose_reviewer({"runner": "agy"}, "auto", {}, runners) == "opencode"


def test_reviewer_edits_are_reverted_by_ask(branch_repo, monkeypatch):
    """Inspect guard results before the temporary branch worktree is removed."""
    project, entry = branch_repo
    original_ask = review_module.agent_ask.ask
    observed = []

    def observe_guard(job, runner):
        """Observe restored contents before review removes its detached worktree."""
        answer = original_ask(job, runner)
        observed.append((job.project / "module.py").read_text(encoding="utf-8"))
        return answer

    monkeypatch.setattr(review_module.agent_ask, "ask", observe_guard)
    result = review(ReviewJob(entry, project), {"codex": FakeRunner(edits=True)}, {})
    assert result["status"] == "failed" and observed == ["VALUE = 2\n"]


def test_model_review_event_is_separate_from_owner_verdict(branch_repo, tmp_path):
    """The model log never records an owner accept or reject decision."""
    project, entry = branch_repo
    review(ReviewJob(entry, project), {"codex": FakeRunner()}, {})
    rows = (tmp_path / "state" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    event = json.loads(rows[-1])
    assert (event["event"], event["branch"], event["reviewer"]) == (
        "agent_run_model_review", entry["branch"], "codex")



def test_report_entry_resolves_by_task_id(branch_repo):
    """A batch report can identify a branch without a separate branch argument."""
    project, entry = branch_repo
    report_path = project / "report.json"
    report_path.write_text(json.dumps({"tasks": [{"id": "test", **entry}]}), encoding="utf-8")
    result = review(ReviewJob(f"{report_path}#test", project), {"codex": FakeRunner()}, {})
    assert result["status"] == "ok" and result["branch"] == entry["branch"]


def test_internal_data_excludes_ineligible_reviewers():
    """Vendor independence cannot override a runner's data ceiling."""
    assert choose_reviewer({"runner": "agy", "sensitivity": "internal"}, "auto", {},
                           {"codex": FakeRunner()}) is None


def test_explicit_same_vendor_can_be_allowed_by_policy():
    """The legacy policy exception applies only to explicit reviewer requests."""
    config = {"routing": {"review": {"require_other_vendor": False}}}
    assert choose_reviewer({"runner": "agy"}, "agy", config,
                           {"agy": FakeRunner("agy")}) == "agy"


def test_findings_are_sorted_high_first():
    """A lower severity finding cannot hide an urgent defect in text output."""
    answer = json.dumps([{**FINDING, "severity": "low"}, FINDING])
    assert parse_findings(answer) == [FINDING, {**FINDING, "severity": "low"}]

