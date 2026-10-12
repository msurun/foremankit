"""Preliminary code review by a second model: who reviews, which findings count, what happens to them."""
from __future__ import annotations

import sys

import pytest
from conftest import chat_json

from foremankit import review
from foremankit.formatting import format_result, format_review
from foremankit.records import ReviewJob

GROQ = "https://api.groq.com"
GEMINI = "https://generativelanguage.googleapis.com"
CODE = "def add(first, second):\n    return first + second\n"
AUTHOR_ANSWER = f"```python\n{CODE}```"
GROUNDED_BLOCKER = "BLOCKER | add | fails for None | `return first + second` | guard against None"
INVENTED_BLOCKER = "BLOCKER | add | uses a global | `global counter` | remove it"
MINOR_NOTE = "MINOR | add | no type hints | `def add(first, second):` | add hints"
PYTHON = sys.executable.replace("\\", "/")


# ---------------------------------------------------------------- reading the reviewer's answer

def test_a_finding_with_a_quote_from_the_code_is_grounded():
    [finding] = review.parse_findings(GROUNDED_BLOCKER, CODE)
    assert (finding.severity, finding.where, finding.grounded) == ("blocker", "add", True)
    assert finding.fix == "guard against None"


def test_a_finding_whose_quote_is_not_in_the_code_is_not_grounded():
    [finding] = review.parse_findings(INVENTED_BLOCKER, CODE)
    assert finding.grounded is False


def test_quotes_match_whatever_the_whitespace_and_backticks():
    [finding] = review.parse_findings("MAJOR | add | x | `return    first  +   second` | y", CODE)
    assert finding.grounded


def test_bullets_bold_severity_and_extra_pipes_in_the_fix_are_tolerated():
    answer = "- **Major** | add | problem | `first` | use a | b | c"
    [finding] = review.parse_findings(answer, CODE)
    assert finding.severity == "major" and finding.fix == "use a | b | c"


def test_evidence_that_contains_a_pipe_is_read_as_one_quote():
    code = "text = first | second\n"
    [finding] = review.parse_findings("MAJOR | merge | bitwise or on strings | `first | second` | use +", code)
    assert finding.evidence == "first | second" and finding.fix == "use +" and finding.grounded


def test_no_findings_and_garbage_give_an_empty_list():
    assert review.parse_findings(review.NO_FINDINGS, CODE) == []
    assert review.parse_findings("Looks great!\nBLOCKER but no fields", CODE) == []


def test_only_grounded_findings_of_the_configured_severities_are_sent_back():
    findings = review.parse_findings(f"{GROUNDED_BLOCKER}\n{INVENTED_BLOCKER}\n{MINOR_NOTE}", CODE)
    assert [f.where for f in review.to_fix(findings, ["blocker"])] == ["add"]
    assert len(review.to_fix(findings, ["blocker", "minor"])) == 2


def test_the_fix_request_lists_the_findings_and_the_target():
    findings = review.parse_findings(GROUNDED_BLOCKER, CODE)
    request = review.fix_request(findings, "calc.py")
    assert "guard against None" in request and "complete corrected calc.py" in request


# ---------------------------------------------------------------- the stage inside delegate

@pytest.fixture
def paid_reviewer(monkeypatch, mock):
    """Author = the free Groq; the only reviewer left is the paid Gemini flash (this machine has no local LLM)."""
    monkeypatch.setenv("GROQ_API_KEY", "q")
    monkeypatch.setenv("GEMINI_API_KEY", "g")


def author_and_reviewer(mock, review_answer, *author_answers):
    mock.on(GROQ, "/chat/completions", *(chat_json(a) for a in (author_answers or (AUTHOR_ANSWER,))))
    mock.on(GEMINI, "/chat/completions", chat_json(review_answer))


def calls_to(mock, host):
    return sum(1 for call in mock.calls if call.startswith(host) and call.endswith("chat/completions"))


def test_a_grounded_blocker_goes_back_to_the_author_once(paid_reviewer, make_router, mock):
    fixed_answer = "```python\ndef add(first, second):\n    return (first or 0) + (second or 0)\n```"
    author_and_reviewer(mock, GROUNDED_BLOCKER, AUTHOR_ANSWER, fixed_answer)
    result = make_router().delegate("write add", kind="bulk_code", review=True)
    assert result["review"]["fixed"] is True and result["review"]["reviewer"].startswith("gemini/")
    assert calls_to(mock, GROQ) == 2 and calls_to(mock, GEMINI) == 1
    assert "(first or 0)" in result["preview"]
    assert "blockers sent back to the author" in format_result(result)


def test_minor_findings_are_only_reported(paid_reviewer, make_router, mock):
    author_and_reviewer(mock, MINOR_NOTE)
    result = make_router().delegate("write add", kind="bulk_code", review=True)
    assert result["review"]["fixed"] is False and calls_to(mock, GROQ) == 1
    assert "1 minor" in "\n".join(format_review(result["review"]))


def test_an_invented_blocker_is_reported_as_unverified_and_not_acted_on(paid_reviewer, make_router, mock):
    author_and_reviewer(mock, INVENTED_BLOCKER)
    result = make_router().delegate("write add", kind="bulk_code", review=True)
    assert result["review"]["fixed"] is False and calls_to(mock, GROQ) == 1
    assert "unverified" in "\n".join(format_review(result["review"]))


def test_a_clean_review_costs_the_reviewers_tokens_and_changes_nothing(paid_reviewer, make_router, mock):
    author_and_reviewer(mock, review.NO_FINDINGS)
    result = make_router().delegate("write add", kind="bulk_code", review=True)
    assert result["review"]["findings"] == [] and result["tokens_out"] == 100       # author 50 + reviewer 50
    assert "no findings" in "\n".join(format_review(result["review"]))


def test_the_author_never_reviews_its_own_code(monkeypatch, make_router, mock):
    monkeypatch.setenv("GEMINI_API_KEY", "g")                  # the only source: it writes the code, so it cannot review
    mock.on(GEMINI, "/chat/completions", chat_json(AUTHOR_ANSWER))
    result = make_router().delegate("write add", kind="bulk_code", review=True)
    assert "review" in result and "SKIPPED" in "\n".join(format_review(result["review"]))
    assert calls_to(mock, GEMINI) == 1


def test_a_missing_reviewer_does_not_fail_the_delegation(monkeypatch, make_router, mock):
    monkeypatch.setenv("GROQ_API_KEY", "q")                    # no local LLM, no paid key: nobody can review
    mock.on(GROQ, "/chat/completions", chat_json(AUTHOR_ANSWER))
    result = make_router().delegate("write add", kind="bulk_code", review=True)
    assert result["provider"] == "groq" and result["review"]["skipped"]


def test_free_clouds_never_review(monkeypatch, make_router, mock):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    mock.on(GROQ, "/chat/completions", chat_json(AUTHOR_ANSWER))
    mock.on("https://api.mistral.ai", "/chat/completions", chat_json(review.NO_FINDINGS))
    router = make_router()
    assert router.delegate("write add", kind="quick", review=True)["review"]["skipped"]
    assert "https://api.mistral.ai" not in mock.hosts()


def test_private_code_is_reviewed_only_on_own_hardware(monkeypatch, make_router, mock):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(GEMINI, "/chat/completions", chat_json(GROUNDED_BLOCKER))
    report = make_router().review_code(ReviewJob(CODE, "add.py", "write add", sensitivity="private"))
    assert report["skipped"] and report["findings"] == []
    assert GEMINI not in " ".join(mock.calls)


def test_reviewing_files_costs_are_reported_with_the_reply(paid_reviewer, make_router, mock):
    mock.on(GEMINI, "/chat/completions", chat_json(GROUNDED_BLOCKER))
    report = make_router().review_code(ReviewJob(CODE, "add.py", "write add", sensitivity="normal"))
    assert report["findings"][0].grounded and report["reply"]["provider"] == "gemini"


# ---------------------------------------------------------------- review together with verify

CHECK_PASSES = "```python\nassert 1 + 1 == 2\n```"
CHECK_FAILS = "```python\nassert 1 + 1 == 3\n```"


@pytest.fixture
def project(tmp_path, cfg):
    cfg["routing"]["verify"]["allowed_commands"] = [PYTHON]
    cfg["routing"]["verify"]["conventions"] = False   # these tests are about the review loop, not the clean-code rules
    (tmp_path / "check.py").write_text("print('original')\n", encoding="utf-8")
    return tmp_path


def delegate_with_review(make_router, cfg):
    return make_router(cfg).delegate("write check.py", kind="bulk_code", verify=f"{PYTHON} check.py",
                                     verify_target="check.py", review=True)


def test_the_review_sees_the_checked_file_and_a_fix_is_checked_again(project, paid_reviewer, make_router, mock, cfg):
    reviewer_answer = "BLOCKER | module | weak assertion | `assert 1 + 1 == 2` | assert something real"
    author_and_reviewer(mock, reviewer_answer, CHECK_PASSES, "```python\nassert 2 * 2 == 4\n```")
    result = delegate_with_review(make_router, cfg)
    assert result["review"]["fixed"] and result["verify"]["passed"]
    assert (project / "check.py").read_text(encoding="utf-8") == "assert 2 * 2 == 4\n"
    assert result["verify_before_review"]["passed"]
    review_request = next(b for b in mock.bodies if "Code under review" in b["messages"][-1]["content"])
    assert "assert 1 + 1 == 2" in review_request["messages"][-1]["content"]      # the reviewer saw the checked file


def test_a_fix_that_breaks_the_check_leaves_the_reviewed_version_in_place(project, paid_reviewer, make_router, mock, cfg):
    reviewer_answer = "BLOCKER | module | weak assertion | `assert 1 + 1 == 2` | assert something else"
    author_and_reviewer(mock, reviewer_answer, CHECK_PASSES, CHECK_FAILS, CHECK_FAILS, CHECK_FAILS)
    result = delegate_with_review(make_router, cfg)
    assert result["review"]["fixed"] and not result["verify"]["passed"]
    assert (project / "check.py").read_text(encoding="utf-8") == "assert 1 + 1 == 2\n"      # the version that had passed
