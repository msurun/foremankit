"""The shared coding standard: injected into code prompts, overridable per project, and checked mechanically."""
from __future__ import annotations

import textwrap

import pytest
from conftest import chat_json

from foremankit import codecheck, conventions
from foremankit.cli import main

GROQ = "https://api.groq.com"
CLEAN_FILE = textwrap.dedent('''
    """A tidy module."""


    def add_numbers(first: int, second: int) -> int:
        """Return the sum of the two numbers."""
        return first + second
''')


def rules_broken(source: str, **limits) -> set[str]:
    return {issue.rule for issue in codecheck.check_source(textwrap.dedent(source), "sample.py", limits)}


# ---------------------------------------------------------------- the mechanical check

def test_a_clean_file_has_no_issues():
    assert codecheck.check_source(CLEAN_FILE, "clean.py") == []


def test_a_function_over_the_line_limit_is_reported():
    body = "\n".join(f"    value_{n} = {n}" for n in range(30))
    assert "function-too-long" in rules_broken(f'"""m."""\n\ndef long_one():\n    """Doc."""\n{body}\n')


def test_comments_blank_lines_and_the_docstring_do_not_count_towards_the_length_limit():
    commented = "\n".join(f"    # explains step {n}\n\n    value_{n} = {n}" for n in range(18))
    docstring = '    """Doc.\n\n    More doc.\n    """'
    source = f'"""m."""\n\ndef documented():\n{docstring}\n{commented}\n'
    assert "function-too-long" not in rules_broken(source)


def test_only_code_lines_count_so_21_code_lines_are_still_too_long():
    body = "\n".join(f"    # why\n    value_{n} = {n}" for n in range(21))
    assert "function-too-long" in rules_broken(f'"""m."""\n\ndef long_one():\n    """Doc."""\n{body}\n')


def test_too_many_parameters_are_reported_but_self_does_not_count():
    assert "too-many-parameters" in rules_broken('"""m."""\n\ndef f(a1, b1, c1, d1, e1):\n    """Doc."""\n')
    assert "too-many-parameters" not in rules_broken(
        '"""m."""\n\nclass Box:\n    """Doc."""\n\n    def f(self, one, two, three, four):\n        """Doc."""\n')


def test_deep_nesting_is_reported():
    deep = '''
        """m."""

        def nested(items):
            """Doc."""
            for item in items:
                if item:
                    while item:
                        with open(item) as handle:
                            handle.read()
    '''
    assert "too-deep-nesting" in rules_broken(deep)
    assert "too-deep-nesting" not in rules_broken(deep, max_nesting=4)


def test_nesting_inside_try_and_except_counts():
    source = '''
        """m."""

        def risky(items):
            """Doc."""
            for item in items:
                try:
                    item.run()
                except ValueError:
                    if item.retry:
                        item.run()
    '''
    assert "too-deep-nesting" in rules_broken(source, max_nesting=2)


def test_public_things_need_a_docstring_but_private_and_test_functions_do_not():
    missing = rules_broken("def public_function():\n    return 1\n")
    assert missing == {"missing-docstring"}
    quiet = '"""m."""\n\ndef _helper():\n    return 1\n\ndef test_it_works():\n    assert True\n'
    assert rules_broken(quiet) == set()


def test_a_class_without_a_docstring_is_reported():
    assert "missing-docstring" in rules_broken('"""m."""\n\nclass Thing:\n    pass\n')


def test_a_bare_except_is_reported():
    source = '"""m."""\n\ndef f():\n    """Doc."""\n    try:\n        return 1\n    except:\n        return 2\n'
    assert "bare-except" in rules_broken(source)


def test_one_letter_names_are_reported_except_the_allowed_ones():
    source = '"""m."""\n\ndef f(x):\n    """Doc."""\n    for i in range(x):\n        t = i\n    return t\n'
    names = [issue.message for issue in codecheck.check_source(source, "s.py") if issue.rule == "short-name"]
    assert any("`x`" in m for m in names) and any("`t`" in m for m in names) and not any("`i`" in m for m in names)


def test_a_syntax_error_is_an_issue_not_a_crash():
    issues = codecheck.check_source("def broken(:\n", "bad.py")
    assert [issue.rule for issue in issues] == ["syntax-error"]


def test_limits_come_from_the_routing_config_on_top_of_the_defaults():
    limits = codecheck.limits_from({"conventions": {"limits": {"max_parameters": 9}}})
    assert limits["max_parameters"] == 9 and limits["max_function_lines"] == 20
    assert codecheck.limits_from(None) == codecheck.DEFAULT_LIMITS


def test_an_unreadable_file_is_an_issue(tmp_path):
    issues = codecheck.check_files([str(tmp_path / "missing.py")])
    assert [issue.rule for issue in issues] == ["unreadable"]


# ---------------------------------------------------------------- the command

def test_the_check_command_exits_0_for_a_clean_file_and_1_for_a_dirty_one(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("foremankit.router.load_env", lambda: None)       # never load the real .env into the test process
    (tmp_path / "clean.py").write_text(CLEAN_FILE, encoding="utf-8")
    (tmp_path / "dirty.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    assert main(["check", str(tmp_path / "clean.py")]) == 0
    assert main(["check", str(tmp_path / "dirty.py")]) == 1
    assert "missing-docstring" in capsys.readouterr().out


# ---------------------------------------------------------------- the standard goes into the prompt

def system_prompt_sent(mock) -> str:
    return mock.bodies[-1]["messages"][0]["content"]


def test_code_kinds_get_the_standard_and_other_kinds_do_not(monkeypatch, make_router, mock):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json())
    router = make_router()
    router.delegate("write code", kind="bulk_code")
    assert "Code conventions" in system_prompt_sent(mock)
    router.delegate("quick question", kind="quick")
    assert "Code conventions" not in system_prompt_sent(mock)


def test_a_project_can_override_the_standard(monkeypatch, make_router, mock, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json())
    (tmp_path / ".foremankit").mkdir()
    (tmp_path / ".foremankit" / "conventions.md").write_text("Use tabs. Always.", encoding="utf-8")
    make_router().delegate("write code", kind="bulk_code")
    prompt = system_prompt_sent(mock)
    assert "Use tabs. Always." in prompt and "Code conventions" not in prompt


def test_without_any_standard_file_the_prompt_is_unchanged(cfg):
    cfg["routing"]["conventions"]["file"] = "no-such-file.md"
    assert conventions.load_text(cfg["routing"], conventions.Path("/nonexistent-project")) == ""


@pytest.mark.parametrize("kind, applies", [("bulk_code", True), ("data_script", True), ("hard_code", True),
                                           ("quick", False), ("summarize", False), ("private", False)])
def test_which_kinds_follow_the_standard(cfg, kind, applies):
    assert conventions.conventions_apply(cfg["routing"], kind) is applies


WIDE_FUNCTION = textwrap.dedent('''
    """Module."""


    # codecheck: allow too-many-parameters (the protocol fixes them)
    def wide(first, second, third, fourth, fifth):
        """Take five values."""
        return first
''')


def test_a_rule_can_be_waived_on_purpose_for_one_function():
    assert codecheck.check_source(WIDE_FUNCTION, "wide.py") == []


def test_the_same_function_without_the_waiver_is_reported():
    unwaived = WIDE_FUNCTION.replace("# codecheck: allow too-many-parameters (the protocol fixes them)", "")
    assert [issue.rule for issue in codecheck.check_source(unwaived, "wide.py")] == ["too-many-parameters"]


def test_a_waiver_without_a_reason_is_still_reported():
    without_reason = WIDE_FUNCTION.replace("# codecheck: allow too-many-parameters (the protocol fixes them)",
                                           "# codecheck: allow too-many-parameters")
    assert [issue.rule for issue in codecheck.check_source(without_reason, "wide.py")] == ["too-many-parameters"]


def test_a_waiver_for_another_rule_is_still_reported():
    body = "\n".join(f"    step_{number} = {number}" for number in range(25))
    long_function = ('"""Module."""\n\n\n# codecheck: allow function-too-long (the protocol fixes them)\n'
                     f'def long_one(value):\n    """Take one value."""\n{body}\n    return value\n')
    assert [issue.rule for issue in codecheck.check_source(long_function, "long.py")] == ["function-too-long"]
