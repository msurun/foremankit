"""Tests for hooks/shell_words.py, the shell command splitter shared by the guard hooks.

Each test feeds a command line an AI agent might type (bash or PowerShell) and checks the exact
simple commands the guard hooks will inspect. Expected values are written as plain lists of tokens.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from shell_words import command_name, simple_commands


def test_and_operator_splits_two_commands_and_keeps_quoted_flag():
    assert simple_commands('git commit -m "a --no-verify" && git push') == [
        ["git", "commit", "-m", "a --no-verify"],
        ["git", "push"],
    ]


def test_heredoc_body_is_dropped_from_the_command_line():
    command = "git commit -F - <<EOF\ngit push --force\nEOF"
    assert simple_commands(command) == [["git", "commit", "-F", "-"]]


def test_quoted_heredoc_tag_ends_the_body_and_later_commands_survive():
    command = "cat <<'EOF'\nbody line\nEOF\ngit status"
    assert simple_commands(command) == [["cat"], ["git", "status"]]


def test_unterminated_heredoc_swallows_the_rest_of_the_input():
    command = "cat <<EOF\nrm -rf x\ngit push --force"
    assert simple_commands(command) == [["cat"]]


def test_assignment_and_quoted_semicolon_and_pipe():
    assert simple_commands("FOO=1 git push --force; echo 'a;b' | cat") == [
        ["git", "push", "--force"],
        ["echo", "a;b"],
        ["cat"],
    ]


def test_or_and_background_operators_split_three_commands():
    assert simple_commands("a || b & c") == [["a"], ["b"], ["c"]]


def test_newlines_separate_commands():
    assert simple_commands("git status\ngit push") == [["git", "status"], ["git", "push"]]


def test_backslash_newline_continuation_joins_lines():
    assert simple_commands("git push \\\n--force") == [["git", "push", "--force"]]


def test_operators_inside_double_quotes_do_not_split():
    assert simple_commands('echo "a && b"') == [["echo", "a && b"]]


def test_sudo_prefix_is_removed():
    assert simple_commands("sudo rm -rf x") == [["rm", "-rf", "x"]]


def test_env_with_assignment_is_removed():
    assert simple_commands("env FOO=1 python x.py") == [["python", "x.py"]]


def test_powershell_command_wrapper_is_unwrapped():
    assert simple_commands('powershell -Command "git reset --hard"') == [
        ["git", "reset", "--hard"]
    ]


def test_bash_c_wrapper_is_unwrapped_and_its_operators_split():
    assert simple_commands('bash -c "git push --force && ls"') == [
        ["git", "push", "--force"],
        ["ls"],
    ]


def test_cmd_c_wrapper_is_unwrapped():
    assert simple_commands('cmd /c "dir"') == [["dir"]]


def test_wrapper_nested_deeper_than_max_depth_stops_unwrapping():
    nested = 'bash -c "' + 'bash -c "' * 4 + "git push" + '"' * 5
    result = simple_commands(nested)
    assert result
    assert result[0][0] in ("bash", "git")


def test_windows_path_keeps_its_separators():
    assert simple_commands("rm -rf C:\\Users\\x\\tmp") == [["rm", "-rf", "C:/Users/x/tmp"]]


def test_unbalanced_quote_does_not_raise_and_returns_one_command():
    result = simple_commands('echo "abc')
    assert len(result) == 1
    assert result[0][0] == "echo"


def test_empty_string_returns_no_commands():
    assert simple_commands("") == []


def test_whitespace_only_returns_no_commands():
    assert simple_commands("   \n\t ") == []


def test_command_name_lowercases_basename_and_strips_exe():
    assert command_name(["C:/x/Git.exe", "a"]) == "git"


def test_command_name_keeps_unix_basename():
    assert command_name(["/usr/bin/rm"]) == "rm"


def test_command_name_of_empty_list_is_empty_string():
    assert command_name([]) == ""


def test_command_name_lowercases_plain_word():
    assert command_name(["GIT"]) == "git"
