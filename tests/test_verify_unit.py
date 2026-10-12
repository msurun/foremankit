import sys

import pytest

from foremankit import verify


def test_split_command_allowed_prefix_returns_tokens():
    allowed_prefixes = ["python -m pytest"]
    command = "python -m pytest tests/test_x.py -q"
    assert verify.split_command(command, allowed_prefixes) == ["python", "-m", "pytest", "tests/test_x.py", "-q"]


def test_split_command_case_insensitive_match():
    allowed_prefixes = ["Python -M PyTest"]
    command = "python -m pytest tests/test_x.py -q"
    assert verify.split_command(command, allowed_prefixes) == ["python", "-m", "pytest", "tests/test_x.py", "-q"]


def test_split_command_windows_backslashes_turned_into_slashes():
    allowed_prefixes = ["python -m pytest"]
    command = r"python -m pytest tests\test_x.py -q"
    assert verify.split_command(command, allowed_prefixes) == ["python", "-m", "pytest", "tests/test_x.py", "-q"]


def test_split_command_non_allowed_prefix_raises_verify_error():
    allowed_prefixes = ["python -m pytest"]
    command = "python -m unittest tests/test_x.py -q"
    with pytest.raises(verify.VerifyError):
        verify.split_command(command, allowed_prefixes)


def test_split_command_empty_allowed_list_refuses_everything():
    allowed_prefixes = []
    command = "python -m pytest tests/test_x.py -q"
    with pytest.raises(verify.VerifyError):
        verify.split_command(command, allowed_prefixes)


def test_check_target_relative_file_inside_project_is_returned_resolved(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = "file.txt"
    assert verify.check_target(target, project) == project / target


def test_check_target_absolute_path_raises_verify_error(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = "/absolute/path/file.txt"
    with pytest.raises(verify.VerifyError):
        verify.check_target(target, project)


def test_check_target_dot_dot_escape_raises_verify_error(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = "../file.txt"
    with pytest.raises(verify.VerifyError):
        verify.check_target(target, project)


def test_check_target_dot_env_raises_verify_error(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = ".env"
    with pytest.raises(verify.VerifyError):
        verify.check_target(target, project)


def test_check_target_dot_git_config_raises_verify_error(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = ".git/config"
    with pytest.raises(verify.VerifyError):
        verify.check_target(target, project)


def test_check_target_sub_dot_env_local_raises_verify_error(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = "sub/.env.local"
    with pytest.raises(verify.VerifyError):
        verify.check_target(target, project)


def test_extract_code_picks_block_under_heading_naming_target_file(tmp_path):
    target = tmp_path / "file.txt"
    answer = """# file.txt
```python
print("Hello, World!")
```"""
    assert verify.extract_code(answer, str(target)) == 'print("Hello, World!")\n'


def test_extract_code_falls_back_to_longest_block(tmp_path):
    target = tmp_path / "file.txt"
    answer = """```python
print("Hello, World!")
```
```python
print("Hello, World! Again!")
```"""
    assert verify.extract_code(answer, str(target)) == 'print("Hello, World! Again!")\n'


def test_extract_code_no_fenced_block_returns_none(tmp_path):
    target = tmp_path / "file.txt"
    answer = "This is a test."
    assert verify.extract_code(answer, str(target)) is None


def test_extract_code_result_ends_with_exactly_one_newline(tmp_path):
    target = tmp_path / "file.txt"
    answer = """```python
print("Hello, World!")
```"""
    assert verify.extract_code(answer, str(target)) == 'print("Hello, World!")\n'


def test_run_command_exit_0_returns_true_and_output(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    script = project / "script.py"
    script.write_text('print("Hello, World!")')
    tokens = [sys.executable, str(script)]
    passed, output = verify.run_command(tokens, project, timeout_s=1)
    assert passed is True
    assert "Hello, World!" in output


def test_run_command_failing_script_returns_false_and_stderr(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    script = project / "script.py"
    script.write_text('import sys; sys.stderr.write("Error!"); sys.exit(1)')
    tokens = [sys.executable, str(script)]
    passed, output = verify.run_command(tokens, project, timeout_s=1)
    assert passed is False
    assert "Error!" in output


def test_run_command_script_sleeping_longer_than_timeout_returns_false_and_message(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    script = project / "script.py"
    script.write_text('import time; time.sleep(2)')
    tokens = [sys.executable, str(script)]
    passed, output = verify.run_command(tokens, project, timeout_s=1)
    assert passed is False
    assert "did not finish" in output


def test_run_command_missing_executable_raises_verify_error(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    tokens = ["missing_executable"]
    with pytest.raises(verify.VerifyError):
        verify.run_command(tokens, project, timeout_s=1)


def test_file_swap_write_replaces_content(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("Original content")
    swap = verify.FileSwap(path)
    swap.write("New content")
    assert path.read_text() == "New content"


def test_file_swap_restore_brings_original_back(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("Original content")
    swap = verify.FileSwap(path)
    swap.write("New content")
    swap.restore()
    assert path.read_text() == "Original content"


def test_file_swap_restore_removes_file_that_did_not_exist_before(tmp_path):
    path = tmp_path / "file.txt"
    swap = verify.FileSwap(path)
    swap.write("New content")
    swap.restore()
    assert not path.exists()


def test_file_swap_restore_after_write_of_new_nested_path(tmp_path):
    path = tmp_path / "sub" / "file.txt"
    swap = verify.FileSwap(path)
    swap.write("New content")
    swap.restore()
    assert not path.exists()
