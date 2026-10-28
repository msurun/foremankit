"""Tests for check_folder(folder, project) in verify.py."""

import pytest

from foremankit import verify

def test_check_folder_empty_string_returns_project(tmp_path):
    """Empty string -> project."""
    project = tmp_path / "project"
    project.mkdir()
    assert verify.check_folder("", project) == project

def test_check_folder_existing_subfolder_returns_resolved_path(tmp_path):
    """Existing subfolder "ForemanKit" -> resolved path."""
    project = tmp_path / "project"
    project.mkdir()
    subfolder = project / "ForemanKit"
    subfolder.mkdir()
    assert verify.check_folder("ForemanKit", project) == subfolder

def test_check_folder_absolute_path_raises_verify_error(tmp_path):
    """Absolute path raises VerifyError."""
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(verify.VerifyError):
        verify.check_folder("/absolute/path", project)

def test_check_folder_dot_dot_escape_raises_verify_error(tmp_path):
    """Dot-dot escape raises VerifyError."""
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(verify.VerifyError):
        verify.check_folder("../file.txt", project)

def test_check_folder_missing_folder_raises_verify_error(tmp_path):
    """Missing folder raises VerifyError."""
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(verify.VerifyError):
        verify.check_folder("missing", project)

def test_check_folder_dot_git_subfolder_raises_verify_error(tmp_path):
    """Dot-git subfolder raises VerifyError."""
    project = tmp_path / "project"
    project.mkdir()
    subfolder = project / ".git"
    subfolder.mkdir()
    with pytest.raises(verify.VerifyError):
        verify.check_folder(".git", project)


def test_normalize_program_relative_exe(tmp_path):
    """".venv/Scripts/python.exe" -> ".venv/Scripts/python"."""
    assert verify.normalize_program(".venv/Scripts/python.exe", None) == ".venv/Scripts/python"

def test_normalize_program_uppercase_exe(tmp_path):
    """ "PYTHON.EXE" -> "PYTHON"."""
    assert verify.normalize_program("PYTHON.EXE", None) == "PYTHON"

def test_normalize_program_absolute_inside_project(tmp_path):
    """Absolute path inside project becomes relative."""
    abs_path = str(tmp_path / "ForemanKit/.venv/Scripts/python.exe")
    expected = "ForemanKit/.venv/Scripts/python"
    assert verify.normalize_program(abs_path, tmp_path) == expected

def test_normalize_program_absolute_outside_project(tmp_path):
    """Absolute path outside project stays absolute minus .exe."""
    abs_path = "C:/Some/Other/folder/python.exe"
    expected = "C:/Some/Other/folder/python"
    assert verify.normalize_program(abs_path, tmp_path) == expected

def test_normalize_program_no_project(tmp_path):
    """project=None leaves absolute path absolute minus .exe."""
    abs_path = "C:/some/path/python.exe"
    expected = "C:/some/path/python"
    assert verify.normalize_program(abs_path, None) == expected

def test_split_command_relative_prefix(tmp_path):
    """r".venv\\Scripts\\python.exe -m pytest -q tests/x.py" with allowed_prefixes [".venv/Scripts/python -m pytest"]."""
    allowed = [".venv/Scripts/python -m pytest"]
    cmd = r".venv\Scripts\python.exe -m pytest -q tests/x.py"
    expected = [".venv/Scripts/python", "-m", "pytest", "-q", "tests/x.py"]
    assert verify.split_command(cmd, allowed, project=tmp_path) == expected

def test_split_command_absolute_inside_project(tmp_path):
    """Full absolute Windows-style path inside tmp_path matches allowed prefix."""
    allowed = ["ForemanKit/.venv/Scripts/python -m pytest"]
    cmd = f"{tmp_path / 'ForemanKit/.venv/Scripts/python.exe'} -m pytest -q tests/x.py"
    expected = ["ForemanKit/.venv/Scripts/python", "-m", "pytest", "-q", "tests/x.py"]
    assert verify.split_command(cmd, allowed, project=tmp_path) == expected

def test_split_command_absolute_outside_raises(tmp_path):
    """Absolute path outside the project raises VerifyError."""
    allowed = [".venv/Scripts/python -m pytest"]
    cmd = "C:/some/other/folder/python.exe -m pytest -q tests/x.py"
    with pytest.raises(verify.VerifyError):
        verify.split_command(cmd, allowed, project=tmp_path)
