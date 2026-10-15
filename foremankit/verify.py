"""Check a delegated file with a command and let the cheap model fix what fails.

Used by Router.delegate(verify=..., verify_target=...). The caller names ONE file and ONE check command (for example
`python -m pytest tests/test_x.py -q`). The model's answer is written to that file, the command runs, and when it fails
the error output goes back to the same model for another try. The file is only left changed when the check passes.

Safety: the command comes from an agent, so it must start with one of the prefixes allowed in routing.yaml
(`verify.allowed_commands`), it is run WITHOUT a shell, and the target must be a plain file inside the project folder.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

OUTPUT_TAIL_CHARS = 6000          # how much of a failing command's output is sent back to the model
FORBIDDEN_NAMES = {".env", ".git", ".ssh"}


class VerifyError(ValueError):
    """The verify request itself is not acceptable (command not allowed, target outside the project...)."""


def check_target(target: str, project: Path) -> Path:
    """The file to replace: relative, inside `project`, not a hidden/secret file. Returns its absolute path."""
    if not target or Path(target).is_absolute():
        raise VerifyError("verify_target must be a path relative to the project folder")
    path = (project / target).resolve()
    if project.resolve() not in path.parents:
        raise VerifyError(f"verify_target {target!r} is outside the project folder")
    if any(part in FORBIDDEN_NAMES or part.startswith(".env") for part in path.relative_to(project.resolve()).parts):
        raise VerifyError(f"verify_target {target!r} points at a protected file")
    return path


def split_command(command: str, allowed_prefixes: list[str]) -> list[str]:
    """The command as a token list, or VerifyError when it does not start with an allowed prefix.
    Backslashes are turned into slashes first so Windows paths survive shlex."""
    tokens = shlex.split(command.replace("\\", "/"))
    for prefix in allowed_prefixes:
        wanted = shlex.split(prefix.replace("\\", "/"))
        if wanted and [token.lower() for token in tokens[:len(wanted)]] == [word.lower() for word in wanted]:
            return tokens
    raise VerifyError(f"command not allowed: {command!r}. Allowed prefixes (routing.yaml verify.allowed_commands): "
                      + ", ".join(allowed_prefixes))


FENCE_WITH_LANGUAGE = re.compile(r"```[\w+.-]+")     # a line that opens a block: ```python
BARE_FENCE = "```"                                    # a line that closes one (or opens a block without a language)


def line_offsets(lines: list[str]) -> list[int]:
    """The character offset where each line starts, plus one more entry for the end of the text."""
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    return offsets


def fence_positions(lines: list[str]) -> list[tuple[int, bool]]:
    """(line number, True when it opens an inner block) for every fence line; other lines are skipped."""
    positions = []
    for number, line in enumerate(lines):
        stripped = line.strip()
        if FENCE_WITH_LANGUAGE.fullmatch(stripped):
            positions.append((number, True))
        elif stripped == BARE_FENCE:
            positions.append((number, False))
    return positions


def depth_after_fence(depth: int, opens_inner: bool) -> int:
    """The nesting depth after one fence line: a fence opens a block when none is open, a line with a language opens
    an inner block, and a bare fence otherwise closes the innermost open block."""
    if depth == 0 or opens_inner:
        return depth + 1
    return depth - 1


def unclosed_block(lines: list[str], offsets: list[int], start: int) -> tuple[int, str]:
    """The block that never closed: it runs to the LAST bare fence line after `start`, or to the end of the answer."""
    closing = [index for index in range(start + 1, len(lines)) if lines[index].strip() == BARE_FENCE]
    end = closing[-1] if closing else len(lines)
    return offsets[start], "".join(lines[start + 1:end])


def fenced_blocks(answer: str) -> list[tuple[int, str]]:
    """Every top-level fenced code block of `answer` as (offset of its opening line, content).
    Fences may nest: a file that contains markdown inside a string has balanced fence lines of its own, so a line with a
    language (```python) opens an inner block and a bare ``` line closes the innermost open one.
    When the fences do not balance (the file has an odd fence inside a string, or the answer was cut off) the block
    runs from its opening line to the LAST bare fence line, or to the end when there is none."""
    lines = answer.splitlines(keepends=True)
    offsets = line_offsets(lines)
    blocks: list[tuple[int, str]] = []
    depth, start = 0, 0
    for number, opens_inner in fence_positions(lines):
        if depth == 0:
            start = number
        depth = depth_after_fence(depth, opens_inner)
        if depth == 0:
            blocks.append((offsets[start], "".join(lines[start + 1:number])))
    if depth:
        blocks.append(unclosed_block(lines, offsets, start))
    return blocks


def names_target(answer: str, block: tuple[int, str], name: str) -> bool:
    """Whether a fenced block says which file it holds: the name is in its info string (```python x.py), in its
    first line (a comment such as `# file: x.py`) or in a markdown heading directly above it (`### x.py`).
    Other prose before the block does not count, because a model may name one file in a sentence and then
    answer with another block."""
    offset, content = block
    info_string = answer[offset:].split("\n", 1)[0]
    first_line = content.lstrip().split("\n", 1)[0]
    line_above = answer[:offset].rstrip("\n").rsplit("\n", 1)[-1].strip()
    heading_above = line_above if line_above.startswith("#") else ""
    return any(name in label for label in (info_string, first_line, heading_above))


def extract_code(answer: str, target: str) -> str | None:
    """The file content inside the model's answer: the fenced block that names the target file (see names_target),
    else the longest fenced block. None when the answer has no code block."""
    blocks = fenced_blocks(answer)
    if not blocks:
        return None
    name = Path(target).name
    named = [candidate for candidate in blocks if names_target(answer, candidate, name)]
    return max(named or blocks, key=lambda block: len(block[1]))[1].rstrip() + "\n"


def resolve_executable(name: str, project: Path) -> str:
    """The program to start. A path with a slash (`.venv/Scripts/python`) is taken relative to the PROJECT folder, not to
    the folder foremankit was started from, and on Windows the missing `.exe` is added; a bare name is looked up on PATH."""
    if "/" not in name:
        return shutil.which(name) or name
    if Path(name).is_absolute():
        return name
    path = project / name
    if os.name == "nt" and not path.suffix:
        path = path.with_suffix(".exe")
    return str(path)


def run_command(tokens: list[str], project: Path, timeout_s: float) -> tuple[bool, str]:
    """Run the check without a shell. Returns (passed, tail of the combined output).
    Raises VerifyError when the program cannot be started at all: that is no fault of the model's answer."""
    command = [resolve_executable(tokens[0], project), *tokens[1:]]
    try:
        done = subprocess.run(command, cwd=project, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout_s, env=os.environ.copy(), check=False)
    except subprocess.TimeoutExpired:
        return False, f"the check did not finish within {timeout_s:.0f} s"
    except OSError as exc:
        raise VerifyError(f"the check could not be started ({command[0]}): {exc}") from exc
    return done.returncode == 0, (done.stdout + done.stderr)[-OUTPUT_TAIL_CHARS:]


class FileSwap:
    """Temporarily replaces one file. `restore()` brings the original back (or removes a file that did not exist);
    simply not calling it keeps the new content."""

    def __init__(self, path: Path):
        self.path = path
        self.original = path.read_bytes() if path.is_file() else None

    def write(self, code: str) -> None:
        """Replace the file's content with `code` (UTF-8, LF line endings), creating folders as needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(code, encoding="utf-8", newline="\n")

    def restore(self) -> None:
        """Put the original bytes back, or delete the file when it did not exist before."""
        if self.original is None:
            self.path.unlink(missing_ok=True)
        else:
            self.path.write_bytes(self.original)
