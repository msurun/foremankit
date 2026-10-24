"""Split a shell command line (bash or PowerShell, as typed by an AI agent) into simple commands and tokens.

Used by the guard hooks in this folder, so that they inspect real command words and flags and are not fooled by
text inside quotes or heredoc bodies. Standard library only; a hook must never fail, so every error path degrades.
"""
import re
import shlex

OPERATORS = ("&&", "||", ";", "|", "&", "\n")
HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
WINDOWS_PATH = re.compile(r"[A-Za-z]:\\|\\\\")
WRAPPER_FLAGS = {"cmd": ("/c", "/k"), "powershell": ("-command", "-c"), "pwsh": ("-command", "-c"),
                 "bash": ("-c",), "sh": ("-c",)}
PREFIX_WORDS = ("sudo", "env", "command")
MAX_DEPTH = 3


def command_name(tokens: list[str]) -> str:
    """Lowercase basename of the first token without a `.exe` suffix; "" for an empty list."""
    if not tokens:
        return ""
    name = re.split(r"[\\/]", tokens[0])[-1].lower()
    return name[:-4] if name.endswith(".exe") else name


def simple_commands(command: str, depth: int = 0) -> list[list[str]]:
    """The simple commands of `command`, each as a token list; quoted text stays one token.

    Heredoc bodies are dropped. Operators `&& || ; | &` and newlines split only outside quotes. Leading `VAR=x`,
    `sudo`/`env`/`command` words are removed, and `cmd /c`, `powershell -Command`, `bash -c` wrappers are replaced
    by the command string they run (up to MAX_DEPTH levels)."""
    text = _remove_heredocs(command.replace("\\\n", ""))
    result: list[list[str]] = []
    for segment in _split_operators(text):
        tokens = _strip_prefixes(_tokenize(segment))
        result.extend(_unwrap(tokens, depth))
    return result


def _remove_heredocs(text: str) -> str:
    """Cut `<<TAG` markers and every line up to the terminator; a missing terminator swallows the rest."""
    kept: list[str] = []
    tag = None
    for line in text.split("\n"):
        if tag is not None:
            tag = None if line.strip() == tag else tag
            continue
        match = HEREDOC.search(line)
        tag = match.group(2) if match else None
        kept.append(HEREDOC.sub("", line))
    return "\n".join(kept)


def _operator_at(text: str, index: int) -> str:
    """The operator that starts at `index`, or "" (two-character operators win over one-character ones)."""
    for operator in OPERATORS:
        if text.startswith(operator, index):
            return operator
    return ""


def _split_operators(text: str) -> list[str]:
    """Split at operators that are outside single and double quotes."""
    parts: list[str] = []
    current: list[str] = []
    quote = ""
    index = 0
    while index < len(text):
        char = text[index]
        operator = "" if quote else _operator_at(text, index)
        if operator:
            parts.append("".join(current))
            current, index = [], index + len(operator)
            continue
        quote = _next_quote(quote, char)
        current.append(char)
        index += 1
    parts.append("".join(current))
    return [part for part in parts if part.strip()]


def _next_quote(quote: str, char: str) -> str:
    """The quote state after `char`: "" outside, otherwise the opening quote character."""
    if quote:
        return "" if char == quote else quote
    return char if char in "'\"" else ""


def _tokenize(segment: str) -> list[str]:
    """Tokens of one simple command; Windows paths keep their separators, a broken quote falls back to split()."""
    if WINDOWS_PATH.search(segment):
        segment = re.sub(r"\\(?!['\"])", "/", segment)
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _strip_prefixes(tokens: list[str]) -> list[str]:
    """Remove leading `VAR=value` assignments and sudo/env/command words."""
    index = 0
    while index < len(tokens) and (re.match(r"[A-Za-z_]\w*=", tokens[index]) or tokens[index] in PREFIX_WORDS):
        index += 1
    return tokens[index:]


def _unwrap(tokens: list[str], depth: int) -> list[list[str]]:
    """The commands behind a `cmd /c "..."`-style wrapper, or the tokens themselves when there is none."""
    flags = WRAPPER_FLAGS.get(command_name(tokens), ())
    if not tokens:
        return []
    if depth < MAX_DEPTH and len(tokens) > 2 and tokens[1].lower() in flags:
        return simple_commands(" ".join(tokens[2:]), depth + 1)
    return [tokens]
