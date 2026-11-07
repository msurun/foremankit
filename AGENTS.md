# Instructions for coding agents (Codex, opencode, Antigravity)

This file is read automatically by Codex CLI and opencode when they work in this repository. It is short on purpose: the
full coding standard is `config/conventions.md`. Read that file before you write code. It applies to every file you
create or change, tests included.

## The rules that matter most (details in config/conventions.md)
- Functions: at most 20 lines of code, 4 parameters, 3 levels of nesting; one job each. Group parameters in a small
  frozen dataclass instead of adding a fifth.
- Names say what a thing is (`remaining_budget_usd`, not `rb`); no one-letter names except `i`, `j`, `k`.
- Every file starts with a 2-4 line docstring (what it does and why it exists); every public function and class has a
  docstring.
- Comments teach: explain the WHY, the units, the protocol quirk, the reason for a workaround. Put a plain-words comment
  above logic that is hard to follow. Never restate what the line says and never leave commented-out code.
- Raise specific exceptions with a message that says what to do; never swallow an error; no bare `except:`.
- Constants for meaningful numbers, with the unit in the name or the comment (`TIMEOUT_S = 30`).
- Tests: one behaviour per test, the name is a sentence, no loops or conditions inside a test.

## Check your work before you finish
- `.venv\Scripts\python -m ruff check <files>`
- `.venv\Scripts\foremankit check <python files you wrote>`: the mechanical part of the standard (function size,
  parameters, nesting, names, docstrings). It must report no issues; a rule may be waived only with
  `# codecheck: allow <rule> (<reason>)` and a real reason.
- Run the tests named in your task.

## Working rules for this repository
- Only edit the files your task lists. Do not commit, do not switch branches, do not push.
- Read the real source files your task points at before you write code that depends on them; never guess a contract.
- Do not reformat or rewrite code you were not asked to touch.
- Runners start with stdin closed; do not wait for interactive input.
