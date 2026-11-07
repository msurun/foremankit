# Codex runner verification

Measured on 2026-11-06 with the installed VS Code extension binary,
`codex-cli 0.164.0-alpha.11.1`, using the owner's existing subscription login (`codex login status`: `Logged in using ChatGPT`).
Both `codex --help` and `codex exec --help` were read before choosing flags.
The [official non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode)
also describes `exec --json`.

The runner invokes `codex --ask-for-approval never exec --ignore-user-config
--sandbox workspace-write --json --cd <folder> --model gpt-6.1-sol -- <prompt>`.
The approval option belongs before `exec` in this installed CLI. `exec` is headless;
`--json` emits JSONL; `--cd` sets the workspace root; `workspace-write` bounds edits.
Personal config is ignored so extra writable roots and configured tools cannot
widen the invocation. Authentication still uses the existing Codex home.
Two configuration overrides exclude TMPDIR and `/tmp` from writable roots;
Windows also selects `windows.sandbox="elevated"` for the native sandbox.
No sandbox or approval bypass is used. The inherited `CODEX_PERMISSION_PROFILE`
is removed, since ForemanKit may itself run inside an unrestricted Codex session.

Every process receives `stdin=DEVNULL`, `cwd=folder`, and `PWD=folder`.
The installed CLI explicitly appends piped stdin to a supplied prompt: leaving
stdin open can both block the run and add unrelated input. It logged
`Reading additional input from stdin...` with DEVNULL, but reached EOF immediately.
The PowerShell shell-snapshot warning did not prevent command execution.

The first synthetic run took **8.81 seconds**, exited 0 and read `answer.txt`
containing `42` through one read-only PowerShell command. The untouched stdout is
stored in `tests/fixtures/codex_exec.jsonl`. Its exact sequence is:

1. `thread.started` with `thread_id`.
2. `turn.started`.
3. `item.completed` with an `agent_message` commentary (`item_0`, `text`).
4. `item.started` with `command_execution` (`item_1`, `command`, empty
   `aggregated_output`, null `exit_code`, `status: in_progress`).
5. `item.completed` with that same command ID, output `42\r\n`, exit code 0
   and `status: completed`.
6. `item.completed` with the final `agent_message` (`item_2`,
   `text: "42\nNo files changed."`).
7. `turn.completed` with `usage: {input_tokens: 29444,
   cached_input_tokens: 27008, cache_write_input_tokens: 0, output_tokens: 63,
   reasoning_output_tokens: 0}`.

Parsing takes the last completed agent message, preserving full answer whitespace;
the coding-run reply flattens it to one line. Tool IDs are counted once across
lifecycle events, and completed-turn counters are summed. Missing or invalid
token fields stay unknown. Tests replay the capture without live calls.

The end-to-end command `foremankit agent-ask --task-file <question.txt>
--project <scratch-git-repo> --runner codex --json` also exited 0 in **7.9 seconds**.
It returned `ok: true`, `answer: "42"`, `tokens_in: 29487`, `tokens_out: 57`,
`tool_calls: 1`, `thinking_tokens: 0`, `cache_read_tokens: 27008`, `turns: 1`,
`modified_files: []` and an empty error. An independent `git status --porcelain`
afterward was empty. The question file lived outside the project.

The VS Code webview was not opened interactively; automated checks verify its
third row, status rendering, toggle wiring and JavaScript syntax.

Final validation from the isolated worktree, with `PYTHONPATH` set to that
worktree and the main repository's shared `.venv/Scripts/python`:

- `python -m pytest tests -q`: **1788 passed in 140.00s**.
- `python -m ruff check <all touched Python files>`: **All checks passed**.
- `node --check vscode-extension/extension.js`: exited 0.
- `git diff --check`: clean.
- `graphify update .`: regenerated the local AST graph without API calls.
