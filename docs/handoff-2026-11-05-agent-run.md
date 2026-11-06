# Handoff: `foremankit agent-run` (state 2026-11-06)

`agent-run` is built and committed (94ce42c + the follow-up commit with `tests/test_agent_run.py`). Not pushed.

## What it does

`foremankit agent-run --task-file F --project <repo root> --scope GLOB [--scope ...] --check "<allowed command>" [--runner agy] [--model alias] [--sensitivity normal|internal] [--timeout 1800] [--check-cwd DIR] [--keep-worktree] [--json]`

1. Refuses `private` data, an empty scope or task, and a runner that may not see the data (`internal` only for an eligible runner, same rule as `supervise`).
2. `git worktree` of `--project` (must be the repo root) on `agent-run/<id>` in `%TEMP%/foremankit-agent-run/` (outside OneDrive). The main working directory is never touched (uncommitted changes in the main checkout are NOT in the worktree: it starts from HEAD).
3. Runs the runner headless (`AgyRunner`: `agy -p ... --sandbox --dangerously-skip-permissions`, model alias `flash_high`). Tasks above ~22 500 chars go to `.agent-task.md` in the worktree (removed before the commit).
4. Scope guard: everything is staged and compared with the base commit (an agent that commits on its own cannot hide changes). Changes outside `--scope` are reverted and reported; modified/deleted TRACKED files outside scope make the run FAIL; new files outside scope are only reverted and reported.
5. Our own re-run of the check through `verify.py` (never the agent's PASS), with `PYTHONPATH` = worktree, `worktree/src`, `worktree/*/src`; the venv program is looked up in the worktree, the project and the project's parent (`locate_program`).
6. Commits the in-scope patch on the branch, removes the worktree (unless `--keep-worktree`), prints a 5-line report (or `--json`): verdict, branch, files, diff --stat, check, reverted paths, reply, `REVIEW REQUIRED` when a changed path matches `routing.yaml agent_run.review_paths`. No change = FAIL. Exit 0 PASS / 1 FAIL / 2 refused.
7. Logs an `outcome` event (kind `agent_run`) so stats and pass rates include it.

Code: `agent_run.py` (orchestration, report), `agent_git.py`, `agent_scope.py` (globs), `agent_runner.py` (`Runner` protocol, `AgyRunner`, `make_runner`). Tests: `tests/test_agent_run.py`, `tests/test_agent_scope.py` (fake runner, no network).

## Measured 2026-11-05/06

- Editable-install trap is real: in a worktree the shared venv's editable install resolves the MAIN checkout, plain `pytest` passed on the wrong code. `PYTHONPATH` prepended with the worktree fixes it (probe test `test_check_runs_the_worktree_code_not_the_main_checkout`).
- Live agy runs in a scratch repo (4 runs, all PASS, main tree and `git worktree list` clean afterwards): trivial task 180.7 s (just over the 3-minute target; agy start-up dominates), 29 480-char task via file indirection 182 s, two tasks that ordered edits outside scope 185 to 222 s. In both of those the agent OBEYED the scope rule in the prompt and refused the extra edits, so the revert path was exercised only by unit tests (fake runner), not live.
- Suite: 1424 passed, 17 failed. The 17 (8 `test_delegation.py`, 8 `test_router.py`, 1 `test_panel.py`) fail identically at 35af60c: they come from fc651c3 (split paid delegation into deepseek/gemini; `category_of` reads `provider.name` that the test fakes lack, router tests expect the old categories). Not caused by agent-run; not fixed here.

## Open

- Owner: install opencode, then add `OpenCodeRunner` (check `opencode run --help`; same members as `AgyRunner`; free models only for `normal`). `make_runner('opencode')` raises today.
- Owner: agree to push (`git push` not done), and re-confirm `--dangerously-skip-permissions` for multi-file runs (larger blast radius than `supervise`; worktree limits file damage, not command damage).
- First real task: `docs/examples/task-crosscut-policy-editor-ui.md` in `CrosscutProcessSuite`; needs `npx vitest run` / `tsc` / `eslint` in `verify.allowed_commands` with `--check-cwd frontend`, and `node_modules` (not in the worktree; a junction was NOT added because removing a worktree through a junction can delete the target's files).
- Not built: MCP tool `agent_run` / non-blocking job id (use the CLI with `run_in_background`), `agent_run` pass-rate view by model (events are logged), 17 pre-existing test failures above.
- Review stays mandatory: every Antigravity slice so far had real defects. Task files must point at the real source files for contracts and list edge cases.
- Unrelated uncommitted work in `foremankit/openai_bridge.py` and `tests/test_openai_bridge.py` was left alone.
