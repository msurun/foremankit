# Handoff: `foremankit agent-run` (state 2026-11-07, UPDATED)

**Update 2026-11-07 (late): Codex is the third runner and the fallback logic exists.** Codex CLI (ChatGPT subscription) is a runner (`--runner codex`, `runner: codex`), may receive `internal` data (`routing.yaml agent_run.codex.max_sensitivity`), the binary is a glob (newest VS Code extension version). Automatic order for `normal`: opencode, then codex (big task or 2 opencode failures), then agy (codex failed twice or off); `internal`: codex, then agy. A usage limit is TEMPORARY: backoff 30 min/1 h/2 h/5 h or the reset time parsed from the message, one probe run after the pause, `auto` tasks are rerouted, pinned/internal tasks become `waiting` (`retry_after`), `agent-run-batch --resume REPORT` skips the passed ones; panel shows the pause and has "Try now". agy daily limit 15 is only SOFT (display). Limit detection ignores a failed run that changed files (a quota task whose check fails is not a limit). The scope guard now reverts nested repositories (sandboxed pytest leaves `pytest-of-*` folders in the worktree). Suite: 1886 passed. All of it is unpushed. The order is a hypothesis from small samples: judge by `clean_rate`, and measure the real Codex limit. The quota file is `%TEMP%/foremankit-agent-run/quota-state.json` (shared, NOT under FOREMANKIT_STATE_DIR: do not write to it from experiments).

**Current state (read this first).** Built, committed locally on main (NOT pushed; 45 commits ahead of origin):
agent-run (`--base`), `agent_quota` (slots, daily limit, cooldown), `agent_plan`, `agent_batch` (`foremankit agent-run-batch`), `agent_merge` (`foremankit agent-merge`), `agent_review_log` (clean-merge rate), `agent_switch` (panel on/off: master + agy + opencode, enforced in agent_run/batch/auto choice), `claude_usage` (`claude -p /usage`, baseline + delta), `panel_agent_runs` + "Agent runs" section in `vscode-extension/extension.js` (reload the VS Code window to see it). Full suite 1711 passed.
Key fix: runners start with `stdin=DEVNULL` (opencode hung forever on an inherited open pipe when run from a background process). `agent_lock` no longer deletes a live lock whose PID is still being written.
Dogfood record (reviewed by the lead): opencode wrote agent_quota, agent_review_log, agent_switch, claude_usage, panel_agent_runs, README/nudge (clean, fast: 17-80 s); agy wrote agent_batch, agent_merge, the panel UI (about 10 min each; 3 real defects in batch/merge, all fixed). Failed/empty runs left branches `agent-run/35a78f6f aa96bb30 bc7dd927 7ebe6da9 f6bf442c` and worktrees in %TEMP%/foremankit-agent-run: delete when the owner agrees (a hook blocks `git branch -D`).
Open: (1) `agent-ask` / `agent-ask-batch` (read-only question mode for the Crosscut graphify A/B; owner agreed 2026-11-07; spec below), (2) push when the owner agrees, (3) the Claude login expires 7-10 Nov (`claude auth login`), (4) measure clean_rate per runner over a week, (5) agy quota 15/day is a guess.

**agent-ask spec (Crosscut request):** `agent-ask --task-file F --project <path> --runner agy|opencode --json`, batch `agent-ask-batch --tasks-dir D --out-dir O --runner R`; runs IN the given path (no worktree); per task JSON: answer, tokens_in/out (null if not reported), tool_calls (or null), seconds, modified_files (git status after; any change = FAIL + revert), raw_path. To check: can agy --sandbox / opencode run `graphify.exe` (C:/Users/regiz/.local/bin), does opencode report token usage.

---- older text below (design and measurements of 2026-11-05/06) ----


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

- opencode 2.2.0 CLI at `~/AppData/Local/Programs/ai.opencode.desktop/resources/opencode-cli.exe` (settings: routing.yaml `agent_run.opencode`, default model `opencode/deepseek-v4.1-flash-free`). `OpenCodeRunner` (`opencode run --auto --model M --format json`, `normal` data only) ran live 2026-11-06 with owner approval of `--auto`: trivial task PASS in 17.7 s, scope-override task PASS in 14.7 s. The scope guard reverted the stray `__pycache__/*.pyc` the agent's own pytest run created (first LIVE revert); the agent again refused the out-of-scope edits. Pass rate of the free model over more runs is still unmeasured.
- Owner: agree to push (`git push` not done), and re-confirm `--dangerously-skip-permissions` for multi-file runs (larger blast radius than `supervise`; worktree limits file damage, not command damage).
- First real task: `docs/examples/task-crosscut-policy-editor-ui.md` in `CrosscutProcessSuite`; needs `npx vitest run` / `tsc` / `eslint` in `verify.allowed_commands` with `--check-cwd frontend`, and `node_modules` (not in the worktree; a junction was NOT added because removing a worktree through a junction can delete the target's files).
- Not built: MCP tool `agent_run` / non-blocking job id (use the CLI with `run_in_background`), `agent_run` pass-rate view by model (events are logged), 17 pre-existing test failures above.
- Review stays mandatory: every Antigravity slice so far had real defects. Task files must point at the real source files for contracts and list edge cases.
- Unrelated uncommitted work in `foremankit/openai_bridge.py` and `tests/test_openai_bridge.py` was left alone.

## Lessons from using Antigravity in the Crosscut project (2026-11-06, sent by the Crosscut session)

1. Quality: every Antigravity slice had real defects found by independent review. Frontends were written against a guessed backend contract; safety-critical code missed the core safety points. Its adversarial TEST files were the best output (found two real bugs in Claude-written code).
2. Cost (estimate, not measured): the worker saved implementation tokens but review (Opus ~106k tokens + foremankit reviews ~0.1 USD) and broad fix agents (~150-250k each) ate most of it; net saving probably 20-40 percent, uncertain. Review is needed for Claude-written code too, so the real extra cost is fix volume and after-the-fact design decisions.
3. Worth it for: UI slices with a real contract, test files (edge-case lists), docs and deployment files, mechanical edits. Not for: gateway/auth/safety-critical logic where the hard part is a decision.
4. agent-run design implications: task files must POINT at real source files for every contract ("read them first"), list edge cases, forbid commits (Antigravity committed to main with the owner's identity when run freely; keep the worktree+branch design) and edits outside scope; print REVIEW REQUIRED for gateway/auth/policy/tuning/supervisor paths with a second-model review mandatory; frontend checks need node_modules (not in the worktree); gemini refuses "security/vulnerability review" prompts but accepts "code review of logic errors and missing edge cases"; foremankit delegate review on a diff passed via context_files is a cheap second opinion (~0.05 USD, about 3 of 10 findings real, some wrong): treat as leads; run checks under WSL/Linux when tests touch permissions/paths (Ubuntu CI failed 10+ pushes on POSIX key-file modes that passed on Windows).
