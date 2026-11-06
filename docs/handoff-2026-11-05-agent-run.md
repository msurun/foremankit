# Handoff 2026-11-05: `foremankit agent-run` (multi-file agentic tasks on Antigravity / opencode)

Written for a NEW session that works only on ForemanKit. Nothing of this is implemented yet. It comes out of a long Crosscut session where the owner pasted prompts into Antigravity by hand and wants that automated.

## Why

During the Crosscut night run (2026-11-02 to 11-05) the lead (Claude) wrote complete prompts for Antigravity (Gemini Flash), the owner pasted them into the Antigravity app, and the lead then reviewed the result. Three slices were produced this way (write-policy validator, L5 test matrix, role/auth frontend, adversarial gateway tests, deployment runbook). Each needed a review pass: an Opus review found real defects in every slice (vacuous fuzz tests, wrong CLI flags in docs, a frontend written against a guessed API, edge cases). The adversarial tests written by Antigravity also found two real bugs in Claude-written code. Conclusion: the worker is useful and cheap, the review must stay, the copy-paste should go.

## What exists (verified 2026-11-05)

- `agy.exe` at `C:\Users\regiz\.gemini\bin\agy.exe` (not on PATH). Headless use: `agy -p "<prompt>" --model <id> --output-format json|text|stream-json`. Useful flags from `agy --help`: `--sandbox`, `--dangerously-skip-permissions`, `--mode accept-edits|auto|plan`, `--add-dir` (repeatable), `--print-timeout`, `--effort low..max`, `--continue`, `--conversation <id>`, `--json-schema`, `--settings` (inline JSON or file), `--agent`, `--project`, `--log-file`. Subcommands: `acp` (Agent Client Protocol stdio server), `models`, `mcp`, `plugin`, `agents`.
- `foremankit/agy_provider.py`: provider type `agy`, one prompt in, text out, `MAX_PROMPT_CHARS = 24000` (Windows command line limit, no stdin in agy). Used by `delegate` (text only, one file per call).
- `foremankit/supervise.py`: `foremankit supervise --task-file F --target T --check C [--sensitivity internal]`. agy flash_high with `--sandbox --dangerously-skip-permissions` writes or fixes ONE file and runs the check; ForemanKit re-runs the check itself through `verify.py` and never trusts agy's PASS. Task file is `.agent-task.md`, timeouts 900 s agent / 300 s check.
- `foremankit/supervisor_choice.py`: `foremankit supervisor auto|agy|haiku`, picks agy vs Haiku supervisor from task features.
- `foremankit/verify.py`: allowed check commands only (`config/routing.yaml` `verify.allowed_commands`: pytest, ruff check, foremankit check, with the `ForemanKit/.venv/...` variants). `python -c` and `verify` without `verify_target` are refused (this broke `bulk-supervisor` twice in Crosscut).
- Memory notes: `project-foremankit-agy-provider-2026-10-31.md` (provider, risks accepted by the owner, quota unknown, ToS grey area), `project-foremankit-state-2026-10-13.md`.
- `opencode`: NOT installed on this PC (not in PATH or global npm). Owner has an account with free models. Not verified here; from general knowledge it has a non-interactive `opencode run "<prompt>"`, check `opencode run --help` once installed.

## The gap

`delegate` returns text for ONE file. `supervise` handles ONE file. The Crosscut slices are multi-file agentic jobs: read the repo, create 4 to 8 new files, edit locale files, run vitest/pytest/ruff/tsc/eslint, report. Today the owner copies the prompt into the Antigravity app by hand.

## Proposed feature: `foremankit agent-run`

```
foremankit agent-run --task-file docs/examples/task.md --project <repo> \
    --scope "gateway/tests/test_adv_*.py" --scope "schemas/foo.json" \
    --check "ForemanKit/.venv/Scripts/python -m pytest gateway/tests -q" \
    [--runner agy|opencode] [--model M] [--sensitivity normal|internal] [--timeout 1800]
```

Behaviour:
1. Refuse `private` data. `internal` only if the runner is eligible (same rule as `supervise`).
2. Create a git worktree of `--project` on a temp branch (`agent-run/<id>`), so `--dangerously-skip-permissions` never touches the owner's working directory or concurrent sessions. Share or link `.venv` and `node_modules` (or run checks from the main checkout against the worktree; decide and measure: the Crosscut frontend needs `node_modules`, the Python side needs the repo `.venv` with editable installs that point to the main checkout, which is a trap: editable installs resolve `src/` of the MAIN checkout, so tests in the worktree could silently test the wrong code. Verify with a probe test or set `PYTHONPATH` to the worktree `src` dirs).
3. Run the runner headless with the prompt taken from the task file. Prompts above 24000 characters: write the full prompt to `.agent-task.md` in the worktree and send a short prompt that says "read .agent-task.md and do exactly that".
4. After the run: compute `git diff --name-status` against the base. Files outside `--scope` are reverted and reported (today scope is only text in the prompt). New files in the scope are allowed. Deleting or modifying tracked files outside scope is a failure.
5. Re-run the check commands itself through `verify.py`. Never trust the agent's own PASS.
6. Output: short report (JSON + 5-line text): branch, files changed, `diff --stat`, check result, runner reply (first 200 chars), seconds, cost/quota note. The patch stays on the branch; the lead reviews and merges (`git merge --no-ff` or `git diff | git apply`). The worktree is removed after merge or on request.
7. Logging into the existing call log so `router_status` and pass rates (`passrates.py`) include agent-run results by task kind.

Runner abstraction: `Runner` protocol with `run(prompt_file, cwd, timeout) -> RunnerResult`. `AgyRunner` first (reuse flags from `supervise.py`). `OpenCodeRunner` second, once the owner has installed it: model list from `opencode models`, free models only for `normal` data, pass rate tracked per model.

MCP and hooks: expose as an MCP tool `agent_run` next to `delegate`, or as a CLI that the nudge hook tells the lead to run (as `supervise` already is). The MCP route is preferable if the call can be non-blocking (jobs take 5 to 25 minutes): return a job id, poll with `agent_run_status`, notify on completion. Claude Code has background-task notifications; a CLI started with `run_in_background` already gets that for free, so CLI first, MCP later.

## Risks and decisions to confirm with the owner

- `--dangerously-skip-permissions` lets the agent run any command on this PC. The worktree limits file damage, not command damage (network, other folders). Accepted by the owner for `supervise` on 2026-10-31; ask again for multi-file runs (larger blast radius). Consider running inside WSL or a Windows Sandbox later.
- Antigravity terms section 6 (third-party tool access) is a grey area; the Google AI Pro quota in agy is unknown (reported low). Measure over the first 10 runs and revert if the pass rate or the quota is bad (`enabled: false` in providers.yaml).
- Free-model providers train on data: only `normal` (synthetic, public) content. Security-critical code (Write Gateway) may be given to a worker only with a mandatory independent review (Opus advisor) before merge.
- Review is not optional. Measured on 2026-11-02..05: every Antigravity slice had real defects. Add to the report a reminder line when the touched paths match a security glob list in routing.yaml (`gateway/`, `auth`, `policy`), and make `agent-run` print "REVIEW REQUIRED".
- Prompt quality matters more than the model: the slice that failed worst was written against an assumed backend contract (the lead gave a wrong status code and payload shape). Rule for task files: state the contract by pointing at the real source files and tell the worker to read them first; list edge cases explicitly; forbid editing files outside scope.
- Concurrency: several agent-runs in parallel on disjoint scopes must use separate worktrees and must not share a database or port (Crosscut e2e starts servers on free ports; fine).

## Acceptance for the first version

1. `agent-run` on a trivial task (create one test file in a scratch repo) passes end to end in under 3 minutes, patch on a branch, main working directory untouched.
2. Out-of-scope edit is reverted and reported (test with a task that tells the agent to also edit README).
3. Failing check is reported as FAIL with the output tail; the branch is kept.
4. A 30000-character task file works through the file indirection.
5. Unit tests with a fake runner (no network): scope guard, revert, report shape, private refusal, timeout, worktree cleanup. Whole suite stays green (1316 tests at last count).
6. Docs: README section and `config/orchestrator.md` nudge text for the lead (when to use `agent-run` vs `delegate` vs `supervise` vs a Claude subagent: Claude subagents remain better for tasks that need the lead's context, many existing files edited consistently, or live debugging).

## First real task to run with it

The Crosscut policy-editor UI slice, prompt saved at `ForemanKit/docs/examples/task-crosscut-policy-editor-ui.md` (also needs the backend contract finished in Crosscut; check `git log` in `CrosscutProcessSuite` for the commit that adds `GET /gateway/policy` and the `digest` field of `GET /gateway/policy/draft`). Project path `C:\Users\regiz\OneDrive\Documents\Python Scripts\CrosscutProcessSuite`, scope: `frontend/src/policyApi.ts`, `policyModel.ts`, `PolicyPanel.tsx`, `components/policy/*`, three test files, and the three locale files. Check commands must be in `verify.allowed_commands` (vitest/tsc/eslint are NOT there today: add `npx vitest run`, `npx tsc --noEmit`, `npx eslint` with `verify_cwd=frontend`, or use the Python-side wrappers).

## Where the owner works

- Crosscut repo: `CrosscutProcessSuite` (this session's other project, do not mix), head around `61c0b7e` on `main` at the time of writing, Phase 2 gateway + roles done, policy editor next, then #48 tuning to write path.
- ForemanKit venv: `ForemanKit/.venv/Scripts/python`; plain `python` lacks yaml. Restart the MCP server to pick up new code.
