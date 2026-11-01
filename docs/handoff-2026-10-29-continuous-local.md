# Handoff 2026-10-29: continuous local-orchestrator session (plan 2 + Stop hook)

For the next Claude session. Read first, then `memory/project-foremankit-state-2026-10-13.md` (last paragraph).
Owner decision (2026-10-29): continuous unattended work with the local model as orchestrator, VISIBLE live in the
VS Code Claude Code panel. At 85 % of the local window: stop what it does, write a handoff, "clear", load the handoff,
go on. A real `/clear` cannot be triggered by a hook or the gateway, so the clear is done in the gateway (plan 2).
Rejected: plan 1 (`claude -p` runner loop) because headless sessions are not visible live in VS Code.

## Why (night 2026-10-28)
Local Gemma main loop ended its turn at 19:50 ("I will verify when the agent finishes") and never woke again. Its
subagents ran until 07:23, chaining ~70 "final report" agents after successful SubagentHandback calls, spent
USD 1.38 / 1.50 of the daily delegation cap (hard_code on deepseek-v4-pro, gemini-3.1-pro) and the Claude
subscription (15 Haiku bulk-supervisor, 1 opus-advisor, ~80 subagent turns spilled to cloud Sonnet).

## Already done (committed, NOT pushed)
- `9a8275c` / `1e80baa` / `354fe1b`: step 4b completed and verified. `night_run.should_continue()` allows configurable driver kinds (`night_drivers`, default `["local", "external:kaggle"]`), pings the Kaggle tunnel (`FOREMANKIT_NODE_KAGGLE_LAN` / `KAGGLE_LLM_KEY`) via `/v1/models` with bounded retries and backoff (reason `"tunnel"` on failure), and stops continuation if `context_stop` flag is present in `night_run.json` (reason `"context_stop"`). Strict clean code enforced (functions <= 20 lines, `foremankit check`, `ruff check`). Full suite: 1380 tests passed.
- `558f4c2` step 4 core (foremankit/night_run.py, hooks/night_continue.py Stop hook, installed via install.py), `bfe707d` panel switch (panel.py night_status/set_night, CLI `panel night-run on|off|status`), `477cc1b` driver.json now has `kind` (local / external:<mode>) and Kaggle main loop goes through virtual_clear_body in prepare_external.
- `4c096fd`: gateway strips Agent/Task from subagent requests (they carry the `SubagentHandback` tool) and answers a
  synthetic end_turn after a successful handback (`guard_subagent`, `handed_back` in foremankit/gateway.py);
  routing.yaml `budget.local_orchestrator_paid: false`; hard_code chain has free sources after the paid ones.
- `2987cd1`: foremankit/driver.py (`driver.json` in the state dir, last main-loop request wins, `local_drives()`);
  candidates.py uses it; orchestrator.yaml `cloud_when_local: false` -> Haiku/Opus subagent ids served locally,
  an overflowing SUBAGENT gets a stop answer instead of a spill (`overflow_stop`).
- `19917b3` and `0c674b8`: plan steps 0-3 done (confirmed hook schemas, no main-loop spill while local drives, context_stop.ratio 0.85, virtual clear in the gateway).
- Step 6 LIVE TEST COMPLETED (night 2026-10-31 to 2026-11-01): 8.5-hour unattended run on Kaggle (`qwen3.8-27b`) executing 4,093 calls. Virtual clear rotated context at ~163k tokens (86%), generated `handoff-20261101-005033.md`, Stop hook blocked stop and resumed session at 14.8k tokens. When the Kaggle 9-hour runtime limit ended the tunnel with HTTP 530 at 06:13, `should_continue()` ping caught it, returned reason `"tunnel"`, and prevented continuation loops. Zero spills to cloud Sonnet. Full telemetry in `docs/night-run-live-log.md`.
- Working tree uncommitted files belonging to others: do not commit or revert (`config/orchestrator.yaml`, `config/routing.yaml`, `kaggle/gpu/serve_gpu.py`, `tests/test_gateway_side_cloud.py`, `tests/test_gateway_upstreams.py`, `tests/test_serve_gpu.py`, `vscode-extension/extension.js`).

## Plan (in order; one step = one delegation + tests) — NEXT: step 5 (steps 0-4b and 6 done)
0. DONE: Confirm hook schemas (Stop hook `decision: "block"` + `reason`, SessionStart `source` values).
1. DONE: No main-loop spill while local model drives (`prepare_body` in gateway.py).
2. DONE: `context_stop.ratio: 0.85`.
3. DONE: Virtual clear in the gateway (`foremankit/virtual_clear.py`).
4. DONE: Stop hook `hooks/night_continue.py`, `foremankit/night_run.py`, panel switch (`panel night-run ...`).
4b. DONE: Driver kinds `local` / `external:kaggle`, tunnel ping with backoff (reason `"tunnel"`), `context_stop` suppression.
5. NEXT: SessionStart hook (sources startup and clear): if a handoff for this cwd is newer than N hours, inject its path and
   a bounded excerpt, so a manual `/clear` also resumes in one step.
6. DONE: Live test passed with full context reset and graceful tunnel drop handling (see Already done).

## Rules for the session
- Orchestrator does not write code: delegate (bulk-supervisor / delegate / Haiku agents), report in Polish, short.
- Clean code rules from CLAUDE.md; full suite `.venv/Scripts/python -m pytest -q` (1380 green). Commit only after green; push only when the owner says so.
