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

## Already done (committed, NOT pushed, NOT live-tested; needs router + MCP + VS Code restart)
- `558f4c2` step 4 core (foremankit/night_run.py, hooks/night_continue.py Stop hook, installed via install.py, run done 2026-10-29), `bfe707d` panel switch (panel.py night_status/set_night, CLI `panel night-run on|off|status`), `477cc1b` driver.json now has `kind` (local / external:<mode>) and Kaggle main loop goes through virtual_clear_body in prepare_external. 1299 tests green. NOT pushed.
- `4c096fd`: gateway strips Agent/Task from subagent requests (they carry the `SubagentHandback` tool) and answers a
  synthetic end_turn after a successful handback (`guard_subagent`, `handed_back` in foremankit/gateway.py);
  routing.yaml `budget.local_orchestrator_paid: false`; hard_code chain has free sources after the paid ones.
- `2987cd1`: foremankit/driver.py (`driver.json` in the state dir, last main-loop request wins, `local_drives()`);
  candidates.py uses it; orchestrator.yaml `cloud_when_local: false` -> Haiku/Opus subagent ids served locally,
  an overflowing SUBAGENT gets a stop answer instead of a spill (`overflow_stop`).
- Still going to the cloud: tool-less side requests (auto-mode classifier -> claude-sonnet-5-5), and the MAIN LOOP
  spill between `overflow_ratio` 0.70 and `context_stop.ratio` 0.90.
- `tests/test_gateway_side_cloud.py` has someone else's uncommitted change: do not commit or revert it.
- `19917b3` and `0c674b8`: plan steps 0-3 done (confirmed hook schemas, no main-loop spill while local drives, context_stop.ratio 0.85, virtual clear in the gateway). Full suite 1267 green. NOT pushed.
- Uncommitted: vscode-extension/extension.js holds the Night run block + tip9 help (mine) mixed with someone else's changes; also others' changes in config/routing.yaml, config/orchestrator.yaml, kaggle/gpu/serve_gpu.py, tests/test_gateway_side_cloud.py, tests/test_gateway_upstreams.py, tests/test_serve_gpu.py - do not commit or revert blindly.

## Plan (in order; one step = one delegation + tests) — NEXT: step 4b (steps 0-3 done, step 4 done except as noted)
0. Before coding, confirm with the `claude-code-guide` agent: Stop hook output schema (`decision: "block"` +
   `reason`, the `stop_hook_active` input field), SessionStart `source` values (startup/resume/clear/compact) and how
   it injects context (`additionalContext` / stdout). Do not code these from memory.
1. No main-loop spill while the local model drives: in `prepare_body` (gateway.py), when `cloud_when_local` is false
   and `driver.local_drives()`, an overflowing main-loop request goes to step 3 instead of `spill_to_cloud`.
2. `context_stop.ratio: 0.85` in config/orchestrator.yaml (the 0.70 spill must no longer pre-empt it, see step 1).
3. Virtual clear in the gateway (new module, e.g. foremankit/virtual_clear.py; call it from prepare_body before
   `context_stop_response`): when a local main-loop request reaches the ratio, write the handoff with the existing
   `context_stop.handoff_markdown` / `write_handoff`, store a cut marker in the state dir, and from then on send the
   model: system + tools + one user message "handoff: <text>" + the messages after the cut. Rules:
   - the kept tail must start at a valid point: never a `tool_result` without its `tool_use`; every kept assistant
     `tool_use` keeps its result;
   - the marker must identify the session (look for a session id in the request, e.g. `metadata.user_id`; check a
     real dump via the `DUMP_ENV` variable in gateway.py) and the cut message (fingerprint), so a new session or a
     real /clear does not inherit it;
   - rolling: when the trimmed request reaches the ratio again, cut again with a new handoff;
   - leave Claude Code's reported usage alone (the panel context meter will look low: known, cosmetic);
   - the hard `context_stop` stays as the last resort when even the trimmed request does not fit.
   Open choice: handoff stays mechanical (from the request body, as now) or a direct tool-less call to the local
   llama-server writes a summary. Start mechanical.
4. DONE (see Already done), except: night_run.should_continue still requires driver.local_drives().
   NEXT 4b: allow driver kind local OR external:kaggle (config night_drivers), ping the Kaggle tunnel before each
   continue, stop continuing after context_stop (Opus advisor 2026-10-29: Kaggle never wrote driver.json before,
   stale local=true let night-run fire by accident; Kaggle windows 9-12h are the owner's preferred night setup;
   URL/key do not change between handoffs). Then step 5, step 6 (nothing at night before it).
   Original spec: Stop hook `hooks/night_continue.py` (install via install.py like the other hooks): when a night run is ON
   (state file e.g. `night_run.json`: enabled, until time, max_continues, count) and `driver.local_drives()` and no
   DONE file and count < max and now < until -> block the stop with a reason like "Continue the task from the plan
   and the latest handoff; when everything is finished create the DONE file <path>". Count every block. Off by
   default. Panel switch + CLI (`panel night-run ...`), strings PL/EN like the other panel items.
5. SessionStart hook (sources startup and clear): if a handoff for this cwd is newer than N hours, inject its path and
   a bounded excerpt, so a manual `/clear` also resumes in one step.
6. Live test before any night: set ratio low (e.g. 0.30) and max_continues 3 on a short task; check in the gateway
   log that the cut happens, tool pairs stay valid, the Stop hook continues and stops at the cap/DONE, and that no
   request reaches the Anthropic cloud except tool-less side requests.

## Rules for the session
- Orchestrator does not write code: delegate (bulk-supervisor / delegate / Haiku agents), report in Polish, short.
- Clean code rules from CLAUDE.md; full suite `.venv/Scripts/python -m pytest -q` (1267 green at 0c674b8; system
  python lacks yaml). Commit only after green; push only when the owner says so.
- Do not run a night session until step 6 passed.

## Also pending (other project)
RougherSim: 3 local commits not pushed; mill/cyclone/flotation physics must be redone with sources before pushing
(see memory `project-roughersim-m0-state-2026-10-27`).
