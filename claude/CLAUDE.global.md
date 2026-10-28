## foremankit – delegation rules (details in ~/.claude/foremankit-reference.md, read only when a rule is not enough)

Delegate by default; write yourself only what needs your own context (the weekly Claude limit is finite). Check BEFORE every Write/Edit or code-writing shell script (a hook counts it):
1. bulk_code (tests, docs/translations, a new file above ~50 lines, mechanical edits across files) -> `bulk-supervisor` agent: task, absolute target path, check command, sensitivity; one file per agent, independent ones in parallel in ONE message. Then diff the target only.
2. Other kinds -> `delegate` (data_script, hard_code, long_context, quick, summarize; one file per call; `verify` + `verify_target` when a command can check it). Small fixes in existing code -> Haiku agent (absolute paths, check command, "small Edit-tool edits, no rewrite").
3. Keep yourself: integration, live debugging, decisions, code needing an API you just changed.
4. Security / data / shared interfaces -> `review=true`. Architecture or two failed attempts -> `opus-advisor`. Wrong code twice -> `escalate=true` (not after TaskTooLarge / empty / cut-off: split instead). One failed PAID call -> stop and diagnose.
5. Sensitivity: process data (plant tags, PI/SQL, internal docs) = `internal`; credentials, personal data, recipes = `private`; `normal` = schema + a few anonymised rows.

Minimal context and minimal text (tokens are the budget):
- Read only what decides the next step: Grep first, then Read with offset/limit; never open `.agent-out/` when verify PASSED; no re-reading files you just edited; no `router_status` unless a delegation failed.
- Ask helpers for short reports (agents return 1-3 lines) and do not echo their output back.
- Write the minimum: no narration between tool calls, no announcing delegations, no recap of what the user saw, no options you will not take. Final report <=5 lines, in the user's language; ask one question only when blocked.

Secondary (rare):
- `allow_cost=true` only after the user agreed to the amount.
- Code standard: R. Martin clean code (functions <=20 lines / 4 params / 3 nesting, docstrings, 2-4 line file header, specific exceptions, no duplication); `foremankit check <file>`; do not reformat untouched code.
- Start a local LLM server ONLY through `llama/start-llama.cmd` (check `-List` first).
- Long session at a natural breakpoint: update the handoff (memory/WORKLOG/repo notes: rewrite to current state, drop finished points), then suggest a NEW session once.
