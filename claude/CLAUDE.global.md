## foremankit – delegation rules (short; details in ~/.claude/foremankit-reference.md - read it when a rule below is not enough)

**Delegate by default; write yourself only what needs your own context** (the weekly Claude limit is finite). Before writing a large file ask "is this bulk_code / data_script / summarize?" If yes, `delegate` it first.

Checklist - apply BEFORE every Write/Edit and before every shell script that writes code (a hook counts what you write):
1. `delegate` first (kind: bulk_code, data_script, hard_code, long_context, quick, summarize; ONE file per call; independent calls in parallel in the SAME message; `verify` + `verify_target` where a command can check it): tests, documentation/translations, a self-contained new file above ~50 lines, a mechanical change across several files. Small mechanical fixes in existing code -> a Haiku agent (`model: haiku`, absolute paths, the exact check command, "small Edit-tool edits, no rewrite"); review the diff and run the tests afterwards.
2. You keep: integration, debugging on live systems, decisions, anything needing the API you changed this minute.
3. Security / data / shared interfaces -> `review=true`. Architecture decision or two failed attempts -> `opus-advisor`. Wrong code twice -> `escalate=true`, never after TaskTooLarge / empty / cut-off answers (split the task). After one failed PAID call stop and diagnose.
4. Sensitivity: process data (plant tags, PI/SQL, internal docs) = `internal`; credentials, personal data, recipes = `private` (local only, fails closed); `normal` only for schema + a few anonymised rows.
5. Session start: `router_status` once. Before a delegation one line `→ calling foremankit: <kind>, <sensitivity>`; after it one line with the result header. Results land in `.agent-out/<id>.md`: read selectively.
6. Reports: at most ~8 lines or a short table, helpers in one line.

Secondary (rare):
- `allow_cost=true` only after the user agreed to the amount.
- Code standard: clean code after R. Martin - functions <=20 lines / 4 params / 3 nesting, docstring on every module/class/public function, 2-4 line header on every file, specific exceptions, no duplication; check python with `foremankit check <file>`; do not reformat code you were not asked to touch. Full text in the reference.
- Start a local LLM server ONLY through `llama/start-llama.cmd` (check `-List` first).
- Long session at a natural breakpoint: suggest a NEW session once, with state saved to memory/repo.
