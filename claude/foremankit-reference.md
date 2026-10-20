# foremankit – reference (on-demand detail)

This file holds the detail that the short checklist in `CLAUDE.md` does not repeat. Read the matching section when you need it. The wording is kept from the original `CLAUDE.md` rules.

## Task sizing

- Size the task for the helper: ONE file or part per call, answer under ~4000 tokens (`max_output_tokens` close to the real need; the router warns above 8000 and lowers reasoning above 6000). Split bigger jobs.
- Free models cut off or lose track on whole-file rewrites; a task that needs the API you just changed stays with you (write it yourself or give a model the exact new signatures as `context_files`).
- Use relative paths in `context_files` and in delegated tasks.
- Pick `kind`: bulk_code, data_script, hard_code, long_context, quick, summarize.
- Delegate by default; write yourself only what needs your own context. Anything that would be > ~50 lines or spans several files (tests, scripts, benchmark and report code, documentation, summaries and tables of results) goes to foremankit: `delegator` subagent, MCP `delegate` directly, or `foremankit delegate ...` from Bash.
- Before writing a large file ask "is this bulk_code / data_script / summarize?" - if yes, delegate first and review the result.
- Integration, debugging on live systems and decisions stay with you. Small edits that need your context: do them yourself.
- Small MECHANICAL fixes in existing code (rename a variable, add a docstring, split one function, fix the issues listed by `foremankit check`/a linter, update callers after a signature change) go to a Haiku subagent (see "Haiku prompt template"). The weekly limit is finite and Haiku is enough. Several such fixes in different files = several Haiku agents in parallel. Review the diff and run the tests afterwards.
- Which work counts as "documentation" for delegation: README, comments-only files, translations.
- MCP tools may be deferred (listed by name only): load their schemas first with ToolSearch (`select:mcp__foremankit__delegate,mcp__foremankit__router_status`), otherwise the call fails. A `delegator` subagent does not need this.

## Verify and review

- Prefer `verify` + `verify_target` for code that a command can check (e.g. `verify="python -m pytest tests/test_x.py -q"`, `verify_target="tests/test_x.py"`): the router runs the command, sends failures back to the SAME cheap model (2 rounds) and keeps the file only if it passes, so you do not debug delegated code by hand.
- Only prefixes from routing.yaml `verify.allowed_commands` run.
- Check python files with `foremankit check <file>` (or `verify="foremankit check <file>"` to make a delegated file pass it).
- Preliminary review: for code that goes into a repo (more than ~50 lines, or touching security, data or shared interfaces) add `review=true` to the delegation (after `verify`).
- Run `foremankit review --diff <paths>` on your own changes before presenting them.
- The reviewer is a DIFFERENT source: the own local LLM pool or a paid model, never a free cloud.
- Findings are advisory. Only blockers whose quoted evidence really appears in the code go back to the author (one round, then the check runs again). Treat the rest as a checklist to verify, not as truth: cheap reviewers also err.
- Code that touches security, data or shared interfaces -> `review=true`.
- Quality problem (wrong code from the default chain, rejected twice) -> retry with `escalate=true` (DeepSeek Pro).
- Do NOT escalate after `TaskTooLarge`, an empty answer or a cut-off answer: the task is too big, split it.
- After one failed PAID call, stop and diagnose before spending again.
- Two failed attempts at the same problem, or an architecture decision -> ask the `opus-advisor` subagent.
- Cost: a paid call whose worst case is above `budget.per_call_usd` (0.10) is refused. Pass `allow_cost=true` only after the user agreed to the amount.
- Results land in `.agent-out/<id>.md`. Read them selectively (Grep/offset), verify with tests, lint or spot checks – not line by line.

## Sensitivity details

- Process data (plant tags, PI/SQL extracts, internal docs) = `"internal"`: paid no-train Gemini, else the local LLM; fails closed, never reaches free/training providers.
- DeepSeek is used for `"internal"` only when this machine accepts it via FOREMANKIT_ALLOW_INTERNAL_TRAINING. Its policy allows training on prompts and stores data in China, so never for credentials, personal data or trade secrets.
- Credentials, personal data, recipes/trade secrets = `"private"`: own local LLM only, fails closed.
- To `"normal"` (free/external) models send only schema + a few anonymised rows, never whole datasets or credentials.
- DeepSeek is preferred over paid Gemini whenever it is reachable. It is probed with a free GET /models, re-probed every 10 min. If it is blocked on the current network, just continue (it is skipped automatically). Never try to bypass a network block.
- `router_status` shows remaining free quota and paid spend; paid providers stop hard at the configured USD budget.

## Local LLM

- `internal` tries the local LLM first (free, private) but ONLY for small `quick`, `summarize`, `bulk_code`, `data_script` tasks (limits in routing.yaml `llm.kinds`).
- Why: the local model is slower than the clouds (roughly 30-55 tok/s on the Legion, 15-20 on the Firefly; see benchmarks/LATENCY_RESULTS.md). So `hard_code`, `long_context`, big inputs and `escalate=true` go straight to the strongest allowed cloud model (paid Gemini, or DeepSeek where accepted).
- Keep `max_output_tokens` close to the real need.
- The local LLM serves one request at a time, so a second concurrent call moves on to the next allowed source.
- Start a local server ONLY through `llama/start-llama.cmd` (never call llama-server.exe directly).
  - It reuses an identical running server.
  - It refuses a second different one: each holds ~25 GB of RAM. Windows even allows two on the same port with different bind addresses, so the script is the guard.
  - It holds a machine-wide lock.
  - `-List` shows what runs. `-Replace` swaps it. `-Force` is for a deliberate second server on another port.
- Check `-List` before asking another machine's session to start one.

## Parallel batches

- Whenever a job splits into independent units (files, packages, directories, test groups), run the units in parallel: several `delegate` calls in one message, or several background processes of a batch script (one worker per package/directory).
- Independent tasks: issue several `delegate` calls in the SAME message (they run in parallel) instead of one after another.
- Serialize only what shares state: the same file, a shared build directory, tests that import code another unit is changing. Say which and why.
- A batch tool or script you write must be built for that layout (a worker per package, not one sequential loop over everything).
- For long jobs run `foremankit delegate ...` in the background and keep working. Use one background wait instead of polling loops.
- Before starting a long job state in one line what runs in parallel and what must wait.

## Reporting

- Session start: call `router_status` once.
- Before a delegation write one short line: `→ calling foremankit: <kind>, <sensitivity>` and the expected helper. For parallel calls, one line listing all of them.
- When the result arrives, one line from the result header: `← response from <provider/model> in <s>, <cost>`. Mention fallbacks and failures honestly.
- Final reports are terse: at most ~8 lines or a short table - what is done, what is pending, the one decision needed. No recap of steps the user saw, no per-helper essays. Helper list = one line or a compact table with cost. Details live in the repo log, not in chat.
- In the final summary list every helper used with its role, tokens and cost (`foremankit stats --days 1`), and say which ones were useful and which were wasted.
- Keep replies short. The user's weekly Claude limit is finite and every turn re-reads the whole conversation.
- Long conversation breakpoint: when a conversation has become long (many tool results, a few hours or several dozen turns) and a natural breakpoint arrives (a task finished and committed, results delivered, a long background job started), say so in ONE line and suggest continuing in a NEW session only AFTER you have updated the handoff yourself, without being asked or reminded: save the state (what is done, where the files are, what is pending) to memory or the repo, REWRITE the existing handoff to the current state and delete points that are outdated, done or superseded (no stale leftovers, no append-only log); then say in the same line that the handoff is current. Mention it at most once per breakpoint and never in the middle of a task.
- Escalation and opinion: two failed attempts or an architecture decision -> `opus-advisor` (see "Verify and review").

## Haiku prompt template

Use for small MECHANICAL fixes in existing code only (see "Task sizing").

- Launch with `Agent`, `model: haiku`.
- The prompt must be self-contained: include the task, absolute paths to every file, the exact check command, and the instruction "small Edit-tool edits, no rewrite".
- Several fixes in different files = several Haiku agents in parallel.
- Afterwards: review the diff and run the tests.

Template:

```
Task: <one-sentence mechanical fix, e.g. rename variable `x` to `pressure_kpa` in the function `read_tag`>
Files (absolute paths): C:\...\<file>.py
Change only what the task says. Small Edit-tool edits, no rewrite.
Check command (run it, report the result): <exact command, e.g. foremankit check C:\...\<file>.py>
Report: one line - the files changed and the check result.
```

## Full code standard

Code standard (same in every session and tool): clean code after R. Martin, readable by a beginner.

- Names that reveal intent.
- Functions <= 20 code lines (comments and docstrings do not count), <= 4 parameters, <= 3 nesting levels.
- No duplication and no magic values.
- Specific exceptions (never bare `except`).
- Logic apart from I/O.
- A docstring on every module, class and public function.
- Comments that explain WHY (units, formulas, workarounds).
- In numerical or domain code, teaching comments: teach the reader the model, the formula and the data structure.
- Every file starts with a 2-4 line header saying what it does and why. Module docstring in Python, `//` block in TS/C++.
- Teaching comments are written together with the code and never added later by a model that sees one function. Review them with whole-file context.
- Tests named as sentences.
- No commented-out code.
- Do not reformat code you were not asked to touch.
- The full text is foremankit's `config/conventions.md`. Kinds `bulk_code`, `data_script` and `hard_code` receive it automatically. A project may override it in `.foremankit/conventions.md`.
