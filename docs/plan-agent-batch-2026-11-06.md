# Plan 2026-11-06: delegation to agy and opencode in parallel, run back to back

Source: Opus advisor review of `agent-run` (2026-11-06). Facts: agy is on the owner's Google AI Pro subscription (flat fee, quota unknown, reportedly low); opencode is used for its FREE models only (no cost, they may train on prompts). The repo `msurun/foremankit` is PUBLIC, so its code is `normal` data and may go to opencode.

## 1. What to delegate (decision rules; `foremankit/agent_choice.py` will encode them)

Hard gates, first match decides:
- G1 `private` data -> lead or a Claude subagent, never a runner.
- G2 `internal` data -> agy only (opencode: `normal` only).
- G3 scope matches `routing.yaml agent_run.review_paths`, or the task changes a shared interface (dataclass, Protocol, CLI, config schema) or git plumbing -> lead (or Sonnet), then Opus review.
- G4 no mechanical check command -> not for agent-run.
- G5 needs live context of the session (an API just changed, debugging) -> lead.

Score 0-2 each: contract (signatures given, real files cited as path:line), edge cases listed, targeted check, isolation (mostly new files, no overlap with parallel runs), size (2 = up to 300 changed lines). Delegate at 7 or more.

Runner: opencode = `normal` data, 1-3 files, up to ~200 lines, pure logic plus tests (fast, can retry). agy = 3-8 files, more context, or `internal` data; at most 1-2 at a time. Haiku subagent = small edits to existing code and CLI wiring. Split when: more than 8 files or ~400 lines, two independent checks, mixed security and ordinary paths, task text over 22k chars.

A good task file: one-line goal, scope globs, files to read (path:line), exact signatures, edge-case list, check command plus the test names that must pass, a "do not" list, 1-2 input/output examples, sensitivity label. Never put `agent_git.py`, `agent_run.py`, `agent_batch.py`, `agent_merge.py` in a dogfood scope.

## 2. Chain (in order; D = dogfood through agent-run, then the lead only verifies)

0. D opencode: fix the 17 pre-existing failing tests (fc651c3 split of delegation categories; tests only, scope tests/test_delegation.py, tests/test_router.py, tests/test_panel.py). Check: those three files green. Needed first, otherwise new regressions hide in the noise.
1. Lead + Opus review: `foremankit/agent_lock.py` (global git lock with O_EXCL + stale-PID detection around worktree add/remove/prune, in OneDrive `.git` parallel runs race) and `agent_run.max_parallel: {agy: 1, opencode: 3}`. Check: `tests/test_agent_lock.py` with 4 threads.
2. Lead: `AgentRunJob.base` / `--base` (a dependent task starts from its predecessor's branch). Check: `tests/test_agent_run.py`.
3. D opencode: `foremankit/agent_overlap.py` `scopes_overlap(a, b, tracked) -> list[str]` (overlap if a tracked file matches both, or the literal prefix of one is a prefix of the other). Check `tests/test_agent_overlap.py`. Lead verifies `**` and new-file cases.
4. D opencode (parallel with 3 = live test of the slots): `foremankit/agent_choice.py` `decide_runner(facts) -> (runner|"haiku"|"lead", reasons)`, mirror `supervisor_choice.py`. Lead verifies gates before score, `private` never reaches a runner.
5. D agy: `foremankit/agent_plan.py`: YAML plan (id, task_file, scope, check, runner, model, sensitivity, depends_on), validate unknown deps and cycles, topological levels, overlap check per level (uses 3). Opus review.
6. Lead + Opus review: `foremankit/agent_batch.py`: run each level in a thread pool within the slots, dependents get `base`, stop on a quota error, JSON report plus summary. Check: fake runner, 3 tasks, one dependent.
7. Lead + Opus review: `foremankit/agent_merge.py`: preconditions (main clean, report PASS, REVIEW REQUIRED needs `--reviewed`), `git merge --no-ff`, on conflict `merge --abort` and report, re-run the check on main, never push. Check: temp repo with and without conflict.
8. D opencode: review events: `agent-merge`/`agent-reject` log `agent_run_review {run_id, verdict, blockers}`; `passrates.py` gets a "merged without fix" rate per runner/model (today `accepted` = check passed only, and every agy slice so far had real defects despite passing). Opus review.
9. Haiku subagent: CLI `agent-run-batch`, `agent-merge` in `cli.py`. Check: `--help` and a dry run.
10. D opencode: README section and `config/orchestrator.md` nudge, through the new batch.

Merge each dogfood branch only after the lead has read the diff.

## 3. Risks

agy quota unknown (start with max_parallel 1, stop on the first quota error, log timestamps); free models train on prompts (fine here: public repo; not for Crosscut, internal or private data); parallel git races and OneDrive locks on `.git` (global lock + retry; after every batch `git worktree list` shows only main and `git status` on main is clean); Windows `worktree remove` fails while a process holds a file (already non-fatal; add a cleanup command for old worktrees); free-model `--auto` has no sandbox: it can run any command on this PC (owner approved 2026-11-06).

## 4. Metrics of the experiment

Check pass@1 per runner/model; share merged without fixes (goal at least 60%); blockers per slice found by review; scope reverts; wall time; lead tokens per merged item versus writing it by hand; 0 corruptions of the main tree.

## 5. Update after the owner's note (agy = AI Pro quota, opencode = free models)

Neither costs per call, so the limits are: agy is the only runner allowed `internal` data and has an unknown quota (probably shared with the owner's own Gemini use); opencode free has no known quota but `normal` data only. So **opencode is the default, agy is kept for what only it can do**.
- `decide_runner` after the gates: `internal` -> agy always (never opencode; if agy quota is used up the task waits in a queue). `normal` -> opencode for 1-5 files, up to ~400 changed lines, context under ~1500 lines; agy only for 6-8 files, 1500+ lines of context, task text above 8k chars, or after opencode failed the same task twice (escalation opencode, opencode, agy is a reason code, not a score; if agy fails too the task file is the problem: split or clarify, do not retry).
- Config (`routing.yaml agent_run`): `max_parallel: {opencode: 3, agy: 1}` (agy 2 only after measuring); `quota: {agy: {max_runs_per_day: 15, cooldown_on_error_s: 1800}}` (15 is a guess). A quota/429/"limit" pattern in agy's reply or stderr marks the run `quota_exhausted` (not FAIL) and pauses agy for the cooldown; independent opencode tasks keep running.
- Batch plan: each task is pinned (`runner: agy` mandatory for `internal`; the plan validator rejects `internal` + opencode) or `runner: auto`.
- Metrics: log `quota_exhausted` events and agy runs per day to estimate the real quota after about a week; if opencode's merged-without-fix rate falls below ~40%, move medium jobs to agy; log the model id on every run (done), keep `default_model` configurable (free models can be withdrawn).
- Extra risk: a task file labelled `normal` that contains plant tags; the lead checks every task file before an opencode run.
