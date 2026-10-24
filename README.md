# foremankit

English · [Polski](README.pl.md)

A portable kit for Claude Code that hands off to cheaper or local models the work that is not worth spending your Claude Pro limit on: boilerplate, tests, pandas/SQL scripts, summaries of large logs.

> foremankit is an independent, unofficial add-on for the Claude Code extension (VS Code). It is not affiliated with, endorsed by or sponsored by Anthropic.

**Analogy:** Claude is the shift foreman. It makes the decisions and checks the results, and it assigns the spadework to subcontractors. `foremankit` is the dispatcher: it knows which subcontractor is free today, how much they cost, and who is allowed to see company data (only "our own", meaning Ollama on your own hardware).

| Role | Who | What they do |
|---|---|---|
| Supervisor | the main Claude Code session (Sonnet/Opus) | plans, verifies, makes small fixes |
| Advisor | the `opus-advisor` subagent (Opus, read-only) | called after 2 failed attempts or for an architectural decision |
| Relay | the `delegator` subagent (Haiku) | calls the MCP `delegate` and returns only the path to the result |
| Workers | DeepSeek, Gemini, Groq, OpenRouter, Mistral, (Cerebras), a local LLM server (llama.cpp, optionally Ollama) on your own machine and on a second laptop | write code and analyses |

Priorities: 1) cost and saving your limit, 2) portability, 3) privacy.

---

## Installation

Requirements: Python 3.10+, the Claude Code CLI (`claude`) in PATH. Works on Windows, Linux and macOS.

```bash
cd foremankit
python install.py --dry-run     # shows what will be done (no changes)
python install.py               # venv + package + agents + rules + hooks + MCP registration
python install.py --check       # what from foremankit is currently installed (read-only)
```

`install.py` is idempotent, so you can run it repeatedly. It does the following in order:
1. creates `foremankit/.venv` and installs the package in it (`pip install -e .[dev]`), with no global installation;
2. on the first run it saves a snapshot of the files it will change (`~/.config/foremankit/install-state/`), so you can go back to the state before installation;
3. copies `claude/agents/*.md` to `~/.claude/agents/`;
4. appends `claude/CLAUDE.global.md` to `~/.claude/CLAUDE.md` between the markers `<!-- BEGIN foremankit -->` and `<!-- END foremankit -->`;
5. registers the MCP server: `claude mcp add --scope user foremankit -- <venv>/Scripts/foremankit-mcp` (the console script, not `python -m`, because a project folder named `foremankit/` would shadow the installed package);
6. appends four hooks to `~/.claude/settings.json` (the file is written without a BOM, because Node's parser rejects a BOM; every hook exits with code 0 and never blocks your work):
   - `PreToolUse` and `PostToolUse` on `mcp__foremankit__delegate` (`hooks/notify_delegate.py`): before and after every call it shows a one-line notification (`→ calling foremankit…`, `← response from…`), whether or not the model remembers the rule from `CLAUDE.md`;
   - `UserPromptSubmit` (`hooks/session_length_reminder.py`; for sessions on the local model it also orders a stop and a handoff at 85 % of the context, and never reports the local fill level to a cloud session): when the transcript exceeds 8 MB (`FOREMANKIT_SESSION_REMINDER_MB`), it reminds you once about starting a new session (after that at most every 90 min and after a 4 MB increase). With the local orchestrator it also warns at 60, 75 and 85 % of the model's context window;
   - `PostToolUse` on `mcp__foremankit__delegate` (`hooks/auto_checkpoint.py`): after an agent task it saves a short summary in `<project>/.agent-out/checkpoint-<time>.md` and records it in `state.json`. Nothing is sent outside;
7. checks whether `CLAUDE_CODE_SUBAGENT_MODEL` is set (it overrides the `model` field of subagents, so **do not set it**);
8. prints which keys are missing.

Installation options:

| Option | What it does |
|---|---|
| `--dry-run` | shows every action, changes nothing (also works with `--uninstall`) |
| `--skip-claude` | only the venv and the package; touches nothing in `~/.claude` |
| `--no-panel` | do not install the VS Code panel (the ForemanKit sidebar is installed by default, in direct mode: Claude Code keeps talking to Anthropic) |
| `--local-orchestrator` | additionally checks/downloads llama-server (and installs the panel even with `--skip-claude`); with `--router` it also routes the Claude Code plugin through the gateway to a local Qwen (changes VS Code settings; described in the "Local orchestrator" section) |

Every file modified in `~/.claude` is first copied as `*.bak-foremankit-<date>`.

### Upgrading from the old name (agentkit)

The project used to be called `agentkit`. Running `python install.py` on a machine that still has the old installation migrates it: `~/.config/agentkit` (API keys, state) is moved to `~/.config/foremankit`, the old rules block in `~/.claude/CLAUDE.md`, the old VS Code autostart tasks and the old MCP registration are removed, and the new ones are installed. Variables named `AGENTKIT_*` are still honoured (they are copied to `FOREMANKIT_*` when the new name is not set). A project override `.agentkit/conventions.md` must be renamed to `.foremankit/conventions.md`.

### Uninstalling and restoring

| Command | Effect |
|---|---|
| `python install.py --check` | read-only: a list of foremankit traces that are still installed (exit code 1 when something is left) |
| `python install.py --uninstall` | undoes what foremankit added and leaves your later changes in place (details below) |
| `python install.py --uninstall --stop-services` | additionally stops the running gateway and `llama-server` from this folder |
| `python install.py --uninstall --purge` | additionally deletes the pre-installation snapshot and the gateway's working files |
| `python install.py --restore-original` | restores the files from before the first installation byte for byte (later changes to them are lost) |

`--uninstall` removes: the agents, the rules block in `CLAUDE.md`, the four hooks, the MCP registration and, if `--local-orchestrator` was used, the gateway entries in VS Code `settings.json`, the autostart (Task Scheduler entry and tasks in `tasks.json`) and `task.allowAutomaticTasks` (only if the installer set them). An agent that existed before installation is put back in place; files created by the installer disappear if they are empty. At the end it prints what is still left.

`--stop-services` is safe: the gateway is stopped only after the settings have been cleared and only if nothing still points at it (VS Code settings, `env` in `~/.claude/settings.json`, the `ANTHROPIC_BASE_URL` variable in your shell). Otherwise the plugin would lose its connection to the cloud as well, so the installer refuses and prints what still points at the gateway. Without this flag the gateway and the server keep running.

`--restore-original` takes the files from the snapshot (`CLAUDE.md`, `settings.json`, the agent files and, if the orchestrator was installed, the VS Code files). Before overwriting, every file gets a `*.bak-foremankit-*` copy. The snapshot is made by the first run of `install.py`; the VS Code files are added to it when you first use `--local-orchestrator`. With an older installation that has no snapshot, the oldest `*.bak-foremankit-*` copy is used; a file without a copy is left untouched.

Deliberately left behind: the venv, the API keys in `~/.config/foremankit/.env`, the usage log and the `.agent-out/` files in your projects (including `checkpoint-*.md`). After uninstalling, **restart VS Code**, because the plugin reads the gateway address at startup. Only `--uninstall` undoes the MCP registration.

What happens to the model in the plugin: the "Qwen (local)" entry in the picker is a label for the identifier `claude-sonnet-5`, which the gateway redirects to the local Qwen. After uninstalling, the picker loses that entry, and the identifier `claude-sonnet-5` goes to the cloud, so a remembered selection means the real Sonnet 5. After the restart, pick your target model in the picker.

**Moving to another machine or another project:** copy the `foremankit/` folder (without `.venv`) and run `python install.py`. There are no absolute paths in the code. The only absolute path is the MCP command written by `claude mcp add`, which the installer generates on the current machine. The MCP server runs in the current project directory, so results land in `<project>/.agent-out/`. This folder adds its own `.gitignore`.
### Local orchestrator (Claude Code plugin → gateway → Qwen)

Optionally, the Claude Code plugin in VS Code can use a local Qwen (`llama-server`) as the main conversation model, while the same model continues to work as a subagent/`llm` node of the node pool.

```bash
python install.py --local-orchestrator    # to ~/AppData/Roaming/Code/User (settings.json, tasks.json); --uninstall reverts it
```

- **Gateway** (`foremankit/gateway.py`, `foremankit-gateway`, port 8081): the plugin connects to it through `ANTHROPIC_BASE_URL`. Model requests from `local_models` (the "Qwen (local)" picker entry = `claude-sonnet-5`) go to `llama-server` (the model name is rewritten, credentials are stripped, `FOREMANKIT_PEER_KEY` is added). Everything else goes unchanged to `api.anthropic.com`, so cloud models work as usual. **When the gateway is not running, the plugin reaches no model at all.**
- **Prompt slimming** (`foremankit/gateway_trim.py`, settings in `config/orchestrator.yaml`, no restart needed): Claude Code sends 66–100k tokens per request, and the local model counts them on the CPU. For the plugin's main loop the gateway (1) replaces the system prompt with the contents of `config/orchestrator.md` plus the `# Environment` section from the original, (2) keeps only the tools from `tools_keep` and shortens their descriptions to `tool_description_max_chars`, (3) removes `<system-reminder>` blocks matching `drop_message_blocks` (skill list, MCP instructions, agent list). It recognises the main loop by the `Agent` tool; **subagents and cloud models are left untouched**. A log line shows the effect: `(main loop: ~NNk -> ~MMk tokens)`. `FOREMANKIT_GATEWAY_DUMP=<file>` saves the last transformed request for tuning the tool list.
- **Server**: `llama/start-orchestrator.cmd` = `start-llama.ps1 -NoMenu -Share` (context, offload, KV cache and thinking come from the tuned profile in `llama/profiles.json`). The gateway shrinks requests to about 25k tokens, so 64k is enough, and the KV cache in `q8_0` stays on the GPU. The 196608 variant with the KV cache in RAM (`--no-kv-offload`) gave 16 tok/s on an empty context and 3–4 on a long one (with 32k on the GPU it was 40). `-NCpuMoe` above 34 collapses the speed. Thinking is enabled with a 1024-token limit (`FOREMANKIT_THINK_BUDGET`, `0` = off): in the test it did not slow down generation, and answers less often needed follow-up questions; without a limit the token budget disappears into the `thinking` block. The script refuses to start a second instance.
- **Panel**: `python install.py` (no flag needed; `--no-panel` skips it) links `vscode-extension/` into VS Code (reload the window once). A ForemanKit button appears in the activity bar: status of the router and the llama-server (model, context fill), start/stop of the server with a model picker, one-click tuning (offload, threads and the largest fast context), a per-effort-level switch for local thinking, router start and stop in direct mode (see below) and quick links to `profiles.json`, `orchestrator.yaml` and `router.log`. The logic is in `foremankit/panel.py` (testable); the extension is a thin shell without a build step.
- **llama-server binary**: it is not in git. `python install.py --local-orchestrator --with-llama` (or `python llama/get_llama.py`, or the panel button shown when it is missing) downloads the tested llama.cpp build (`b11429`; `--latest` for the newest) matching the machine: CUDA 13.4 or 12.4 with its runtime for a new enough NVIDIA driver, Vulkan for other GPUs, CPU otherwise. Every archive is checked against the sha256 GitHub publishes for it. Model files (`.gguf`) are not downloaded: put your own in `llama/`.
- **Delegation nudge**: the `delegation_nudge.py` hook (PostToolUse on Write, Edit and Bash) counts the lines the session writes itself, including shell scripts written through a heredoc, since its last `delegate` or agent call. After 80 lines (`FOREMANKIT_NUDGE_LINES`) it reminds the model, and shows you a one-line note, that tests, documentation, standalone files and mechanical multi-file edits belong to the cheaper models; it repeats only after another 80 lines. While delegation is switched off it is silent, and the `UserPromptSubmit` side tells the model to do everything itself and never to mention delegation. The rules the installer puts into `~/.claude/CLAUDE.md` are a short checklist; the long reference lives in `~/.claude/foremankit-reference.md` (installed with them, read on demand).
- **Delegation switch**: the *Delegation* section has a master checkbox and one per source category (paid clouds, free clouds, local models); they are the existing `foremankit delegation [paid|free|local|all] [on|off]` toggles. With everything off the `delegate` tool answers `DELEGATION IS SWITCHED OFF by the user ... Do this task yourself; do not retry delegation`, so the Claude session does the work itself and spends nothing on other providers. The setting is read on every call: no restart.
- **Live view and costs**: below the status lights the panel shows what the local model does right now: *reading the prompt*, *thinking*, *answering* or *calling a tool*, with tokens per second (the gateway watches the answer stream, `foremankit/live.py`; about one delta per token, so the rate is approximate), or, for delegated tasks, the prompt progress from the server's `/slots`. The *Delegation* section reads the router's call log (`calls.jsonl`): today's and the last 7 days' cost, today's calls by target (provider, node, model) and the latest delegated calls with kind, tokens, cost and time. Free and local calls cost 0.
- **Pairing lab computers**: the pairing buttons in the *Extra workers* section replace copying the lab key by hand (`foremankit/pairing.py`). On the computer that has the key press *Show pairing code* (it creates the key if this is the first computer): an 8-character code is shown for 2 minutes while the computer answers discovery broadcasts (UDP 8766) and waits for one joiner (TCP 8767). On the new computer press *Find computers*, type the code and *Pair*. The key is sent AES-GCM encrypted with a key derived from the code (scrypt), the host hands out one sealed key per session and the joiner proves it opened it, so listening on the network yields nothing usable without the code. Afterwards both computers are registered as workers of each other (private data allowed, priority 5) and the key is stored in `.env`; a different key already present is replaced only after you confirm. Meant for a home network where every computer is yours: broadcast discovery may be blocked on company networks, and Windows asks once to allow Python on private networks. Each computer still has to run its llama-server with "Share with the lab" to be used.
- **Firewall for pairing (Windows)**: pairing needs inbound UDP 8766 and TCP 8767. The Windows "allow access" prompt offers only Private networks by default, and a dismissed prompt leaves an inbound *block* rule for `python.exe` that overrides every allow rule (it also makes Windows stop asking). The *Open firewall (Private + Public)* button in *Extra workers* runs `llama/pair-firewall.ps1` elevated (one UAC consent): it creates allow rules for the venv launcher and for the base interpreter that actually opens the socket, on both Private and Public profiles but limited to the local subnet, and removes those leftover block rules for the same programs. `llama/pair-firewall.ps1 -Remove` deletes the rules again; the ports answer only while a pairing session is open.
- **Orchestrator on another PC**: in the panel, "Orchestrator for Claude Code runs on" lists this PC and every worker of the pool. Choosing a worker points the gateway's local requests at that machine's llama-server (`orchestrator_target.json` in the state dir; no restart). That machine must run its server with `-Share` (`llama/start-orchestrator.cmd` does) and have the same `FOREMANKIT_PEER_KEY`; the context window shown comes from that server. Each worker also has a checkbox that tells the router whether it may send tasks there, and extra workers can be edited or removed (workers of `nodes.yaml` are edited in the file).
- **Roles and workers**: the local server is an *orchestrator* (answers Claude Code through the gateway, no delegated tasks), a *worker* (delegated tasks only, Claude Code gets a clear 503; this is the default) or *both*; the panel switches it (gateway role file plus the router toggle `llm:self`). Extra llama-server workers (other machines or ports) are added in the panel, stored in `~/.config/foremankit/workers.json` and merged into the node pool next to `nodes.yaml`; the panel shows their status. The "Advanced parameters" form writes hand-picked launch parameters (context, experts on CPU, threads, KV cache, thinking, temperature) into the model's profile, as an alternative to autotuning.
- **Router and the connection to Anthropic**: Claude Code reaches Anthropic through the router (`ANTHROPIC_BASE_URL` in the VS Code settings), so a stopped router means no connection. In router mode the Task Scheduler entry restarts a dead router within 5 minutes; **direct mode** ("Stop (direct mode)" in the panel) stops the router and removes it from the VS Code settings, so Claude Code talks straight to Anthropic. Claude Code reads that setting only when a window starts, so **the VS Code window must be reloaded after every switch** (the panel asks first, then reloads the window for you).
- **Direct mode is the default**: `python install.py` only installs the panel (a link in the VS Code extensions folder). Claude Code keeps talking straight to Anthropic; no router, no Task Scheduler entry, no gateway entries in the VS Code settings and no `tasks.json` tasks exist. **Router mode** is opt-in: `--router` or the panel's "Start router". It writes the three gateway entries into the VS Code settings, registers the Task Scheduler entry `ForemanKit router` (logon + every 5 minutes, so a dead router returns; off Windows a folder-open VS Code task instead) and starts the gateway. "Stop (direct mode)" in the panel, or `--uninstall`, removes every one of those traces and stops the router. The llama-server is never started automatically.
outer.log`.
- Known limitations: without `mmproj` the server does not accept images (a screenshot in the history gives HTTP 500, start a new conversation in that case); the first reply in a session processes the whole prompt. To revert: `python install.py --uninstall`.

### Keys

You set keys only in environment variables or in a `.env` file: `foremankit/.env` or `~/.config/foremankit/.env`. There is a template in `.env.example`. Keys never end up in configs, logs or `~/.claude.json`, because MCP registration deliberately does **not** use `-e KEY=...`.

A provider is active only when its key is set.

| Provider | Variable | Type | Notes |
|---|---|---|---|
| deepseek | `DEEPSEEK_API_KEY` | **paid** | flash is the default, pro on escalation; the cost is calculated from `usage` (cache hit/miss). Models: `deepseek-flash` (V4.1 Flash) and `deepseek-v4-pro`; prices per 1M tokens at peak (half off-peak): flash 0.30 input / 1.20 output, pro 1.32 / 3.96. Reasoning is on by default and billed as output, so the router controls it through the level from `routing.yaml`: `minimal` and `low` turn it off, `medium` turns it on with low effort, `high` with high effort. With `medium` and `high` effort I measured 3.5k reasoning tokens for a small task and no answer, hence the low effort. `enabled: auto`: foremankit checks with the free `GET /models` (result cached for 10 min) whether DeepSeek responds, and if it does, **it prefers it over the paid Gemini**. A corporate network filter blocks `api.deepseek.com`, so at work it is silently skipped, and at home it turns itself on. Do not bypass the block. `foremankit status` shows the test result |
| gemini_free | `FREE_GEMINI_API_KEY` | free (second Google account) | a separate quota from the paid key, used together with the other free ones before the paid ones. `trains_on_data: true`. **Disabled by default** (measurement 2026-10-08: 50% of calls ended in an error; together with OpenRouter it handled pennies) |
| gemini | `GEMINI_API_KEY` | **paid** (account with payments enabled) | `flash` for regular work, `pro` replaces DeepSeek Pro in `hard_code`. Every call counts towards the USD budget (we assume the worst case). The prices in the config are valid until 31.12.2026; from 1.01.2027 Flash doubles in price. Never for private data |
| groq | `GROQ_API_KEY` | free | fast, narrow tokens/day limit |
| openrouter | `OPENROUTER_API_KEY` | free (`:free`), **disabled by default** | about 20 rpm, 50 rpd (1000 rpd after a 10 USD payment) |
| mistral | `MISTRAL_API_KEY` | free (Experiment) | |
| cerebras | `CEREBRAS_API_KEY` | free | fast (gpt-oss-120b); free-plan limits to be confirmed |
| llm | `LLAMACPP_URL` (main, default `http://127.0.0.1:8080`), `OLLAMA_LOCAL_URL` (optional fallback, default `http://localhost:11434`) | local | Optionally `LOCAL_LLM_GEN_TPS` and `LOCAL_LLM_PROMPT_TPS` override `speed` from `providers.yaml` on a given machine (the router estimates the task duration from them); take the values from `tg_tok_s`/`pp_tok_s` in `llama/profiles.json` |

---

## Commands

```bash
foremankit status                     # providers, remaining limits, spend today/this month, LLM pool nodes
foremankit models [provider]          # list from /models of every active provider; flags config IDs that are missing
foremankit stats [--days 7]           # cost and calls per provider/kind + estimated savings vs Sonnet API prices + pass@1/pass@3 per kind and model
foremankit nodes                       # status of all nodes: reachability, busyness, speed, rule, accepted data level, netguard
foremankit nodes key                   # creates FOREMANKIT_PEER_KEY in .env
foremankit llm status                  # all nodes
foremankit llm test                     # test all nodes
foremankit llm <node> test              # test a specific node
foremankit llm <node|self|others|all> on|off|auto
foremankit profile [name]            # shows or switches the profile (home|work|...)
foremankit delegation [paid|free|local|all] on|off|status   # switch a whole source category
foremankit check file.py [...]       # checks Python files against the clean-code rules (config/conventions.md)
foremankit regress [--list]        # run the REGRESSION.md commands (| Module | Downstream | Command |) for modules changed in the git tree
foremankit review [--diff] files     # review by a second model (your own LLM or a paid one); --diff reviews the git diff
python install.py --check          # what of foremankit is installed (separately from the CLI, in the foremankit folder)
foremankit delegate "task" --kind quick [--sensitivity private] [--files a.py b.py] [--escalate] [--allow-cost] [--verify "pytest"] [--verify-target "test_a.py"]
```

On Windows without activating the venv: `.venv\Scripts\foremankit.exe status`.

`FOREMANKIT_DRY_RUN=1` returns a dummy response without any API call. Use it to check which provider would be selected.

---
## How it works

### MCP tools (only two, short descriptions, about 250 tokens per session)

- `delegate(task, kind, sensitivity="normal", context_files=[], system_hint="", max_output_tokens=4000, escalate=false, allow_cost=false, verify="", verify_target="")` writes the full response to `.agent-out/<id>.md`. Only the provider, model, tokens, cost, path and a preview (about 600 characters) come back to Claude, **never the whole response**.
- `router_status()` shows the active providers, limits, spending and the nodes of the LLM pool.

### Routing (`config/routing.yaml`)

| kind | chain (the first available one wins) |
|---|---|
| bulk_code – boilerplate, tests, docs, refactor | mistral → groq → llm → deepseek:flash → gemini:flash (the local model only for small inputs, `llm.kinds`; measurement in `benchmarks/CLOUD_VS_LOCAL.md`) |
| data_script – pandas/SQL/cleaning | deepseek:flash → gemini:flash |
| hard_code – complex code | deepseek:pro → gemini:pro |
| long_context – large logs, many files | deepseek:flash → gemini:flash |
| quick / summarize | groq → cerebras → llm → gemini:flash |

| private (or `sensitivity="private"`) | llm, **and nothing else** (only machines with `accept: private`) |

The order is: the free ones → DeepSeek (the preferred paid one) → paid Gemini as the last reserve. An unavailable or blocked DeepSeek is skipped silently. A connection error (e.g. a block) gives a 30 s cooldown so that subsequent tasks do not wait for the timeout.

`escalate=true` replaces the chain with `escalation` (deepseek:pro). Model names live only in `providers.yaml`.

The router skips a provider when:
- there is no key, or the provider is disabled;
- its rpm/rpd/tpd limit is exhausted (counters in `~/.config/foremankit/state.json`, the day is UTC-based);
- a cooldown after a 429 is in progress (backoff 30 s → 2 min → 10 min → 30 min, or `Retry-After`) or after a 5xx;
- for paid ones: the worst possible cost of the call (estimated input + `max_output_tokens`) would exceed the budget. This is a **hard stop**: currently 1 USD/day and 5 USD/month (as much as was paid in to Gemini), set in `routing.yaml → budget`.

Retries: at most 1, and only after a 5xx or connection error. After a timeout (a hung server) and after a 429 the router immediately moves to the next provider.

**What goes to the slow local model.** An explicit list in `routing.yaml → llm.kinds`: currently `quick` (up to 6k input tokens), `summarize` (8k), `bulk_code` and `data_script` (5k each). Kinds outside the list (`hard_code`, `long_context`) and inputs above the limit go **immediately** to the next allowed source (paid Gemini, and once `FOREMANKIT_ALLOW_INTERNAL` is enabled, DeepSeek too), so that a long task does not block the single slot. In addition, there is a time estimate from the node's speed (`nodes.yaml → speed`, limit `max_seconds`). The only exception is `private` data: they have no other allowed source, so the local model gets them regardless of size. Locally, reasoning is disabled for levels below `high` (measured: 23 s instead of 142 s and an empty response).

**Concurrency.** Several `delegate` calls at once (e.g. in a single Claude message) run in parallel, and the router spreads them across the sources. The local server handles one request at a time (`max_concurrent: 1`), so a second parallel task does not wait in a queue but moves to the next allowed source. If that is the only allowed source (e.g. `private`), the router will wait for its free slot instead of raising an error. For `internal` data the local model is tried **first** (free and private), except for `hard_code` and `escalate=true`, where the strongest model is what matters.

### Cost protection

- `budget.per_call_usd = 0.10`: a call to a paid model whose worst possible cost exceeds 0.10 USD is rejected, unless the call passes `allow_cost=true` (CLI `--allow-cost`; ask the user first).
- `reasoning.large_output_tokens 6000 / large_output_level medium`: requesting a response above 6000 tokens lowers the reasoning level to `medium`.
- `calls.warn_output_tokens 8000`: requesting a response above 8000 tokens produces a note saying "split the task".
- `calls.abort_waste_usd 0.02`: a call to a paid model that, after spending more than 0.02 USD, returns an empty response or one truncated inside a code block fails with a `TaskTooLarge` error and is not retried on another model (the partial text stays in `.agent-out`). This rule protects against costly mistakes.

### Verifying the result (`verify`)

For code that can be checked with a command, pass `verify` and `verify_target` (CLI: `--verify`, `--verify-target`):

```bash
foremankit delegate "Napisz testy dla verify.py" --kind bulk_code --files foremankit/verify.py \
  --verify ".venv/Scripts/python -m pytest tests/test_verify_unit.py -q" --verify-target tests/test_verify_unit.py
```

- The router writes the code block from the response to `verify_target` and runs the command (without a shell, in the project folder, with the limit `verify.timeout_s` = 300 s).
- When the command does not pass, its output goes back to the **same** model with a request for a fixed file (at most `verify.max_rounds` = 2 times).
- The file is changed **only** when the command passes. Otherwise the original comes back (a new file is deleted). The result shows `verify: PASSED ...` or `verify: FAILED ... was restored`.
- Safety: the command must start with a prefix from `routing.yaml → verify.allowed_commands` (`python -m pytest`, `.venv/Scripts/python -m pytest`, `.venv/bin/python -m pytest`, `pytest`, `python -m ruff check`, `ruff check`). `verify_target` is a plain relative file inside the project (never `.env`, `.git`, `.ssh`). A bad inference is rejected before you pay for the model, and a command that fails to start ends the loop immediately.
- Paths are resolved from the process working directory: the MCP server works in the Claude session folder, and the CLI in the folder you launch it from.
### One coding standard (clean code)

All code written by foremankit and for it must follow the same rules, regardless of tool and session: clean code per R. Martin, readable for a beginner programmer (names that reveal intent, small functions, no duplication and no magic values, specific exceptions, logic separated from I/O, a docstring for every module, class and public function, comments explaining **why**). The full text is `config/conventions.md`.

- **Prompts:** for the `bulk_code`, `data_script` and `hard_code` kinds, the router appends this file to the system prompt (about 600 tokens per call; the prompt is fixed, so DeepSeek's cache works). A project can override it with the file `<project>/.foremankit/conventions.md`.
- **Claude sessions:** a summary of the standard is in the `claude/CLAUDE.global.md` rules (`install.py` installs them), so the same convention applies in every session.
- **Machine checking:** `foremankit check file.py` enforces what can be measured: function length (20 lines), number of parameters (4), nesting (3), missing docstrings, bare `except`, single-letter names. The limits are in `routing.yaml → conventions.limits`. Use it in `verify`, and the file only enters the repo once the rules are met: `--verify ".venv/Scripts/foremankit check file.py" --verify-target file.py`. Whether a name is *good* or a comment *explains why* is still judged by the prompt and a human.
- All code in `foremankit/` meets these limits (`foremankit check foremankit/*.py` says: clean). The only deliberate exception is `server.delegate`: its 11 parameters are the fields of the MCP tool schema; a comment marks them: `# codecheck: allow too-many-parameters (reason)`. Only this rule can be silenced, and a silencing without a reason in parentheses is ignored.

### Preliminary code review by a second model

`delegate(..., review=true)` (CLI `--review`) runs, after `verify`, a review by a model **other** than the author: its own `llm` (node pool, fast Legion first) or a paid one (`deepseek:flash`, `gemini:flash`). Free clouds never review (shallow, and some train on code), and `private` data stays on your own nodes. The chain is `chains.review` in `routing.yaml`; a slow node gets the review at the end (`llm.kinds.review`, `max_seconds`).

- The reviewer answers with one line per issue: `SEVERITY | where | problem | evidence | fix`, where evidence is a literal quote from the code. A remark is **justified** only when the quote really appears in the code; invented remarks are marked in the report as unverified and never trigger a fix.
- Justified remarks with a severity from `review.fix_severities` (by default only `blocker`) go back to the author **once** (`review.fix_rounds`), and then `verify` checks the file again. If the fix breaks a test, the version from before it is kept. The remaining remarks only go into the report.
- A missing reviewer (nobody is available or the data is too confidential) does not stop the task: the report then has `review: SKIPPED (reason)`.
- `foremankit review file.py ...` or `foremankit review --diff paths` reviews existing code (e.g. your own changes before handing them on); exit code 1 when there is a justified `blocker`. Options: `--task "co ten kod ma robić"`, `--sensitivity`, `--allow-cost`.
- Cost: output up to 1500 tokens (`review.max_output_tokens`); a local node is free, and a paid flash costs a fraction of a cent per file.

### Saving

- Each `kind` has a fixed system prompt at the beginning, and the variable data (files, notes, task) goes at the end. This way DeepSeek hits the prefix cache, which costs about 10% of the input price.
- `max_output_tokens` is capped at 16000, and each provider has a timeout.
- The log `~/.config/foremankit/calls.jsonl` contains only metadata: provider, model, tokens, cost, time, success. **Prompt contents are not logged** (`log_prompts: false`).

---

## Privacy (fail closed)

Three levels of `sensitivity`:

| Level | For which data | Who can get it |
|---|---|---|
| `normal` | code, public or anonymised data | all providers; to external models only the schema and a few rows |
| `internal` | process data (tags, PI/SQL extracts, internal documents) | only providers that **do not train on data** and have an explicit `allow_internal: true` (paid Gemini), or your own LLM pool nodes. Never free tiers. DeepSeek has `trains_on_data: true` (its policy, checked 2026-10-08, permits using prompts for training, and the data is stored in China), so by default it is excluded. You can deliberately allow it **on a given machine** with the variable `FOREMANKIT_ALLOW_INTERNAL_TRAINING=deepseek` in `.env` (the user's decision of 2026-10-08: process data is not a secret). For `private` it never works. The corporate network blocks DeepSeek, so it is worth confirming compliance with company policy. The line limit for data files is lifted |
| `private` | passwords, personal data, recipes and trade secrets | only machines with `accept: private` |

For `internal` there is no network address check (Gemini is a public endpoint). If no authorised provider works, you get an error and nothing is sent anywhere else. Classifying process data as "internal" is decided by your company's policy, so when in doubt ask IT.

- `sensitivity="private"` always goes to machines with `accept: private`, and only to them. When none works, you get a **clear error** and nothing is sent to the cloud. The router rejects the cloud even when someone adds it to the `private` chain.
- Providers with `trains_on_data: true` and DeepSeek never get private data.
- **netguard**: private data goes only to addresses localhost/127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12 and 192.168.0.0/16. The host name is resolved via DNS and if any address is public, the router refuses. When DNS fails, it also refuses. The workaround is `allow_public_endpoint: true` on a given instance, with a warning in the log. Do not do this without a clear reason.
- File guard: a data file (`.csv`, `.tsv`, `.jsonl` etc.) longer than 60 lines is rejected for an external model with a request for a schema and a few anonymised rows. Binary files (parquet, pickle) are always rejected.
- Treat company and process data (PI tags, SQL extracts, recipes) as `private`. This rule is also in the rules for Claude.

---
## LLM node pool (`config/nodes.yaml`)

Local models (`llama-server` from llama.cpp or Ollama; any server answering on `/v1/models` or `/api/tags` is enough) form the **`llm` pool**. Each lab machine is one node, including the one you are working on. The same `nodes.yaml` sits on every machine, and a machine recognises itself by hostname (or by `FOREMANKIT_SELF`). A machine that is not in the file becomes a `local` node and uses the others. The pool is designed for a network of up to about 5 computers. A machine without a GPU or without a server has `serves: false` (it only uses the pool); a CPU-only server is an ordinary node with a low speed.

### How the router picks a node

When the chain reaches the `llm` entry, the router:

1. skips nodes that are disabled, unreachable (all are probed in parallel, the result is remembered for 30 s) and those that do not accept the task's data level (`accept`);
2. for the rest it estimates the completion time: `input_tokens / prompt_tps + min(max_output, expected_output_cap) / gen_tps`, and for a node whose server is currently busy (`GET /slots`) it adds `llm.queue_penalty_s` (30 s). The speed is the stored `speed` moved towards what the node **actually delivered** (see below);
3. it tries nodes from the fastest (a tie is won by the machine with the higher priority). This way the faster machine is used first **on every machine**, not just on one;
4. a node that is too slow for the task, or that is not entitled to the task's kind or size (`routing.yaml → llm.kinds`, overridden by the node's `kinds` field), gets the task only last in line. `private` data has no other source, so there size does not block.

Limits, quotas and cooldown as well as statistics are counted **separately for each node** (`llm@legion`), so a failure of one machine does not disable the others.

### Node fields in `nodes.yaml`

| Field | Meaning |
|---|---|
| `hostname` | machine name by which the file recognises "me" |
| `lan` | addresses checked in order (a name, an IP or a full URL `https://...`, which is used unchanged) |
| `port` | server port, 8080 by default |
| `serves` | `false` = the machine only uses the pool |
| `accept` | the highest data level the node may receive: `normal` < `internal` < `private`. **Defaults to `private`**: lab machines exist precisely so that confidential data does not leave your own hardware. Lower it for a machine you do not trust |
| `speed` | `prompt_tps`, `gen_tps`, `expected_output_cap`, `max_seconds`: the measured speed (`llama/tune_llama.py`, results in `llama/profiles.json`) |
| `learn` | `true` (default): the router learns the node's speed from real calls; `false`: the stored `speed` is final (for nodes measured exactly) |
| `kinds` | optionally: input limits per task kind for this node |
| `max_concurrent` | how many requests the server accepts at once (1 by default) |
| `priority` | a lower number = a more important machine (work before private); it resolves ties and borrowing rules |

The `settings` sections (`key_env`, `loan_to_higher_priority`) and `defaults` (the field value for all nodes) are described in the file's comments.

### Learning the speed

The stored `speed` is an expectation. Another machine's speed (a colleague's laptop, a shared server) depends on its load and the network and **fluctuates**, so after every successful call to a node with `learn: true` the router updates a moving average of what the node delivered: exactly from the `timings` field of the `llama-server` response, and when it is missing, approximately from the response time (for responses of 64 tokens or more). The learned value takes over planning gradually: with `n` measurements it has a weight of `n / min_samples` (3 by default), and a node unused for longer than `max_age_h` (72 h) reverts to the stored speed. The parameters are in `nodes.yaml → settings.speed_learning`. Your own, exactly measured laptops have `learn: false`. `foremankit nodes` shows whether the speed is pinned, learned or still just stored.

### Borrowing from a more important machine

A machine borrowing from a machine of **higher priority** does so only when its server is idle (`/slots`), and only for tasks estimated to take less than `loan_to_higher_priority.max_seconds` (45 s). The owner never waits long for someone else's task. When all machines have the same priority, there are no restrictions.

### Key and network security

Every node server runs with `--api-key`, and the shared key is `FOREMANKIT_PEER_KEY` in `.env` (not in git). You create it with `foremankit nodes key` and you must manually copy the same line into the `.env` of every machine. `private` data goes only to addresses passing netguard (described above) and only to nodes with `accept: private`.

### Switches and diagnostics

```bash
foremankit nodes                        # table: reachable, busy, speed, rule, accepted data level, netguard
foremankit nodes key                    # creates FOREMANKIT_PEER_KEY in .env
foremankit llm                          # state of all nodes (models, selected model)
foremankit llm test                     # probe + short generation on every node (or: foremankit llm legion test)
foremankit llm others off               # turn off other machines; also: self, all, or a node name; on | off | auto
```

Environment variables (priority: CLI state > env > config; the most specific wins: node, then `self`/`others`, then `all`):

- `FOREMANKIT_NODE_<NAME>_ENABLED`, `LLM_SELF_ENABLED`, `LLM_OTHERS_ENABLED`, `LLM_ENABLED` (`auto|on|off`);
- `FOREMANKIT_NODE_<NAME>_LAN="ip1,ip2"` : an address without editing the file, e.g. with a changing DHCP lease;
- `LLAMACPP_URL` (`http://127.0.0.1:8080` by default) and `OLLAMA_LOCAL_URL` (`http://localhost:11434` by default): servers on this machine, checked in that order;
- `LOCAL_LLM_GEN_TPS`, `LOCAL_LLM_PROMPT_TPS`: override the speed of **this machine only**;
- `FOREMANKIT_SELF` (a machine name other than the hostname), `FOREMANKIT_NODES=off` (ignore `nodes.yaml`).

Profiles (`config/profiles.yaml`) use the keys `"llm:self"`, `"llm:others"`, `"llm:all"` or `"llm:<node>"`.

> **Renaming.** The providers `llm_local` / `llm_remote` (previously `ollama_*`), the `peers.yaml` file, the `foremankit ollama` and `foremankit peers` commands, and the variables `LLM_LOCAL_ENABLED`, `LLM_REMOTE_ENABLED`, `LLM_REMOTE_*_URL`, `OLLAMA_REMOTE_*` were removed without aliases. An old config is rejected with an error and a hint (`peers.yaml` → `nodes.yaml`, key `peers:` → `nodes:`).

### Reaching a node (LAN)

`foremankit llm test` checks every node: response time, model list, netguard verdict and a short generation. The commands below concern Ollama (port 11434). For `llama-server` use `--host <LAN address> --api-key <key>` (the launcher `llama/start-llama.ps1 -Share` and the firewall rule `llama/share-firewall.ps1`), port 8080.

**1. LAN (home).** On the remote machine:
```bash
# Linux (systemd): sudo systemctl edit ollama  ->  [Service] Environment="OLLAMA_HOST=0.0.0.0"
sudo ufw allow from 192.168.1.0/24 to any port 11434 proto tcp      # home subnet only
```
```powershell
# Windows: system variable OLLAMA_HOST=0.0.0.0, restart Ollama, then:
New-NetFirewallRule -DisplayName "Ollama LAN" -Direction Inbound -Protocol TCP -LocalPort 11434 -RemoteAddress 192.168.1.0/24 -Action Allow
```
On the laptop set `FOREMANKIT_NODE_<NAME>_LAN=http://192.168.1.50:11434`, where 192.168.1.50 is the address of the remote machine. It is safer to give the LAN address itself instead of `0.0.0.0`, e.g. `OLLAMA_HOST=192.168.1.50`.

**2. Never expose port 11434 to the internet.** Ollama has no authentication and no TLS.

---
## Configuration and files

| What | Where |
|---|---|
| providers, models, prices, limits | `config/providers.yaml` |
| chains, system prompts, budget, file guard | `config/routing.yaml` |
| profiles | `config/profiles.yaml` |
| nodes | `config/nodes.yaml` |
| config override on a given machine | `~/.config/foremankit/config/*.yaml` or `FOREMANKIT_CONFIG_DIR` |
| state (counters, switches, spending, cooldowns) | `~/.config/foremankit/state.json` (`FOREMANKIT_STATE_DIR`) |
| call log | `~/.config/foremankit/calls.jsonl` |
| results | `<project>/.agent-out/` (`FOREMANKIT_OUT_DIR`) |
| auto-checkpoint hook checkpoints | `<project>/.agent-out/checkpoint-<time>.md` |
| state snapshot from before installation | `~/.config/foremankit/install-state/` (used by `install.py --restore-original`) |
| gateway context counter | `~/.config/foremankit/gateway_context.json` |
| gateway logs (Windows) | `%TEMP%/router.log`, `%TEMP%/router.err` |
| VS Code settings changed by `--local-orchestrator` | `settings.json` and `tasks.json` in the VS Code user folder (Windows: `%APPDATA%/Code/User`, Linux: `~/.config/Code/User`, macOS: `~/Library/Application Support/Code/User`) |
| backups before changing a file | next to the file, as `*.bak-foremankit-<date>` |

The files for Claude (`claude/agents/*.md`, `claude/CLAUDE.global.md`) are in English, because they are loaded in every session and that way they use fewer tokens.

### VERIFY items

Everything marked `# VERIFY` in `providers.yaml` (model IDs, limits, DeepSeek prices, `trains_on_data`) and the Sonnet reference price in `routing.yaml` are **unverified**. After you set the keys, run `foremankit models`: it will print the IDs that are missing on the endpoint. The DeepSeek prices in the config are deliberately overestimated so that the budget kicks in sooner rather than later.

## Troubleshooting

| Symptom | Cause and what to do |
|---|---|
| The Claude Code plugin in VS Code does not connect to any model (cloud ones included) | The VS Code settings point to the gateway (`ANTHROPIC_BASE_URL` = `http://127.0.0.1:8081`) and the gateway is not running. Start it: `llama\Start-Router.ps1` (check: `-Status`). If you do not want the gateway, remove the entries: `python install.py --uninstall`, then restart VS Code. |
| After uninstalling, the plugin uses the real Sonnet 5 instead of Qwen | That is expected: without the gateway, the identifier `claude-sonnet-5` goes to the cloud. Pick the target model in the picker. |
| The `delegate` tool is not visible or returns an error | Check the registration: `claude mcp get foremankit` and `python install.py --check`. MCP tools are sometimes "deferred" (visible by name only): load their schemas through ToolSearch before you call them. |
| `install.py` says `settings.json` is not clean JSON | The file contains comments, so the installer leaves it alone. Fix the settings by hand or remove the comments from it and run it again. |
| I do not know what of foremankit is installed | `python install.py --check` (read-only). |
| I want to go back to the state from before installation, down to the byte | `python install.py --restore-original`, then `--uninstall` to unregister the MCP. |
| Two local model servers are running | Start the server only through `llama/start-llama.cmd`. `-List` shows what is running, `-Replace` swaps it, `-Force` is for a deliberate second server on a different port. |
| A provider is skipped | `foremankit status` shows which providers are active, their limits and spending; a blocked corporate network skips a provider automatically. |

## Code structure

`router.py` is a thin facade (`Router`); the work happens in small modules:

| module | job |
|---|---|
| `chain.py` | walking the chain of task sources (`ChainRun`), ordering, deferred sources, an error when nothing responds |
| `candidates.py` | turning a chain entry (a cloud model or a node) into a ready candidate or a reason for skipping |
| `caller.py` | a single call: retries, cooldowns, usage accounting, rejecting empty and truncated responses |
| `probing.py`, `selection.py` | node and cloud probes, node ranking, concurrency limits |
| `postcheck.py` | the `verify` command with fixes by the same model, and `review` by a second model |
| `prompting.py`, `outputs.py`, `reasoning.py` | prompt and context files, saving responses, reasoning level |
| `reports.py`, `formatting.py` | status, models, statistics, node tests and their text display |
| `records.py` | small data classes passed between steps (`TaskRequest`, `CallContext`, `Candidate`, `Answer`) |

`Router(cfg, RouterEnv(...))`: the environment (state, log, network, clock) is passed as a single object. `delegate(task, kind, sensitivity, **options)` accepts the options from `TaskRequest`.

## Tests

```bash
.venv/bin/python -m pytest -q        # Windows: .venv\Scripts\python.exe -m pytest -q
```
HTTP is mocked (`httpx.MockTransport`). The tests cover: routing, skipping providers without a key, fallback after 429, retry after 5xx, hard budget, cost with a cache hit, escalation, rpd limits, independent on/off of both Ollama instances, CLI > env > config priority, private to remote when local is off, refusal for a public address, `allow_private=false`, fail closed without Ollama, the file guard, profiles, dry run and an end-to-end test of the MCP server over stdio. The tests of the `nodes.py` and `verify.py` modules cover: node ranking, CLI > env > config priority, LAN addresses, machine priority, data limit, cost, the file guard, profiles, dry run and an end-to-end test of the MCP server over stdio.

---

## Status

An early, single-author project. It is developed and tested mainly on Windows; the Linux and macOS code paths exist but are exercised less. Interfaces, config keys and defaults may change between versions. Issues and pull requests are welcome.

## Data and privacy notice

The content of every delegated task is sent to the provider that the router selects (for example DeepSeek, Google, Mistral or Groq), under that provider's terms. Free tiers may use prompts for training. Use the `sensitivity` levels described above, keep secrets and personal data on `private`, and check the terms of every provider you enable. You are responsible for what you send.

## Trademarks

foremankit is an independent project and is not affiliated with or endorsed by Anthropic, DeepSeek, Google, Mistral, Groq, Cerebras, OpenRouter or any other provider named here. Claude and Claude Code are trademarks of Anthropic. Other names belong to their owners.

## License

[MIT](LICENSE). Copyright (c) 2026 Marcin Surunowicz.

## External orchestrator (Gemini / DeepSeek instead of Claude)

- When the weekly Claude limit is exhausted, Claude Code can be redirected to the ForemanKit gateway (`ANTHROPIC_BASE_URL=http://127.0.0.1:8081`, `ANTHROPIC_AUTH_TOKEN=foremankit`) and orchestrated by DeepSeek or Gemini while ForemanKit delegation continues to work.  
- Orchestrator selection commands:  
  - `foremankit orchestrator use deepseek|gemini|claude|local`  
  - `foremankit orchestrator status|env` to view current status or environment.  
- Configuration lives in `config/orchestrator.yaml` under `upstreams`, each entry defining:  
  - `url` (provider endpoint)  
  - `key_env` (environment variable holding the provider key)  
  - `ceiling` (maximum allowed data‑sensitivity)  
  - `models` mapping aliases (`opus`, `sonnet`, `haiku`) to provider‑specific model IDs and context.  
- Provider endpoints must implement the Anthropic Messages API:  
  - DeepSeek: `https://api.deepseek.com/anthropic`  
  - Gemini: accessed via a LiteLLM proxy URL placed in the YAML.  
  - Model IDs/URLs are placeholders for verification.  
- The ForemanKit gateway:  
  - Replaces incoming client credentials with the provider key from `key_env` in `.env`.  
  - Strips the `anthropic-beta` header.  
  - Routes tool‑less side requests (e.g., title extraction, auto‑mode checks) to the `haiku` alias.  
  - Appends `config/orchestrator.external.md` to the main loop.  
- Privacy handling:  
  - The orchestrator can read every file it processes.  
  - Each repository must contain `.foremankit/sensitivity` set to `normal`, `internal`, or `private` (missing defaults to `private`).  
  - `orchestrator use` refuses a provider if the repo’s sensitivity exceeds that provider’s `ceiling` (DeepSeek: `normal` only; Gemini paid: `internal`; no external data for `private`).  
- Guard hook: `hooks/orchestrator_guard.py` (installed by `install.py`) implements a `PreToolUse` check that blocks tool calls on paths above the ceiling or matching `deny_globs`, failing closed.  
- Outstanding limitations:  
  - No cost‑metering for orchestrator token usage.  
  - Session‑length hint not yet adapted.  
  - Gemini requires a LiteLLM setup.  
  - Weaker models’ tool‑calling quality still needs smoke‑testing (e.g., 10 edits, a delegate call, router status).
