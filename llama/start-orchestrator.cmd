@echo off
rem Local model as the Claude Code orchestrator (via the gateway :8081); also what the VS Code autostart task runs.
rem Starts the DEFAULT model (Qwen3.6 35B Q4_K_XL) with the values from its tuned profile in profiles.json (ctx, ncmoe, threads,
rem KV cache, thinking off): change them there or with `python sweep_ctx.py`, not here.
rem   -NoMenu          the model menu would wait for a keypress in the VS Code terminal and the server would never start
rem   -Share           lend the server to the other laptop (needs FOREMANKIT_PEER_KEY)
rem   thinking         off at start; the gateway turns it on or off per request from Claude Code's effort slider
rem                    (config/orchestrator.yaml thinking_by_effort). Force a cap instead: set FOREMANKIT_THINK_BUDGET=1024
rem Extra options pass through (e.g. -Replace). Refuses a second server on this machine.
set THINK=
if defined FOREMANKIT_THINK_BUDGET set THINK=-ThinkBudget %FOREMANKIT_THINK_BUDGET%
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-llama.ps1" -NoMenu %THINK% -Share %*
