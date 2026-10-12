@echo off
rem Qwen3.6 as the Claude Code orchestrator (via the gateway :8081); also what the VS Code autostart task runs.
rem   -Ctx 98304       the gateway slims Claude Code's ~100k-token requests to ~25k (config/orchestrator.yaml); 98304 measured: 7.6 of 8.2 GB VRAM, 31-37 tok/s;
rem                    196608 with --no-kv-offload (KV in RAM) measured 16 tok/s on an empty context and 3-4 on a long one (40 at 32k on GPU)
rem   -KvCache q8_0    half the KV memory, so the 64k KV cache stays on the GPU (about 0.7 GB of the ~1.3 GB left at -NCpuMoe 32)
rem   thinking         ON, capped at 1024 tokens per answer (--reasoning-budget); fewer follow-up questions, no measurable slowdown
rem                    in tests (same tok/s, a few hundred extra tokens per answer). Without a cap the budget vanishes into the thinking block.
rem   -Share           lend the server to the other laptop (needs FOREMANKIT_PEER_KEY)
rem   Other cap:  set FOREMANKIT_THINK_BUDGET=2048 && start-orchestrator.cmd -Replace     Thinking off:  set FOREMANKIT_THINK_BUDGET=0 && ...
rem Measured limit of this 8 GB GPU: 98304 holds (34k-token prompt: 202 tok/s prompt, 31 tok/s generation, 386 MiB free); 102400 loads but
rem crashes with CUDA out of memory while processing a long prompt; 106496 and above do not even load. Keep 98304.
rem Do not go above -NCpuMoe 32..34: 36 collapsed the speed in tests. If the load fails with an out-of-memory error, lower -Ctx.
rem Extra options pass through (e.g. -Replace). Refuses a second server on this machine.
if not defined FOREMANKIT_THINK_BUDGET set FOREMANKIT_THINK_BUDGET=1024
set THINK=-ThinkBudget %FOREMANKIT_THINK_BUDGET%
if "%FOREMANKIT_THINK_BUDGET%"=="0" set THINK=-NoThink
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-qwen3.6.ps1" -Ctx 98304 -NCpuMoe 32 -KvCache q8_0 %THINK% -Share %*
