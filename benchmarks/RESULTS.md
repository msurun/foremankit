# Qwen3.6-35B-A3B on HP ZBook Firefly 14 G11 - llama.cpp benchmark (2026-10-08/09)

Hardware: Core Ultra 7 155H (6 P-cores/12 threads, 8 E-cores, 2 LP-E-cores), 64 GB DDR5-5600, RTX A500 Laptop 4 GB.
Build: llama.cpp b11228 (commit 4364bf723), CUDA backend active (ggml-cuda.dll, CPU backend ggml-cpu-alderlake.dll).
Conditions: AC power, Windows plan **Balanced (unchanged)**, `-r 3` repetitions (depth runs `-r 2`), mean ± stddev,
cooldown guard before each run (< 75 C), 3 s delay between tests. CPU zone peaked at 83-93 C under load in every run;
no run showed a thermal collapse (results stable run to run). Raising the laptop on stands did not change the picture
(re-check: same config 17.0 -> 17.4 tok/s, Tmax 88-90 C either way).
Raw data: `results.csv` (bench), `server_results.jsonl` (server), logs `run1.log`, `run2.log`, `srv_*.log`.

## 1. What actually helped / what did not

| change | effect on generation (tg) | verdict |
|---|---|---|
| `--n-cpu-moe 40` (38 ties, 42 slightly slower) | 17.4 | keep 40; 38 gains nothing and leaves ~200 MiB VRAM free |
| `--n-cpu-moe` 37 / 36 | **7.5 / 5.9** (pp 18-20) | VRAM full (3.9 GB) -> spill to shared memory; never go below 38 |
| Pin to the 6 P-cores, SMT on (`-C 0xFFC03 --cpu-strict 1 -t 12`) | **18.2-18.4 vs 17.0** (+7 %), spread 0.14-0.34 vs 0.5-2.3 | **helped** |
| P + E cores (t=18 or 14) | 11.4 / 15.7 | **hurts**: slow cores stall the synchronised threads |
| Threads without a mask: 4...22 | 14.9-17.0, best t=8/12, worst t=22 | no better than the P-core mask |
| One thread per P-core (t=6, 0x57C01) | 16.1 | worse than SMT on |
| Quantization Q4_K_XL instead of Q5_K_M | **19.05-19.23** vs 18.4; pp **192-197** vs 167 | **helped** (+4 % tg, +15 % pp, 3.8 GB less RAM) |
| `-fa off` | 18.15 | no effect on speed (keep on for memory) |
| KV cache q8_0 | 18.26 | no speed gain; only saves VRAM, not needed |
| `-b/-ub` 512...4096 | 17.7-18.6 | noise; ub 2048 slightly worse; keep defaults |
| `--load-mode` mmap / mmap+mlock / auto | tg same, **pp 139-150 vs 167** | `none` is best |
| `--poll 0/100`, `--prio 2` | 18.2-18.3 | no effect |
| `-nopo 1` (no op offload) | pp **115**, tg 17.7 | worse |
| Context depth 4096 / 16384 | tg 18.4 / 18.0; pp 158 / 153 | hybrid model: long context nearly free for generation |
| Laptop on stands | none measurable | - |

Conclusion: the old command was already close to the optimum. Real gains: **P-core pinning (+7 %)** and **Q4_K_XL (+4 % tg, +15 % pp)**,
together about 19 tok/s generation and ~195 tok/s prompt, versus ~17 / 167 before. Nothing else on the list matters.

## 2. Real Claude-Code-like conditions (llama-server, Q4_K_XL, ctx 32768)

| test | configuration | result |
|---|---|---|
| TTFT cold 9.7k tokens | Q4_K_XL, ncmoe 40, P-core mask | **61.8 s** (157 tok/s prompt) |
| TTFT 9.7k, same prefix + new question | same | **0.7 s** (only 24 new tokens processed) |
| TTFT cold 20.0k tokens | same | **128.9 s** (155 tok/s), VRAM 3.3 GB of 4 GB |
| TTFT cold 28.5k tokens (first, mis-sized run) | same | 187 s (153 tok/s); prefix reuse 0.8 s |
| Drop a middle chunk (A+B+C then A+C, 9.4k) | no cache-reuse | everything recomputed: cache_n = 0, 61 s |
| same | `--cache-reuse 256` | **no change**: cache_n = 0, 61 s (hybrid/recurrent model cannot shift or partially reuse KV) |
| Thinking, default | Q4 and Q5 | reasoning_content present, 210-243 tokens, 12-14 s for "17*23" |
| Thinking, `chat_template_kwargs {"enable_thinking": false}` per request | Q4 and Q5 | no reasoning, 4 tokens, 0.6 s |
| Thinking, server flag `--reasoning off` | Q4 | no reasoning without any request parameter, 0.9 s |
| Tool calling, 8 tasks x 3, card "precise coding" sampling (t0.6, pp0) | Q4_K_XL | **24/24**, 0 invalid JSON, 2.6 s/call |
| Tool calling, card "instruct" sampling (t0.7, top_p0.8, pp1.5) | Q4_K_XL | **24/24**, 0 invalid JSON, 2.4 s/call |
| Tool calling, t0.6 pp0 | Q5_K_M | 22/24, 0 invalid JSON (2x called read_file before edit_file - a defensible but different choice) |
| Tool calling, t0.7 pp1.5 | Q5_K_M | 24/24, 0 invalid JSON |

What this means in practice:
* The expensive part is the **first prompt**, not generation: ~6.2 s per 1000 tokens of new prompt. A 10k system prompt costs ~1 min once.
* An unchanged prefix is free, so keep system prompt + tool list identical and only append. Any edit earlier in the context (compaction, reordered tools, a changing timestamp in the prompt) forces a full re-read.
* `--cache-reuse` is useless for this architecture. `-np 1` is right (one slot, one cache).
* To switch thinking off use `--reasoning off` on the server (default for foremankit use); per request `enable_thinking=false` also works.
* Sampling from the Hugging Face model card: precise coding t=0.6 top_p=0.95 top_k=20 min_p=0 presence_penalty=0; instruct/non-thinking t=0.7 top_p=0.8 top_k=20 min_p=0 presence_penalty=1.5; thinking general t=1.0 top_p=0.95 top_k=20 pp=1.5. Both card presets give clean tool-call JSON.

## 3. Recommended command

```
llama-server.exe --model "...\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" --alias qwen3.6-35b-a3b --host 127.0.0.1 --port 8080 -c 32768 -ngl 99 --n-cpu-moe 40 --load-mode none --jinja -fa on -np 1 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --no-webui -t 12 -tb 12 -C 0xFFC03 --cpu-strict 1 --reasoning off
```
Script: `foremankit\llama\start-llama.ps1` (default model Q4_K_XL, `-PrintOnly` shows the command, `-CpuMask ''` and `-Threads 0` for other CPUs; run `start-llama.cmd` or the ps1). Omit `--reasoning off` when thinking is wanted.

## 4. Checked, and not worth it on this hardware

* Fewer `--n-cpu-moe` than 38 (VRAM spill) and any E-core/LP-E-core in the thread set.
* `--mlock`, `mmap`, `--poll`, `--prio`, `-b/-ub` tuning, flash attention off, q8_0 KV cache, `--no-op-offload`: all neutral or worse.
* `--cache-reuse` (hybrid architecture, zero reuse measured).
* A bigger quant: Q5_K_M is slower and no better at tool calling.
* Not tested (stated honestly): Intel iGPU via Vulkan/SYCL (shares the same RAM bandwidth, no gain expected; the shipped build has CUDA/CPU/RPC only), IQ4_XS (file not on disk), battery/High-performance power plans (only Balanced exists; battery lowers speed a lot, seen earlier).
* Windows "CUDA - Sysmem Fallback Policy" cannot be read programmatically. It cannot be proven from here that the setting is off; the measured spill at ncmoe <= 37 shows fallback is currently ON. For a predictable failure instead of a silent 3x slowdown set NVIDIA Control Panel > Manage 3D settings > "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback" (previous value: driver default). I did not change it.

## 5. All benchmark rows (llama-bench)

| # | phase | configuration | model | pp512 t/s | tg128 t/s | depth | CPU Tmax C | VRAM MiB | notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline | base (ncmoe40 fa1 t16 lm none) | Q5_K_M | 167.32 ± 2.16 | 16.98 ± 0.52 | 0 | 90.9 | 2731.0 | cooled 15s; |
| 2 | ncmoe | ncmoe=42 | Q5_K_M | 167.28 ± 2.31 | 16.19 ± 0.65 | 0 | 90.9 | 2731.0 | cooled 30s; |
| 3 | ncmoe | ncmoe=40 | Q5_K_M | 167.71 ± 2.38 | 17.43 ± 0.55 | 0 | 88.9 | 2731.0 | cooled 30s; |
| 4 | ncmoe | ncmoe=38 | Q5_K_M | 163.63 ± 1.4 | 17.01 ± 0.52 | 0 | 88.9 | 3889.0 | cooled 30s; |
| 5 | ncmoe | ncmoe=37 | Q5_K_M | 17.78 ± 0.13 | 7.51 ± 0.05 | 0 | 89.9 | 3913.0 | cooled 75s; |
| 6 | ncmoe | ncmoe=36 | Q5_K_M | 20.54 ± 0.14 | 5.88 ± 0.07 | 0 | 92.9 | 3913.0 | cooled 45s; |
| 7 | threads | t=4 (no mask) | Q5_K_M | 167.73 ± 2.14 | 15.65 ± 0.56 | 0 | 92.9 | 2731.0 | cooled 30s; |
| 8 | threads | t=6 (no mask) | Q5_K_M | 167.92 ± 2.15 | 16.55 ± 0.3 | 0 | 92.9 | 2731.0 | cooled 45s; |
| 9 | threads | t=8 (no mask) | Q5_K_M | 167.65 ± 2.17 | 16.82 ± 0.22 | 0 | 90.9 | 2731.0 | cooled 30s; |
| 10 | threads | t=10 (no mask) | Q5_K_M | 167.73 ± 2.25 | 16.44 ± 0.37 | 0 | 91.9 | 2731.0 | cooled 45s; |
| 11 | threads | t=12 (no mask) | Q5_K_M | 165.3 ± 4.39 | 16.96 ± 0.62 | 0 | 90.9 | 2731.0 | cooled 75s; |
| 12 | threads | t=14 (no mask) | Q5_K_M | 167.49 ± 2.73 | 16.21 ± 0.86 | 0 | 88.9 | 2731.0 | cooled 30s; |
| 13 | threads | t=16 (no mask) | Q5_K_M | 167.81 ± 1.87 | 15.88 ± 2.32 | 0 | 86.9 | 2731.0 | cooled 15s; |
| 14 | threads | t=20 (no mask) | Q5_K_M | 168.06 ± 2.23 | 16.14 ± 0.61 | 0 | 86.9 | 2731.0 | cooled 45s; |
| 15 | threads | t=22 (no mask) | Q5_K_M | 167.85 ± 2.07 | 14.87 ± 0.48 | 0 | 86.9 | 2731.0 | cooled 30s; |
| 16 | masks | P-cores SMT on, t=12 | Q5_K_M | 167.08 ± 2.93 | 18.15 ± 0.14 | 0 | 86.9 | 2731.0 | cooled 30s; |
| 17 | masks | P-cores SMT on, t=6 | Q5_K_M | 166.38 ± 0.71 | 16.64 ± 0.19 | 0 | 89.9 | 2731.0 | cooled 15s; |
| 18 | masks | P-cores 1 thr/core, t=6 | Q5_K_M | 167.89 ± 2.68 | 16.1 ± 1.29 | 0 | 85.9 | 2731.0 | cooled 15s; |
| 19 | masks | P+E (all but LP-E), t=18 | Q5_K_M | 167.46 ± 1.58 | 11.43 ± 0.97 | 0 | 83.9 | 2731.0 | cooled 15s; |
| 20 | masks | P 1thr/core + 8 E, t=14 | Q5_K_M | 167.72 ± 1.81 | 15.65 ± 0.98 | 0 | 89.9 | 2731.0 | cooled 30s; |
| 21 | recheck | OLD base t16 no mask (after cooling change) | Q5_K_M | 167.77 ± 2.11 | 17.42 ± 0.24 | 0 | 87.9 | 2731.0 |  |
| 22 | recheck | NEW base P-mask t12 | Q5_K_M | 167.13 ± 3.33 | 18.44 ± 0.34 | 0 | 89.9 | 2731.0 | cooled 30s; |
| 23 | fa | fa=0 | Q5_K_M | 166.31 ± 2.68 | 18.15 ± 0.21 | 0 | 89.9 | 2773.0 | cooled 30s; |
| 24 | kv | ctk/ctv q8_0 (needs fa) | Q5_K_M | 167.78 ± 2.09 | 18.26 ± 0.43 | 0 | 90.9 | 2763.0 | cooled 30s; |
| 25 | batch | b=512 ub=512 | Q5_K_M | 167.66 ± 1.73 | 18.57 ± 0.4 | 0 | 89.9 | 2731.0 | cooled 30s; |
| 26 | batch | b=1024 ub=512 | Q5_K_M | 167.86 ± 2.01 | 18.3 ± 0.19 | 0 | 89.9 | 2731.0 | cooled 30s; |
| 27 | batch | b=2048 ub=1024 | Q5_K_M | 167.83 ± 2.07 | 18.47 ± 0.43 | 0 | 90.9 | 2731.0 | cooled 30s; |
| 28 | batch | b=2048 ub=2048 | Q5_K_M | 165.13 ± 3.38 | 17.73 ± 0.47 | 0 | 88.9 | 2731.0 | cooled 30s; |
| 29 | batch | b=4096 ub=2048 | Q5_K_M | 167.56 ± 2.48 | 18.32 ± 0.1 | 0 | 89.9 | 2731.0 | cooled 30s; |
| 30 | load | lm=mmap | Q5_K_M | 138.93 ± 8.92 | 18.0 ± 0.18 | 0 | 90.9 | 2731.0 | cooled 30s; |
| 31 | load | lm=mmap+mlock | Q5_K_M | 149.89 ± 2.2 | 18.34 ± 0.44 | 0 | 89.9 | 2731.0 | cooled 30s; |
| 32 | load | lm=auto | Q5_K_M | 143.47 ± 7.92 | 17.73 ± 0.42 | 0 | 91.9 | 2731.0 | cooled 30s; |
| 33 | poll | poll=0 | Q5_K_M | 166.52 ± 3.1 | 18.2 ± 0.19 | 0 | 88.9 | 2731.0 | cooled 15s; |
| 34 | poll | poll=100 | Q5_K_M | 166.98 ± 3.31 | 18.25 ± 0.36 | 0 | 89.9 | 2731.0 | cooled 15s; |
| 35 | prio | prio=2 (high) | Q5_K_M | 166.45 ± 0.74 | 18.32 ± 0.09 | 0 | 89.9 | 2731.0 | cooled 30s; |
| 36 | opoff | no-op-offload=1 | Q5_K_M | 114.86 ± 1.65 | 17.66 ± 0.29 | 0 | 89.9 | 2779.0 | cooled 30s; |
| 37 | quant | Q4_K_XL ncmoe=40 | Q4_K_XL | 192.06 ± 2.29 | 19.05 ± 0.23 | 0 | 88.9 | 2731.0 | cooled 30s; |
| 38 | quant | Q4_K_XL ncmoe=38 | Q4_K_XL | 197.16 ± 1.47 | 19.23 ± 0.07 | 0 | 89.9 | 3727.0 | cooled 30s; |
| 39 | depth | base depth 0/4096/16384 | Q5_K_M | 165.67 ± 1.57 | 18.76 ± 0.85 | 0 | 88.9 | 3115.0 | cooled 30s; |
| 40 | depth | base depth 0/4096/16384 | Q5_K_M | 157.8 ± 0.48 | 18.37 ± 0.52 | 4096 | 88.9 | 3115.0 | cooled 30s; |
| 41 | depth | base depth 0/4096/16384 | Q5_K_M | 152.76 ± 2.66 | 17.96 ± 0.36 | 16384 | 88.9 | 3115.0 | cooled 30s; |
