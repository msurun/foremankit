# Qwen3.6 Q4_K_XL vs Qwen3.6 Q5_K_M vs Gemma 4 26B-A4B Q4_K_XL (2026-10-09)

HP ZBook Firefly 14 G11, AC power, llama.cpp b11228. Every model runs at ITS OWN tuned settings (`llama/profiles.json`,
produced by `llama/tune_llama.py`): P-core pinning (0xFFC03, 12 threads), ctx 32768, `--load-mode none`, fa on.
Gemma needs KV q8_0 to fit 32k in 4 GB VRAM. Sampling per model card (Qwen: t0.6/top_p0.95/top_k20; Gemma: t1.0/top_p0.95/top_k64).
Small samples (short tasks x2, long x1, agent runs x2): a difference of one task is noise.

## Tuned speed (llama-bench, 3 repetitions)

| model | file | n-cpu-moe | generation | prompt |
|---|---|---|---|---|
| Qwen3.6 Q4_K_XL | 20.8 GB | 40 | 19.6 tok/s | 192 tok/s |
| Qwen3.6 Q5_K_M | 24.6 GB | 41 | 18.9 tok/s | 167 tok/s |
| Gemma 4 Q4_K_XL (KV q8_0) | 17.0 GB | 31 | 16.5 tok/s | 215 tok/s |

## Short/long tasks with automatic checking (`quality_bench.py`, 6 short x2 + 3 long)

| model | thinking | short correct | long correct | mean time short / long | tokens per short task |
|---|---|---|---|---|---|
| Q4 | off | 12/12 | 2/3 | 11 s / 35 s | 179 |
| Q4 | on | 10/12 | 3/3 | 82 s / 202 s | 1497 |
| Q5 | off | 12/12 | 2/3 | 8 s / 36 s | 111 |
| Q5 | on | 10/12 | 2/3 | 87 s / 199 s | 1513 |
| Gemma | off | 10/12 | 3/3 | 8 s / 26 s | 95 |
| Gemma | on | 10/12 | 2/3 | 70 s / 210 s | 1082 |

Misses: thinking-on S4 (ISO dates) hit the 3000-token cap on every model; Gemma without thinking answered the pipes problem
in 4 tokens and got it wrong twice; Q5 off failed the TTL cache (IndexError), Q4 off failed the table calculation.

## End-to-end agent task (`agent_task.py`: fix 4 bugs with tools; success = visible 12/12 + hidden 9/9, tests untouched, signatures kept)

| model | thinking | success | mean wall time | to first green tests | completion tokens |
|---|---|---|---|---|---|
| Q4 | off | 2/2 | 112 s | 99 s | 1568 |
| Q4 | on | 2/2 | 154 s | 131 s | 2203 |
| Q5 | off | 1/2 | 128 s | 115 s | 1740 |
| Q5 | on | 2/2 | 227 s | 210 s | 3235 |
| Gemma | off | 2/2 | 103 s | 85 s | 1223 |
| Gemma | on | 2/2 | **508 s** | 388 s | 7242 |

Note: a first series was invalid (the hidden test required sorted days, the docstring did not say so) and is kept as
`agent_results_v1_flawed.jsonl`; the signature check was later relaxed to the four required functions (private helpers allowed).
Q5-off failure: it ignored the sorted-days requirement in one run.
Code review of the final files (`agent_solutions/`): all correct solutions keep the docstrings and edit minimally. Q4's are the most
compact (63-71 lines, one leftover unused import), Gemma's are longer (80-85 lines, trailing whitespace, explicit loops), Q5's in between.

## Firefly (local) vs Legion (over the LAN): same tasks, sent at the same time (`compare_llamas.py`, 2026-10-09)

Firefly: ZBook Firefly 14 G11, **Qwen3.6 Q4_K_XL** (build b11228). Legion: Ryzen 7 5800H + RTX 3070 8 GB, **Qwen3.6 Q5_K_M** (build b11429),
reached through the LAN with the shared API key. So this compares two whole systems (hardware + quantization + build), not the CPU alone.
Both received each task simultaneously, thinking off (plus thinking on for S2 and L3), same sampling, cache_prompt off.
A first run was discarded: two llama-server instances were running on the Legion at the same time (`compare_llamas_run1_two_instances_on_legion.jsonl`).

| set | Firefly correct | Legion correct | Firefly mean time | Legion mean time | Legion faster by | Firefly gen tok/s | Legion gen tok/s |
|---|---|---|---|---|---|---|---|
| short, thinking off | 11/12 | 12/12 | 7.9 s | 5.2 s | 35 % | 17.9 | 30.2 |
| long, thinking off | 5/6 | 3/6 | 26.0 s | 21.8 s | 16 % | 19.1 | 32.9 |
| thinking on (S2, L3) | 2/2 | 2/2 | 86.1 s | 50.1 s | 42 % | 18.9 | 31.8 |
| all | 18/20 | 17/20 | 21.1 s | 14.6 s | 31 % | 18.4 | 31.2 |

* Prompt processing of a 6k-token prompt: Firefly 157 tok/s (38.6 s), Legion 195 tok/s (31.1 s).
* Quality: no difference on the short tasks (11/12 vs 12/12; the only miss is the pipes problem answered in 2-4 tokens). L3 (table arithmetic)
  was missed once by each machine with the very same wrong answer. The one systematic difference: **L2 (TTL cache implementation) failed 2/2 on
  the Legion (Q5_K_M, "IndexError: pop index out of range") and passed 2/2 on Firefly (Q4_K_XL)**; the Q5_K_M model also failed L2 on Firefly in the
  earlier quality run. Small sample, but consistent with "Q5_K_M is not better than Q4_K_XL" from the model comparison above.
* Practical reading: the Legion is ~70 % faster at generating (31 vs 18 tok/s) and ~25 % faster at reading prompts, which cuts wall time by
  16-42 % depending on how generation-heavy the task is; use it first for latency, but quality-sensitive code tasks should not rely on the Q5 file.
