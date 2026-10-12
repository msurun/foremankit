# Model sweep (Legion/Firefly: one machine, one run per file)

Generated 2026-10-10 22:40. Quality = mean of the short-task, long-task and agent-task pass rates (thinking off).
Speed = measured generation tok/s over the answers longer than 30 tokens. Small samples: differences of one task are noise.

## Ranking

| # | model | size GiB | ncmoe | KV | tuner tok/s (gen / prompt) | measured gen tok/s | short | long | agent | quality | s/short | s/long | s/agent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3.6-35B-A3B-UD-Q5_K_M | 24.6 | 34 | f16 | 33.0 / 241 | 33.7 | 17/18 | 14/15 | 2/2 | 0.96 | 4.7 | 19.0 | 61 |
| 2 | Qwen3.6-35B-A3B-UD-Q4_K_XL | 20.8 | 32 | f16 | 40.8 / 296 | 38.7 | 15/18 | 14/15 | 2/2 | 0.92 | 4.4 | 16.6 | 75 |
| 3 | gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.8 | 24 | f16 | 31.0 / 343 | 30.3 | 15/18 | 14/15 | 2/2 | 0.92 | 4.3 | 14.5 | 56 |
| 4 | Qwen3.6-35B-A3B-UD-Q3_K_XL | 15.7 | 31 | f16 | 36.1 / 397 | 38.2 | 17/18 | 11/15 | 2/2 | 0.89 | 3.9 | 14.0 | 52 |
| 5 | Qwen3.6-35B-A3B-UD-Q2_K_XL | 11.4 | 21 | f16 | 53.1 / 601 | 48.4 | 16/18 | 10/15 | 2/2 | 0.85 | 2.9 | 10.3 | 33 |
| 6 | Qwen3.6-35B-A3B-UD-IQ1_M | 9.4 | 15 | f16 | 57.5 / 828 | 55.5 | 15/18 | 10/15 | 2/2 | 0.83 | 2.2 | 7.3 | 32 |
| 7 | Qwen3.5-35B-A3B-UD-Q4_K_XL | 20.7 | 31 | f16 | 41.0 / 304 | 39.7 | 15/18 | 10/15 | 2/2 | 0.83 | 4.3 | 15.0 | 51 |
| 8 | gemma-4-26B-A4B-it-UD-Q3_K_XL | 12.0 | 20 | f16 | 33.6 / 536 | 31.2 | 15/18 | 10/15 | 2/2 | 0.83 | 3.9 | 11.7 | 50 |
| 9 | Qwen3.8-35B-A3B-Q4_K_M | 20.2 | 30 | f16 | 43.8 / 312 | 42.2 | 16/18 | 7/15 | 1/2 | 0.62 | 3.1 | 14.2 | 225 |
| 10 | NVIDIA-Nemotron-3.5-Lightning-30B-A3B-IQ4_XS | 17.6 | 37 | f16 | 38.1 / 297 | 42.2 | 13/18 | 5/15 | 1/2 | 0.52 | 3.8 | 15.5 | 102 |
| 11 | Qwen3-Coder-30B-A3B-Q4_K_M | 17.3 | 36 | q8_0 | 27.1 / 292 | 29.8 | 16/18 | 10/15 | 0/2 | 0.52 | 6.5 | 13.7 | 398 |
| 12 | Qwen3.5-9B-UD-Q4_K_XL | 5.6 | None | f16 | 58.7 / 1884 | 55.6 | 14/18 | 7/15 | 0/2 | 0.41 | 2.2 | 5.5 | 47 |
| 13 | Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL | 21.3 | 43 | f16 | 26.9 / 214 | 29.8 | 11/18 | 1/15 | 0/2 | 0.23 | 4.4 | 16.8 | 128 |

## Pass rate per task (correct / runs)

| model | S1 | S2 | S3 | S4 | S5 | S6 | L1 | L2 | L3 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B-UD-Q5_K_M | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 5/5 | 5/5 | 4/5 |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 5/5 | 5/5 | 4/5 |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 5/5 | 5/5 | 4/5 |
| Qwen3.6-35B-A3B-UD-Q3_K_XL | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 5/5 | 4/5 | 2/5 |
| Qwen3.6-35B-A3B-UD-Q2_K_XL | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 5/5 | 4/5 | 1/5 |
| Qwen3.6-35B-A3B-UD-IQ1_M | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 5/5 | 5/5 | 0/5 |
| Qwen3.5-35B-A3B-UD-Q4_K_XL | 1/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 5/5 | 4/5 | 1/5 |
| gemma-4-26B-A4B-it-UD-Q3_K_XL | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 5/5 | 5/5 | 0/5 |
| Qwen3.8-35B-A3B-Q4_K_M | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 5/5 | 2/5 | 0/5 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-IQ4_XS | 1/3 | 1/3 | 3/3 | 2/3 | 3/3 | 3/3 | 5/5 | 0/5 | 0/5 |
| Qwen3-Coder-30B-A3B-Q4_K_M | 3/3 | 2/3 | 3/3 | 2/3 | 3/3 | 3/3 | 5/5 | 5/5 | 0/5 |
| Qwen3.5-9B-UD-Q4_K_XL | 0/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 5/5 | 2/5 | 0/5 |
| Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL | 1/3 | 0/3 | 3/3 | 1/3 | 3/3 | 3/3 | 0/5 | 1/5 | 0/5 |

Task ids: S1 pipes (math); S2 ordering (logic); S3 balanced brackets (code); S4 ISO dates (code); S5 moving average (code); S6 log to JSON (extraction); L1 needle in 10k-token code (retrieval); L2 TTL cache (implementation); L3 longest run in a table (multi-step reasoning)
