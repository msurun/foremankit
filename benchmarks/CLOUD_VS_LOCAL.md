# bulk_code: cloud providers vs local models (same tasks, same automatic tests)

Generated 2026-10-28 12:13. Tasks: L2 TTL cache (implementation); S3 balanced brackets (code); S4 ISO dates (code); S5 moving average (code). Cloud rows go through foremankit's bulk_code path (system prompt, minimal reasoning); local rows are the raw prompt on llama-server (thinking off). Seconds = median request time (cloud latency_s, local wall_s). Small samples: one task more or less is noise.

| source | L2 | S3 | S4 | S5 | total | median s | cost |
|---|---|---|---|---|---|---|---|
| cloud mistral | 5/5 | 3/3 | 2/3 | 3/3 | 13/14 | 2.4 | $0.0000 |
| cloud groq | 10/10 | 4/4 | 4/4 | 4/4 | 22/22 | 2.3 | $0.0000 |
| cloud deepseek:flash | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 1.7 | $0.0050 |
| cloud gemini:lite25 | - | 0/3 | 0/3 | 0/2 | 0/8 | - | $0.0000 |
| cloud gemini:lite31 | 4/5 | 3/3 | 3/3 | 3/3 | 13/14 | 3.0 | $0.0166 |
| cloud gemini:flash | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 5.5 | $0.0399 |
| cloud cerebras | 4/5 | 1/1 | 1/1 | 1/1 | 7/8 | 3.2 | $0.0000 |
| local Q5_K_M | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 12.2 | $0 (local) |
| local 9B-Q4_K_XL | 2/5 | 3/3 | 2/3 | 3/3 | 10/14 | 6.5 | $0 (local) |
| local Q2_K_XL | 4/5 | 3/3 | 3/3 | 3/3 | 13/14 | 8.8 | $0 (local) |
| local IQ1_M | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 5.2 | $0 (local) |
| local Q4_K_XL | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 12.7 | $0 (local) |
| local Q3_K_XL | 4/5 | 3/3 | 3/3 | 3/3 | 13/14 | 11.3 | $0 (local) |
| local gemma-4-26B-A4B-it-UD-Q4_K_XL | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 9.7 | $0 (local) |
| local Qwen3.8-35B-A3B-Q4_K_M | 2/5 | 3/3 | 2/3 | 3/3 | 10/14 | 7.0 | $0 (local) |
| local Qwen3-Coder-30B-A3B-Q4_K_M | 5/5 | 3/3 | 2/3 | 3/3 | 13/14 | 12.4 | $0 (local) |
| local gemma-4-26B-A4B-it-UD-Q3_K_XL | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 8.9 | $0 (local) |
| local Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL | 1/5 | 3/3 | 1/3 | 3/3 | 8/14 | 5.8 | $0 (local) |
| local Qwen3.5-35B-A3B-UD-Q4_K_XL | 4/5 | 3/3 | 3/3 | 3/3 | 13/14 | 9.8 | $0 (local) |
| local NVIDIA-Nemotron-3.5-Lightning-30B-A3B-IQ4_XS | 0/5 | 3/3 | 2/3 | 3/3 | 8/14 | 8.7 | $0 (local) |

12 cloud request(s) failed (counted as incorrect): no provider available for kind=bulk_code. Tried: cerebras(em; no provider available for kind=bulk_code. Tried: gemini:lite; no provider available for kind=bulk_code. Tried: groq(429 ra