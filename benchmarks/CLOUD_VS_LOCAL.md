# bulk_code: cloud providers vs local models (same tasks, same automatic tests)

Generated 2026-10-10 10:32. Tasks: L2 TTL cache (implementation); S3 balanced brackets (code); S4 ISO dates (code); S5 moving average (code). Cloud rows go through foremankit's bulk_code path (system prompt, minimal reasoning); local rows are the raw prompt on llama-server (thinking off). Seconds = median request time (cloud latency_s, local wall_s). Small samples: one task more or less is noise.

| source | L2 | S3 | S4 | S5 | total | median s | cost |
|---|---|---|---|---|---|---|---|
| cloud mistral | 5/5 | 3/3 | 2/3 | 3/3 | 13/14 | 2.4 | $0.0000 |
| cloud groq | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 2.0 | $0.0000 |
| cloud deepseek:flash | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 1.7 | $0.0050 |
| local Q5_K_M | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 12.2 | $0 (local) |
| local 9B-Q4_K_XL | 2/5 | 3/3 | 2/3 | 3/3 | 10/14 | 6.5 | $0 (local) |
| local Q2_K_XL | 4/5 | 3/3 | 3/3 | 3/3 | 13/14 | 8.8 | $0 (local) |
| local IQ1_M | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 5.2 | $0 (local) |
| local Q4_K_XL | 5/5 | 3/3 | 3/3 | 3/3 | 14/14 | 12.7 | $0 (local) |
| local Q3_K_XL | 4/5 | 3/3 | 3/3 | 3/3 | 13/14 | 11.3 | $0 (local) |