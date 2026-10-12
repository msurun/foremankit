# Latency: seconds from prompt to the finished answer

Generated 2026-10-10 20:50. Median of the repetitions per scenario, thinking OFF, streaming, unique prompt nonce (cold cache) except cached10k.

| model | GiB | quick (s) | summ2k (s) | code (s) | long10k (s) | cached10k (s) | decode200 (s) | long10k first token (s) |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B-UD-Q4_K_XL | 5.6 | 1.2 | 2.3 | 7.8 | 4.3 | 0.5 | 3.8 | 4.2 |
| Qwen3.6-35B-A3B-UD-IQ1_M | 9.4 | 1.4 | 4.1 | 7.5 | 8.4 | 0.9 | 3.9 | 8.3 |
| gemma-4-26B-A4B-it-UD-Q3_K_XL | 12.0 | 2.1 | 6.5 | 16.9 | 11.0 | 0.6 | 7.0 | 10.9 |
| Qwen3.6-35B-A3B-UD-Q2_K_XL | 11.4 | 1.6 | 5.5 | 8.4 | 11.5 | 1.3 | 4.5 | 11.4 |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.8 | 2.5 | 7.5 | 14.8 | 16.2 | 0.5 | 7.4 | 16.1 |
| Qwen3.6-35B-A3B-UD-Q3_K_XL | 15.7 | 1.8 | 8.0 | 10.4 | 18.4 | 1.9 | 6.0 | 18.4 |
| Qwen3-Coder-30B-A3B-Q4_K_M | 17.3 | 2.8 | 8.0 | 12.8 | 19.3 | 0.3 | 7.9 | 19.2 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-IQ4_XS | 17.6 | 2.5 | 7.1 | 9.3 | 21.6 | 13.9 | - | 21.6 |
| Qwen3.8-35B-A3B-Q4_K_M | 20.2 | 2.0 | 8.6 | 9.4 | 23.6 | 2.3 | 5.7 | 23.5 |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 20.8 | 2.1 | 9.1 | 10.5 | 24.9 | 2.5 | 6.2 | 24.8 |
| Qwen3.6-35B-A3B-UD-Q5_K_M | 24.6 | 3.1 | 11.2 | 12.3 | 30.8 | 3.0 | 7.2 | 30.8 |

Typical size (first model): quick: in 41 / out 60 tokens; summ2k: in 1162 / out 142 tokens; code: in 56 / out 357 tokens; long10k: in 6018 / out 3 tokens; cached10k: in 6016 / out 4 tokens; decode200: in 45 / out 200 tokens

5 request(s) ended with an error (excluded from the medians).