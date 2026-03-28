# Benchmark Results — sprint_planner

**Date:** 2026-03-28 00:49 UTC  
**Runs per model:** 3  
**GPU:** NVIDIA RTX 2000 Ada Generation Laptop GPU (8188 MB)  
**CPU:** Intel64 Family 6 Model 183 Stepping 1, GenuineIntel  

| Rank | Model | Params | Quant | Avg Score | Min | Max | Avg Time (s) | Tok/s | Peak VRAM (MB) |
|------|-------|--------|-------|-----------|-----|-----|--------------|-------|----------------|
| 1 | qwen2.5-coder:3b | 3.1B | Q4_K_M | 1.000 | 1.000 | 1.000 | 17.2 | 115.0 | 3990 |
| 2 | qwen2.5-coder:7b | 7.6B | Q4_K_M | 1.000 | 1.000 | 1.000 | 29.3 | 52.7 | 4922 |
| 3 | deepseek-coder-v2:16b | 15.7B | Q4_0 | 1.000 | 1.000 | 1.000 | 62.0 | 25.9 | 7408 |
| 4 | qwen3:14b | 14.8B | Q4_K_M | 0.984 | 0.952 | 1.000 | 467.3 | 12.2 | 7340 |
| 5 | qwen2.5-coder:1.5b | 1.5B | Q4_K_M | 0.951 | 0.929 | 0.972 | 14.2 | 187.0 | 1576 |
| 6 | qwen3:8b | 8.2B | Q4_K_M | 0.897 | 0.738 | 1.000 | 282.1 | 39.9 | 5552 |
| 7 | qwen3:4b | 4.0B | Q4_K_M | 0.809 | 0.726 | 0.869 | 257.3 | 58.9 | 5364 |
