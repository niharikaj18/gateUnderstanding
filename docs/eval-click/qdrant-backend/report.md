## GitContext evaluation (60 questions)

### Retrieval ablation

| config | R@1 | R@3 | R@5 | R@10 | P@3 | P@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.617 | 0.817 | 0.900 | 0.917 | 0.272 | 0.180 | 0.730 | 0.777 | 25 | 32 |
| bm25 | 0.833 | 0.983 | 1.000 | 1.000 | 0.328 | 0.200 | 0.912 | 0.935 | 15 | 50 |
| hybrid (RRF) | 0.817 | 1.000 | 1.000 | 1.000 | 0.333 | 0.200 | 0.908 | 0.932 | 38 | 73 |

### By question type (hybrid (RRF): Recall@5 / MRR)

| type | R@5 | MRR |
|---|---|---|
| semantic | 1.000 | 0.875 |
| exact_error | 1.000 | 0.833 |
| symbol | 1.000 | 0.958 |
| history | 1.000 | 0.917 |
| doc | 1.000 | 0.958 |

### Latency per stage (p50 / p95 ms)

| config | embed_query | dense | lexical | fuse | rerank |
|---|---|---|---|---|---|
| dense | 16 / 22 | 4 / 6 | – | 4 / 5 | – |
| bm25 | – | – | 10 / 44 | 5 / 7 | – |
| hybrid (RRF) | 16 / 20 | 4 / 5 | 10 / 46 | 5 / 6 | – |
