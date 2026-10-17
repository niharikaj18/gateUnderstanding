## GitContext evaluation (60 questions)

### Retrieval ablation

| config | R@1 | R@3 | R@5 | R@10 | P@3 | P@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.617 | 0.817 | 0.900 | 0.917 | 0.272 | 0.180 | 0.730 | 0.777 | 25 | 32 |
| bm25 | 0.850 | 0.967 | 1.000 | 1.000 | 0.322 | 0.200 | 0.914 | 0.936 | 3 | 6 |
| hybrid (RRF) | 0.867 | 0.983 | 1.000 | 1.000 | 0.328 | 0.200 | 0.926 | 0.945 | 26 | 34 |
| hybrid + rerank | 0.733 | 0.883 | 0.933 | 0.967 | 0.294 | 0.187 | 0.818 | 0.854 | 3636 | 5038 |

### By question type (hybrid + rerank: Recall@5 / MRR)

| type | R@5 | MRR |
|---|---|---|
| semantic | 1.000 | 1.000 |
| exact_error | 0.833 | 0.734 |
| symbol | 0.917 | 0.551 |
| history | 0.917 | 0.887 |
| doc | 1.000 | 0.917 |

### Latency per stage (p50 / p95 ms)

| config | embed_query | dense | lexical | fuse | rerank |
|---|---|---|---|---|---|
| dense | 18 / 25 | 5 / 8 | – | 1 / 1 | – |
| bm25 | – | – | 2 / 5 | 1 / 1 | – |
| hybrid (RRF) | 17 / 22 | 5 / 7 | 3 / 6 | 1 / 1 | – |
| hybrid + rerank | 18 / 24 | 5 / 9 | 3 / 7 | 1 / 1 | 3605 / 5011 |
