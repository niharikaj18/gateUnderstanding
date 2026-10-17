## GitContext evaluation (60 questions)

### Retrieval ablation

| config | R@1 | R@3 | R@5 | R@10 | P@3 | P@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.617 | 0.817 | 0.900 | 0.917 | 0.272 | 0.180 | 0.730 | 0.777 | 24 | 29 |
| bm25 | 0.817 | 0.967 | 1.000 | 1.000 | 0.322 | 0.200 | 0.897 | 0.923 | 4 | 8 |
| hybrid (RRF) | 0.833 | 0.983 | 1.000 | 1.000 | 0.328 | 0.200 | 0.910 | 0.933 | 27 | 36 |

### By question type (hybrid (RRF): Recall@5 / MRR)

| type | R@5 | MRR |
|---|---|---|
| semantic | 1.000 | 0.861 |
| exact_error | 1.000 | 0.833 |
| symbol | 1.000 | 1.000 |
| history | 1.000 | 0.896 |
| doc | 1.000 | 0.958 |

### Latency per stage (p50 / p95 ms)

| config | embed_query | dense | lexical | fuse | rerank |
|---|---|---|---|---|---|
| dense | 16 / 20 | 7 / 8 | – | 1 / 1 | – |
| bm25 | – | – | 3 / 7 | 1 / 1 | – |
| hybrid (RRF) | 16 / 21 | 6 / 7 | 3 / 7 | 1 / 1 | – |
