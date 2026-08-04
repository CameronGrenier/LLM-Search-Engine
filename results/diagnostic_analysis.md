# No-Context Diagnostic Analysis

## Purpose

The local Qwen2.5-3B-Instruct model was asked ten answerable factual questions without access to the Material UI corpus. The same questions were then answered using BM25 and dense retrieval with the same generation model and settings.

## Diagnostic composition

- 10 answerable questions
- 7 factoid questions
- 3 multi-hop questions

## Summary

| System | Fully correct | Rate |
|---|---:|---:|
| Naive, no context | 0/10 | 0% |
| BM25 RAG | 3/10 | 30% |
| Dense RAG | 8/10 | 80% |

The naive model produced 2 partially correct answers and 8 incorrect answers. Retrieval improved 9/10 answers.

## Per-question results

| Q | Type | Naive status | Naive score | BM25 score | Dense score |
|---:|---|---|---:|---:|---:|
| 1 | factoid | partially correct | 1/2 | 1/2 | 1/2 |
| 2 | factoid | incorrect | 0/2 | 0/2 | 2/2 |
| 3 | factoid | incorrect | 0/2 | 2/2 | 2/2 |
| 4 | factoid | incorrect | 0/2 | 2/2 | 2/2 |
| 5 | factoid | incorrect | 0/2 | 2/2 | 2/2 |
| 6 | factoid | partially correct | 1/2 | 0/2 | 2/2 |
| 7 | factoid | incorrect | 0/2 | 0/2 | 2/2 |
| 8 | multi-hop | incorrect | 0/2 | 0/2 | 2/2 |
| 9 | multi-hop | incorrect | 0/2 | 1/2 | 2/2 |
| 10 | multi-hop | incorrect | 0/2 | 1/2 | 0/2 |

## Conclusion

The local LLM answered none of the ten diagnostic questions fully correctly without retrieved context. Retrieval improved 9 of the ten answers, demonstrating that the selected corpus contains information the model did not reliably know from its parameters alone.

The model could sometimes produce related or partially correct information, but it did not fully answer any of the ten questions without retrieval. Dense retrieval produced the strongest overall improvement.
