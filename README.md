# LLM-Search-Engine
use python 3.12
## Project Structure
```
root/
├── README.md                  # setup + execution instructions (spec requires)
├── requirements.txt           # pinned deps (spec requires)
├── config.py                  # all knobs: REPO, TAG, KEEP_SUBPATHS, paths, chunk size, top_k, seeds
├── run_all.py                 # THE single command that reproduces every result/table
│
├── data/
│   ├── corpus/                # flat dump from the fetch script: ~151 .md + ~143 API .json
│   ├── processed/
│   │   ├── manifest.json      # {flat_filename: {source_path, doc_type}} — authority for what's a doc
│   │   ├── docs.jsonl         # normalized per-document: {doc_id, text, metadata}
│   │   └── chunks.jsonl       # chunked corpus: {chunk_id, doc_id, text, metadata}
│   ├── eval/
│   │   └── gold.jsonl         # gold questions: {q, answer, gold_chunk_ids, type}
│   └── indexes/               # built retrieval indexes (bm25 pickle, dense vectors)
│
├── src/
│   ├── fetch_corpus.py        # download tarball, flat-dump 2 subtrees, write manifest.json
│   ├── preprocess.py          # driven off manifest: parse md frontmatter + flatten API json, clean, assign IDs
│   ├── chunk.py               # split docs -> chunks, carry doc_id + metadata
│   ├── retrieval/
│   │   ├── base.py            # a Retriever interface (retrieve(query, k) -> chunks)
│   │   ├── bm25.py            # classical: BM25
│   │   └── dense.py           # dense: sentence-transformer embeddings + similarity
│   ├── generate.py            # prompt build + local LLM call, citation + "I don't know"
│   └── evaluate/
│       ├── retrieval_metrics.py   # recall@k, MRR, nDCG over gold_chunk_ids
│       └── generation_metrics.py  # answer correctness / support (auto or human-assist)
│
├── results/                   # output tables/figures land here (git-tracked or regenerated)
│   ├── diagnostic.md          # the 10-question no-context baseline test
│   ├── retrieval_results.csv
│   └── generation_results.csv
│
└── report/                    # the 3-4 page PDF + LaTeX source
```