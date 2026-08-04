# Material UI RAG Search Engine

A reproducible retrieval-augmented generation system created for the CP423 course project.

The project compares three systems:

1. A local language model without retrieved context
2. BM25 retrieval with local generation
3. Dense retrieval with local generation

All systems use the same local generation model and settings. RAG answers cite retrieved chunks inline using citations such as `[1]`. When the retrieved context is insufficient, the system returns exactly `I don't know`.

## Main results

### Retrieval evaluation

The retrieval evaluation uses 10 answerable questions. Three unanswerable questions are excluded from retrieval metrics.

| Retriever | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.1600 | 0.6500 | 0.6736 | 0.5839 |
| Dense | **0.2000** | **0.8000** | **0.6827** | **0.6693** |

Dense retrieval outperformed BM25 on all four retrieval metrics.

### Automated generation evaluation

| System | Exact match | Mean token F1 | Unanswerable refusal accuracy |
|---|---:|---:|---:|
| Naive | 0.0000 | 0.4347 | 0.0% |
| BM25 RAG | **0.2308** | 0.5255 | **100.0%** |
| Dense RAG | 0.1538 | **0.6259** | 66.7% |

Every accepted RAG answer contained at least one inline citation.

### Human evaluation

Human scores range from 0 to 2.

| System | Correctness | Groundedness | Completeness | Fully correct |
|---|---:|---:|---:|---:|
| Naive | 0.1538 | 0.1538 | 0.1538 | 0/13 |
| BM25 RAG | 1.1538 | 1.6923 | 1.1538 | 6/13 |
| Dense RAG | **1.6923** | **1.7692** | **1.6923** | **10/13** |

### No-context diagnostic

The local model was asked the 10 answerable evaluation questions without corpus context.

| System | Fully correct |
|---|---:|
| Naive | 0/10 |
| BM25 RAG | 3/10 |
| Dense RAG | 8/10 |

Retrieval improved 9 of the 10 answers. This shows that the corpus contains information the local model did not reliably know without retrieved context.

## Corpus

The project uses a pinned version of the public Material UI documentation repository.

- Repository: `mui/material-ui`
- Version: `v9.2.0`
- Language: English
- Extracted files: 1,861
- Processed documents: 294
- Referenced demos: 613
- Final chunks: 2,140
- Prose chunks: 1,997
- API chunks: 143

Pinning the corpus version prevents later upstream documentation changes from altering the experiment.

## Models

Dense retrieval model:

    jinaai/jina-embeddings-v3-hf

The dense retriever uses the model's passage and query adapters and stores vectors in a FAISS index.

Local generation model:

    Qwen/Qwen2.5-3B-Instruct

The same generation model and settings are used for the naive, BM25 RAG, and dense RAG systems.

## Evaluation set

The evaluation set is stored in `data/questions.jsonl`.

It contains 13 questions:

- 7 factoid questions
- 3 multi-hop questions
- 3 unanswerable questions

The multi-hop questions require evidence from multiple chunks. The unanswerable questions use `I don't know` as the reference answer.

## Requirements

- Python 3.12
- Internet connection for the first corpus and model download
- Sufficient memory to load the embedding and generation models

The code selects Apple MPS, CUDA, or CPU depending on the available hardware.

All tested dependencies are pinned in `requirements.txt`.

## Installation

Clone the repository:

    git clone https://github.com/CameronGrenier/LLM-Search-Engine.git
    cd LLM-Search-Engine

Create and activate a Python 3.12 virtual environment:

    python3.12 -m venv .venv312
    source .venv312/bin/activate

Install dependencies:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

## Reproduce the complete experiment

Run one command from the repository root:

    python reproduce.py

The reproduction pipeline performs all 10 stages:

1. Fetch the pinned corpus
2. Preprocess the documents
3. Create the chunks
4. Build the BM25 index
5. Build the dense FAISS index
6. Evaluate both retrievers
7. Generate all 39 answers
8. Evaluate the generated answers
9. Regenerate the human-evaluation summary
10. Regenerate the no-context diagnostic analysis

The complete pipeline was successfully tested on Apple Silicon using MPS. It completed in approximately 4.3 minutes after the model files were already cached.

## Run individual stages

    python -m src.fetch_corpus
    python -m src.preprocess
    python -m src.chunk
    python -m src.retrieval.bm25
    python -m src.retrieval.dense
    python -m src.evaluate.retrieval_metrics --system both --k 5 --self-test
    python -m src.llm.llm
    python -m src.evaluate.generation_metrics
    python -m src.evaluate.manual_summary
    python -m src.evaluate.diagnostic

## Generated data and indexes

The reproduction pipeline creates:

    data/processed/manifest.json
    data/processed/docs.jsonl
    data/processed/demos.jsonl
    data/processed/chunks.jsonl
    data/processed/answers.jsonl
    data/indexes/bm25.pkl
    data/indexes/dense.faiss
    data/indexes/dense_ids.json

These derived files are ignored by Git and regenerated by `python reproduce.py`.

## Tracked result files

    results/retrieval_bm25.json
    results/retrieval_bm25.csv
    results/retrieval_dense.json
    results/retrieval_dense.csv
    results/generation_per_answer.json
    results/generation_per_answer.csv
    results/generation_aggregate.json
    results/generation_aggregate.csv
    results/generation_manual_review.csv
    results/generation_manual_summary.json
    results/generation_manual_summary.csv
    results/diagnostic_results.csv
    results/diagnostic_summary.json
    results/diagnostic_analysis.md

## Project structure

    .
    ├── README.md
    ├── requirements.txt
    ├── config.py
    ├── reproduce.py
    ├── data/
    │   ├── questions.jsonl
    │   ├── corpus/
    │   ├── processed/
    │   └── indexes/
    ├── results/
    ├── src/
    │   ├── fetch_corpus.py
    │   ├── preprocess.py
    │   ├── chunk.py
    │   ├── corpus_io.py
    │   ├── retrieval/
    │   │   ├── bm25.py
    │   │   └── dense.py
    │   ├── llm/
    │   │   └── llm.py
    │   └── evaluate/
    │       ├── retrieval_metrics.py
    │       ├── generation_metrics.py
    │       ├── manual_summary.py
    │       └── diagnostic.py
    └── report/

## Reproducibility controls

- Material UI corpus pinned to `v9.2.0`
- Dependencies pinned in `requirements.txt`
- Random seed fixed to `42`
- Greedy generation with sampling disabled
- Stable document and chunk identifiers
- Persisted JSON and CSV result tables
- One command regenerates the experiment outputs

Human evaluation scores are stored in the tracked manual-review CSV. The automated evaluator preserves those scores when regenerating result files.

## Known limitations

- The evaluation set is small.
- Retrieval is limited to the top five chunks.
- Exact gold-chunk matching may miss neighboring chunks containing equivalent evidence.
- The 3B local model sometimes produces incomplete answers or unnecessary refusals.
- Multi-hop questions can fail when only one required chunk appears in the top five.
- Human evaluation contains some unavoidable subjectivity.
