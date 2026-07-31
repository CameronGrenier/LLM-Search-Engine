"""Single source of truth for corpus, paths and model settings.

Every stage of the pipeline imports from here rather than hardcoding paths or
model names, so that a change made once propagates through fetch, preprocess,
chunk, index and evaluation.
"""

from pathlib import Path

# --- Corpus source (defines what corpus we built; all report-relevant) ---
REPO = "mui/material-ui"
TAG = "v9.2.0"

# Two subtrees make up the corpus:
#   - markdown prose/usage docs  -> conceptual & multi-hop questions
#   - API reference JSON (props) -> factoid questions (defaults, types)
KEEP_SUBPATHS = {
    "docs/data/material/": ".md",
    "docs/pages/material-ui/api/": ".json",
}

# File extensions of the demo files in the corpus
DEMO_EXTS = {".js", ".tsx"}

# --- Paths (shared by every stage) ---
DATA_DIR = Path("data")
CORPUS_DIR = DATA_DIR / "corpus"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
DOCS_PATH = PROCESSED_DIR / "docs.jsonl"
INDEX_DIR = DATA_DIR / "indexes"
EVAL_PATH = DATA_DIR / "eval" / "gold.jsonl"
RESULTS_DIR = Path("results")
TARBALL_PATH = DATA_DIR / "cache" / f"{TAG}.tar.gz"
DEMOS_PATH = PROCESSED_DIR / "demos.jsonl"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"
ANSWERS_PATH = PROCESSED_DIR / "answers.jsonl"

# --- Dense retrieval ---
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3-hf"
DENSE_INDEX_PATH = INDEX_DIR / "dense.faiss"
DENSE_INDEX_ID_MAPPING = INDEX_DIR / "dense_ids.json"

# jina-embeddings-v3 task specific LoRA adapters, as named in the -hf port.
# Asymmetric retrieval embeds corpus text with the passage adapter and queries
# with the query adapter.
#
# Only one adapter may be loaded per process. Loading a second re-initializes
# the first, and because LoRA initializes its B matrix to zeros the clobbered
# adapter silently becomes a no-op. The active adapter is therefore selected
# once per process through the JINA_ADAPTER environment variable, which entry
# point scripts set before importing the dense module.
PASSAGE_ADAPTER = "retrieval_passage"
QUERY_ADAPTER = "retrieval_query"
ADAPTER_ENV_VAR = "JINA_ADAPTER"

# --- LLM Model ---
GENERATION_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Model position limit is 8192. A margin is left for the added CLS and SEP.
MAX_CONTEXT_TOKENS = 8000

# Coverage budget per window, deliberately below MAX_CONTEXT_TOKENS so each
# window retains room to extend backwards for context.
COVERAGE_TOKENS = 6000

# --- Sparse retrieval ---
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"

# --- Reproducibility ---
RANDOM_SEED = 42