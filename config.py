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

# --- Paths (shared by every stage) ---
DATA = Path("data")
CORPUS_DIR = DATA / "corpus"
PROCESSED_DIR = DATA / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
DOCS_PATH = PROCESSED_DIR / "docs.jsonl"
INDEX_DIR = DATA / "indexes"
EVAL_PATH = DATA / "eval" / "gold.jsonl"
RESULTS_DIR = Path("results")