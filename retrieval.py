"""Command line search against the dense index.

The dense module loads its model, and its single LoRA adapter, at import time.
The query adapter therefore has to be selected before that import happens,
which is why the environment variable is set at the top of this file rather
than passed in from the shell.

Run from the project root with:
    python3 retrieval.py
    python3 retrieval.py how do I let users type values not in the list
"""

import os
import sys

from config import ADAPTER_ENV_VAR, QUERY_ADAPTER

# torch and faiss each vendor their own OpenMP runtime on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Queries are the query side of asymmetric retrieval. This must be set before
# src.retrieval.dense is imported, because that import loads the model.
os.environ[ADAPTER_ENV_VAR] = QUERY_ADAPTER

from src.retrieval.dense import dense_search, load_dense_index  # noqa: E402
from src.corpus_io import load_chunks  # noqa: E402
from config import CHUNKS_PATH  # noqa: E402

DEFAULT_QUERY = "how do I disable a button?"
PREVIEW_CHARS = 300


def show_results(query, results):
  """Print retrieval results in a readable form.

  Args:
    query: The query that produced the results.
    results: Result dicts from dense_search.
  """
  print(f'\nquery: "{query}"\n')
  for rank, result in enumerate(results, start=1):
    metadata = result["metadata"]
    preview = result["text"][:PREVIEW_CHARS].replace("\n", " ")
    print(f"{rank}. {result['score']:.3f}  {result['chunk_id']}")
    print(f"     component: {metadata.get('component')}")
    print(f"     headings:  {metadata.get('headings')}")
    print(f"     {preview}...")
    print()


def search(query, k=5):
  """Run one query against the dense index and print the results.

  Args:
    query: The raw query string.
    k: Number of chunks to return.
  """
  results = dense_search(query, k=k)
  show_results(query, results)


def interactive(k=5):
  """Run repeated queries against a single loaded index.

  Loading the index and chunk lookup once avoids paying that cost per query.

  Args:
    k: Number of chunks to return per query.
  """
  index, chunk_ids = load_dense_index()
  _, by_id, _ = load_chunks(CHUNKS_PATH)
  print("enter a query, or a blank line to quit")

  while True:
    try:
      query = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
      print()
      return
    if not query:
      return
    results = dense_search(
      query,
      k=k,
      index=index,
      chunk_ids=chunk_ids,
      by_id=by_id,
    )
    show_results(query, results)


if __name__ == "__main__":
  if len(sys.argv) > 1:
    if sys.argv[1] == "-i":
      interactive()
    else:
      search(" ".join(sys.argv[1:]))
  else:
    search(DEFAULT_QUERY)