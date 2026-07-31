"""BM25 sparse retrieval over the chunk index.

Stub. Implement bm25_search to return the same result shape as
dense_search so the generation layer can switch retrievers without
any downstream changes.
"""
from __future__ import annotations


def bm25_search(query: str, k: int = 3) -> list[dict]:
  """Retrieves the top-k chunks for a query by BM25 score.

  Must return the same shape as src.retrieval.dense.dense_search so
  llm.py can treat the two interchangeably:

      [
        {
          "chunk_id": str,
          "text": str,
          "score": float,
          "metadata": dict,   # must include doc_type, url, demo_refs
        },
        ...
      ]

  Args:
    query: Raw user query text.
    k: Number of chunks to return.

  Returns:
    Ranked list of result dicts, highest score first.

  Raises:
    NotImplementedError: Until implemented.
  """
  raise NotImplementedError(
    "bm25_search is not implemented yet. See src/retrieval/dense.py "
    "for the expected result shape."
  )