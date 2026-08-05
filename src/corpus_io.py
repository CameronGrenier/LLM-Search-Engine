"""Helper functions for loading corpus docs and chunks from their jsonl files"""

import json
from collections import defaultdict


def load_docs(path):
    """Load documents from a JSONL file into a dict keyed by doc_id.

    Args:
      path: Path to the documents JSONL file. Each line must be a JSON
        object containing a "doc_id" field.

    Returns:
      A dict mapping each doc_id to its full document dict.
    """
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            docs[doc["doc_id"]] = doc
    return docs


def load_chunks(path):
    """Load chunks from a JSONL file and build lookup views over them.

    Reads the chunks once and derives two in-memory indexes so callers
    avoid re-scanning the file: a chunk_id lookup for retrieval and
    citation, and a doc_id grouping for late chunking.

    Args:
      path: Path to the chunks JSONL file. Each line must be a JSON
        object containing "chunk_id" and "doc_id" fields.

    Returns:
      A tuple (chunks, by_id, by_doc) where:
        chunks: List of all chunk dicts in file order.
        by_id: Dict mapping chunk_id to its chunk dict, used for
          citation and post-retrieval lookup.
        by_doc: Dict mapping doc_id to the list of its chunk dicts in
          file order, used to group chunks per document for late
          chunking.
    """
    chunks = [json.loads(l) for l in open(path, encoding="utf-8")]
    by_id = {c["chunk_id"]: c for c in chunks}  # citations, retrieval lookup
    by_doc = defaultdict(list)  # late-chunking grouping
    for c in chunks:
        by_doc[c["doc_id"]].append(c)
    return chunks, by_id, dict(by_doc)
