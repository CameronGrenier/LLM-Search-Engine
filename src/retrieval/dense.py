"""Dense retrieval indexing and search.

This module builds the dense half of the retrieval system. Chunk vectors are
produced by late chunking: the model runs over a long span of document text in
a single forward pass, and each chunk vector is the mean of that chunk's own
token vectors taken from that pass. Context therefore reaches a chunk through
attention rather than through concatenation of a separate document vector.

Windowing
---------
Documents that fit inside the model context limit are embedded in one pass.
Longer documents are covered by windows built in two stages:

- Coverage packing decides ownership. Starting at the end of the document and
  working backwards, whole chunks are packed into a window until the next
  chunk would push the covered span past COVERAGE_TOKENS. Because only whole
  chunks are ever added, both window edges land on chunk boundaries, so no
  chunk is split across a boundary and every chunk is owned by exactly one
  window.
- Context extension decides what the model reads. Once ownership is fixed, the
  window start is pushed further back toward MAX_CONTEXT_TOKENS. Those extra
  tokens supply preceding context only and never transfer ownership.

Documentation is read top to bottom, so context preceding a chunk carries more
signal than context following it. Windows are packed backwards and extended
backwards for that reason.

Task adapters
-------------
jina-embeddings-v3 ships task specific LoRA adapters. Asymmetric retrieval uses
retrieval_passage for corpus text and retrieval_query for queries, which places
question shaped text near answer shaped text.

Two constraints govern how they are loaded here:

- Exactly one adapter may be loaded per process. Loading a second adapter
  re-initializes the weights of the first, and since LoRA initializes its B
  matrix to zeros, the clobbered adapter silently becomes a no-op. The corpus
  would then be embedded with the plain base model while queries used a
  trained adapter, putting the two sides in different spaces. The adapter is
  therefore chosen once, through the environment variable named by
  ADAPTER_ENV_VAR, which entry point scripts set before importing this module.
- Adapters are injected while the model is still on CPU, and the model is
  moved to the accelerator only afterwards. Injecting into a model already
  resident on MPS has been observed to hang in a native call that ignores
  interrupts.

Usage
-----
Build the index, which defaults to the passage adapter:
    python3 -m src.retrieval.dense

Search, which selects the query adapter for itself:
    python3 retrieval.py

A clean load prints no MISSING rows in the model load report. If MISSING rows
appear even with a single adapter, the checkpoint does not ship those weights;
set USE_ADAPTERS to False and record it as a limitation.
"""

from __future__ import annotations

import os

# torch and faiss each vendor their own OpenMP runtime on macOS. Importing
# torch first and permitting the duplicate avoids an abort at import time.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402  must precede faiss

import json  # noqa: E402
import time  # noqa: E402

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from config import (  # noqa: E402
    ADAPTER_ENV_VAR,
    CHUNKS_PATH,
    COVERAGE_TOKENS,
    DENSE_INDEX_ID_MAPPING,
    DENSE_INDEX_PATH,
    DOCS_PATH,
    EMBEDDING_MODEL,
    INDEX_DIR,
    MAX_CONTEXT_TOKENS,
    PASSAGE_ADAPTER,
    QUERY_ADAPTER,
)
from src.corpus_io import load_chunks, load_docs  # noqa: E402
from src.chunk import normalize, strip_demo_markers  # noqa: E402

# Which adapter this process runs with. Indexing leaves the default in place;
# the search entry point sets the variable before importing this module.
ACTIVE_ADAPTER = os.environ.get(ADAPTER_ENV_VAR, PASSAGE_ADAPTER)

# Debug switch. Set False to skip adapters and use the plain base model.
USE_ADAPTERS = True


def log(message):
    """Print a timestamped progress message.

    Args:
      message: Text to print.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def get_device():
    """Pick the best available torch device.

    Returns:
      One of "cuda", "mps" or "cpu", in order of preference.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def verify_adapters():
    """Report which adapter files the model repo actually ships.

    Useful when adapter loading fails, to confirm the subfolder names assumed by
    this module exist in the repo at all.

    Returns:
      List of matching file paths in the repo, or an empty list on failure.
    """
    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files(EMBEDDING_MODEL)
        matches = [name for name in files if "retrieval" in name]
        log(f"adapter files found in repo: {matches}")
        return matches
    except Exception as error:
        log(f"could not list repo files: {error}")
        return []


def adapter_is_active(adapter_name):
    """Check whether a loaded adapter has non-zero weights.

    LoRA initializes its B matrix to zeros, so a freshly initialized adapter
    contributes nothing. A sum of zero across every B matrix means the adapter
    is a no-op and the model is effectively running unadapted.

    Args:
      adapter_name: Name of the adapter to inspect.

    Returns:
      True if any B matrix for the adapter holds non-zero weights.
    """
    for name, param in model.named_parameters():
        if "lora_B" in name and adapter_name in name:
            if float(param.detach().abs().sum()) > 0.0:
                return True
    return False


def load_model(adapter_name):
    """Load the tokenizer and model with exactly one LoRA adapter.

    The model is built on CPU, the adapter is injected there, and only then is
    the whole thing moved to the accelerator.

    Args:
      adapter_name: PASSAGE_ADAPTER for indexing, QUERY_ADAPTER for search.

    Returns:
      A tuple (tokenizer, model, device).
    """
    device = get_device()
    log(f"selected device: {device}")

    log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    log(f"tokenizer loaded (fast={tokenizer.is_fast})")

    log("loading base model on cpu")
    model = AutoModel.from_pretrained(EMBEDDING_MODEL, dtype=torch.float32)
    log("base model loaded on cpu")

    if USE_ADAPTERS:
        log(f"loading single adapter: {adapter_name}")
        model.load_adapter(
            EMBEDDING_MODEL,
            adapter_name=adapter_name,
            adapter_kwargs={"subfolder": adapter_name},
        )
        model.set_adapter(adapter_name)
        log(f"adapter active: {adapter_name}")
    else:
        log("adapters disabled, using plain base model")

    model.eval()

    log(f"moving model to {device}")
    model = model.to(device)
    log("model ready")

    return tokenizer, model, device


log("initialising dense retrieval module")
tokenizer, model, device = load_model(ACTIVE_ADAPTER)

if USE_ADAPTERS and not adapter_is_active(ACTIVE_ADAPTER):
    log(
        f"WARNING: adapter {ACTIVE_ADAPTER} has all-zero B matrices and is a "
        "no-op. The model is running unadapted."
    )


# ---------------------------------------------------------------------------
# Tokenization and span mapping
# ---------------------------------------------------------------------------


def tokenize_document(doc_text):
    """Tokenize a full document without truncation.

    Args:
      doc_text: The normalized document text. This must be the same string the
        chunker computed its char offsets against, or spans will not align.

    Returns:
      A tuple (input_ids, offsets) where input_ids is a list of token ids and
      offsets is a list of (start_char, end_char) pairs, one per token.
    """
    encoded = tokenizer(
        doc_text,
        return_offsets_mapping=True,
        truncation=False,
    )
    return encoded["input_ids"], encoded["offset_mapping"]


def chunk_token_ranges(doc_chunks, offsets):
    """Map each chunk's char span onto a token index range.

    The packer reasons in token budgets, so each chunk's char span has to be
    translated into token indices first.

    Args:
      doc_chunks: Chunk records for one document, each carrying char_start and
        char_end offsets into the document text.
      offsets: Per token (start_char, end_char) pairs for the whole document.

    Returns:
      A tuple (spans, unmapped) where spans is a list of
      (chunk, token_start, token_end) sorted by token_start, and unmapped lists
      any chunks whose span matched no tokens.
    """
    starts = np.array([pair[0] for pair in offsets])
    ends = np.array([pair[1] for pair in offsets])
    # special tokens carry (0, 0) offsets and never belong to a chunk
    real = ends > starts

    spans = []
    unmapped = []
    for chunk in doc_chunks:
        inside = real & (starts >= chunk["char_start"]) & (ends <= chunk["char_end"])
        positions = np.nonzero(inside)[0]
        if positions.size == 0:
            unmapped.append(chunk)
            continue
        spans.append((chunk, int(positions[0]), int(positions[-1]) + 1))

    spans.sort(key=lambda item: item[1])
    return spans, unmapped


# ---------------------------------------------------------------------------
# Window packing
# ---------------------------------------------------------------------------


def pack_windows(spans, coverage=COVERAGE_TOKENS, max_context=MAX_CONTEXT_TOKENS):
    """Group chunks into windows by packing whole chunks backwards.

    Ownership is decided here and nowhere else. A window closes when the next
    earlier chunk would push the covered token span past `coverage`, so both
    window edges land on chunk boundaries. The window start is then extended
    backwards toward `max_context` purely to supply preceding context; that
    extension never changes which chunks the window owns.

    Deciding ownership at packing time rather than by containment at pooling
    time matters, because extended regions overlap and containment would be
    ambiguous.

    Args:
      spans: List of (chunk, token_start, token_end) sorted by token_start.
      coverage: Token budget governing how many whole chunks a window owns.
      max_context: Hard ceiling on tokens fed to the model in one pass.

    Returns:
      List of dicts with keys "owned" (spans in document order), "context_start"
      and "window_end", both token indices.
    """
    windows = []
    index = len(spans) - 1

    while index >= 0:
        window_end = spans[index][2]
        owned = [spans[index]]

        earlier = index - 1
        while earlier >= 0 and (window_end - spans[earlier][1]) <= coverage:
            owned.append(spans[earlier])
            earlier -= 1

        owned.reverse()
        coverage_start = owned[0][1]

        # extend backwards for context, never past the coverage start
        context_start = min(coverage_start, max(0, window_end - max_context))

        windows.append(
            {
                "owned": owned,
                "context_start": context_start,
                "window_end": window_end,
            }
        )
        index = earlier

    windows.reverse()
    return windows


# ---------------------------------------------------------------------------
# Forward passes and pooling
# ---------------------------------------------------------------------------


def forward_window(input_ids, offsets, context_start, window_end):
    """Run one window through the model and return its token vectors.

    Slicing a document mid sequence strips the CLS and SEP tokens the model
    expects, so they are re-added and the offset list is padded with (0, 0)
    sentinels to stay aligned row for row. Those sentinels are excluded from
    pooling by the end > start test.

    Args:
      input_ids: Token ids for the whole document.
      offsets: Per token (start_char, end_char) pairs for the whole document.
      context_start: First token index of the window.
      window_end: Token index just past the last token of the window.

    Returns:
      A tuple (token_vectors, window_offsets), both on CPU and row aligned.
    """
    slice_ids = list(input_ids[context_start:window_end])
    slice_offsets = list(offsets[context_start:window_end])

    if tokenizer.cls_token_id is not None:
        slice_ids = [tokenizer.cls_token_id] + slice_ids
        slice_offsets = [(0, 0)] + slice_offsets
    if tokenizer.sep_token_id is not None:
        slice_ids = slice_ids + [tokenizer.sep_token_id]
        slice_offsets = slice_offsets + [(0, 0)]

    ids_tensor = torch.tensor([slice_ids], device=device)
    attention = torch.ones_like(ids_tensor, device=device)

    with torch.no_grad():
        outputs = model(input_ids=ids_tensor, attention_mask=attention)

    token_vectors = outputs.last_hidden_state[0].float().cpu()
    window_offsets = torch.tensor(slice_offsets)
    return token_vectors, window_offsets


def pool_chunk_span(token_vectors, offsets, char_start, char_end):
    """Mean-pool the token vectors whose char span falls within a chunk.

    Args:
      token_vectors: Tensor of shape (num_tokens, dim) from one forward pass.
      offsets: Tensor of shape (num_tokens, 2) giving each token's
        (start_char, end_char) in the document text.
      char_start: Start char offset of the chunk in the document text.
      char_end: End char offset, exclusive, of the chunk.

    Returns:
      A 1D L2 normalized float32 tensor of shape (dim,), or None if no tokens
      fall inside the span.
    """
    token_vectors = token_vectors.float().cpu()
    offsets = offsets.cpu()
    starts = offsets[:, 0]
    ends = offsets[:, 1]
    # a token belongs to the chunk when its span lies inside [char_start, char_end)
    # special tokens have offset (0, 0) and are excluded by requiring end > start
    mask = (starts >= char_start) & (ends <= char_end) & (ends > starts)
    selected = token_vectors[mask]
    if selected.shape[0] == 0:
        return None
    pooled = selected.mean(dim=0)
    return pooled / pooled.norm(p=2)


def embed_independently(text):
    """Embed a chunk from its own text, with no surrounding document context.

    This is a fallback only. It is reached when a chunk's span maps to no tokens
    or when a single chunk exceeds the context ceiling. Vectors produced here
    lack the cross chunk context that late chunking provides, so the count of
    such chunks is reported and belongs in the limitations discussion.

    Args:
      text: The chunk text.

    Returns:
      A 1D L2 normalized float32 tensor of shape (dim,).
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    token_vectors = outputs.last_hidden_state[0].float().cpu()
    pooled = token_vectors.mean(dim=0)
    return pooled / pooled.norm(p=2)


def embed_document(doc_text, doc_chunks, stats):
    """Embed every chunk of one document using late chunking.

    Short documents take a single forward pass, which gives true whole document
    context. Longer documents are split into backward packed coverage windows,
    each extended backwards for additional context.

    Args:
      doc_text: The normalized document text.
      doc_chunks: Chunk records belonging to this document.
      stats: Mutable dict of counters, updated in place.

    Returns:
      List of (chunk_id, vector) pairs.
    """
    input_ids, offsets = tokenize_document(doc_text)
    spans, unmapped = chunk_token_ranges(doc_chunks, offsets)

    results = []
    for chunk in unmapped:
        stats["fallback"] += 1
        results.append((chunk["chunk_id"], embed_independently(chunk["text"])))

    if not spans:
        return results

    if len(input_ids) <= MAX_CONTEXT_TOKENS:
        windows = [{"owned": spans, "context_start": 0, "window_end": len(input_ids)}]
        stats["single_pass_docs"] += 1
    else:
        windows = pack_windows(spans)
        stats["windowed_docs"] += 1

    stats["windows"] += len(windows)

    for window in windows:
        token_vectors, window_offsets = forward_window(
            input_ids,
            offsets,
            window["context_start"],
            window["window_end"],
        )
        for chunk, _, _ in window["owned"]:
            vector = pool_chunk_span(
                token_vectors,
                window_offsets,
                chunk["char_start"],
                chunk["char_end"],
            )
            if vector is None:
                stats["fallback"] += 1
                vector = embed_independently(chunk["text"])
            else:
                stats["late_chunked"] += 1
            results.append((chunk["chunk_id"], vector))

    return results


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def create_embeddings():
    """Embed every chunk in the corpus.

    The process must have been started with the passage adapter selected, which
    is the default when the adapter environment variable is unset.

    Returns:
      A tuple (chunk_ids, vectors) in matching order.
    """
    if USE_ADAPTERS and ACTIVE_ADAPTER != PASSAGE_ADAPTER:
        log(
            f"WARNING: indexing with adapter {ACTIVE_ADAPTER}. Corpus text should "
            f"be embedded with {PASSAGE_ADAPTER}. Unset {ADAPTER_ENV_VAR} or set it "
            f"to {PASSAGE_ADAPTER}."
        )

    log(f"loading documents from {DOCS_PATH}")
    docs = load_docs(DOCS_PATH)
    log(f"loaded {len(docs)} documents")

    log(f"loading chunks from {CHUNKS_PATH}")
    all_chunks, _, by_doc = load_chunks(CHUNKS_PATH)
    log(f"loaded {len(all_chunks)} chunks across {len(by_doc)} documents")

    stats = {
        "single_pass_docs": 0,
        "windowed_docs": 0,
        "windows": 0,
        "late_chunked": 0,
        "fallback": 0,
    }

    chunk_ids = []
    vectors = []
    started = time.time()

    for doc_id, doc_chunks in tqdm(by_doc.items(), desc="embedding docs"):
        normalized_text = normalize(docs[doc_id]["text"])
        doc_text, _ = strip_demo_markers(normalized_text)
        for chunk_id, vector in embed_document(doc_text, doc_chunks, stats):
            chunk_ids.append(chunk_id)
            vectors.append(vector.numpy())

    log(f"embedding finished in {time.time() - started:.1f}s")
    log(f"documents embedded in a single pass: {stats['single_pass_docs']}")
    log(f"documents requiring windows: {stats['windowed_docs']}")
    log(f"forward passes over windows: {stats['windows']}")
    log(f"chunks pooled from a window: {stats['late_chunked']}")
    log(f"chunks embedded independently: {stats['fallback']}")
    log(f"chunks embedded in total: {len(chunk_ids)}")

    if len(chunk_ids) != len(all_chunks):
        log(
            "WARNING: embedded chunk count does not match corpus chunk count "
            f"({len(chunk_ids)} vs {len(all_chunks)})"
        )

    return chunk_ids, vectors


def store_embeddings(chunk_ids, vectors):
    """Build the FAISS index and write it to disk alongside its id mapping.

    FAISS addresses vectors by integer row position and has no knowledge of the
    string chunk ids, so the id list is persisted separately in insertion order
    to translate search results back.

    Args:
      chunk_ids: Chunk ids in the same order as vectors.
      vectors: List of 1D normalized float32 arrays.
    """
    log("stacking vectors")
    matrix = np.vstack(vectors).astype("float32")
    log(f"vector matrix shape: {matrix.shape}")

    # vectors are L2 normalized, so inner product equals cosine similarity
    log("building FAISS index")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    log(f"index contains {index.ntotal} vectors")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(DENSE_INDEX_PATH))
    log(f"wrote dense index to {DENSE_INDEX_PATH}")

    with open(DENSE_INDEX_ID_MAPPING, "w") as file:
        json.dump(chunk_ids, file)
    log(f"wrote id mapping to {DENSE_INDEX_ID_MAPPING}")


def create_dense_index():
    """Embed the corpus and persist the dense index."""
    chunk_ids, vectors = create_embeddings()
    store_embeddings(chunk_ids, vectors)
    log("dense index build complete")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def load_dense_index():
    """Load the FAISS index and its position to chunk_id mapping.

    Returns:
      A tuple (index, chunk_ids) where chunk_ids[i] names FAISS row i.
    """
    index = faiss.read_index(str(DENSE_INDEX_PATH))
    with open(DENSE_INDEX_ID_MAPPING) as file:
        chunk_ids = json.load(file)
    return index, chunk_ids


def embed_query(query):
    """Embed a query into one normalized vector in the corpus vector space.

    The process must have been started with the query adapter selected. The
    query is short, so there is nothing late about its chunking; it is a single
    forward pass mean pooled over all real tokens.

    Args:
      query: The raw query string.

    Returns:
      A 1D float32 numpy array of shape (dim,).
    """
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    token_vectors = outputs.last_hidden_state[0].float().cpu()
    pooled = token_vectors.mean(dim=0)
    pooled = pooled / pooled.norm(p=2)
    return pooled.numpy().astype("float32")


def dense_search(query, k=10, index=None, chunk_ids=None, by_id=None):
    """Retrieve the top k chunks for a query by vector similarity.

    The index, id mapping and chunk lookup can be passed in to avoid reloading
    them on every call, which matters when running an evaluation loop.

    Args:
      query: The raw query string.
      k: Number of chunks to return.
      index: Optional preloaded FAISS index.
      chunk_ids: Optional preloaded position to chunk_id list.
      by_id: Optional preloaded chunk_id to chunk record mapping.

    Returns:
      List of result dicts with chunk_id, score, text and metadata, ordered by
      descending score.
    """
    if USE_ADAPTERS and ACTIVE_ADAPTER != QUERY_ADAPTER:
        log(
            f"WARNING: searching with adapter {ACTIVE_ADAPTER}. Queries should use "
            f"{QUERY_ADAPTER}. Set {ADAPTER_ENV_VAR} before importing this module."
        )

    if index is None or chunk_ids is None:
        index, chunk_ids = load_dense_index()
    if by_id is None:
        _, by_id, _ = load_chunks(CHUNKS_PATH)

    query_vector = embed_query(query).reshape(1, -1)  # FAISS expects 2D input
    scores, positions = index.search(query_vector, k)

    results = []
    for score, position in zip(scores[0], positions[0]):
        chunk_id = chunk_ids[position]
        chunk = by_id[chunk_id]
        results.append(
            {
                "chunk_id": chunk_id,
                "score": float(score),
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            }
        )
    return results


if __name__ == "__main__":
    create_dense_index()
