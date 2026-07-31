"""Batch question answering, with and without retrieved context.

Reads questions from config.QUESTIONS_PATH and writes one answer record
per (question, mode) pair to config.ANSWERS_PATH.

Retriever selection is a module-level flag so the dense and sparse arms
of the evaluation run through identical prompt assembly and identical
output formatting. Any difference in the resulting answers is then
attributable to retrieval alone, which is what makes the comparison
meaningful.

Run from the project root with:
    python -m src.llm.llm
"""

from __future__ import annotations
import os
import json
import re
import sys
from pathlib import Path
import posixpath

from config import (
    ADAPTER_ENV_VAR,
    QUERY_ADAPTER,
    GENERATION_MODEL,
    DEMOS_PATH,
    QUESTIONS_PATH,
    ANSWERS_PATH,
    MANIFEST_PATH,
    CORPUS_DIR,
    REPO,
    TAG,
)

os.environ[ADAPTER_ENV_VAR] = QUERY_ADAPTER  # must precede the dense import

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)
from src.retrieval.dense import get_device, dense_search  # noqa: E402
from src.retrieval.bm25 import bm25_search  # noqa: E402

# ============================================================
# Run configuration
# ============================================================

# "dense" or "bm25". Selects which retriever supplies context for the
# rag mode. Naive mode ignores it.
RETRIEVER = "dense"

# Which arms to run per question. Naive is the ungrounded control: it
# isolates how much of a correct answer came from retrieval rather than
# from the generation model's parametric knowledge.
MODES = ("naive", "rag")

TOP_K = 5
MAX_NEW_TOKENS = 512
STREAM = False  # noisy across a full question set; enable to watch one run
SHOW_CONTEXT = False  # print the assembled passages before each answer

RETRIEVERS = {
    "dense": dense_search,
    "bm25": bm25_search,
}

# Matches the inline citation markers the RAG system prompt asks for.
CITATION_RE = re.compile(r"\[(\d+)\]")


# ============================================================
# Model
# ============================================================

device = get_device()
MODEL = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL, dtype="auto").to(device)
TOKENIZER = AutoTokenizer.from_pretrained(GENERATION_MODEL)
MODEL.eval()


RAG_SYSTEM = """You answer questions about Material UI using only the numbered passages provided.

Rules:
- Use only information stated in the passages. Do not use prior knowledge about Material UI.
- Cite the passage number inline after each claim, like [2]. Cite every claim.
- If the passages do not contain the answer, reply exactly: I don't know.
- Do not guess prop names, default values, or types that are not stated in the passages.
- Keep the answer to a few sentences."""

NAIVE_SYSTEM = (
    "Answer the question about Material UI as accurately as you can. "
    "Keep it concise and to the point."
)


# ============================================================
# Loading
# ============================================================


def load_questions(path: Path) -> list[dict]:
    """Reads the evaluation question set.

    Args:
      path: Path to questions.jsonl, one object per line with
        question_id (int) and question_text (str).

    Returns:
      List of question dicts in file order.

    Raises:
      ValueError: If a line is malformed or missing a required field, or
        if a question_id is duplicated. Duplicate ids would silently make
        answer records ambiguous, so they fail loudly here instead.
    """
    questions: list[dict] = []
    seen: set[int] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}"
                ) from error

            if "question_id" not in row or "question_text" not in row:
                raise ValueError(
                    f"Line {line_num} of {path} needs question_id and question_text"
                )

            qid = int(row["question_id"])
            if qid in seen:
                raise ValueError(f"Duplicate question_id {qid} on line {line_num}")
            seen.add(qid)

            questions.append(
                {"question_id": qid, "question_text": row["question_text"]}
            )

    return questions


def load_demos(path: Path) -> dict[str, dict]:
    """Loads the demo source store into a lookup keyed by demo_id.

    Args:
      path: Path to demos.jsonl.

    Returns:
      Mapping of demo_id to its record.
    """
    with open(path, "r", encoding="utf-8") as f:
        return {d["demo_id"]: d for d in map(json.loads, f)}


def load_corpus_index(path: Path) -> dict[str, str]:
    """Inverts the manifest into repo path -> flat corpus filename.

    Chunk metadata carries the original repo path, not the flat on-disk
    name, and collisions mean the flat name cannot be derived from the
    basename alone. Going through the manifest gives an exact corpus
    path for every citation.

    Args:
      path: Path to manifest.json.

    Returns:
      Mapping of normalized repo path to flat corpus filename.
    """
    import posixpath

    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {posixpath.normpath(v["url"]): flat for flat, v in manifest.items()}


DEMOS = load_demos(DEMOS_PATH)
CORPUS_INDEX = load_corpus_index(MANIFEST_PATH)


# ============================================================
# Prompt assembly
# ============================================================


def hydrate(chunk: dict, demos: dict[str, dict]) -> str:
    """Re-attaches demo source to a retrieved chunk for prompt assembly.

    Retrieval matched on prose alone; generation needs the code. Splicing
    it in here keeps identifier-heavy boilerplate out of the index while
    still giving the model the concrete implementation the prose names.

    Args:
      chunk: A retrieved chunk carrying metadata.demo_refs.
      demos: The demo store from load_demos.

    Returns:
      Chunk text with each referenced demo appended as a fenced code block.
    """
    parts = [chunk["text"]]
    for ref in chunk["metadata"].get("demo_refs", []):
        demo = demos.get(ref)
        if demo is None:
            continue
        parts.append(f"```{demo['lang']}\n{demo['code']}\n```")
    return "\n\n".join(parts)


def format_context(results: list[dict], demos: dict[str, dict]) -> str:
    """Formats retrieved chunks as numbered passages for the prompt.

    Passage numbers are 1-based and positional, so passage n corresponds
    to results[n - 1]. That correspondence is what lets the inline [n]
    markers in the answer be resolved back to chunk ids.

    Args:
      results: Result dicts from a retrieval function.
      demos: The demo store from load_demos.

    Returns:
      A numbered passage block with chunk ids retained for traceability.
    """
    blocks = []
    for number, result in enumerate(results, start=1):
        body = hydrate(result, demos)
        blocks.append(f"[{number}] ({result['chunk_id']})\n{body}")
    return "\n\n".join(blocks)


# ============================================================
# Citations
# ============================================================


def parse_cited_numbers(answer_text: str, n_passages: int) -> set[int]:
    """Extracts the passage numbers the model actually cited.

    Numbers outside the range of supplied passages are discarded: a model
    citing [7] when three passages were given has hallucinated the marker,
    and recording it would corrupt the citation trace.

    Args:
      answer_text: The generated answer.
      n_passages: How many passages were supplied.

    Returns:
      Set of valid 1-based passage numbers found in the text.
    """
    found = {int(m) for m in CITATION_RE.findall(answer_text)}
    return {n for n in found if 1 <= n <= n_passages}


def build_citations(results: list[dict], answer_text: str) -> list[dict]:
    """Builds the citation records for one answer.

    Every retrieved passage is recorded, with cited_in_text marking those
    the model referenced. Keeping both makes it possible to measure how
    faithfully the model cites what it was given, which a list of cited
    chunks alone would discard.

    Args:
      results: The retrieved chunks, in rank order.
      answer_text: The generated answer, scanned for [n] markers.

    Returns:
      List of citation dicts, one per retrieved passage.
    """
    cited = parse_cited_numbers(answer_text, len(results))
    citations = []

    for number, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        repo_path = posixpath.normpath(metadata.get("url", ""))
        flat = CORPUS_INDEX.get(repo_path)

        citations.append(
            {
                "passage_number": number,
                "chunk_id": result["chunk_id"],
                "doc_id": result.get("doc_id"),
                "score": float(result.get("score", 0.0)),
                "cited_in_text": number in cited,
                "source": {
                    "repo_path": repo_path,
                    "corpus_path": str(CORPUS_DIR / flat) if flat else None,
                    "github_url": (
                        f"https://github.com/{REPO}/blob/{TAG}/{repo_path}"
                        if repo_path
                        else None
                    ),
                },
                "demo_refs": metadata.get("demo_refs", []),
            }
        )

    return citations


# ============================================================
# Generation
# ============================================================


def ask(system: str, user: str, stream: bool = STREAM) -> str:
    """Generates one answer with greedy decoding.

    Greedy rather than sampled so that reruns are reproducible, which
    matters when the point of the run is to compare retrievers rather
    than to compare samples.

    Args:
      system: System prompt.
      user: User message.
      stream: Whether to print tokens as they generate.

    Returns:
      The generated text with the prompt removed.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = TOKENIZER(text, return_tensors="pt").to(MODEL.device)

    outputs = MODEL.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        streamer=(
            TextStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
            if stream
            else None
        ),
    )
    # generate returns prompt + completion, so drop the prompt tokens
    completion = outputs[0][inputs["input_ids"].shape[1] :]
    return TOKENIZER.decode(completion, skip_special_tokens=True)


def answer_naive(question: dict) -> dict:
    """Answers a question with no retrieved context.

    Args:
      question: A question dict with question_id and question_text.

    Returns:
      An answer record with retriever None and no citations.
    """
    answer_text = ask(NAIVE_SYSTEM, question["question_text"])
    return {
        "answer_id": question["question_id"],
        "question_text": question["question_text"],
        "mode": "naive",
        "retriever": None,
        "answer_text": answer_text.strip(),
        "citations": [],
    }


def answer_rag(question: dict, retriever: str, k: int = TOP_K) -> dict:
    """Answers a question grounded in retrieved passages.

    Args:
      question: A question dict with question_id and question_text.
      retriever: Key into RETRIEVERS, either "dense" or "bm25".
      k: Number of passages to retrieve.

    Returns:
      An answer record naming the retriever used and its citations.
    """
    search = RETRIEVERS[retriever]
    results = search(question["question_text"], k=k)
    context = format_context(results, DEMOS)

    if SHOW_CONTEXT:
        print(f"\n--- context ({retriever})\n{context}\n--- end context\n")

    prompt = f"Passages:\n{context}\n\nQuestion: {question['question_text']}"
    answer_text = ask(RAG_SYSTEM, prompt)

    return {
        "answer_id": question["question_id"],
        "question_text": question["question_text"],
        "mode": "rag",
        "retriever": retriever,
        "answer_text": answer_text.strip(),
        "citations": build_citations(results, answer_text),
    }


# ============================================================
# Output
# ============================================================


def write_answers(records: list[dict], out_path: Path) -> None:
    """Writes one answer record per line, creating parents if needed.

    Args:
      records: Answer records to write.
      out_path: Destination .jsonl path (config.ANSWERS_PATH).

    Returns:
      None.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(retriever: str = RETRIEVER, modes: tuple[str, ...] = MODES) -> None:
    """Answers every question in the question set and writes the results.

    Args:
      retriever: Which retriever supplies context for the rag mode.
      modes: Which arms to run per question.

    Returns:
      None. Writes config.ANSWERS_PATH as a side effect.

    Raises:
      ValueError: If retriever is not a known key.
    """
    if retriever not in RETRIEVERS:
        raise ValueError(
            f"Unknown retriever '{retriever}'. Choose from {list(RETRIEVERS)}"
        )

    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")
    print(f"Retriever: {retriever}   modes: {', '.join(modes)}   k={TOP_K}")

    records: list[dict] = []
    total = len(questions) * len(modes)
    done = 0

    for question in questions:
        for mode in modes:
            done += 1
            sys.stdout.write(
                f"\r  [{done}/{total}] q{question['question_id']} ({mode})".ljust(60)
            )
            sys.stdout.flush()

            if mode == "naive":
                records.append(answer_naive(question))
            else:
                records.append(answer_rag(question, retriever))

    sys.stdout.write("\r" + " " * 60 + "\r")

    write_answers(records, ANSWERS_PATH)

    n_rag = sum(1 for r in records if r["mode"] == "rag")
    n_grounded = sum(
        1
        for r in records
        if r["mode"] == "rag" and any(c["cited_in_text"] for c in r["citations"])
    )
    n_idk = sum(
        1
        for r in records
        if r["answer_text"].strip().lower().startswith("i don't know")
    )

    print(f"Wrote {len(records)} answers to {ANSWERS_PATH}")
    print(f"  RAG answers with at least one inline citation: {n_grounded}/{n_rag}")
    print(f"  Answers declining to answer: {n_idk}")


if __name__ == "__main__":
    run()
