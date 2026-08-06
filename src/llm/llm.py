"""Batch question answering, with and without retrieved context.

Reads questions from config.QUESTIONS_PATH and writes one answer record
per (question, mode, retriever) triple to config.ANSWERS_PATH.

The rag mode runs once per retriever in RETRIEVERS_TO_RUN, so a single
invocation produces every arm of the comparison. All arms share identical
prompt assembly and identical output formatting, so any difference in the
resulting answers is attributable to retrieval alone, which is what makes
the comparison meaningful.

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
from src.seeding import seed_everything  # noqa: E402

# ============================================================
# Run configuration
# ============================================================

# Which retrievers supply context for the rag mode. Each one produces its
# own rag record per question, so the arms can be compared directly. Naive
# mode ignores this. Narrow it to a single entry to run one arm.
RETRIEVERS_TO_RUN = ("dense", "bm25")

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

seed_everything()

device = get_device()
MODEL = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL, dtype="auto").to(device)
TOKENIZER = AutoTokenizer.from_pretrained(GENERATION_MODEL)
MODEL.eval()


GROUNDED_ANSWER_SYSTEM = """Answer the question using only the supplied numbered passages.

If the passages do not explicitly answer the question, your entire reply must be
this exact sentence:

I don't know

Write it exactly that way: a straight apostrophe, no trailing period, no citation,
no explanation, and nothing before or after it.

Otherwise, answer the question and follow these rules:
- Answer every part of the question that is explicitly supported.
- Place an inline citation such as [1] immediately after every factual sentence.
- Use only passage numbers that were supplied.
- Do not use prior knowledge.
- Do not infer prop names, defaults, variants, or component behaviour from related components.
- Do not treat a passing mention, dependency list, or example name as an answer.
- Never reply with citation markers alone; every reply must contain a sentence.
- Keep the answer concise."""


NAIVE_SYSTEM = """Answer the question about Material UI as accurately as you can. \
Keep it concise and to the point."""


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
    """Format retrieved chunks as numbered passages carrying their demo code.

    The split between what is indexed and what is prompted is deliberate.
    Demo source stays out of the index, because every MUI demo opens with a
    near-identical import and export block and hundreds of chunks sharing that
    boilerplate compress the distances retrieval depends on. It goes into the
    prompt, because "how do I do X in MUI" is answered by the code, not by the
    prose describing it.

    The code is appended inside the passage that referenced it rather than as
    a separate block, so a citation of [n] still points at one source.

    Args:
      results: Result dicts from a retrieval function.
      demos: The demo store from load_demos.

    Returns:
      Numbered passages, each holding its chunk id, chunk text, and any demo
      source the chunk referenced, as fenced code blocks.
    """
    blocks = []

    for number, result in enumerate(results, start=1):
        blocks.append(
            f"[{number}] ({result['chunk_id']})\n"
            f"{hydrate(result, demos)}"
        )

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
                # Subscripted, not .get(). A retriever that stops emitting
                # doc_id should fail here rather than write null into every
                # citation record, which is how this went unnoticed before.
                "doc_id": result["doc_id"],
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
        "answer_id": f"{question['question_id']}-naive",
        "question_id": question["question_id"],
        "question_text": question["question_text"],
        "mode": "naive",
        "retriever": None,
        "answer_text": answer_text.strip(),
        "citations": [],
    }


def answer_rag(question: dict, retriever: str, k: int = TOP_K) -> dict:
    """Answer one question from retrieved context, returning the model's reply.

    One generation call, one answer, recorded verbatim. Nothing between the
    model and the result file rewrites, re-scores, or suppresses what it said,
    so a refusal is a refusal the model chose to produce, which is what
    GROUNDED_ANSWER_SYSTEM instructs it to do when the passages fall short.

    Grounding is judged by a human, in results/generation_manual_review.csv.
    An earlier version of this function asked the same model to verify its own
    answers and rewrote the ones it rejected. That gated the output on a judge
    no more reliable than the thing it judged, and it made the inline-citation
    rate true by construction. A person reads the passages instead.

    Args:
      question: A question dict with question_id and question_text.
      retriever: Key into RETRIEVERS naming the context source.
      k: Number of chunks to retrieve.

    Returns:
      An answer record carrying the answer and one citation entry per
      retrieved passage.
    """
    search = RETRIEVERS[retriever]
    results = search(question["question_text"], k=k)
    context = format_context(results, DEMOS)

    if SHOW_CONTEXT:
        print(
            f"\n--- context ({retriever})\n"
            f"{context}\n"
            "--- end context\n"
        )

    answer_prompt = (
        f"Numbered passages:\n{context}\n\n"
        f"Question:\n{question['question_text']}\n\n"
        "Answer:"
    )

    answer_text = ask(
        GROUNDED_ANSWER_SYSTEM,
        answer_prompt,
    ).strip()

    return {
        "answer_id": f"{question['question_id']}-rag-{retriever}",
        "question_id": question["question_id"],
        "question_text": question["question_text"],
        "mode": "rag",
        "retriever": retriever,
        "answer_text": answer_text,
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


def run(
    retrievers: tuple[str, ...] = RETRIEVERS_TO_RUN,
    modes: tuple[str, ...] = MODES,
) -> None:
    """Answers every question in the question set and writes the results.

    The rag mode runs once per retriever, so each question yields one naive
    record plus one rag record per retriever. Every arm is written to the
    same file, distinguished by the mode and retriever fields.

    Args:
      retrievers: Which retrievers supply context for the rag mode.
      modes: Which arms to run per question.

    Returns:
      None. Writes config.ANSWERS_PATH as a side effect.

    Raises:
      ValueError: If any entry of retrievers is not a known key.
    """
    unknown = [r for r in retrievers if r not in RETRIEVERS]
    if unknown:
        raise ValueError(
            f"Unknown retriever(s) {unknown}. Choose from {list(RETRIEVERS)}"
        )

    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")
    print(
        f"Retrievers: {', '.join(retrievers)}   "
        f"modes: {', '.join(modes)}   k={TOP_K}"
    )

    # One naive record per question, plus one rag record per retriever.
    per_question = [("naive", None)] if "naive" in modes else []
    if "rag" in modes:
        per_question += [("rag", retriever) for retriever in retrievers]

    records: list[dict] = []
    total = len(questions) * len(per_question)
    done = 0

    for question in questions:
        for mode, retriever in per_question:
            done += 1
            label = mode if retriever is None else f"{mode}/{retriever}"
            sys.stdout.write(
                f"\r  [{done}/{total}] q{question['question_id']} ({label})".ljust(60)
            )
            sys.stdout.flush()

            if mode == "naive":
                records.append(answer_naive(question))
            else:
                records.append(answer_rag(question, retriever))

    sys.stdout.write("\r" + " " * 60 + "\r")

    write_answers(records, ANSWERS_PATH)

    n_idk = sum(
        1
        for r in records
        if r["answer_text"].strip().lower().startswith("i don't know")
    )

    print(f"Wrote {len(records)} answers to {ANSWERS_PATH}")
    for retriever in retrievers:
        arm = [
            r for r in records if r["mode"] == "rag" and r["retriever"] == retriever
        ]
        if not arm:
            continue
        n_grounded = sum(
            1 for r in arm if any(c["cited_in_text"] for c in r["citations"])
        )
        print(
            f"  {retriever}: RAG answers with at least one inline citation: "
            f"{n_grounded}/{len(arm)}"
        )
    print(f"  Answers declining to answer: {n_idk}")


if __name__ == "__main__":
    run()
