"""Evaluate generated naive, BM25-RAG, and dense-RAG answers."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


QUESTIONS_PATH = Path("data/questions.jsonl")
ANSWERS_PATH = Path("data/processed/answers.jsonl")
RESULTS_DIR = Path("results")

PER_ANSWER_JSON = RESULTS_DIR / "generation_per_answer.json"
PER_ANSWER_CSV = RESULTS_DIR / "generation_per_answer.csv"
AGGREGATE_JSON = RESULTS_DIR / "generation_aggregate.json"
AGGREGATE_CSV = RESULTS_DIR / "generation_aggregate.csv"
REVIEW_CSV = RESULTS_DIR / "generation_manual_review.csv"

EXACT_FALLBACK = "I don't know"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON Lines records."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

    return records


def normalize_text(text: str) -> str:
    """Normalize text for exact-match and token-overlap evaluation."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(prediction: str, reference: str) -> float:
    """Compute bag-of-words token F1."""
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    if not prediction_tokens and not reference_tokens:
        return 1.0

    if not prediction_tokens or not reference_tokens:
        return 0.0

    prediction_counts: dict[str, int] = defaultdict(int)
    reference_counts: dict[str, int] = defaultdict(int)

    for token in prediction_tokens:
        prediction_counts[token] += 1

    for token in reference_tokens:
        reference_counts[token] += 1

    overlap = sum(
        min(count, reference_counts[token])
        for token, count in prediction_counts.items()
    )

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> bool:
    """Return normalized exact match."""
    return normalize_text(prediction) == normalize_text(reference)


def arm_name(record: dict[str, Any]) -> str:
    """Return a stable evaluation arm name."""
    if record["mode"] == "naive":
        return "naive"

    return f"rag_{record['retriever']}"


def cited_numbers_in_text(text: str) -> set[int]:
    """Extract all bracketed numeric citation references."""
    return {
        int(number)
        for number in re.findall(r"\[(\d+)\]", text)
    }


def evaluate_record(
    answer: dict[str, Any],
    question: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one generated answer."""
    answer_text = answer["answer_text"].strip()
    reference_answer = question["reference_answer"].strip()

    citations = answer.get("citations", [])
    retrieved_chunk_ids = [
        citation["chunk_id"]
        for citation in citations
    ]
    cited_chunk_ids = [
        citation["chunk_id"]
        for citation in citations
        if citation.get("cited_in_text", False)
    ]

    citation_numbers = cited_numbers_in_text(answer_text)
    valid_numbers = set(range(1, len(citations) + 1))
    invalid_citation_numbers = sorted(citation_numbers - valid_numbers)

    gold_chunk_ids = list(question["gold_chunk_ids"])
    gold_chunk_set = set(gold_chunk_ids)
    retrieved_chunk_set = set(retrieved_chunk_ids)
    cited_chunk_set = set(cited_chunk_ids)

    is_unanswerable = question["type"] == "unanswerable"
    is_exact_refusal = answer_text == EXACT_FALLBACK

    context_has_all_gold = (
        not is_unanswerable
        and gold_chunk_set.issubset(retrieved_chunk_set)
    )

    if answer["mode"] == "rag":
        expected_refusal = (
            is_unanswerable
            or not context_has_all_gold
        )
        support_gate_correct: bool | None = (
            is_exact_refusal == expected_refusal
        )
        unsupported_answer = (
            expected_refusal
            and not is_exact_refusal
        )
        supported_refusal = (
            context_has_all_gold
            and is_exact_refusal
        )
    else:
        expected_refusal = None
        support_gate_correct = None
        unsupported_answer = None
        supported_refusal = None

    if cited_chunk_ids:
        citation_precision = (
            len(cited_chunk_set & gold_chunk_set)
            / len(cited_chunk_set)
        )
    else:
        citation_precision = 0.0

    if gold_chunk_ids:
        citation_recall = (
            len(cited_chunk_set & gold_chunk_set)
            / len(gold_chunk_set)
        )
    else:
        citation_recall = 0.0

    has_valid_inline_citation = (
        bool(cited_chunk_ids)
        and not invalid_citation_numbers
    )

    return {
        "answer_id": answer["answer_id"],
        "question_id": question["question_id"],
        "question_type": question["type"],
        "arm": arm_name(answer),
        "mode": answer["mode"],
        "retriever": answer["retriever"],
        "question_text": question["question_text"],
        "reference_answer": reference_answer,
        "answer_text": answer_text,
        "exact_match": exact_match(answer_text, reference_answer),
        "token_f1": token_f1(answer_text, reference_answer),
        "is_exact_refusal": is_exact_refusal,
        "is_unanswerable": is_unanswerable,
        "context_has_all_gold": (
            context_has_all_gold
            if answer["mode"] == "rag"
            else None
        ),
        "expected_refusal": expected_refusal,
        "support_gate_correct": support_gate_correct,
        "unsupported_answer": unsupported_answer,
        "supported_refusal": supported_refusal,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "gold_chunk_ids": gold_chunk_ids,
        "cited_chunk_ids": cited_chunk_ids,
        "has_valid_inline_citation": has_valid_inline_citation,
        "invalid_citation_numbers": invalid_citation_numbers,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
    }


def safe_mean(values: list[float]) -> float:
    """Return arithmetic mean, or zero for an empty sequence."""
    return mean(values) if values else 0.0


def aggregate_arm(
    arm: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate generation metrics for one system arm."""
    answerable = [
        record
        for record in records
        if not record["is_unanswerable"]
    ]
    unanswerable = [
        record
        for record in records
        if record["is_unanswerable"]
    ]
    non_refusals = [
        record
        for record in records
        if not record["is_exact_refusal"]
    ]

    aggregate: dict[str, Any] = {
        "arm": arm,
        "count": len(records),
        "exact_match": safe_mean([
            float(record["exact_match"])
            for record in records
        ]),
        "mean_token_f1": safe_mean([
            record["token_f1"]
            for record in records
        ]),
        "answerable_mean_token_f1": safe_mean([
            record["token_f1"]
            for record in answerable
        ]),
        "unanswerable_exact_refusal_accuracy": safe_mean([
            float(record["is_exact_refusal"])
            for record in unanswerable
        ]),
        "exact_refusal_count": sum(
            record["is_exact_refusal"]
            for record in records
        ),
    }

    if arm.startswith("rag_"):
        supported = [
            record
            for record in records
            if record["context_has_all_gold"]
        ]
        unsupported = [
            record
            for record in records
            if not record["context_has_all_gold"]
        ]

        aggregate.update({
            "context_supported_count": len(supported),
            "context_unsupported_count": len(unsupported),
            "support_gate_accuracy": safe_mean([
                float(record["support_gate_correct"])
                for record in records
            ]),
            "supported_answer_rate": safe_mean([
                float(not record["is_exact_refusal"])
                for record in supported
            ]),
            "unsupported_refusal_rate": safe_mean([
                float(record["is_exact_refusal"])
                for record in unsupported
            ]),
            "unsupported_answer_rate": safe_mean([
                float(record["unsupported_answer"])
                for record in records
            ]),
            "supported_refusal_rate": safe_mean([
                float(record["supported_refusal"])
                for record in records
            ]),
            "non_refusal_inline_citation_rate": safe_mean([
                float(record["has_valid_inline_citation"])
                for record in non_refusals
            ]),
            "mean_citation_precision": safe_mean([
                record["citation_precision"]
                for record in non_refusals
            ]),
            "mean_citation_recall": safe_mean([
                record["citation_recall"]
                for record in answerable
            ]),
            "invalid_citation_answer_count": sum(
                bool(record["invalid_citation_numbers"])
                for record in records
            ),
        })

    return aggregate


def write_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Write dictionaries to CSV, JSON-encoding nested values."""
    if not records:
        raise ValueError(f"Cannot write empty CSV: {path}")

    fields: list[str] = []

    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()

        for record in records:
            row = {}

            for key, value in record.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(
                        value,
                        ensure_ascii=False,
                    )
                else:
                    row[key] = value

            writer.writerow(row)


def main() -> None:
    """Run generation evaluation and save all result tables."""
    questions = load_jsonl(QUESTIONS_PATH)
    answers = load_jsonl(ANSWERS_PATH)

    question_lookup = {
        question["question_id"]: question
        for question in questions
    }

    if len(questions) != 13:
        raise ValueError(
            f"Expected 13 questions, found {len(questions)}"
        )

    if len(answers) != 39:
        raise ValueError(
            f"Expected 39 answers, found {len(answers)}"
        )

    answer_ids = [
        answer["answer_id"]
        for answer in answers
    ]

    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("Duplicate answer IDs found.")

    per_answer = []

    for answer in answers:
        question_id = answer["question_id"]

        if question_id not in question_lookup:
            raise ValueError(
                f"Unknown question ID in answers: {question_id}"
            )

        per_answer.append(
            evaluate_record(
                answer,
                question_lookup[question_id],
            )
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in per_answer:
        grouped[record["arm"]].append(record)

    expected_arms = {"naive", "rag_bm25", "rag_dense"}

    if set(grouped) != expected_arms:
        raise ValueError(
            f"Expected arms {sorted(expected_arms)}, "
            f"found {sorted(grouped)}"
        )

    aggregate = [
        aggregate_arm(arm, grouped[arm])
        for arm in ["naive", "rag_bm25", "rag_dense"]
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    PER_ANSWER_JSON.write_text(
        json.dumps(
            per_answer,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    AGGREGATE_JSON.write_text(
        json.dumps(
            aggregate,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(PER_ANSWER_CSV, per_answer)
    write_csv(AGGREGATE_CSV, aggregate)

    review_rows = []

    for record in per_answer:
        review_rows.append({
            "answer_id": record["answer_id"],
            "question_id": record["question_id"],
            "arm": record["arm"],
            "question_type": record["question_type"],
            "question_text": record["question_text"],
            "reference_answer": record["reference_answer"],
            "answer_text": record["answer_text"],
            "retrieved_chunk_ids": record["retrieved_chunk_ids"],
            "cited_chunk_ids": record["cited_chunk_ids"],
            "automated_token_f1": record["token_f1"],
            "automated_exact_match": record["exact_match"],
            "manual_correctness_0_to_2": "",
            "manual_groundedness_0_to_2": "",
            "manual_completeness_0_to_2": "",
            "manual_notes": "",
        })

    manual_fields = [
        "manual_correctness_0_to_2",
        "manual_groundedness_0_to_2",
        "manual_completeness_0_to_2",
        "manual_notes",
    ]

    if REVIEW_CSV.exists():
        with REVIEW_CSV.open(
            encoding="utf-8",
            newline="",
        ) as file:
            existing_reviews = {
                row["answer_id"]: row
                for row in csv.DictReader(file)
            }

        for row in review_rows:
            previous = existing_reviews.get(row["answer_id"])

            if previous is None:
                continue

            for field in manual_fields:
                row[field] = previous.get(field, "")

    write_csv(REVIEW_CSV, review_rows)

    print("Generation evaluation complete.")
    print(f"Evaluated answers: {len(per_answer)}")
    print()

    for row in aggregate:
        print(row["arm"])
        print(
            f"  Exact match: {row['exact_match']:.4f}"
        )
        print(
            f"  Mean token F1: {row['mean_token_f1']:.4f}"
        )
        print(
            "  Unanswerable exact-refusal accuracy: "
            f"{row['unanswerable_exact_refusal_accuracy']:.4f}"
        )

        if row["arm"].startswith("rag_"):
            print(
                f"  Support-gate accuracy: "
                f"{row['support_gate_accuracy']:.4f}"
            )
            print(
                f"  Supported-answer rate: "
                f"{row['supported_answer_rate']:.4f}"
            )
            print(
                f"  Unsupported-refusal rate: "
                f"{row['unsupported_refusal_rate']:.4f}"
            )
            print(
                f"  Inline-citation rate: "
                f"{row['non_refusal_inline_citation_rate']:.4f}"
            )
            print(
                f"  Mean citation precision: "
                f"{row['mean_citation_precision']:.4f}"
            )
            print(
                f"  Mean citation recall: "
                f"{row['mean_citation_recall']:.4f}"
            )

        print()

    print("Wrote:")
    print(f"  {PER_ANSWER_JSON}")
    print(f"  {PER_ANSWER_CSV}")
    print(f"  {AGGREGATE_JSON}")
    print(f"  {AGGREGATE_CSV}")
    print(f"  {REVIEW_CSV}")


if __name__ == "__main__":
    main()
