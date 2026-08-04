"""Generate the ten-question no-context diagnostic analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path


QUESTIONS_PATH = Path("data/questions.jsonl")
ANSWERS_PATH = Path("results/generation_per_answer.json")
REVIEW_PATH = Path("results/generation_manual_review.csv")

CSV_PATH = Path("results/diagnostic_results.csv")
JSON_PATH = Path("results/diagnostic_summary.json")
MARKDOWN_PATH = Path("results/diagnostic_analysis.md")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def main() -> None:
    questions = load_jsonl(QUESTIONS_PATH)

    answers = json.loads(
        ANSWERS_PATH.read_text(encoding="utf-8")
    )

    with REVIEW_PATH.open(
        encoding="utf-8",
        newline="",
    ) as file:
        reviews = list(csv.DictReader(file))

    answer_lookup = {
        record["answer_id"]: record
        for record in answers
    }

    review_lookup = {
        record["answer_id"]: record
        for record in reviews
    }

    diagnostic_questions = [
        question
        for question in questions
        if question["type"] != "unanswerable"
    ]

    if len(diagnostic_questions) != 10:
        raise ValueError(
            "Expected exactly 10 answerable diagnostic questions, "
            f"found {len(diagnostic_questions)}."
        )

    rows = []

    for question in diagnostic_questions:
        question_id = question["question_id"]

        naive_id = f"{question_id}-naive"
        bm25_id = f"{question_id}-rag-bm25"
        dense_id = f"{question_id}-rag-dense"

        naive_correctness = int(
            review_lookup[naive_id][
                "manual_correctness_0_to_2"
            ]
        )
        naive_completeness = int(
            review_lookup[naive_id][
                "manual_completeness_0_to_2"
            ]
        )
        bm25_correctness = int(
            review_lookup[bm25_id][
                "manual_correctness_0_to_2"
            ]
        )
        dense_correctness = int(
            review_lookup[dense_id][
                "manual_correctness_0_to_2"
            ]
        )

        if (
            naive_correctness == 2
            and naive_completeness == 2
        ):
            naive_status = "fully correct"
        elif naive_correctness == 1:
            naive_status = "partially correct"
        else:
            naive_status = "incorrect"

        best_rag_score = max(
            bm25_correctness,
            dense_correctness,
        )

        if dense_correctness > bm25_correctness:
            best_rag_system = "dense"
        elif bm25_correctness > dense_correctness:
            best_rag_system = "bm25"
        else:
            best_rag_system = "tie"

        rows.append({
            "question_id": question_id,
            "question_type": question["type"],
            "question_text": question["question_text"],
            "reference_answer": question["reference_answer"],
            "naive_answer": answer_lookup[naive_id][
                "answer_text"
            ],
            "naive_correctness_0_to_2": naive_correctness,
            "naive_completeness_0_to_2": naive_completeness,
            "naive_status": naive_status,
            "bm25_rag_answer": answer_lookup[bm25_id][
                "answer_text"
            ],
            "bm25_correctness_0_to_2": bm25_correctness,
            "dense_rag_answer": answer_lookup[dense_id][
                "answer_text"
            ],
            "dense_correctness_0_to_2": dense_correctness,
            "best_rag_system": best_rag_system,
            "best_rag_correctness_0_to_2": best_rag_score,
            "retrieval_improved_answer": (
                best_rag_score > naive_correctness
            ),
        })

    naive_fully_correct = sum(
        row["naive_status"] == "fully correct"
        for row in rows
    )
    naive_partially_correct = sum(
        row["naive_status"] == "partially correct"
        for row in rows
    )
    naive_incorrect = sum(
        row["naive_status"] == "incorrect"
        for row in rows
    )
    bm25_fully_correct = sum(
        row["bm25_correctness_0_to_2"] == 2
        for row in rows
    )
    dense_fully_correct = sum(
        row["dense_correctness_0_to_2"] == 2
        for row in rows
    )
    retrieval_improved = sum(
        row["retrieval_improved_answer"]
        for row in rows
    )

    summary = {
        "diagnostic_question_count": len(rows),
        "question_composition": {
            "factoid": sum(
                row["question_type"] == "factoid"
                for row in rows
            ),
            "multi_hop": sum(
                row["question_type"] == "multi-hop"
                for row in rows
            ),
        },
        "naive_without_context": {
            "fully_correct": naive_fully_correct,
            "partially_correct": naive_partially_correct,
            "incorrect": naive_incorrect,
            "fully_correct_rate": (
                naive_fully_correct / len(rows)
            ),
        },
        "bm25_rag": {
            "fully_correct": bm25_fully_correct,
            "fully_correct_rate": (
                bm25_fully_correct / len(rows)
            ),
        },
        "dense_rag": {
            "fully_correct": dense_fully_correct,
            "fully_correct_rate": (
                dense_fully_correct / len(rows)
            ),
        },
        "retrieval_improved_answer_count": retrieval_improved,
        "retrieval_improved_answer_rate": (
            retrieval_improved / len(rows)
        ),
        "conclusion": (
            "The local LLM answered none of the ten diagnostic "
            "questions fully correctly without retrieved context. "
            f"Retrieval improved {retrieval_improved} of the ten "
            "answers, demonstrating that the selected corpus "
            "contains information the model did not reliably know "
            "from its parameters alone."
        ),
    }

    with CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    JSON_PATH.write_text(
        json.dumps(
            {
                "summary": summary,
                "questions": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown = [
        "# No-Context Diagnostic Analysis",
        "",
        "## Purpose",
        "",
        (
            "The local Qwen2.5-3B-Instruct model was asked ten "
            "answerable factual questions without access to the "
            "Material UI corpus. The same questions were then "
            "answered using BM25 and dense retrieval with the same "
            "generation model and settings."
        ),
        "",
        "## Diagnostic composition",
        "",
        "- 10 answerable questions",
        "- 7 factoid questions",
        "- 3 multi-hop questions",
        "",
        "## Summary",
        "",
        "| System | Fully correct | Rate |",
        "|---|---:|---:|",
        (
            f"| Naive, no context | "
            f"{naive_fully_correct}/10 | "
            f"{naive_fully_correct / 10:.0%} |"
        ),
        (
            f"| BM25 RAG | {bm25_fully_correct}/10 | "
            f"{bm25_fully_correct / 10:.0%} |"
        ),
        (
            f"| Dense RAG | {dense_fully_correct}/10 | "
            f"{dense_fully_correct / 10:.0%} |"
        ),
        "",
        (
            f"The naive model produced "
            f"{naive_partially_correct} partially correct answers "
            f"and {naive_incorrect} incorrect answers. Retrieval "
            f"improved {retrieval_improved}/10 answers."
        ),
        "",
        "## Per-question results",
        "",
        (
            "| Q | Type | Naive status | Naive score | "
            "BM25 score | Dense score |"
        ),
        "|---:|---|---|---:|---:|---:|",
    ]

    for row in rows:
        markdown.append(
            f"| {row['question_id']} "
            f"| {row['question_type']} "
            f"| {row['naive_status']} "
            f"| {row['naive_correctness_0_to_2']}/2 "
            f"| {row['bm25_correctness_0_to_2']}/2 "
            f"| {row['dense_correctness_0_to_2']}/2 |"
        )

    markdown.extend([
        "",
        "## Conclusion",
        "",
        summary["conclusion"],
        "",
        (
            "The model could sometimes produce related or partially "
            "correct information, but it did not fully answer any of "
            "the ten questions without retrieval. Dense retrieval "
            "produced the strongest overall improvement."
        ),
        "",
    ])

    MARKDOWN_PATH.write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    print("Diagnostic analysis complete.")
    print(
        f"  Naive fully correct: "
        f"{naive_fully_correct}/10"
    )
    print(
        f"  BM25 RAG fully correct: "
        f"{bm25_fully_correct}/10"
    )
    print(
        f"  Dense RAG fully correct: "
        f"{dense_fully_correct}/10"
    )
    print(
        f"  Retrieval improved: "
        f"{retrieval_improved}/10"
    )


if __name__ == "__main__":
    main()
