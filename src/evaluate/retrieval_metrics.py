"""Retrieval evaluation for the CP423 RAG project."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean

from config import (
    ADAPTER_ENV_VAR,
    CHUNKS_PATH,
    QUESTIONS_PATH,
    QUERY_ADAPTER,
    RESULTS_DIR,
)
from src.corpus_io import load_chunks


SearchFunction = Callable[[str, int], list[dict]]


def _validate_ranking(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int,
) -> None:
    """Validate inputs shared by the retrieval metrics."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not gold_ids:
        raise ValueError(
            "gold_ids cannot be empty for retrieval evaluation. "
            "Exclude unanswerable questions from retrieval metrics."
        )

    if len(gold_ids) != len(set(gold_ids)):
        raise ValueError("gold_ids contains duplicate chunk IDs.")

    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError("retrieved_ids contains duplicate chunk IDs.")


def precision_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Calculate binary Precision@k."""
    _validate_ranking(retrieved_ids, gold_ids, k)

    gold = set(gold_ids)
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(chunk_id in gold for chunk_id in top_k)

    return relevant_retrieved / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Calculate binary Recall@k."""
    _validate_ranking(retrieved_ids, gold_ids, k)

    gold = set(gold_ids)
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(chunk_id in gold for chunk_id in top_k)

    return relevant_retrieved / len(gold)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
) -> float:
    """Calculate reciprocal rank of the first relevant chunk."""
    _validate_ranking(retrieved_ids, gold_ids, k=1)

    gold = set(gold_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Calculate binary normalized discounted cumulative gain at k."""
    _validate_ranking(retrieved_ids, gold_ids, k)

    gold = set(gold_ids)
    top_k = retrieved_ids[:k]

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(top_k, start=1)
        if chunk_id in gold
    )

    ideal_relevant = min(len(gold), k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )

    return dcg / ideal_dcg


def evaluate_ranking(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int = 5,
) -> dict[str, float]:
    """Calculate all required retrieval metrics for one question."""
    return {
        f"precision_at_{k}": precision_at_k(retrieved_ids, gold_ids, k),
        f"recall_at_{k}": recall_at_k(retrieved_ids, gold_ids, k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, gold_ids),
        f"ndcg_at_{k}": ndcg_at_k(retrieved_ids, gold_ids, k),
    }


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    """Load the gold questions from their JSONL file."""
    with path.open("r", encoding="utf-8") as file:
        questions = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not questions:
        raise ValueError(f"No questions found in {path}.")

    return questions


def evaluate_questions(
    system_name: str,
    search: SearchFunction,
    questions: Sequence[dict],
    full_ranking_depth: int,
    k: int = 5,
) -> dict:
    """Evaluate one retrieval system on all answerable questions."""
    if full_ranking_depth < k:
        raise ValueError(
            "The full ranking depth must be greater than or equal to k."
        )

    answerable = [
        question
        for question in questions
        if question["type"] != "unanswerable"
    ]

    if not answerable:
        raise ValueError("No answerable questions were found.")

    records = []

    for question in answerable:
        results = search(
            question["question_text"],
            full_ranking_depth,
        )

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        gold_ids = question["gold_chunk_ids"]
        metrics = evaluate_ranking(retrieved_ids, gold_ids, k)

        position_by_id = {
            chunk_id: rank
            for rank, chunk_id in enumerate(retrieved_ids, start=1)
        }

        gold_ranks = {
            chunk_id: position_by_id.get(chunk_id)
            for chunk_id in gold_ids
        }

        found_ranks = [
            rank
            for rank in gold_ranks.values()
            if rank is not None
        ]

        first_relevant_rank = min(found_ranks) if found_ranks else None

        records.append(
            {
                "question_id": question["question_id"],
                "question_type": question["type"],
                "question_text": question["question_text"],
                "gold_chunk_ids": gold_ids,
                "top_k_chunk_ids": retrieved_ids[:k],
                "gold_ranks": gold_ranks,
                "first_relevant_rank": first_relevant_rank,
                "metrics": metrics,
            }
        )

    precision_key = f"precision_at_{k}"
    recall_key = f"recall_at_{k}"
    ndcg_key = f"ndcg_at_{k}"

    aggregate = {
        precision_key: fmean(
            record["metrics"][precision_key]
            for record in records
        ),
        recall_key: fmean(
            record["metrics"][recall_key]
            for record in records
        ),
        "mrr": fmean(
            record["metrics"]["reciprocal_rank"]
            for record in records
        ),
        ndcg_key: fmean(
            record["metrics"][ndcg_key]
            for record in records
        ),
    }

    return {
        "system": system_name,
        "cutoff": k,
        "mrr_ranking_depth": full_ranking_depth,
        "total_gold_questions": len(questions),
        "evaluated_answerable_questions": len(answerable),
        "excluded_unanswerable_questions": len(questions) - len(answerable),
        "aggregate": aggregate,
        "questions": records,
    }


def save_results(payload: dict) -> tuple[Path, Path]:
    """Save retrieval results as JSON and CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    system_name = payload["system"]
    json_path = RESULTS_DIR / f"retrieval_{system_name}.json"
    csv_path = RESULTS_DIR / f"retrieval_{system_name}.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")

    k = payload["cutoff"]
    precision_key = f"precision_at_{k}"
    recall_key = f"recall_at_{k}"
    ndcg_key = f"ndcg_at_{k}"

    fieldnames = [
        "question_id",
        "question_type",
        "question_text",
        "gold_chunk_ids",
        f"top_{k}_chunk_ids",
        "gold_ranks",
        "first_relevant_rank",
        precision_key,
        recall_key,
        "reciprocal_rank",
        ndcg_key,
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in payload["questions"]:
            writer.writerow(
                {
                    "question_id": record["question_id"],
                    "question_type": record["question_type"],
                    "question_text": record["question_text"],
                    "gold_chunk_ids": json.dumps(
                        record["gold_chunk_ids"],
                        ensure_ascii=False,
                    ),
                    f"top_{k}_chunk_ids": json.dumps(
                        record["top_k_chunk_ids"],
                        ensure_ascii=False,
                    ),
                    "gold_ranks": json.dumps(
                        record["gold_ranks"],
                        ensure_ascii=False,
                    ),
                    "first_relevant_rank": record["first_relevant_rank"],
                    precision_key: record["metrics"][precision_key],
                    recall_key: record["metrics"][recall_key],
                    "reciprocal_rank": record["metrics"][
                        "reciprocal_rank"
                    ],
                    ndcg_key: record["metrics"][ndcg_key],
                }
            )

        writer.writerow(
            {
                "question_id": "MEAN",
                precision_key: payload["aggregate"][precision_key],
                recall_key: payload["aggregate"][recall_key],
                "reciprocal_rank": payload["aggregate"]["mrr"],
                ndcg_key: payload["aggregate"][ndcg_key],
            }
        )

    return json_path, csv_path


def print_summary(payload: dict) -> None:
    """Print per-question and aggregate retrieval results."""
    system_name = payload["system"].upper()
    k = payload["cutoff"]

    precision_key = f"precision_at_{k}"
    recall_key = f"recall_at_{k}"
    ndcg_key = f"ndcg_at_{k}"

    print(f"\n{system_name} retrieval evaluation")
    print("=" * 72)
    print(
        "Evaluated answerable questions:",
        payload["evaluated_answerable_questions"],
    )
    print(
        "Excluded unanswerable questions:",
        payload["excluded_unanswerable_questions"],
    )
    print("MRR ranking depth:", payload["mrr_ranking_depth"])
    print()

    for record in payload["questions"]:
        metrics = record["metrics"]

        print(
            f"Q{record['question_id']:>2} "
            f"{record['question_type']:<10} "
            f"P@{k}={metrics[precision_key]:.3f} "
            f"R@{k}={metrics[recall_key]:.3f} "
            f"RR={metrics['reciprocal_rank']:.3f} "
            f"nDCG@{k}={metrics[ndcg_key]:.3f} "
            f"gold_ranks={record['gold_ranks']}"
        )

    aggregate = payload["aggregate"]

    print("\nAggregate metrics")
    print("-" * 72)
    print(f"Precision@{k}: {aggregate[precision_key]:.6f}")
    print(f"Recall@{k}:    {aggregate[recall_key]:.6f}")
    print(f"MRR:           {aggregate['mrr']:.6f}")
    print(f"nDCG@{k}:      {aggregate[ndcg_key]:.6f}")


def run_bm25(k: int = 5) -> dict:
    """Run the complete BM25 retrieval evaluation."""
    from src.retrieval.bm25 import bm25_search

    chunks, _, _ = load_chunks(CHUNKS_PATH)
    questions = load_questions()

    def search(query: str, depth: int) -> list[dict]:
        return bm25_search(query, k=depth)

    return evaluate_questions(
        system_name="bm25",
        search=search,
        questions=questions,
        full_ranking_depth=len(chunks),
        k=k,
    )


def run_dense(k: int = 5) -> dict:
    """Run the complete dense retrieval evaluation."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ[ADAPTER_ENV_VAR] = QUERY_ADAPTER

    from src.retrieval.dense import dense_search, load_dense_index

    index, chunk_ids = load_dense_index()
    _, by_id, _ = load_chunks(CHUNKS_PATH)
    questions = load_questions()

    if index.ntotal != len(chunk_ids):
        raise ValueError(
            "Dense FAISS index size does not match the chunk ID mapping."
        )

    def search(query: str, depth: int) -> list[dict]:
        return dense_search(
            query,
            k=depth,
            index=index,
            chunk_ids=chunk_ids,
            by_id=by_id,
        )

    return evaluate_questions(
        system_name="dense",
        search=search,
        questions=questions,
        full_ranking_depth=len(chunk_ids),
        k=k,
    )


def _run_self_tests() -> None:
    """Run deterministic checks for each metric formula."""
    ranking = ["a", "x", "b", "y", "z"]
    gold = ["a", "b"]

    metrics = evaluate_ranking(ranking, gold, k=5)

    assert math.isclose(metrics["precision_at_5"], 0.4)
    assert math.isclose(metrics["recall_at_5"], 1.0)
    assert math.isclose(metrics["reciprocal_rank"], 1.0)
    assert math.isclose(metrics["ndcg_at_5"], 0.9197207891481876)

    partial_ranking = ["x", "a", "y", "z", "w", "b"]
    partial_metrics = evaluate_ranking(partial_ranking, gold, k=5)

    assert math.isclose(partial_metrics["precision_at_5"], 0.2)
    assert math.isclose(partial_metrics["recall_at_5"], 0.5)
    assert math.isclose(partial_metrics["reciprocal_rank"], 0.5)
    assert math.isclose(
        partial_metrics["ndcg_at_5"],
        0.38685280723454163,
    )

    missing_ranking = ["x", "y", "z"]
    missing_metrics = evaluate_ranking(missing_ranking, gold, k=3)

    assert math.isclose(missing_metrics["precision_at_3"], 0.0)
    assert math.isclose(missing_metrics["recall_at_3"], 0.0)
    assert math.isclose(missing_metrics["reciprocal_rank"], 0.0)
    assert math.isclose(missing_metrics["ndcg_at_3"], 0.0)

    print("Retrieval metric self-tests PASSED.")


def main() -> None:
    """Run self-tests or retrieval evaluation from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 and dense retrieval against the gold set."
    )
    parser.add_argument(
        "--system",
        choices=("bm25", "dense", "both"),
        help="Retrieval system to evaluate.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Metric cutoff. Default: 5.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic metric formula tests.",
    )

    args = parser.parse_args()

    if args.self_test or args.system is None:
        _run_self_tests()

        if args.system is None:
            return

    systems = (
        ("bm25", "dense")
        if args.system == "both"
        else (args.system,)
    )

    for system_name in systems:
        payload = (
            run_bm25(k=args.k)
            if system_name == "bm25"
            else run_dense(k=args.k)
        )

        print_summary(payload)
        json_path, csv_path = save_results(payload)

        print("\nSaved:")
        print(json_path)
        print(csv_path)


if __name__ == "__main__":
    main()
