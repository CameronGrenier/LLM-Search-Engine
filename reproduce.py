"""Run the complete CP423 experiment and regenerate result tables."""

from __future__ import annotations
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


DERIVED_PATHS = [
    Path("data/corpus"),
    Path("data/processed/manifest.json"),
    Path("data/processed/docs.jsonl"),
    Path("data/processed/demos.jsonl"),
    Path("data/processed/chunks.jsonl"),
    Path("data/processed/answers.jsonl"),
    Path("data/indexes/bm25.pkl"),
    Path("data/indexes/dense.faiss"),
    Path("data/indexes/dense_ids.json"),
]

STAGES = [
    ("Fetch corpus", ["-m", "src.fetch_corpus"]),
    ("Preprocess corpus", ["-m", "src.preprocess"]),
    ("Chunk documents", ["-m", "src.chunk"]),
    ("Build BM25 index", ["-m", "src.retrieval.bm25"]),
    ("Build dense index", ["-m", "src.retrieval.dense"]),
    (
        "Evaluate retrieval",
        [
            "-m",
            "src.evaluate.retrieval_metrics",
            "--system",
            "both",
            "--k",
            "5",
            "--self-test",
        ],
    ),
    ("Generate answers", ["-m", "src.llm.llm"]),
    (
        "Evaluate generation",
        ["-m", "src.evaluate.generation_metrics"],
    ),
    (
        "Summarize human evaluation",
        ["-m", "src.evaluate.manual_summary"],
    ),
    (
        "Generate diagnostic analysis",
        ["-m", "src.evaluate.diagnostic"],
    ),
]

REQUIRED_OUTPUTS = [
    Path("data/processed/manifest.json"),
    Path("data/processed/docs.jsonl"),
    Path("data/processed/demos.jsonl"),
    Path("data/processed/chunks.jsonl"),
    Path("data/processed/answers.jsonl"),
    Path("data/indexes/bm25.pkl"),
    Path("data/indexes/dense.faiss"),
    Path("data/indexes/dense_ids.json"),
    Path("results/generation_per_answer.json"),
    Path("results/generation_aggregate.json"),
    Path("results/generation_manual_review.csv"),
    Path("results/generation_manual_summary.json"),
    Path("results/diagnostic_results.csv"),
    Path("results/diagnostic_summary.json"),
    Path("results/diagnostic_analysis.md"),
]


def clean_derived_files() -> None:
    print("Removing derived corpus, indexes, and generated answers...")
    for path in DERIVED_PATHS:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

def run_stage(
    number: int,
    total: int,
    name: str,
    arguments: list[str],
    environment: dict[str, str],
) -> None:
    print()
    print("=" * 100)
    print(f"[{number}/{total}] {name}")
    print("=" * 100)
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, *arguments],
        check=True,
        env=environment,
    )
    elapsed = time.perf_counter() - started
    print(f"Completed {name} in {elapsed:.1f} seconds.")


def validate_outputs() -> None:
    errors = []
    for path in REQUIRED_OUTPUTS:
        if not path.exists():
            errors.append(f"Missing output: {path}")
        elif path.is_file() and path.stat().st_size == 0:
            errors.append(f"Empty output: {path}")
    if errors:
        raise RuntimeError(
            "Reproduction validation failed:\n- "
            + "\n- ".join(errors)
        )
    print()
    print("Output validation passed.")
    print(f"Validated {len(REQUIRED_OUTPUTS)} required outputs.")

def main() -> None:
    overall_start = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "42"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    clean_derived_files()
    for number, (name, arguments) in enumerate(
        STAGES,
        start=1,
    ):
        run_stage(
            number,
            len(STAGES),
            name,
            arguments,
            environment,
        )
    validate_outputs()
    elapsed = time.perf_counter() - overall_start
    print()
    print("=" * 100)
    print("FULL REPRODUCTION COMPLETE")
    print("=" * 100)
    print(f"Total runtime: {elapsed / 60:.1f} minutes")
    print("All indexes, answers, metrics, and tables were regenerated.")

if __name__ == "__main__":
    main()
