"""Summarize the completed human evaluation scores."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


REVIEW_PATH = Path("results/generation_manual_review.csv")
SUMMARY_JSON_PATH = Path("results/generation_manual_summary.json")
SUMMARY_CSV_PATH = Path("results/generation_manual_summary.csv")

SCORE_FIELDS = [
    "manual_correctness_0_to_2",
    "manual_groundedness_0_to_2",
    "manual_completeness_0_to_2",
]


def main() -> None:
    if not REVIEW_PATH.exists():
        raise FileNotFoundError(
            f"Missing manual review file: {REVIEW_PATH}"
        )

    with REVIEW_PATH.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 39:
        raise ValueError(
            f"Expected 39 reviewed answers, found {len(rows)}."
        )

    errors: list[str] = []

    for row in rows:
        for field in SCORE_FIELDS:
            value = row.get(field, "")

            try:
                score = int(value)
            except ValueError:
                errors.append(
                    f"{row['answer_id']}: invalid {field}={value!r}"
                )
                continue

            if score not in {0, 1, 2}:
                errors.append(
                    f"{row['answer_id']}: out-of-range {field}={score}"
                )

    if errors:
        raise ValueError(
            "Manual-review validation failed:\n- "
            + "\n- ".join(errors)
        )

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[row["arm"]].append(row)

    expected_arms = {"naive", "rag_bm25", "rag_dense"}

    if set(grouped) != expected_arms:
        raise ValueError(
            f"Expected arms {sorted(expected_arms)}, "
            f"found {sorted(grouped)}."
        )

    summary = []

    for arm in ["naive", "rag_bm25", "rag_dense"]:
        arm_rows = grouped[arm]

        correctness = [
            int(row["manual_correctness_0_to_2"])
            for row in arm_rows
        ]
        groundedness = [
            int(row["manual_groundedness_0_to_2"])
            for row in arm_rows
        ]
        completeness = [
            int(row["manual_completeness_0_to_2"])
            for row in arm_rows
        ]

        summary.append({
            "arm": arm,
            "count": len(arm_rows),
            "mean_correctness_0_to_2": (
                sum(correctness) / len(correctness)
            ),
            "mean_groundedness_0_to_2": (
                sum(groundedness) / len(groundedness)
            ),
            "mean_completeness_0_to_2": (
                sum(completeness) / len(completeness)
            ),
            "fully_correct_count": sum(
                score == 2
                for score in correctness
            ),
            "fully_grounded_count": sum(
                score == 2
                for score in groundedness
            ),
            "fully_complete_count": sum(
                score == 2
                for score in completeness
            ),
        })

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with SUMMARY_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary)

    print("Manual evaluation summary complete.")

    for record in summary:
        print(
            f"  {record['arm']}: "
            f"correctness="
            f"{record['mean_correctness_0_to_2']:.4f}/2, "
            f"groundedness="
            f"{record['mean_groundedness_0_to_2']:.4f}/2, "
            f"completeness="
            f"{record['mean_completeness_0_to_2']:.4f}/2"
        )


if __name__ == "__main__":
    main()
