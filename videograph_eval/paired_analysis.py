"""Paired statistical analysis for EGC-ON and EGC-OFF predictions.

This module implements the controlled comparison reported under "Effect of
EGC" in the paper and "Paired analysis and caveats" in the supplement. It
aligns predictions by video and question, applies exact paired tests, and
resamples complete video clusters for the confidence interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from .datasets import get_nextqa_category


PredictionKey = tuple[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_predictions(path: Path) -> Dict[PredictionKey, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Predictions must be a JSON list: {path}")

    indexed: Dict[PredictionKey, dict] = {}
    for position, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Prediction {position} is not an object: {path}")
        missing = {
            key
            for key in ("video_id", "qid", "answer", "predicted")
            if key not in row
        }
        if missing:
            raise ValueError(
                f"Prediction {position} is missing {sorted(missing)}: {path}"
            )
        key = (str(row["video_id"]), str(row["qid"]))
        if key in indexed:
            raise ValueError(f"Duplicate prediction key {key}: {path}")
        indexed[key] = row
    return indexed


def exact_two_sided_binomial_p(successes: int, trials: int) -> float:
    """Return the exact two-sided p-value under Binomial(n, 0.5)."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Expected 0 <= successes <= trials")
    if trials == 0:
        return 1.0
    lower_tail = min(successes, trials - successes)
    tail_probability = sum(
        math.comb(trials, value) for value in range(lower_tail + 1)
    ) / (2**trials)
    return min(1.0, 2.0 * tail_probability)


def _accuracy(correct: Iterable[int]) -> float:
    values = list(correct)
    return sum(values) / len(values) if values else 0.0


def _cluster_bootstrap(
    video_counts: Sequence[tuple[int, int, int]],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap the ON-minus-OFF accuracy difference by complete videos."""
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not video_counts:
        raise ValueError("At least one video is required")

    on_correct = np.asarray([row[0] for row in video_counts], dtype=np.int64)
    off_correct = np.asarray([row[1] for row in video_counts], dtype=np.int64)
    question_counts = np.asarray([row[2] for row in video_counts], dtype=np.int64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)

    batch_size = min(1000, samples)
    video_count = len(video_counts)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(
            0,
            video_count,
            size=(stop - start, video_count),
        )
        numerator = (
            on_correct[indices].sum(axis=1) - off_correct[indices].sum(axis=1)
        )
        denominator = question_counts[indices].sum(axis=1)
        deltas[start:stop] = 100.0 * numerator / denominator

    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def analyze_paired_predictions(
    on_path: Path,
    off_path: Path,
    bootstrap_samples: int = 50_000,
    seed: int = 0,
) -> dict:
    """Analyze two complete, question-aligned prediction files."""
    on_path = Path(on_path).resolve()
    off_path = Path(off_path).resolve()
    on_rows = _load_predictions(on_path)
    off_rows = _load_predictions(off_path)

    on_keys = set(on_rows)
    off_keys = set(off_rows)
    if on_keys != off_keys:
        only_on = sorted(on_keys - off_keys)[:5]
        only_off = sorted(off_keys - on_keys)[:5]
        raise ValueError(
            "Prediction sets are not aligned: "
            f"{len(on_keys - off_keys)} ON-only and "
            f"{len(off_keys - on_keys)} OFF-only keys; "
            f"examples ON-only={only_on}, OFF-only={only_off}"
        )

    ordered_keys = sorted(on_keys)
    records = []
    for key in ordered_keys:
        on_row = on_rows[key]
        off_row = off_rows[key]
        if on_row["answer"] != off_row["answer"]:
            raise ValueError(f"Ground-truth answer mismatch for {key}")
        if on_row.get("question_type") != off_row.get("question_type"):
            raise ValueError(f"Question-type mismatch for {key}")
        records.append(
            {
                "video_id": key[0],
                "qid": key[1],
                "question_type": str(on_row.get("question_type") or ""),
                "on_correct": int(on_row["predicted"] == on_row["answer"]),
                "off_correct": int(off_row["predicted"] == off_row["answer"]),
            }
        )

    both_correct = sum(
        row["on_correct"] and row["off_correct"] for row in records
    )
    both_wrong = sum(
        not row["on_correct"] and not row["off_correct"] for row in records
    )
    on_only = sum(
        row["on_correct"] and not row["off_correct"] for row in records
    )
    off_only = sum(
        row["off_correct"] and not row["on_correct"] for row in records
    )

    by_video: dict[str, list[dict]] = defaultdict(list)
    by_type: dict[str, list[dict]] = defaultdict(list)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_video[row["video_id"]].append(row)
        by_type[row["question_type"]].append(row)
        by_category[get_nextqa_category(row["question_type"])].append(row)

    video_counts = []
    video_on_wins = 0
    video_off_wins = 0
    video_ties = 0
    for video_id in sorted(by_video):
        video_records = by_video[video_id]
        on_correct = sum(row["on_correct"] for row in video_records)
        off_correct = sum(row["off_correct"] for row in video_records)
        video_counts.append((on_correct, off_correct, len(video_records)))
        if on_correct > off_correct:
            video_on_wins += 1
        elif off_correct > on_correct:
            video_off_wins += 1
        else:
            video_ties += 1

    bootstrap_low, bootstrap_high = _cluster_bootstrap(
        video_counts,
        samples=bootstrap_samples,
        seed=seed,
    )

    def breakdown(groups: dict[str, list[dict]]) -> dict:
        output = {}
        for label, rows in sorted(groups.items()):
            on_accuracy = _accuracy(row["on_correct"] for row in rows)
            off_accuracy = _accuracy(row["off_correct"] for row in rows)
            output[label] = {
                "questions": len(rows),
                "on_accuracy_percent": round(100.0 * on_accuracy, 6),
                "off_accuracy_percent": round(100.0 * off_accuracy, 6),
                "delta_points": round(100.0 * (on_accuracy - off_accuracy), 6),
            }
        return output

    on_accuracy = _accuracy(row["on_correct"] for row in records)
    off_accuracy = _accuracy(row["off_correct"] for row in records)
    return {
        "inputs": {
            "egc_on": {
                "path": str(on_path),
                "sha256": _sha256(on_path),
            },
            "egc_off": {
                "path": str(off_path),
                "sha256": _sha256(off_path),
            },
        },
        "settings": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "confidence_level": 0.95,
        },
        "coverage": {
            "questions": len(records),
            "videos": len(by_video),
        },
        "accuracy": {
            "egc_on_percent": round(100.0 * on_accuracy, 6),
            "egc_off_percent": round(100.0 * off_accuracy, 6),
            "delta_points": round(100.0 * (on_accuracy - off_accuracy), 6),
        },
        "question_pairs": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "egc_on_only_correct": on_only,
            "egc_off_only_correct": off_only,
        },
        "mcnemar_exact_two_sided": {
            "discordant_pairs": on_only + off_only,
            "p_value": exact_two_sided_binomial_p(
                min(on_only, off_only),
                on_only + off_only,
            ),
        },
        "video_sign_exact_two_sided": {
            "egc_on_wins": video_on_wins,
            "egc_off_wins": video_off_wins,
            "ties": video_ties,
            "p_value": exact_two_sided_binomial_p(
                min(video_on_wins, video_off_wins),
                video_on_wins + video_off_wins,
            ),
        },
        "video_cluster_bootstrap": {
            "delta_points_ci_95": [
                round(bootstrap_low, 6),
                round(bootstrap_high, 6),
            ],
        },
        "by_category": breakdown(by_category),
        "by_question_type": breakdown(by_type),
    }


def _format_markdown(result: dict) -> str:
    accuracy = result["accuracy"]
    pairs = result["question_pairs"]
    mcnemar = result["mcnemar_exact_two_sided"]
    sign = result["video_sign_exact_two_sided"]
    ci_low, ci_high = result["video_cluster_bootstrap"]["delta_points_ci_95"]
    settings = result["settings"]
    coverage = result["coverage"]
    return "\n".join(
        [
            "# EGC Paired Analysis",
            "",
            f"- Questions: {coverage['questions']}",
            f"- Videos: {coverage['videos']}",
            f"- EGC-ON accuracy: {accuracy['egc_on_percent']:.4f}%",
            f"- EGC-OFF accuracy: {accuracy['egc_off_percent']:.4f}%",
            f"- Difference: {accuracy['delta_points']:.4f} points",
            "",
            "## Paired Tests",
            "",
            f"- Both correct: {pairs['both_correct']}",
            f"- Both wrong: {pairs['both_wrong']}",
            f"- EGC-ON only correct: {pairs['egc_on_only_correct']}",
            f"- EGC-OFF only correct: {pairs['egc_off_only_correct']}",
            f"- Exact two-sided McNemar p-value: {mcnemar['p_value']:.10g}",
            (
                "- Video wins (ON/OFF/tie): "
                f"{sign['egc_on_wins']}/{sign['egc_off_wins']}/{sign['ties']}"
            ),
            f"- Exact two-sided video sign-test p-value: {sign['p_value']:.10g}",
            (
                f"- {settings['bootstrap_samples']:,}-sample video-cluster "
                f"bootstrap 95% CI (seed {settings['bootstrap_seed']}): "
                f"[{ci_low:.4f}, {ci_high:.4f}] points"
            ),
            "",
            "Input paths and SHA-256 hashes are recorded in `paired_analysis.json`.",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run paired EGC-ON versus EGC-OFF statistical analysis."
    )
    parser.add_argument("--on-predictions", type=Path, required=True)
    parser.add_argument("--off-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    result = analyze_paired_predictions(
        args.on_predictions,
        args.off_predictions,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "paired_analysis.md").write_text(
        _format_markdown(result),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
