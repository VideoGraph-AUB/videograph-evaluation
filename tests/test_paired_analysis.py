from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from videograph_eval.paired_analysis import (
    analyze_paired_predictions,
    exact_two_sided_binomial_p,
)


class PairedAnalysisTests(unittest.TestCase):
    def _write_predictions(self, path: Path, predictions: list[int]) -> None:
        rows = []
        answers = [0] * len(predictions)
        video_ids = ["v1", "v1", "v2", "v3"]
        question_types = ["CH", "TN", "DC", "CH"]
        for index, predicted in enumerate(predictions):
            rows.append(
                {
                    "video_id": video_ids[index],
                    "qid": str(index),
                    "answer": answers[index],
                    "predicted": predicted,
                    "question_type": question_types[index],
                }
            )
        path.write_text(json.dumps(rows), encoding="utf-8")

    def test_exact_two_sided_binomial(self) -> None:
        self.assertAlmostEqual(
            exact_two_sided_binomial_p(0, 4),
            0.125,
        )
        self.assertEqual(exact_two_sided_binomial_p(2, 4), 1.0)

    def test_analysis_aligns_pairs_and_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            on_path = root / "on.json"
            off_path = root / "off.json"
            self._write_predictions(on_path, [0, 0, 0, 1])
            self._write_predictions(off_path, [0, 1, 1, 0])

            first = analyze_paired_predictions(
                on_path,
                off_path,
                bootstrap_samples=500,
                seed=7,
            )
            second = analyze_paired_predictions(
                on_path,
                off_path,
                bootstrap_samples=500,
                seed=7,
            )

        self.assertEqual(first["coverage"], {"questions": 4, "videos": 3})
        self.assertEqual(first["question_pairs"]["both_correct"], 1)
        self.assertEqual(first["question_pairs"]["egc_on_only_correct"], 2)
        self.assertEqual(first["question_pairs"]["egc_off_only_correct"], 1)
        self.assertEqual(
            first["video_cluster_bootstrap"],
            second["video_cluster_bootstrap"],
        )

    def test_analysis_rejects_unaligned_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            on_path = root / "on.json"
            off_path = root / "off.json"
            self._write_predictions(on_path, [0, 0, 0, 1])
            self._write_predictions(off_path, [0, 1, 1, 0])
            rows = json.loads(off_path.read_text(encoding="utf-8"))
            rows.pop()
            off_path.write_text(json.dumps(rows), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not aligned"):
                analyze_paired_predictions(
                    on_path,
                    off_path,
                    bootstrap_samples=10,
                    seed=0,
                )


if __name__ == "__main__":
    unittest.main()
