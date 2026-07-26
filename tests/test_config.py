from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from videograph.config_loader import resolve_evidence_construction
from videograph_eval.pipeline import (
    load_config,
    resolve_retrieval_settings,
    save_effective_config,
)
from videograph_eval.report import generate_report


class EvaluationConfigurationTests(unittest.TestCase):
    def test_egc_off_preset_merges_with_defaults(self) -> None:
        preset = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "ablations"
            / "egc_off.yaml"
        )

        config = load_config(preset)
        resolved = resolve_evidence_construction(config)

        self.assertEqual(config["retrieval"]["top_k"], 7)
        self.assertEqual(
            config["video_ingestion"]["fixed_segmentation"]["window_duration_s"],
            10.0,
        )
        self.assertFalse(resolved["enabled"])
        self.assertTrue(all(not value for value in resolved.values()))

    def test_effective_config_allows_identical_resume(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            first = save_effective_config(config, temp_dir)
            second = save_effective_config(config, temp_dir)

            self.assertEqual(first, second)
            self.assertTrue(first.is_file())

    def test_effective_config_rejects_mixed_run(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            save_effective_config(config, temp_dir)
            changed = load_config()
            changed["evidence_construction"]["enabled"] = False

            with self.assertRaises(RuntimeError):
                save_effective_config(changed, temp_dir)

    def test_retrieval_ablation_presets(self) -> None:
        presets_dir = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "ablations"
            / "retrieval"
        )
        expected = {
            "full.yaml": (1, None, True, None),
            "no_graph_expansion.yaml": (0, None, True, None),
            "flat.yaml": (
                0,
                ["TranscriptNode", "VisualNode"],
                False,
                [],
            ),
            "transcript_only.yaml": (
                1,
                ["TranscriptNode"],
                False,
                ["TEMPORAL_NEXT"],
            ),
            "visual_only.yaml": (
                1,
                ["VisualNode"],
                True,
                ["TEMPORAL_NEXT"],
            ),
            "temporal_only_expansion.yaml": (
                1,
                None,
                True,
                ["TEMPORAL_NEXT"],
            ),
            "alignment_only_expansion.yaml": (
                1,
                None,
                True,
                ["ALIGNED_TO"],
            ),
        }

        for filename, values in expected.items():
            with self.subTest(filename=filename):
                settings = resolve_retrieval_settings(load_config(presets_dir / filename))
                self.assertEqual(
                    (
                        settings["hop_expansion"],
                        settings["allowed_node_types"],
                        settings["use_state_change_channel"],
                        settings["expansion_edge_types"],
                    ),
                    values,
                )

    def test_retrieval_settings_reject_unknown_node_type(self) -> None:
        config = load_config()
        config["retrieval"]["allowed_node_types"] = ["UnknownNode"]

        with self.assertRaisesRegex(ValueError, "Unsupported value"):
            resolve_retrieval_settings(config)

    def test_report_records_egc_status(self) -> None:
        results = {
            "_meta": {
                "evidence_construction": {
                    "enabled": False,
                },
                "ablation": "flat_retrieval",
            },
            "nextqa-val": {
                "accuracy": {"overall": 0.5, "count": 2, "failed": 0},
                "total_predictions": 2,
                "qa_performance": {
                    "sample_videos": 1,
                    "total_api_calls": 4,
                    "total_cost_usd": 0.0123,
                    "total_wall_time_s": 3.5,
                    "avg_answer_time_per_question_s": 1.25,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.md"
            generate_report(results, str(report_path), version="test")
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("**Evidence construction**: OFF", report)
        self.assertIn("**Ablation**: flat_retrieval", report)
        self.assertIn("## Question-Answering Performance", report)
        self.assertIn("| nextqa-val | 2 | 1 | 4 | $0.0123 | 3.5 | 1.250 |", report)


if __name__ == "__main__":
    unittest.main()
