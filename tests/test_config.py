from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from videograph.config_loader import resolve_evidence_construction
from videograph_eval.pipeline import load_config, save_effective_config
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

    def test_report_records_egc_status(self) -> None:
        results = {
            "_meta": {
                "evidence_construction": {
                    "enabled": False,
                }
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.md"
            generate_report(results, str(report_path), version="test")
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("**Evidence construction**: OFF", report)


if __name__ == "__main__":
    unittest.main()
