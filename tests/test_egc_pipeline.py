from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videograph_eval.pipeline import _run_pipeline, load_config


class EGCPipelineTests(unittest.TestCase):
    def test_default_pipeline_keeps_all_egc_stages_enabled(self) -> None:
        config = load_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "audio.wav").write_bytes(b"audio")

            with (
                patch(
                    "videograph.video.adaptive_ingest.process_local_video_adaptive"
                ),
                patch("videograph.video.transcribe.transcribe_audio") as transcribe,
                patch(
                    "videograph.visual.adaptive_processing.analyze_adaptive_clips"
                ) as analyze,
                patch(
                    "videograph.visual.adaptive_processing."
                    "update_adaptive_visual_json_with_ocr"
                ) as ocr,
                patch(
                    "videograph.graph.reinforce.reinforce_video_graph"
                ) as reinforce,
                patch(
                    "videograph.visual.adaptive_processing.append_video_summary_node"
                ) as summarize,
                patch("videograph.graph.builder.build_video_graph"),
            ):
                _run_pipeline(
                    video_path="video.mp4",
                    output_dir=output_dir,
                    video_id="video",
                    config=config,
                )

        self.assertTrue(transcribe.call_args.kwargs["filter_hallucinations"])
        self.assertTrue(analyze.call_args.kwargs["use_previous_clip_context"])
        self.assertTrue(ocr.call_args.kwargs["gate_on_readable_text"])
        reinforce.assert_called_once()
        summarize.assert_called_once()

    def test_egc_off_propagates_flags_and_skips_completion_stages(self) -> None:
        preset = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "ablations"
            / "egc_off.yaml"
        )
        config = load_config(preset)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "audio.wav").write_bytes(b"audio")

            with (
                patch(
                    "videograph.video.adaptive_ingest.process_local_video_adaptive"
                ) as ingest,
                patch("videograph.video.transcribe.transcribe_audio") as transcribe,
                patch(
                    "videograph.visual.adaptive_processing.analyze_adaptive_clips"
                ) as analyze,
                patch(
                    "videograph.visual.adaptive_processing."
                    "update_adaptive_visual_json_with_ocr"
                ) as ocr,
                patch(
                    "videograph.graph.reinforce.reinforce_video_graph"
                ) as reinforce,
                patch(
                    "videograph.visual.adaptive_processing.append_video_summary_node"
                ) as summarize,
                patch("videograph.graph.builder.build_video_graph") as build,
            ):
                _run_pipeline(
                    video_path="video.mp4",
                    output_dir=output_dir,
                    video_id="video",
                    config=config,
                )

        ingest.assert_called_once()
        ingest_config = ingest.call_args.kwargs["config"]
        self.assertFalse(
            ingest_config["evidence_construction"]["enabled"]
        )
        self.assertEqual(
            ingest_config["video_ingestion"]["fixed_segmentation"][
                "window_duration_s"
            ],
            10.0,
        )
        self.assertFalse(transcribe.call_args.kwargs["filter_hallucinations"])
        self.assertFalse(analyze.call_args.kwargs["use_previous_clip_context"])
        self.assertFalse(ocr.call_args.kwargs["gate_on_readable_text"])
        reinforce.assert_not_called()
        summarize.assert_not_called()
        build.assert_called_once()


if __name__ == "__main__":
    unittest.main()
