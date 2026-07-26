from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videograph_eval.datasets import Question
from videograph_eval.pipeline import answer_questions, evaluate_dataset, load_config


class RetrievalAblationTests(unittest.TestCase):
    def test_external_graph_root_requires_skip_processing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "requires skip_processing"):
                evaluate_dataset(
                    dataset_name="nextqa-val",
                    questions=[],
                    videos_dir=temp_dir,
                    output_dir=temp_dir,
                    config=load_config(),
                    graphs_root=temp_dir,
                )

    def test_flat_preset_reaches_answerer_and_preserves_source_graph(self) -> None:
        preset = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "ablations"
            / "retrieval"
            / "flat.yaml"
        )
        question = Question(
            qid="q1",
            video_id="video-1",
            question="What happened?",
            options=["A", "B"],
            answer=0,
            question_type="DC",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "video-1" / "graph.json"
            graph_path.parent.mkdir(parents=True)
            graph_path.write_text("{}", encoding="utf-8")

            mocked_result = {
                "predicted": 0,
                "raw_response": "0",
                "answer_time_s": 0.1,
            }
            with patch(
                "videograph_eval.mc_answer.answer_mc",
                return_value=mocked_result,
            ) as mock_answer:
                predictions = answer_questions(
                    [question],
                    temp_dir,
                    load_config(preset),
                    dataset_name="nextqa-val",
                    read_only_graphs=True,
                )

        self.assertEqual(predictions[0]["predicted"], 0)
        kwargs = mock_answer.call_args.kwargs
        self.assertEqual(kwargs["hop_expansion"], 0)
        self.assertEqual(
            kwargs["allowed_node_types"],
            ["TranscriptNode", "VisualNode"],
        )
        self.assertFalse(kwargs["use_state_change_channel"])
        self.assertEqual(kwargs["expansion_edge_types"], [])
        self.assertFalse(kwargs["persist_visual_channel_embeddings"])


if __name__ == "__main__":
    unittest.main()
