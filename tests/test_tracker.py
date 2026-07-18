from __future__ import annotations

import unittest

from videograph_eval.tracker import _estimate_cost


class TrackerPricingTests(unittest.TestCase):
    def test_openrouter_chat_and_embedding_slugs_keep_existing_prices(self) -> None:
        self.assertEqual(
            _estimate_cost("openai/gpt-4o-2024-08-06", 1_000_000, 1_000_000),
            12.5,
        )
        self.assertEqual(
            _estimate_cost("openai/text-embedding-3-small", 1_000_000, 0),
            0.02,
        )

    def test_openrouter_whisper_price_is_duration_based(self) -> None:
        self.assertEqual(
            _estimate_cost(
                "openai/whisper-large-v3", 0, 0, audio_duration_s=60.0
            ),
            0.0015,
        )


if __name__ == "__main__":
    unittest.main()
