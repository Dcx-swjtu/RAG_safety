import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

import torch

from verirag.policy_network import VerificationPolicyNetwork
from verirag.text_features import TextFeatureExtractor


class TestTextFeatureExtractor(unittest.TestCase):
    def test_hashed_embeddings_are_deterministic_and_nonrandom(self):
        extractor = TextFeatureExtractor({"embedding_dim": 768})
        first = extractor.encode_texts(["Paris is the capital of France."])
        second = extractor.encode_texts(["Paris is the capital of France."])
        other = extractor.encode_texts(["Jupiter is the largest planet."])

        self.assertEqual(first.shape, (1, 768))
        self.assertTrue((first == second).all())
        self.assertGreater(float(abs(first).sum()), 0.0)
        self.assertFalse((first == other).all())

    def test_policy_inputs_use_document_text_features(self):
        policy = VerificationPolicyNetwork({"state_dim": 512, "action_dim": 5, "max_docs": 5})
        extractor = TextFeatureExtractor({"embedding_dim": 768, "max_docs": 5})
        inputs = extractor.build_policy_inputs(
            "What is the capital of France?",
            [{"doc_id": "d1", "text": "Paris is the capital of France."}],
            policy,
            torch.device("cpu"),
        )

        self.assertIn("query_tokens", inputs)
        self.assertEqual(inputs["doc_embeddings"].shape, (1, 1, 768))
        self.assertGreater(float(inputs["doc_embeddings"].abs().sum()), 0.0)
        self.assertGreater(float(inputs["doc_scores"].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
