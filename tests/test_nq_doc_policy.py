import unittest

import torch

from verirag.nq_doc_policy import NQDocumentActionPolicy, NQDocRiskEncoder


class NQDocPolicyTest(unittest.TestCase):
    def test_encoder_and_policy_shapes(self):
        encoder = NQDocRiskEncoder(input_dim=24, hidden_dim=32, doc_state_dim=16, dropout=0.0)
        features = torch.randn(2, 5, 24)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
        out = encoder(features, mask)
        self.assertEqual(out["doc_state"].shape, (2, 5, 16))
        self.assertTrue(torch.all(out["attack_prob"][~mask] == 0))

        policy = NQDocumentActionPolicy(input_dim=24, hidden_dim=32, doc_state_dim=16, global_dim=32, dropout=0.0)
        action = policy.select_action({"doc_features": features, "doc_mask": mask}, deterministic=True)
        self.assertEqual(action["keep_mask"].shape, (2, 5))
        self.assertEqual(action["abstain"].shape, (2,))
        self.assertTrue(torch.all(action["keep_mask"][~mask] == 0))

        eval_out = policy.evaluate_actions(
            {"doc_features": features, "doc_mask": mask},
            action["keep_mask"],
            action["abstain"],
        )
        self.assertEqual(eval_out["log_probs"].shape, (2,))
        self.assertEqual(eval_out["values"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
