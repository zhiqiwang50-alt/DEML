import unittest

import torch

from pia_server_attn_pia import (
    residual_attention,
    rollout_from_attentions,
    token_weights_from_rollout,
    weighted_activation_loss,
)


class ServerAttentionPIAUnitTest(unittest.TestCase):
    def test_residual_attention_row_normalizes_and_keeps_causal_mask(self):
        attn = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.1, 0.3, 0.6]])
        out = residual_attention(attn, residual=True)
        self.assertTrue(torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-6))
        self.assertLessEqual(float(torch.triu(out, diagonal=1).abs().max()), 1e-6)
        self.assertGreater(float(out.diag().min()), 0.0)

    def test_rollout_uses_forward_composition_order(self):
        a1 = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
        a2 = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
        got = rollout_from_attentions([a1, a2], residual=False)
        expected = a2 @ a1
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_token_weights_last_and_mean_query_have_mean_one_with_floor(self):
        rollout = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.3, 0.7, 0.0],
                [0.1, 0.2, 0.7],
            ]
        )
        mask = torch.ones(1, 3, dtype=torch.long)
        for source in ["last_query", "mean_query"]:
            weights, stats = token_weights_from_rollout(rollout, mask, source, 0.05)
            self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
            self.assertGreaterEqual(float(weights.min()), 0.05)
            self.assertEqual(stats["source"], source)

    def test_uniform_weighted_loss_equals_unweighted_mean(self):
        pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        target = torch.zeros_like(pred)
        weights = torch.ones(2)
        self.assertAlmostEqual(
            float(weighted_activation_loss(pred, target, weights)),
            float(weighted_activation_loss(pred, target, None)),
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
