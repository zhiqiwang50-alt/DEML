import unittest

import torch

import pia_attention_weighted_pia as awp


class AttentionWeightedPIAPureTests(unittest.TestCase):
    def test_schedule_uses_uniform_before_attention_start(self):
        variable = torch.tensor([False, True, True, False])
        attn = torch.tensor([0.0, 0.5, 1.5, 0.0])

        weights = awp.scheduled_weights(attn, variable, step=69, epoch=100, start_ratio=0.7, gate=None)

        self.assertEqual(weights.tolist(), [0.0, 1.0, 1.0, 0.0])

    def test_schedule_uses_attention_after_start(self):
        variable = torch.tensor([False, True, True, False])
        attn = torch.tensor([0.0, 0.5, 1.5, 0.0])

        weights = awp.scheduled_weights(attn, variable, step=70, epoch=100, start_ratio=0.7, gate=None)

        self.assertEqual(weights.tolist(), [0.0, 0.5, 1.5, 0.0])

    def test_gate_keeps_confident_positions_uniform(self):
        variable = torch.tensor([False, True, True, True])
        attn = torch.tensor([0.0, 0.5, 1.5, 1.0])
        gate = torch.tensor([False, True, False, True])

        weights = awp.scheduled_weights(attn, variable, step=70, epoch=100, start_ratio=0.7, gate=gate)

        self.assertEqual(weights.tolist(), [0.0, 0.5, 1.0, 1.0])

    def test_all_valid_loss_weights_leave_fixed_positions_at_one(self):
        valid = torch.tensor([True, True, True])
        variable = torch.tensor([False, True, True])
        variable_weights = torch.tensor([0.0, 0.75, 1.25])

        weights = awp.all_valid_weights_from_variable_weights(variable_weights, valid, variable)

        self.assertEqual(weights.tolist(), [1.0, 0.75, 1.25])
        self.assertAlmostEqual(float(weights[valid].mean()), 1.0, places=7)

    def test_attack_api_has_no_ground_truth_inputs(self):
        check = awp.validate_attack_api()

        self.assertTrue(check["passes"], check)


if __name__ == "__main__":
    unittest.main()
