import unittest

import torch

import pia_server_attention_consistency_pia as sacr


class ServerAttentionConsistencyPureTests(unittest.TestCase):
    def test_tail_consistency_is_zero_when_hidden_states_match(self):
        hidden = torch.randn(1, 4, 3)
        variable_mask = torch.tensor([False, True, True, False])

        loss = sacr.tail_consistency_loss(hidden, hidden.clone(), variable_mask)

        self.assertAlmostEqual(float(loss), 0.0, places=8)

    def test_attention_js_is_zero_when_attention_maps_match(self):
        attn = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
                [0.2, 0.3, 0.5, 0.0],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        variable_mask = torch.tensor([False, True, True, True])

        loss, stats = sacr.attention_js_consistency_loss([(18, attn)], [(18, attn.clone())], variable_mask)

        self.assertAlmostEqual(float(loss), 0.0, places=8)
        self.assertGreater(stats["used_rows"], 0)
        self.assertEqual(stats["skipped_rows"], 0)

    def test_attention_js_excludes_fixed_public_query_and_key_positions(self):
        ref = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.8, 0.1, 0.1],
            ]
        )
        pred = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.1, 0.1, 0.8],
            ]
        )
        variable_mask = torch.tensor([False, True, True])

        loss, stats = sacr.attention_js_consistency_loss([(18, ref)], [(18, pred)], variable_mask)

        expected_ref_rows = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
        expected_pred_rows = torch.tensor([[1.0, 0.0], [0.11111111, 0.88888889]])
        expected = sacr.jensen_shannon_divergence(expected_ref_rows, expected_pred_rows).mean()
        self.assertAlmostEqual(float(loss), float(expected), places=6)
        self.assertEqual(stats["used_rows"], 2)

    def test_gradient_cap_limits_auxiliary_contribution(self):
        base_grad = torch.tensor([3.0, 4.0])
        aux_grad = torch.tensor([10.0, 0.0])

        eff = sacr.capped_auxiliary_lambda(base_grad, aux_grad, raw_lambda=1.0, rho=0.1)

        self.assertAlmostEqual(eff, 0.05, places=8)

    def test_zero_lambda_total_loss_matches_b0v(self):
        base = torch.tensor(2.5)
        tail = torch.tensor(3.0)
        attn = torch.tensor(4.0)

        total = sacr.sacr_total_loss(
            base_loss=base,
            vocab_loss=torch.tensor(0.0),
            tail_loss=tail,
            attn_loss=attn,
            tail_initial=torch.tensor(3.0),
            attn_initial=torch.tensor(4.0),
            lambda_vocab=0.0,
            lambda_tail_eff=0.0,
            lambda_attn_eff=0.0,
        )

        self.assertAlmostEqual(float(total), float(base), places=8)

    def test_attack_api_does_not_accept_ground_truth_or_dummy_inputs(self):
        check = sacr.validate_attack_api()

        self.assertTrue(check["passes"], check)
        self.assertEqual(check["banned_signature_hits"], [])
        self.assertEqual(check["banned_global_reference_hits"], [])
        self.assertEqual(check["dummy_or_gradient_rerank_hits"], [])


if __name__ == "__main__":
    unittest.main()
