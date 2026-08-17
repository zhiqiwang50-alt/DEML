import unittest
import contextlib
import io
import tempfile
from pathlib import Path

import torch

import pia_masked_server_attn_pia as mts
from pia_masked_server_attn_pia import (
    SAWConfig,
    build_mts_token_weights,
    build_variable_mask,
    naive_discretization,
    stage_b_optimize,
    uniform_all_token_activation_loss,
    weighted_variable_activation_loss,
)


class MTSPureFunctionTests(unittest.TestCase):
    def test_method_registry_excludes_dummy_and_alpha_combos(self):
        legacy_methods = {
            "dummy_init_existing",
            "alpha_nn_init_only",
            "attention_context_existing",
            "alpha_init_plus_server_attn",
            "server_attn_weighted_mean",
        }

        self.assertTrue(legacy_methods.isdisjoint(set(mts.METHODS)))
        self.assertEqual(mts.DUMMY_METHODS, set())
        self.assertEqual(mts.ALPHA_METHODS, set())

    def test_attack_api_does_not_accept_ground_truth_prompt_or_ids(self):
        check = mts.validate_attack_api()

        self.assertTrue(check["passes"], check)
        self.assertEqual(check["banned_signature_hits"], [])
        self.assertEqual(check["banned_global_reference_hits"], [])
        self.assertEqual(check["server_attention_dummy_proxy_hits"], [])

    def test_b0v_variable_positions_have_uniform_weight_one(self):
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
        variable = build_variable_mask(
            attention_mask=attention_mask,
            fixed_public={1: 100},
            special_token_ids={0, 2},
            token_ids=torch.tensor([[0, 100, 2, 55, 99]]),
        )

        self.assertEqual(variable.variable_count, 1)
        self.assertEqual(variable.variable_mask.tolist(), [False, False, False, True, False])
        self.assertEqual(variable.uniform_weights.tolist(), [0.0, 0.0, 0.0, 1.0, 0.0])

    def test_fixed_and_special_final_weights_sum_to_zero(self):
        rollout = torch.eye(5)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        variable = build_variable_mask(
            attention_mask=attention_mask,
            fixed_public={1: 100},
            special_token_ids={0, 2, 4},
            token_ids=torch.tensor([[0, 100, 2, 55, 4]]),
        )

        weights, stats = build_mts_token_weights(
            rollout=rollout,
            attention_mask=attention_mask,
            variable_mask=variable.variable_mask,
            fixed_public={1: 100},
            source="mean_query",
            power=0.5,
            weight_min=0.25,
            weight_max=4.0,
            beta=0.5,
            last_window_size=8,
        )

        excluded = ~variable.variable_mask
        self.assertAlmostEqual(float(weights[excluded].sum()), 0.0, places=8)
        self.assertAlmostEqual(stats["final_fixed_weight_sum"], 0.0, places=8)
        self.assertAlmostEqual(stats["final_bos_weight"], 0.0, places=8)
        self.assertIn("raw_last_valid_mass", stats)
        self.assertIn("final_last_valid_weight", stats)
        self.assertNotIn("raw_eos_mass", stats)
        self.assertNotIn("final_eos_weight", stats)

    def test_variable_mask_audit_distinguishes_public_framing_from_token_id_specials(self):
        attention_mask = torch.tensor([[1, 1, 1, 0]])
        variable = build_variable_mask(
            attention_mask=attention_mask,
            fixed_public={0: 1},
            special_token_ids={0, 1, 2},
            token_ids=None,
        )

        self.assertEqual(variable.known_public_special_positions, [0])
        self.assertFalse(variable.token_id_based_special_mask_available)
        self.assertEqual(variable.last_valid_position, 2)
        self.assertTrue(variable.variable_mask.tolist()[2])

    def test_mts_weights_mean_one_and_max_clipped(self):
        rollout = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.7, 0.3, 0.0, 0.0],
                [0.8, 0.1, 0.1, 0.0],
                [0.7, 0.1, 0.1, 0.1],
            ]
        )
        attention_mask = torch.tensor([[1, 1, 1, 1]])
        variable = torch.tensor([False, True, True, True])

        weights, _stats = build_mts_token_weights(
            rollout=rollout,
            attention_mask=attention_mask,
            variable_mask=variable,
            fixed_public={0: 1},
            source="mean_query",
            power=0.25,
            weight_min=0.25,
            weight_max=1.5,
            beta=1.0,
            last_window_size=8,
        )

        self.assertAlmostEqual(float(weights[variable].mean()), 1.0, places=6)
        self.assertLessEqual(float(weights[variable].max()), 1.5 + 1e-6)

    def test_beta_zero_mts_loss_matches_b0v_loss(self):
        pred = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]])
        target = torch.zeros_like(pred)
        variable = torch.tensor([False, True, True])
        attention_mask = torch.tensor([[1, 1, 1]])
        rollout = torch.eye(3)
        weights, _stats = build_mts_token_weights(
            rollout=rollout,
            attention_mask=attention_mask,
            variable_mask=variable,
            fixed_public={0: 1},
            source="mean_query",
            power=0.5,
            weight_min=0.25,
            weight_max=4.0,
            beta=0.0,
            last_window_size=8,
        )

        b0v = weighted_variable_activation_loss(pred, target, variable, None)
        mts = weighted_variable_activation_loss(pred, target, variable, weights)
        self.assertAlmostEqual(float(mts), float(b0v), places=8)

    def test_last_window_mean_uses_only_variable_query_rows(self):
        rollout = torch.tensor(
            [
                [100.0, 0.0, 0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        attention_mask = torch.ones((1, 5), dtype=torch.long)
        variable = torch.tensor([False, False, True, True, True])

        weights, stats = build_mts_token_weights(
            rollout=rollout,
            attention_mask=attention_mask,
            variable_mask=variable,
            fixed_public={0: 1, 1: 2},
            source="last_window_mean",
            power=1.0,
            weight_min=0.25,
            weight_max=4.0,
            beta=1.0,
            last_window_size=2,
        )

        self.assertEqual(stats["query_positions"], [3, 4])
        self.assertAlmostEqual(float(weights[:2].sum()), 0.0, places=8)

    def test_uniform_all_token_loss_matches_plain_valid_mean(self):
        pred = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [9.0, 0.0]]])
        target = torch.zeros_like(pred)
        attention_mask = torch.tensor([[1, 1, 0]])

        loss = uniform_all_token_activation_loss(pred, target, attention_mask)

        expected = torch.tensor([(1.0**2 + 0.0) / 2.0, (2.0**2 + 0.0) / 2.0]).mean()
        self.assertAlmostEqual(float(loss), float(expected), places=8)

    def test_beta_zero_recovered_ids_match_b0v_with_same_initialization(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4, 2)
                with torch.no_grad():
                    self.embedding.weight.copy_(
                        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
                    )

            def get_input_embeddings(self):
                return self.embedding

        def cfg(method: str, beta: float) -> SAWConfig:
            return SAWConfig(
                method=method,
                run_name="unit",
                output_dir="unit",
                dataset_name="unit",
                dataset_path="unit",
                dataset_len=1,
                seed=42,
                participant_number=4,
                attacker_position=4,
                inverted_block_count=1,
                target_layer=0,
                epoch=4,
                stage_a_epoch=4,
                lr=0.05,
                lambda_vocab=0.0,
                lambda_dummy=0.0,
                lambda_context=0.0,
                top_k_embedding=1,
                top_y_semantic=1,
                gamma=0.0,
                max_token_len=8,
                grad_clip=0.0,
                weight_source="mean_query",
                weight_floor=0.0,
                weight_power=0.5,
                weight_min=0.25,
                weight_max=4.0,
                alpha_min=0.5,
                alpha_max=2.0,
                attention_start_ratio=0.7,
                attention_full_ratio=0.7,
                uncertainty_fraction=0.3,
                adaptive_min_beta=0.05,
                refine_epoch=10,
                lambda_projection=0.1,
                refine_lr_scale=0.25,
                beta=beta,
                last_window_size=8,
                residual_rollout=True,
                server_rollout_depth="all",
                adaptive_discretization=False,
                semantic_speculation=False,
                local_files_only=True,
            )

        original_capture = mts.capture_prefix_activation
        original_nearest = mts.nearest_embedding_loss
        try:
            mts.capture_prefix_activation = lambda model, target_layer, inputs_embeds, attention_mask: inputs_embeds.float()
            mts.nearest_embedding_loss = lambda learnable, embed_weight: torch.tensor(0.0, device=learnable.device)
            model = FakeModel()
            device = torch.device("cpu")
            fixed_public = {0: 0}
            attention_mask = torch.ones((1, 3), dtype=torch.long)
            variable = build_variable_mask(attention_mask, fixed_public=fixed_public, special_token_ids={0}, token_ids=None)
            init = torch.tensor([[[0.0, 0.0], [0.8, 0.2], [0.2, 0.8]]], dtype=torch.float32)
            target = torch.zeros_like(init)
            rollout = torch.eye(3)
            beta_zero_weights, _stats = build_mts_token_weights(
                rollout,
                attention_mask,
                variable.variable_mask,
                fixed_public,
                source="mean_query",
                power=0.5,
                weight_min=0.25,
                weight_max=4.0,
                beta=0.0,
                last_window_size=8,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                z_b0v, _loss_b0v, _hist_b0v = stage_b_optimize(
                    model, None, cfg("variable_only_uniform", 0.0), target, 3, device, fixed_public, variable, init, None
                )
                z_p2, _loss_p2, _hist_p2 = stage_b_optimize(
                    model, None, cfg("mts_mean_query", 0.0), target, 3, device, fixed_public, variable, init, beta_zero_weights
                )

            ids_b0v = naive_discretization(z_b0v, model.embedding.weight)
            ids_p2 = naive_discretization(z_p2, model.embedding.weight)
            self.assertEqual(ids_p2, ids_b0v)
        finally:
            mts.capture_prefix_activation = original_capture
            mts.nearest_embedding_loss = original_nearest


class AttentionScalePureFunctionTests(unittest.TestCase):
    def test_p4_p5_p6_are_registered_without_dummy_or_alpha_init(self):
        self.assertIn("P4", mts.METHODS)
        self.assertIn("P4D", mts.METHODS)
        self.assertIn("P4DL", mts.METHODS)
        self.assertIn("P4LWR", mts.METHODS)
        self.assertIn("P4DR", mts.METHODS)
        self.assertIn("P4C", mts.METHODS)
        self.assertIn("P4DG", mts.METHODS)
        self.assertIn("P4SRES", mts.METHODS)
        self.assertIn("P4SR", mts.METHODS)
        self.assertIn("P5", mts.METHODS)
        self.assertIn("P6", mts.METHODS)
        self.assertEqual(mts.canonical_method("P4"), "attn_scale_mean_query")
        self.assertEqual(mts.canonical_method("P4D"), "attn_weighted_residual_schedule")
        self.assertEqual(mts.canonical_method("P4DL"), "attn_linear_weighted_residual_schedule")
        self.assertEqual(mts.canonical_method("P4LWR"), "attn_last_window_linear_residual_schedule")
        self.assertEqual(mts.canonical_method("P4DR"), "attn_rho_weighted_residual_schedule")
        self.assertEqual(mts.canonical_method("P4DG"), "attn_gate_weighted_residual_schedule")
        self.assertEqual(mts.canonical_method("P4C"), "attn_calibrated_weighted_residual_schedule")
        self.assertEqual(mts.canonical_method("P4SRES"), "attn_scale_mean_query_schedule_residual")
        self.assertEqual(mts.canonical_method("P4SR"), "attn_scale_mean_query_schedule_refine")
        self.assertEqual(mts.canonical_method("P5"), "attn_scale_last_window")
        self.assertEqual(mts.canonical_method("P6"), "attn_scale_last_window_gate")
        self.assertTrue(
            {
                "attn_scale_mean_query",
                "attn_weighted_residual_schedule",
                "attn_linear_weighted_residual_schedule",
                "attn_last_window_linear_residual_schedule",
                "attn_rho_weighted_residual_schedule",
                "attn_gate_weighted_residual_schedule",
                "attn_calibrated_weighted_residual_schedule",
                "attn_scale_mean_query_schedule_residual",
                "attn_scale_mean_query_schedule_refine",
                "attn_scale_last_window",
                "attn_scale_last_window_gate",
            }.issubset(mts.ATTN_SCALE_METHODS)
        )
        self.assertIn("attn_scale_mean_query_schedule_refine", mts.PROJECTION_REFINE_METHODS)
        self.assertNotIn("attn_scale_mean_query_schedule_residual", mts.PROJECTION_REFINE_METHODS)
        self.assertTrue(mts.ATTN_SCALE_METHODS.isdisjoint(mts.DUMMY_METHODS))
        self.assertTrue(mts.ATTN_SCALE_METHODS.isdisjoint(mts.ALPHA_METHODS))

    def test_p4lwr_uses_last_window_attention_source(self):
        self.assertEqual(mts.attention_weight_source_for_method("P4LWR"), "last_window_mean")
        self.assertEqual(
            mts.attention_weight_source_for_method("attn_last_window_linear_residual_schedule"),
            "last_window_mean",
        )
        self.assertEqual(mts.attention_weight_source_for_method("P4DL"), "mean_query")

    def test_calibrated_residual_alpha_gates_and_preserves_mean_one(self):
        variable = torch.tensor([False, True, True, True, True])
        gate = torch.tensor([False, True, False, True, False])
        alpha = torch.tensor([0.0, 0.5, 1.4, 1.5, 0.6])

        calibrated, stats = mts.calibrated_residual_alpha(
            alpha, variable, gate, rho=0.5, alpha_min=0.5, alpha_max=1.5
        )

        self.assertEqual(float(calibrated[0]), 0.0)
        self.assertEqual(float(calibrated[2]), 1.0)
        self.assertEqual(float(calibrated[4]), 1.0)
        self.assertTrue(torch.all(calibrated[variable] >= 0.5))
        self.assertTrue(torch.all(calibrated[variable] <= 1.5))
        self.assertAlmostEqual(float(calibrated[variable].mean()), 1.0, places=7)
        self.assertGreater(float(calibrated[3]), 1.0)
        self.assertLess(float(calibrated[1]), 1.0)
        self.assertAlmostEqual(stats["gate_open_rate"], 0.5, places=7)

    def test_residual_alpha_rho_interpolates_between_uniform_and_p4s_alpha(self):
        variable = torch.tensor([False, True, True, True])
        alpha = torch.tensor([0.0, 0.5, 1.0, 1.5])

        rho_zero = mts.residualize_attention_alpha(alpha, variable, rho=0.0, alpha_min=0.5, alpha_max=1.5)
        rho_half = mts.residualize_attention_alpha(alpha, variable, rho=0.5, alpha_min=0.5, alpha_max=1.5)
        rho_one = mts.residualize_attention_alpha(alpha, variable, rho=1.0, alpha_min=0.5, alpha_max=1.5)

        self.assertEqual(float(rho_zero[0]), 0.0)
        self.assertTrue(torch.allclose(rho_zero[variable], torch.ones(3), atol=1e-7))
        self.assertTrue(torch.allclose(rho_half, torch.tensor([0.0, 0.75, 1.0, 1.25]), atol=1e-7))
        self.assertTrue(torch.allclose(rho_one, alpha, atol=1e-7))
        self.assertAlmostEqual(float(rho_half[variable].mean()), 1.0, places=7)

    def test_alpha_one_scale_loss_matches_b0v_loss(self):
        pred = torch.tensor([[[1.0, 0.0], [2.0, 1.0], [4.0, 0.0]]])
        target = torch.zeros_like(pred)
        variable = torch.tensor([False, True, True])
        alpha = torch.tensor([0.0, 1.0, 1.0])

        b0v = weighted_variable_activation_loss(pred, target, variable, None)
        scaled = mts.attention_scaled_activation_loss(pred, target, variable, alpha)

        self.assertAlmostEqual(float(scaled), float(b0v), places=8)

    def test_scale_loss_multiplies_hidden_not_error(self):
        pred = torch.tensor([[[2.0, 0.0]]])
        target = torch.tensor([[[1.0, 0.0]]])
        variable = torch.tensor([True])
        alpha = torch.tensor([2.0])

        scaled = mts.attention_scaled_activation_loss(pred, target, variable, alpha)
        old_external_weight = weighted_variable_activation_loss(pred, target, variable, alpha)

        self.assertAlmostEqual(float(scaled), 4.5, places=8)
        self.assertAlmostEqual(float(old_external_weight), 0.5, places=8)
        self.assertNotAlmostEqual(float(scaled), float(old_external_weight), places=8)

    def test_attention_weighted_residual_loss_multiplies_error_before_square(self):
        pred = torch.tensor([[[2.0, 0.0]]])
        target = torch.tensor([[[1.0, 0.0]]])
        variable = torch.tensor([True])
        alpha = torch.tensor([2.0])

        weighted_residual = mts.attention_weighted_residual_loss(pred, target, variable, alpha)
        old_p4s = mts.attention_scaled_activation_loss(pred, target, variable, alpha)
        old_external_weight = weighted_variable_activation_loss(pred, target, variable, alpha)

        self.assertAlmostEqual(float(weighted_residual), 2.0, places=8)
        self.assertAlmostEqual(float(old_p4s), 4.5, places=8)
        self.assertAlmostEqual(float(old_external_weight), 0.5, places=8)

    def test_attention_linear_weighted_residual_loss_uses_single_alpha_power(self):
        pred = torch.tensor([[[2.0, 0.0]]])
        target = torch.tensor([[[1.0, 0.0]]])
        variable = torch.tensor([True])
        alpha = torch.tensor([2.0])

        linear_residual = mts.attention_linear_weighted_residual_loss(pred, target, variable, alpha)
        quadratic_residual = mts.attention_weighted_residual_loss(pred, target, variable, alpha)
        old_p4s = mts.attention_scaled_activation_loss(pred, target, variable, alpha)
        old_external_weight = weighted_variable_activation_loss(pred, target, variable, alpha)

        self.assertAlmostEqual(float(linear_residual), 1.0, places=6)
        self.assertAlmostEqual(float(quadratic_residual), 2.0, places=8)
        self.assertAlmostEqual(float(old_p4s), 4.5, places=8)
        self.assertAlmostEqual(float(old_external_weight), 0.5, places=8)

    def test_p4c_uses_same_weighted_residual_formula_as_p4d(self):
        pred = torch.tensor([[[2.0, 0.0]]])
        target = torch.tensor([[[1.0, 0.0]]])
        variable = torch.tensor([True])
        alpha = torch.tensor([1.25])

        p4c_loss = mts.attention_weighted_residual_loss(pred, target, variable, alpha)
        p4d_loss = mts.attention_weighted_residual_loss(pred, target, variable, alpha)
        p4s_loss = mts.attention_scaled_activation_loss(pred, target, variable, alpha)

        self.assertAlmostEqual(float(p4c_loss), float(p4d_loss), places=8)
        self.assertNotAlmostEqual(float(p4c_loss), float(p4s_loss), places=8)

    def test_attention_weighted_residual_alpha_one_matches_b0v_loss(self):
        pred = torch.tensor([[[1.0, 0.0], [2.0, 1.0], [4.0, 0.0]]])
        target = torch.zeros_like(pred)
        variable = torch.tensor([False, True, True])
        alpha = torch.tensor([0.0, 1.0, 1.0])

        b0v = weighted_variable_activation_loss(pred, target, variable, None)
        weighted_residual = mts.attention_weighted_residual_loss(pred, target, variable, alpha)

        self.assertAlmostEqual(float(weighted_residual), float(b0v), places=8)

    def test_attention_weighted_residual_zeroes_when_activation_matches(self):
        pred = torch.tensor([[[3.0, -1.0], [2.0, 5.0]]])
        target = pred.clone()
        variable = torch.tensor([True, True])
        alpha = torch.tensor([0.5, 1.5])

        weighted_residual = mts.attention_weighted_residual_loss(pred, target, variable, alpha)

        self.assertAlmostEqual(float(weighted_residual), 0.0, places=8)

    def test_attention_weighted_residual_excludes_fixed_and_invalid_positions(self):
        pred = torch.tensor([[[100.0, 0.0], [2.0, 0.0], [100.0, 0.0]]])
        target = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])
        variable = torch.tensor([False, True, False])
        alpha = torch.tensor([9.0, 2.0, 9.0])

        loss = mts.attention_weighted_residual_loss(pred, target, variable, alpha)

        self.assertAlmostEqual(float(loss), 2.0, places=8)

    def test_scale_loss_excludes_fixed_and_invalid_positions(self):
        pred = torch.tensor([[[100.0, 0.0], [2.0, 0.0], [100.0, 0.0]]])
        target = torch.zeros_like(pred)
        variable = torch.tensor([False, True, False])
        alpha = torch.tensor([9.0, 1.0, 9.0])

        loss = mts.attention_scaled_activation_loss(pred, target, variable, alpha)

        self.assertAlmostEqual(float(loss), 2.0, places=8)

    def test_build_attention_scale_alpha_mean_one_clipped_and_public_zero(self):
        rollout = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.1, 0.1, 0.0],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.1, 0.7],
            ]
        )
        attention_mask = torch.ones((1, 4), dtype=torch.long)
        variable = torch.tensor([False, True, True, True])

        alpha, stats = mts.build_attention_scale_alpha(
            rollout=rollout,
            attention_mask=attention_mask,
            variable_mask=variable,
            fixed_public={0: 1},
            source="mean_query",
            power=0.5,
            alpha_min=0.5,
            alpha_max=1.5,
            beta=0.5,
            last_window_size=16,
        )

        self.assertAlmostEqual(float(alpha[variable].mean()), 1.0, places=6)
        self.assertGreaterEqual(float(alpha[variable].min()), 0.5 - 1e-6)
        self.assertLessEqual(float(alpha[variable].max()), 1.5 + 1e-6)
        self.assertAlmostEqual(float(alpha[~variable].sum()), 0.0, places=8)
        self.assertEqual(stats["final_bos_alpha"], 0.0)
        self.assertEqual(stats["fixed_public_alpha_sum"], 0.0)
        self.assertIn("raw_last_valid_mass", stats)
        self.assertNotIn("raw_eos_mass", stats)

    def test_gate_false_positions_alpha_one_and_global_mean_one(self):
        variable = torch.tensor([False, True, True, True])
        alpha = torch.tensor([0.0, 0.5, 1.5, 1.2])
        gate = torch.tensor([False, True, False, True])

        gated, stats = mts.apply_attention_scale_gate(alpha, variable, gate, alpha_min=0.5, alpha_max=1.5)

        self.assertEqual(float(gated[2]), 1.0)
        self.assertAlmostEqual(float(gated[variable].mean()), 1.0, places=6)
        self.assertAlmostEqual(stats["gate_open_rate"], 2.0 / 3.0, places=8)
        self.assertEqual(stats["alpha_active_count"], 2)

    def test_schedule_activates_only_after_configured_ratio(self):
        self.assertFalse(mts.attention_scale_schedule_active(step_index=13, epoch=20, start_ratio=0.7))
        self.assertTrue(mts.attention_scale_schedule_active(step_index=14, epoch=20, start_ratio=0.7))

    def test_uncertainty_gate_opens_low_margin_variable_positions_only(self):
        z = torch.tensor([[[0.0, 0.0], [0.51, 0.49], [0.98, 0.02], [0.55, 0.45]]])
        embed_weight = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        variable = torch.tensor([False, True, True, True])

        gate, stats = mts.embedding_uncertainty_gate(z, embed_weight, variable, uncertainty_fraction=1.0 / 3.0)

        self.assertFalse(bool(gate[0]))
        self.assertEqual(int(gate[variable].sum()), 1)
        self.assertIn("uncertainty_fraction", stats)


    def test_linear_schedule_ramps_between_start_and_full(self):
        self.assertAlmostEqual(
            mts.attention_scale_linear_schedule_beta(
                step_index=1,
                epoch=100,
                start_ratio=0.5,
                full_ratio=0.7,
                max_beta=0.25,
            ),
            0.0,
            places=8,
        )
        self.assertAlmostEqual(
            mts.attention_scale_linear_schedule_beta(
                step_index=50,
                epoch=100,
                start_ratio=0.5,
                full_ratio=0.7,
                max_beta=0.25,
            ),
            0.0,
            places=8,
        )
        self.assertAlmostEqual(
            mts.attention_scale_linear_schedule_beta(
                step_index=60,
                epoch=100,
                start_ratio=0.5,
                full_ratio=0.7,
                max_beta=0.25,
            ),
            0.125,
            places=8,
        )
        self.assertAlmostEqual(
            mts.attention_scale_linear_schedule_beta(
                step_index=70,
                epoch=100,
                start_ratio=0.5,
                full_ratio=0.7,
                max_beta=0.25,
            ),
            0.25,
            places=8,
        )

    def test_adaptive_beta_keeps_certain_tokens_near_uniform(self):
        alpha = torch.tensor([0.0, 0.8, 1.2, 1.5])
        variable = torch.tensor([False, True, True, True])
        uncertainty = torch.tensor([0.0, 0.0, 0.5, 1.0])

        out, stats = mts.apply_adaptive_attention_beta(
            alpha,
            variable,
            uncertainty,
            base_beta=0.25,
            min_beta=0.05,
            alpha_min=0.5,
            alpha_max=1.5,
        )

        self.assertEqual(float(out[0]), 0.0)
        self.assertLess(abs(float(out[1]) - 1.0), abs(float(alpha[1]) - 1.0))
        self.assertGreater(abs(float(out[3]) - 1.0), abs(float(out[1]) - 1.0))
        self.assertAlmostEqual(float(out[variable].mean()), 1.0, places=6)
        self.assertGreater(stats["adaptive_beta_mean"], 0.05)
        self.assertLessEqual(stats["adaptive_beta_max"], 0.25)


    def test_projection_refinement_loss_uses_variable_top1_projection(self):
        z = torch.tensor([[[9.0, 9.0], [0.9, 0.1], [0.2, 0.8]]], requires_grad=True)
        embed_weight = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        variable = torch.tensor([False, True, True])

        loss, stats = mts.projection_refinement_loss(z, embed_weight, variable)

        self.assertAlmostEqual(float(loss), 0.025, places=8)
        self.assertEqual(stats["projection_variable_count"], 2)
        self.assertEqual(stats["projection_top1_only"], True)
        loss.backward()
        self.assertAlmostEqual(float(z.grad[0, 0].abs().sum()), 0.0, places=8)
        self.assertGreater(float(z.grad[0, 1].abs().sum()), 0.0)



class MTSHeldoutAnalysisTests(unittest.TestCase):
    def test_ensure_jsonl_file_creates_empty_failure_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.jsonl"

            mts.ensure_jsonl_file(path)

            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_paired_summary_reports_deltas_wins_and_bootstrap_ci(self):
        baseline = [
            {"prompt_id": 0, "token_accuracy": 0.80, "bleu": 0.70},
            {"prompt_id": 1, "token_accuracy": 0.60, "bleu": 0.50},
            {"prompt_id": 2, "token_accuracy": 0.90, "bleu": 0.80},
        ]
        candidate = [
            {"prompt_id": 0, "token_accuracy": 0.90, "bleu": 0.75},
            {"prompt_id": 1, "token_accuracy": 0.60, "bleu": 0.50},
            {"prompt_id": 2, "token_accuracy": 0.70, "bleu": 0.70},
        ]

        summary = mts.compute_paired_delta_summary(
            method="P2",
            baseline_rows=baseline,
            candidate_rows=candidate,
            bootstrap_iters=200,
            bootstrap_seed=7,
        )

        self.assertAlmostEqual(summary["mean_token_accuracy_delta"], (-0.1 / 3.0), places=8)
        self.assertAlmostEqual(summary["mean_bleu_delta"], (-0.05 / 3.0), places=8)
        self.assertEqual(summary["win_count"], 1)
        self.assertEqual(summary["tie_count"], 1)
        self.assertEqual(summary["loss_count"], 1)
        self.assertLessEqual(summary["token_accuracy_ci_low"], summary["mean_token_accuracy_delta"])
        self.assertGreaterEqual(summary["token_accuracy_ci_high"], summary["mean_token_accuracy_delta"])
        self.assertLessEqual(summary["bleu_ci_low"], summary["mean_bleu_delta"])
        self.assertGreaterEqual(summary["bleu_ci_high"], summary["mean_bleu_delta"])

    def test_stop_rule_blocks_audit_violations_even_when_one_candidate_improves(self):
        p2 = {
            "method": "P2",
            "mean_token_accuracy_delta": 0.01,
            "mean_bleu_delta": 0.01,
            "win_count": 2,
            "loss_count": 1,
            "has_nan": False,
            "failed_sample_count": 0,
            "fixed_public_final_weight_sum": 1.0e-5,
            "max_final_weight": 1.5,
            "configured_weight_max": 4.0,
            "attack_api_passes": True,
            "leakage_verification_passes": True,
        }
        p3 = {
            "method": "P3",
            "mean_token_accuracy_delta": 0.02,
            "mean_bleu_delta": 0.02,
            "win_count": 3,
            "loss_count": 0,
            "has_nan": False,
            "failed_sample_count": 0,
            "fixed_public_final_weight_sum": 0.0,
            "max_final_weight": 1.25,
            "configured_weight_max": 2.0,
            "attack_api_passes": True,
            "leakage_verification_passes": True,
        }

        decision = mts.heldout_seed_stop_decision([p2, p3])

        self.assertTrue(decision["stop"])
        self.assertEqual(decision["rule"], "B")
        self.assertIn("fixed public final weight sum", decision["reasons"][0])


class TestCarPtrStrictTop1(unittest.TestCase):
    def test_car_bundle_stats_require_both_rollout_views(self):
        import torch
        import pia_masked_server_attn_pia as pia

        r_all = torch.tensor([0.0, 0.2, 0.5, 0.3], dtype=torch.float32)
        r_last2 = torch.tensor([0.0, 0.3, 0.4, 0.3], dtype=torch.float32)
        variable_mask = torch.tensor([False, True, True, True])
        alpha, confidence, stats = pia.build_car_attention_bundle_from_rollouts(
            r_all=r_all,
            r_last2=r_last2,
            variable_mask=variable_mask,
            weight_min=0.5,
            weight_max=1.5,
            strength=0.25,
            power=0.5,
        )

        self.assertEqual(alpha.shape, r_all.shape)
        self.assertEqual(confidence.shape, r_all.shape)
        self.assertIn("rollout_all_variable_mass", stats)
        self.assertIn("rollout_last2_variable_mass", stats)
        self.assertEqual(stats["fixed_public_final_weight_sum"], 0.0)

    def test_car_methods_use_rollout_bundle_even_with_uniform_source(self):
        from unittest.mock import patch
        import torch
        import pia_masked_server_attn_pia as pia

        seq = 4
        calls = []

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.model = type("ModelState", (), {"layers": [object(), object(), object(), object()]})()

        def fake_server_forward_with_attention(*_args, **kwargs):
            calls.append(kwargs)
            attn_a = torch.tril(torch.ones(seq, seq, dtype=torch.float32))
            attn_a = attn_a / attn_a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            attn_b = torch.tril(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.6, 1.0, 0.0, 0.0],
                        [0.2, 0.6, 1.0, 0.0],
                        [0.1, 0.2, 0.6, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )
            attn_b = attn_b / attn_b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            attn_c = torch.tril(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.2, 1.0, 0.0, 0.0],
                        [0.6, 0.2, 1.0, 0.0],
                        [0.1, 0.6, 0.2, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )
            attn_c = attn_c / attn_c.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return (
                torch.zeros(1, seq, 2),
                torch.zeros(1, seq, 3),
                [(1, attn_a), (2, attn_b), (3, attn_c)],
            )

        cfg = SAWConfig(
            method="confidence_aware_rollout_residual",
            run_name="car_bundle_test",
            output_dir=".",
            dataset_name="Skytrax",
            dataset_path="data/airline.json",
            dataset_len=1,
            seed=42,
            participant_number=4,
            attacker_position=4,
            inverted_block_count=1,
            target_layer=0,
            epoch=5,
            stage_a_epoch=0,
            lr=0.1,
            lambda_vocab=0.1,
            lambda_dummy=0.0,
            lambda_context=0.0,
            top_k_embedding=1,
            top_y_semantic=0,
            gamma=0.0,
            max_token_len=16,
            grad_clip=1.0,
            weight_source="uniform",
            weight_floor=0.05,
            weight_power=0.5,
            weight_min=0.25,
            weight_max=4.0,
            alpha_min=0.5,
            alpha_max=1.5,
            attention_start_ratio=0.5,
            attention_full_ratio=0.7,
            uncertainty_fraction=0.3,
            adaptive_min_beta=0.05,
            refine_epoch=10,
            lambda_projection=0.1,
            refine_lr_scale=0.25,
            beta=0.25,
            last_window_size=2,
            residual_rollout=True,
            server_rollout_depth="last2",
            adaptive_discretization=True,
            semantic_speculation=False,
            local_files_only=True,
        )
        variable_mask = torch.tensor([False, True, True, True])
        audit = pia.VariableMaskAudit(
            valid_mask=torch.ones(seq, dtype=torch.bool),
            fixed_public_mask=torch.tensor([True, False, False, False]),
            special_mask=torch.zeros(seq, dtype=torch.bool),
            variable_mask=variable_mask,
            uniform_weights=variable_mask.float(),
            variable_count=3,
            known_public_special_positions=[0],
            token_id_based_special_mask_available=False,
            last_valid_position=3,
            rows=[],
        )

        with patch.object(pia, "server_forward_with_attention", side_effect=fake_server_forward_with_attention):
            weights, server_stats, rollout_stats, weight_stats = pia.server_attention_bundle(
                FakeModel(),
                torch.zeros(1, seq, 2),
                torch.ones(1, seq, dtype=torch.long),
                cfg,
                {0: 1},
                audit,
            )

        self.assertEqual(calls[0]["rollout_depth"], "all")
        self.assertEqual(tuple(weights.shape), (seq,))
        self.assertEqual(weight_stats["rollout_depth"], "all+last2")
        self.assertEqual(weight_stats["attention_formula"], "confidence_aware_rollout_residual")
        self.assertEqual(server_stats["server_layers"], [1, 2, 3])
        self.assertEqual(rollout_stats["rollout_shape"], [seq, seq])

    def test_new_method_aliases_resolve(self):
        import pia_masked_server_attn_pia as pia

        self.assertEqual(
            pia.METHOD_ALIASES["P4CAR"],
            "confidence_aware_rollout_residual",
        )
        self.assertEqual(
            pia.METHOD_ALIASES["CAR"],
            "confidence_aware_rollout_residual",
        )
        self.assertEqual(
            pia.METHOD_ALIASES["P4DPTR"],
            "attn_weighted_residual_ptr",
        )
        self.assertEqual(
            pia.METHOD_ALIASES["P4CARPTR"],
            "confidence_aware_rollout_residual_ptr",
        )
        self.assertEqual(
            pia.METHOD_ALIASES["CARPTR"],
            "confidence_aware_rollout_residual_ptr",
        )
        self.assertTrue(
            {
                "confidence_aware_rollout_residual",
                "attn_weighted_residual_ptr",
                "confidence_aware_rollout_residual_ptr",
            }.issubset(pia.SERVER_ATTENTION_METHODS)
        )
        self.assertTrue(
            {
                "attn_weighted_residual_ptr",
                "confidence_aware_rollout_residual_ptr",
            }.issubset(pia.PTR_METHODS)
        )
        self.assertTrue(
            {
                "confidence_aware_rollout_residual",
                "attn_weighted_residual_ptr",
                "confidence_aware_rollout_residual_ptr",
            }.issubset(pia.ATTENTION_WEIGHTED_RESIDUAL_METHODS)
        )

    def test_car_schedule_uses_uniform_then_ramp_then_full_weight(self):
        import torch
        import pia_masked_server_attn_pia as pia

        alpha = torch.tensor([0.0, 0.8, 1.2], dtype=torch.float32)
        variable_mask = torch.tensor([False, True, True])

        warmup = pia.schedule_residual_alpha(
            alpha=alpha,
            variable_mask=variable_mask,
            epoch_index=50,
            warmup_epochs=50,
            ramp_end_epoch=70,
            weight_min=0.5,
            weight_max=1.5,
        )
        mid = pia.schedule_residual_alpha(
            alpha=alpha,
            variable_mask=variable_mask,
            epoch_index=60,
            warmup_epochs=50,
            ramp_end_epoch=70,
            weight_min=0.5,
            weight_max=1.5,
        )
        full = pia.schedule_residual_alpha(
            alpha=alpha,
            variable_mask=variable_mask,
            epoch_index=70,
            warmup_epochs=50,
            ramp_end_epoch=70,
            weight_min=0.5,
            weight_max=1.5,
        )

        self.assertEqual(float(warmup[0]), 0.0)
        self.assertTrue(torch.allclose(warmup[variable_mask].pow(2), torch.ones(2), atol=1e-6))
        self.assertGreater(float(mid[2].pow(2)), 1.0)
        self.assertLess(float(mid[2].pow(2)), float(full[2].pow(2)))
        self.assertAlmostEqual(float(full[variable_mask].pow(2).mean()), 1.0, places=6)

    def test_car_schedule_no_variable_positions_returns_all_zeros(self):
        import torch
        import pia_masked_server_attn_pia as pia

        alpha = torch.tensor([0.7, 1.2, 1.8], dtype=torch.float32)
        variable_mask = torch.tensor([False, False, False])

        scheduled = pia.schedule_residual_alpha(
            alpha=alpha,
            variable_mask=variable_mask,
            epoch_index=80,
            warmup_epochs=50,
            ramp_end_epoch=70,
        )

        self.assertTrue(torch.equal(scheduled, torch.zeros_like(alpha)))

    def test_car_schedule_invalid_bounds_raise_when_variable_positions_exist(self):
        import torch
        import pia_masked_server_attn_pia as pia

        alpha = torch.tensor([0.0, 0.8, 1.2], dtype=torch.float32)
        variable_mask = torch.tensor([False, True, True])

        with self.assertRaises(ValueError):
            pia.schedule_residual_alpha(
                alpha=alpha,
                variable_mask=variable_mask,
                epoch_index=70,
                warmup_epochs=50,
                ramp_end_epoch=70,
                weight_min=1.1,
                weight_max=1.5,
            )
        with self.assertRaises(ValueError):
            pia.schedule_residual_alpha(
                alpha=alpha,
                variable_mask=variable_mask,
                epoch_index=70,
                warmup_epochs=50,
                ramp_end_epoch=70,
                weight_min=0.5,
                weight_max=0.9,
            )

    def test_car_schedule_degenerate_ramp_jumps_to_full_after_warmup(self):
        import torch
        import pia_masked_server_attn_pia as pia

        alpha = torch.tensor([0.0, 0.8, 1.2], dtype=torch.float32)
        variable_mask = torch.tensor([False, True, True])

        warmup = pia.schedule_residual_alpha(
            alpha=alpha,
            variable_mask=variable_mask,
            epoch_index=50,
            warmup_epochs=50,
            ramp_end_epoch=50,
            weight_min=0.5,
            weight_max=1.5,
        )
        after_warmup = pia.schedule_residual_alpha(
            alpha=alpha,
            variable_mask=variable_mask,
            epoch_index=51,
            warmup_epochs=50,
            ramp_end_epoch=50,
            weight_min=0.5,
            weight_max=1.5,
        )
        expected_full = pia.schedule_residual_alpha(
            alpha=alpha,
            variable_mask=variable_mask,
            epoch_index=70,
            warmup_epochs=50,
            ramp_end_epoch=70,
            weight_min=0.5,
            weight_max=1.5,
        )

        self.assertTrue(torch.allclose(warmup[variable_mask].pow(2), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(after_warmup, expected_full, atol=1e-6))

    def test_car_schedule_detaches_alpha_grad_input(self):
        import torch
        import pia_masked_server_attn_pia as pia

        alpha = torch.tensor([0.0, 0.8, 1.2], dtype=torch.float32, requires_grad=True)
        variable_mask = torch.tensor([False, True, True])

        scheduled = pia.schedule_residual_alpha(
            alpha=alpha,
            variable_mask=variable_mask,
            epoch_index=70,
            warmup_epochs=50,
            ramp_end_epoch=70,
        )

        self.assertFalse(scheduled.requires_grad)

    def test_car_residual_loss_matches_manual_alpha_weighted_residual_formula(self):
        import torch
        import pia_masked_server_attn_pia as pia

        hidden = torch.tensor([[[2.0, 0.0], [3.0, 1.0], [4.0, 2.0]]], dtype=torch.float32)
        target = torch.tensor([[[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
        variable_mask = torch.tensor([False, True, True])
        alpha = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float32)

        residual_loss = pia.attention_weighted_residual_loss(hidden, target, variable_mask, alpha)
        scaled_activation_loss = pia.attention_scaled_activation_loss(hidden, target, variable_mask, alpha)
        manual = (((hidden - target) * alpha.view(1, -1, 1)) ** 2).mean(dim=-1)[:, variable_mask].mean()

        self.assertAlmostEqual(float(residual_loss), float(manual), places=8)
        self.assertNotAlmostEqual(float(residual_loss), float(scaled_activation_loss), places=8)

    def test_strict_top1_config_accepts_only_k1_y0_no_semantic(self):
        import argparse
        import pia_masked_server_attn_pia as pia

        ok = argparse.Namespace(
            method="confidence_aware_rollout_residual_ptr",
            k=1,
            y=0,
            semantic_speculation=False,
        )
        pia.validate_strict_top1_config(ok)

        bad_k = argparse.Namespace(
            method="confidence_aware_rollout_residual_ptr",
            k=10,
            y=0,
            semantic_speculation=False,
        )
        with self.assertRaises(ValueError):
            pia.validate_strict_top1_config(bad_k)

        bad_y = argparse.Namespace(
            method="confidence_aware_rollout_residual_ptr",
            k=1,
            y=1,
            semantic_speculation=False,
        )
        with self.assertRaises(ValueError):
            pia.validate_strict_top1_config(bad_y)

        bad_semantic = argparse.Namespace(
            method="confidence_aware_rollout_residual_ptr",
            k=1,
            y=0,
            semantic_speculation=True,
        )
        with self.assertRaises(ValueError):
            pia.validate_strict_top1_config(bad_semantic)

    def test_strict_top1_methods_checks_multi_layer_methods(self):
        import argparse
        import pia_masked_server_attn_pia as pia

        bad_methods = argparse.Namespace(
            method="B0",
            methods=["P4CAR"],
            k=1,
            y=10,
            semantic_speculation=False,
        )
        with self.assertRaises(ValueError):
            pia.validate_strict_top1_methods(bad_methods)

        ok_methods = argparse.Namespace(
            method="B0",
            methods=["P4CAR"],
            k=1,
            y=0,
            semantic_speculation=False,
        )
        pia.validate_strict_top1_methods(ok_methods)

    def test_strict_top1_for_mode_ignores_unused_summarize_methods(self):
        import argparse
        import pia_masked_server_attn_pia as pia

        summarize_args = argparse.Namespace(
            mode="summarize",
            method="B0",
            methods=["P4CAR"],
            k=1,
            y=10,
            semantic_speculation=True,
        )
        pia.validate_strict_top1_for_mode(summarize_args)

        unused_single_method_args = argparse.Namespace(
            mode="multi-layer",
            method="P4CAR",
            methods=["B0"],
            k=1,
            y=10,
            semantic_speculation=True,
        )
        pia.validate_strict_top1_for_mode(unused_single_method_args)

        multi_layer_args = argparse.Namespace(
            mode="multi-layer",
            method="B0",
            methods=["P4CAR"],
            k=1,
            y=10,
            semantic_speculation=False,
        )
        with self.assertRaises(ValueError):
            pia.validate_strict_top1_for_mode(multi_layer_args)


    def test_car_attention_identical_rollouts_has_confidence_one(self):
        import torch
        import pia_masked_server_attn_pia as pia

        r_all = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
        r_last2 = r_all.clone()
        variable_mask = torch.tensor([False, True, True, True])
        alpha, confidence, stats = pia.build_confidence_aware_residual_alpha(
            r_all=r_all,
            r_last2=r_last2,
            variable_mask=variable_mask,
            weight_min=0.5,
            weight_max=1.5,
            strength=0.25,
            power=0.5,
            eps=1e-8,
        )

        self.assertEqual(float(alpha[0]), 0.0)
        self.assertTrue(torch.allclose(confidence[variable_mask], torch.ones(3), atol=1e-6))
        w = alpha[variable_mask].pow(2)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=6)
        self.assertGreaterEqual(float(w.min()), 0.5 - 1e-6)
        self.assertLessEqual(float(w.max()), 1.5 + 1e-6)
        self.assertEqual(stats["fixed_public_final_weight_sum"], 0.0)

    def test_car_attention_disagreement_moves_weights_toward_one(self):
        import torch
        import pia_masked_server_attn_pia as pia

        variable_mask = torch.tensor([False, True, True, True])
        r_all = torch.tensor([0.0, 0.8, 0.1, 0.1], dtype=torch.float32)
        r_last2 = torch.tensor([0.0, 0.1, 0.8, 0.1], dtype=torch.float32)
        alpha, confidence, _ = pia.build_confidence_aware_residual_alpha(
            r_all=r_all,
            r_last2=r_last2,
            variable_mask=variable_mask,
            weight_min=0.5,
            weight_max=1.5,
            strength=0.25,
            power=0.5,
            eps=1e-8,
        )

        self.assertLess(float(confidence[1]), 1.0)
        self.assertLess(float(abs(alpha[1].pow(2) - 1.0)), 0.5)
        self.assertAlmostEqual(float(alpha[variable_mask].pow(2).mean()), 1.0, places=6)

    def test_car_attention_no_variable_positions_returns_zero_stats(self):
        import torch
        import pia_masked_server_attn_pia as pia

        variable_mask = torch.zeros(3, dtype=torch.bool)
        alpha, confidence, stats = pia.build_confidence_aware_residual_alpha(
            r_all=torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32),
            r_last2=torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32),
            variable_mask=variable_mask,
        )

        self.assertTrue(torch.equal(alpha, torch.zeros(3, dtype=torch.float32)))
        self.assertTrue(torch.equal(confidence, torch.zeros(3, dtype=torch.float32)))
        expected_keys = {
            'variable_final_weight_mean',
            'variable_final_weight_std',
            'fixed_public_final_weight_sum',
            'final_max_weight',
            'final_min_weight',
            'confidence_mean',
            'confidence_min',
            'confidence_max',
            'rollout_disagreement_mean',
            'rollout_disagreement_median',
        }
        self.assertTrue(expected_keys.issubset(stats.keys()))
        for key in expected_keys:
            self.assertEqual(stats[key], 0.0)


    def test_car_attention_extreme_weights_remain_bounded_mean_one(self):
        import torch
        import pia_masked_server_attn_pia as pia

        r_all = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        r_last2 = r_all.clone()
        variable_mask = torch.tensor([True, True, True])
        alpha, _confidence, _stats = pia.build_confidence_aware_residual_alpha(
            r_all=r_all,
            r_last2=r_last2,
            variable_mask=variable_mask,
            weight_min=0.5,
            weight_max=1.5,
            strength=1.0,
            power=1.0,
            eps=1e-8,
        )

        w = alpha[variable_mask].pow(2)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=6)
        self.assertGreaterEqual(float(w.min()), 0.5 - 1e-6)
        self.assertLessEqual(float(w.max()), 1.5 + 1e-6)


    def test_car_attention_stats_include_compatibility_aliases(self):
        import torch
        import pia_masked_server_attn_pia as pia

        variable_mask = torch.tensor([False, True, True, True])
        alpha, _confidence, stats = pia.build_confidence_aware_residual_alpha(
            r_all=torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32),
            r_last2=torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32),
            variable_mask=variable_mask,
        )
        weights = alpha[variable_mask].pow(2)

        self.assertEqual(stats['final_fixed_weight_sum'], stats['fixed_public_final_weight_sum'])
        self.assertEqual(stats['max'], stats['final_max_weight'])
        self.assertEqual(stats['min'], stats['final_min_weight'])
        self.assertAlmostEqual(stats['max'], float(weights.max()), places=6)
        self.assertAlmostEqual(stats['min'], float(weights.min()), places=6)

    def test_car_attention_no_variable_stats_include_compatibility_aliases(self):
        import torch
        import pia_masked_server_attn_pia as pia

        _alpha, _confidence, stats = pia.build_confidence_aware_residual_alpha(
            r_all=torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32),
            r_last2=torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32),
            variable_mask=torch.zeros(3, dtype=torch.bool),
        )

        self.assertEqual(stats['final_fixed_weight_sum'], stats['fixed_public_final_weight_sum'])
        self.assertEqual(stats['max'], stats['final_max_weight'])
        self.assertEqual(stats['min'], stats['final_min_weight'])
        self.assertEqual(stats['final_fixed_weight_sum'], 0.0)
        self.assertEqual(stats['max'], 0.0)
        self.assertEqual(stats['min'], 0.0)

    def test_bounded_mean_one_project_empty_input_returns_empty(self):
        import torch
        import pia_masked_server_attn_pia as pia

        projected = pia.bounded_mean_one_project(torch.empty(0, dtype=torch.float32), 0.5, 1.5)

        self.assertEqual(projected.numel(), 0)
        self.assertEqual(projected.dtype, torch.float32)

    def test_bounded_mean_one_project_invalid_bounds_raise(self):
        import torch
        import pia_masked_server_attn_pia as pia

        with self.assertRaises(ValueError):
            pia.bounded_mean_one_project(torch.ones(3), 1.1, 1.5)
        with self.assertRaises(ValueError):
            pia.bounded_mean_one_project(torch.ones(3), 0.5, 0.9)

    def test_bounded_mean_one_project_extreme_vector_is_bounded_mean_one(self):
        import torch
        import pia_masked_server_attn_pia as pia

        projected = pia.bounded_mean_one_project(
            torch.tensor([3.0, 1e-8, 1e-8], dtype=torch.float32),
            0.5,
            1.5,
        )

        self.assertAlmostEqual(float(projected.mean()), 1.0, places=6)
        self.assertGreaterEqual(float(projected.min()), 0.5 - 1e-6)
        self.assertLessEqual(float(projected.max()), 1.5 + 1e-6)

    def test_bounded_mean_one_project_detaches_grad_input(self):
        import torch
        import pia_masked_server_attn_pia as pia

        weights = torch.tensor([3.0, 1.0, 1.0], dtype=torch.float32, requires_grad=True)
        projected = pia.bounded_mean_one_project(weights, 0.5, 1.5)

        self.assertFalse(projected.requires_grad)

if __name__ == "__main__":
    unittest.main()
