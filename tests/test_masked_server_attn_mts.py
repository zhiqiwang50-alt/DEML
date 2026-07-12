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


if __name__ == "__main__":
    unittest.main()
