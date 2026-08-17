import inspect
import unittest
from pathlib import Path

import torch

import pia_air_ptr as air


class StrictTop1Tests(unittest.TestCase):
    def test_strict_top1_accepts_only_k1_y0_semantic_off(self):
        air.validate_strict_top1(k=1, y=0, semantic_speculation=False)
        for values in [(2, 0, False), (1, 1, False), (1, 0, True)]:
            with self.assertRaises(ValueError):
                air.validate_strict_top1(
                    k=values[0],
                    y=values[1],
                    semantic_speculation=values[2],
                )

    def test_attack_api_has_no_ground_truth_inputs(self):
        names = set(inspect.signature(air.invert_observed).parameters)
        banned = {
            "prompt",
            "prompt_text",
            "original_ids",
            "original_tokens",
            "ground_truth_ids",
            "ground_truth_embedding",
            "reference_token",
            "reference_prompt",
        }
        self.assertFalse(names & banned)

    def test_launcher_exports_existing_offline_model_cache(self):
        launcher = Path("scripts/run_air_ptr.sh").read_text(encoding="utf-8")
        self.assertIn("TRANSFORMERS_CACHE", launcher)
        self.assertIn("/home/zhiqi/data/hf_cache/transformers", launcher)


class LarWeightTests(unittest.TestCase):
    def setUp(self):
        self.variable = torch.tensor([False, True, True, True])
        self.importance = torch.tensor([0.0, 0.5, 1.0, 1.5])

    def test_rho_zero_is_exact_b0v_weighting(self):
        weights, audit = air.layer_adaptive_weights(
            normalized_importance=self.importance,
            variable_mask=self.variable,
            step_index=90,
            epoch=100,
            rho=0.0,
            weight_min=0.5,
            weight_max=1.5,
        )
        self.assertTrue(torch.equal(weights, torch.tensor([0.0, 1.0, 1.0, 1.0])))
        self.assertEqual(audit["schedule"], 1.0)
        self.assertEqual(audit["fixed_public_final_weight_sum"], 0.0)

    def test_late_schedule_and_mean_one_invariant(self):
        early, early_audit = air.layer_adaptive_weights(
            normalized_importance=self.importance,
            variable_mask=self.variable,
            step_index=25,
            epoch=100,
            rho=1.0,
            weight_min=0.5,
            weight_max=1.5,
        )
        late, late_audit = air.layer_adaptive_weights(
            normalized_importance=self.importance,
            variable_mask=self.variable,
            step_index=90,
            epoch=100,
            rho=1.0,
            weight_min=0.5,
            weight_max=1.5,
        )
        self.assertTrue(torch.equal(early, torch.tensor([0.0, 1.0, 1.0, 1.0])))
        self.assertEqual(early_audit["schedule"], 0.0)
        self.assertAlmostEqual(float(late[self.variable].mean()), 1.0, places=6)
        self.assertGreater(float(late[self.variable].std(unbiased=False)), 0.0)
        self.assertGreaterEqual(late_audit["min_variable_weight"], 0.5)
        self.assertLessEqual(late_audit["max_variable_weight"], 1.5)

    def test_linear_residual_schedule_applies_residual_alpha_rho(self):
        import pia_masked_server_attn_pia as masked

        source = inspect.getsource(masked.stage_b_optimize)
        residualize_call = source.index("residualize_attention_alpha(")
        branch_start = source.rfind("if cfg.method in {", 0, residualize_call)
        branch_end = source.index("}:", branch_start)
        branch = source[branch_start:branch_end]
        self.assertIn('"attn_linear_weighted_residual_schedule"', branch)


class ProjectionTimeRepairTests(unittest.TestCase):
    @staticmethod
    def objective_to_target(target):
        target_tensor = torch.tensor(target, dtype=torch.float32)

        def objective(embeds):
            residual = ((embeds[0] - target_tensor.to(embeds.device)) ** 2).mean(dim=-1)
            return residual.mean(), residual

        return objective

    def setUp(self):
        self.embedding_weight = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

    def test_eta_zero_keeps_ids(self):
        result = air.projection_time_repair(
            initial_ids=[2, 1],
            embedding_weight=self.embedding_weight,
            variable_mask=torch.tensor([True, True]),
            objective_from_embeddings=self.objective_to_target([[0.0, 1.0], [0.0, 1.0]]),
            eta=0.0,
            repair_fraction=1.0,
            max_positions_per_pass=2,
            max_passes=2,
            epsilon_accept=1e-6,
        )
        self.assertEqual(result.token_ids, [2, 1])
        self.assertEqual(result.stats["accepted_count"], 0)

    def test_zero_passes_is_exact_no_repair(self):
        result = air.projection_time_repair(
            initial_ids=[2, 1],
            embedding_weight=self.embedding_weight,
            variable_mask=torch.tensor([True, True]),
            objective_from_embeddings=self.objective_to_target([[0.0, 1.0], [0.0, 1.0]]),
            eta=1.0,
            repair_fraction=1.0,
            max_positions_per_pass=2,
            max_passes=0,
            epsilon_accept=1e-6,
        )
        self.assertEqual(result.token_ids, [2, 1])
        self.assertEqual(result.stats["proposal_count"], 0)

    def test_accepted_proposal_strictly_decreases_loss(self):
        result = air.projection_time_repair(
            initial_ids=[1],
            embedding_weight=self.embedding_weight,
            variable_mask=torch.tensor([True]),
            objective_from_embeddings=self.objective_to_target([[0.0, 1.0]]),
            eta=1.0,
            repair_fraction=1.0,
            max_positions_per_pass=1,
            max_passes=2,
            epsilon_accept=1e-6,
        )
        self.assertEqual(result.token_ids, [0])
        self.assertEqual(result.stats["accepted_count"], 1)
        event = result.events[0]
        self.assertLess(event["new_activation_loss"], event["old_activation_loss"])

    def test_rejected_proposal_does_not_change_token(self):
        result = air.projection_time_repair(
            initial_ids=[1],
            embedding_weight=self.embedding_weight,
            variable_mask=torch.tensor([True]),
            objective_from_embeddings=self.objective_to_target([[0.9, 0.1]]),
            eta=1.0,
            repair_fraction=1.0,
            max_positions_per_pass=1,
            max_passes=1,
            epsilon_accept=1e-6,
        )
        self.assertEqual(result.token_ids, [1])
        self.assertEqual(result.stats["rejected_count"], 1)

    def test_fixed_position_is_never_proposed(self):
        result = air.projection_time_repair(
            initial_ids=[2, 1],
            embedding_weight=self.embedding_weight,
            variable_mask=torch.tensor([False, True]),
            objective_from_embeddings=self.objective_to_target([[0.0, 1.0], [0.0, 1.0]]),
            eta=1.0,
            repair_fraction=1.0,
            max_positions_per_pass=2,
            max_passes=2,
            epsilon_accept=1e-6,
        )
        self.assertEqual(result.token_ids[0], 2)
        self.assertNotIn(0, result.stats["proposed_positions"])

    def test_each_position_gets_at_most_one_nn_proposal(self):
        result = air.projection_time_repair(
            initial_ids=[2, 2],
            embedding_weight=self.embedding_weight,
            variable_mask=torch.tensor([True, True]),
            objective_from_embeddings=self.objective_to_target([[0.0, 1.0], [0.0, 1.0]]),
            eta=1.0,
            repair_fraction=1.0,
            max_positions_per_pass=2,
            max_passes=2,
            epsilon_accept=1e-6,
        )
        positions = result.stats["proposed_positions"]
        self.assertEqual(len(positions), len(set(positions)))
        self.assertTrue(all(event["candidate_count"] == 1 for event in result.events))


if __name__ == "__main__":
    unittest.main()
