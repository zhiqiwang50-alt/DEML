import inspect
import unittest

import torch

import pia_b0_laer as laer
import pia_attention_validation_audit as audit


class B0LAERTests(unittest.TestCase):
    def test_strict_top1_config(self):
        args = laer.parse_args([])
        cfg = laer.build_cfg(args, "b0_laer", __import__("pathlib").Path("tmp"))
        self.assertEqual(cfg.top_k_embedding, 1)
        self.assertEqual(cfg.top_y_semantic, 0)
        self.assertFalse(cfg.semantic_speculation)
        self.assertFalse(cfg.adaptive_discretization)

    def test_no_ground_truth_in_attack_api(self):
        sig = str(inspect.signature(laer.invert_observed))
        for banned in ["original", "prompt", "token_ids", "ground", "gold"]:
            self.assertNotIn(banned, sig)
        self.assertNotIn("gt_gain", inspect.getsource(laer.repair_with_laer_gate))

    def test_attention_not_in_candidate_generation_or_selection(self):
        src = inspect.getsource(laer.repair_with_laer_gate)
        self.assertIn("selected = min(c_plus, key=lambda cand: float(cand[\"cut_loss\"]))", src)
        self.assertLess(src.index("selected = min(c_plus"), src.index("local_relation_scores"))
        self.assertIn('"attention_used_for_candidate_generation": False', src)
        self.assertIn('"attention_used_for_candidate_ordering": False', src)
        self.assertIn('"attention_used_for_candidate_selection": False', src)

    def test_changed_set_uses_actual_top1_changes(self):
        variable_mask = torch.tensor([False, True, True, True])
        changed = audit.changed_top1_positions([1, 2, 3, 4], [1, 2, 8, 9], variable_mask, {3: 9})
        self.assertEqual(changed, [2])

    def test_delta_laer_formula_and_gate(self):
        obs = [(18, torch.zeros(1, 4, 4))]
        before = [(18, torch.zeros(1, 4, 4))]
        cand = [(18, torch.zeros(1, 4, 4))]
        obs[0][1][0, 3, 1] = 0.8
        before[0][1][0, 3, 1] = 0.2
        cand[0][1][0, 3, 1] = 0.7
        rows = [{"layer": 18, "head": 0, "q": 3}]
        variable_mask = torch.tensor([False, True, True, True])
        scores = audit.local_relation_scores(before, cand, obs, rows, [1], variable_mask, {}, False, 1)
        self.assertLess(scores["delta_laer"], 0)
        src = inspect.getsource(laer.repair_with_laer_gate)
        self.assertIn('float(relation["delta_laer"]) < 0', src)
        self.assertNotIn("delta_js", src)
        self.assertNotIn("delta_mass", src)

    def test_laer_edges_are_causal_and_no_fixed_keys(self):
        obs = [(18, torch.zeros(1, 5, 5))]
        state = [(18, torch.zeros(1, 5, 5))]
        obs[0][1][0, 3, 1] = 0.8
        state[0][1][0, 3, 1] = 0.2
        rows = [{"layer": 18, "head": 0, "q": 1}, {"layer": 18, "head": 0, "q": 3}]
        variable_mask = torch.tensor([False, True, True, True, True])
        value, info = audit.local_attention_edge_residual(state, obs, rows, [1, 2], variable_mask, {2: 10}, False, 7)
        self.assertAlmostEqual(value, 0.6, places=6)
        self.assertEqual(info["laer_illegal_q_le_k_count"], 0)
        self.assertEqual(info["laer_fixed_public_key_hit_count"], 0)

    def test_state_persistence_and_rollback_are_explicit(self):
        src = inspect.getsource(laer.repair_with_laer_gate)
        self.assertIn('z = selected["z"].detach().clone()', src)
        self.assertIn('"state_hash_after_gate"', src)
        self.assertIn('"next_patch_start_ids_hash"', src)
        self.assertIn('reason = "laer_not_improved"', src)

    def test_shuffled_preserves_row_distribution_by_permutation(self):
        src = inspect.getsource(audit.shuffled_obs_row)
        self.assertIn("torch.randperm", src)
        self.assertIn("shuffled[valid_idx]", src)

    def test_pre_divergence_audit_requires_candidate_matching(self):
        src = inspect.getsource(laer.pre_divergence_audit)
        self.assertIn("candidate_ids_hash", src)
        self.assertIn("pre_divergence_candidate_hash_agreement_rate", src)
        self.assertIn("required_pre_divergence_agreement_rate", src)

    def test_b0_regression_path_delegates_to_same_stage1(self):
        src = inspect.getsource(laer.invert_observed)
        self.assertIn("acdr.optimize_stage1_b0", src)
        self.assertIn('method == "b0"', src)

    def test_resume_skips_existing_predictions_and_archives_failures(self):
        src = inspect.getsource(laer.run_method)
        self.assertIn("completed_prompt_ids", src)
        self.assertIn("skipped_existing_prediction", src)
        self.assertIn("failures.previous_", src)


if __name__ == "__main__":
    unittest.main()
