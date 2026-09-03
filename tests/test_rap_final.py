import inspect
import unittest

import torch

import pia_rap_final as final


class RAPFinalTests(unittest.TestCase):
    def test_strict_top1_rejects_k_y_semantic_and_calibration(self):
        for kwargs in [
            {"top_k_embedding": 2},
            {"top_y_semantic": 1},
            {"semantic_speculation": True},
            {"adaptive_discretization": True},
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    final.require_strict_top1(self.cfg(**kwargs))

    def test_attack_api_has_no_ground_truth_parameters(self):
        names = set(inspect.signature(final.invert_observed).parameters)
        banned = {"prompt", "original_ids", "original_tokens", "input_ids", "token_ids", "ground_truth", "reference"}
        self.assertTrue(names.isdisjoint(banned))

    def test_contribution_formula_is_attention_times_value_norm(self):
        attention = torch.tensor([[[1.0, 0.0], [0.25, 0.75]]])
        value_norm = torch.tensor([[2.0, 4.0]])
        score = final.contribution_matrix(attention, value_norm)
        self.assertTrue(torch.allclose(score, torch.tensor([[2.0, 0.0], [0.5, 3.0]])))

    def test_layer_score_normalization_uses_legal_variable_causal_edges(self):
        score = torch.tensor([
            [10.0, 0.0, 0.0],
            [2.0, 5.0, 0.0],
            [4.0, 6.0, 7.0],
        ])
        variable = torch.tensor([True, True, True])
        normalized, stats = final.normalize_layer_score(score, variable)
        self.assertAlmostEqual(stats["raw_mean"], 4.0)
        self.assertAlmostEqual(float(normalized[1, 0]), 0.5)
        self.assertAlmostEqual(stats["normalized_mean"], 1.0)

    def test_fixed_size_patches_do_not_use_relational_partition(self):
        variable = torch.tensor([False, True, True, True, False, True, True])
        patches = final.fixed_size_patches(7, variable, 3)
        self.assertEqual(patches, [[1, 2], [3, 5], [6]])

    def test_build_patch_plan_keeps_fixed_boundaries_for_relational_methods(self):
        variable = torch.ones(8, dtype=torch.bool)
        uncertainty = torch.ones(8)
        aggregate = torch.zeros(8, 8)
        aggregate[:, 6] = 100.0
        sparse = final.build_patch_plan("sparse_fixed", 8, variable, aggregate, {1, 6}, uncertainty, 4, 4, 4)
        rel = final.build_patch_plan("rel_order", 8, variable, aggregate, {1, 6}, uncertainty, 4, 4, 4)
        self.assertEqual(rel["patches"], sparse["patches"])
        self.assertFalse(rel["partition_uses_relational_graph"])
        self.assertTrue(rel["ordering_uses_relational_graph"])

    def test_joint_uncertainty_combines_margin_and_residual_rank(self):
        margins = torch.tensor([0.9, 0.1, 0.2, 0.3])
        residual = torch.tensor([10.0, 1.0, 2.0, 3.0])
        variable = torch.tensor([True, True, True, True])
        _scores, chosen, stats = final.build_uncertainty_score(margins, variable, 0.25, residual, detector="joint")
        self.assertEqual(chosen, [2])
        self.assertEqual(stats["uncertainty_detector"], "joint")

    def test_shuffle_relational_score_preserves_legal_edge_count(self):
        score = torch.arange(25, dtype=torch.float32).view(5, 5)
        variable = torch.ones(5, dtype=torch.bool)
        shuffled = final.shuffle_relational_score(score, variable, seed=123)
        legal = final.legal_variable_causal_mask(score, variable)
        self.assertEqual(int(torch.isfinite(shuffled[legal]).sum()), int(legal.sum()))
        self.assertEqual(sorted(shuffled[legal].tolist()), sorted(score[legal].tolist()))

    def test_discrete_change_detector_only_counts_variable_positions(self):
        variable = torch.tensor([False, True, True])
        self.assertFalse(final.variable_token_sequence_changed([1, 2, 3], [9, 2, 3], variable))
        self.assertTrue(final.variable_token_sequence_changed([1, 2, 3], [1, 4, 3], variable))

    def test_no_dummy_attention_and_no_scalar_attention_loss_flags(self):
        audit = final.method_integrity_audit()
        self.assertFalse(audit["uses_dummy_attention"])
        self.assertFalse(audit["uses_scalar_attention_weighted_activation_loss"])
        self.assertTrue(audit["method_integrity_check"]["passes"])

    def test_method_integrity_ast_audit_detects_dummy_reference(self):
        def bad_attack_path():
            dummy_attention = None
            return dummy_attention

        audit = final.validate_method_integrity([bad_attack_path])
        self.assertFalse(audit["passes"])
        self.assertTrue(audit["dummy_reference_hits"])

    def test_last_valid_not_called_eos(self):
        note = final.special_token_semantic_note()
        self.assertIn("last_valid_position", note)
        self.assertNotIn("detected EOS", note)


    def test_accepted_candidate_persists_to_final_return(self):
        cfg = self.cfg(method="sparse_fixed", max_repair_passes=1)
        model = TinyEmbeddingModel()
        variable = torch.tensor([False, True])
        stage1 = torch.zeros(1, 2, 2)
        candidate = torch.tensor([[[0., 0.], [1., 0.]]])
        old = self._patch_state_flow_dependencies(candidate_sequence=[candidate])
        try:
            final_z, stats, events, _priority, _extra = final.run_sparse_repair(
                model, cfg, torch.zeros(1, 2, 2), stage1, variable, {0: 0},
                {18: torch.zeros(2, 2)}, torch.zeros(2, 2), {}, [1], torch.ones(2)
            )
        finally:
            self._restore_state_flow_dependencies(old)
        self.assertTrue(torch.equal(final_z, candidate))
        self.assertTrue(events[0]["accepted"])
        self.assertEqual(stats["returned_state_policy"], "final_current_state_not_best_continuous_state")

    def test_two_accepted_patches_are_stateful(self):
        cfg = self.cfg(method="sparse_fixed", max_repair_passes=1)
        model = TinyEmbeddingModel()
        variable = torch.tensor([False, True, True])
        stage1 = torch.zeros(1, 3, 2)
        x1 = torch.tensor([[[0., 0.], [1., 0.], [0., 0.]]])
        x2 = torch.tensor([[[0., 0.], [1., 0.], [2., 0.]]])
        old = self._patch_state_flow_dependencies(candidate_sequence=[x1, x2], patches=[[1], [2]], uncertain={1, 2})
        try:
            final_z, _stats, events, _priority, _extra = final.run_sparse_repair(
                model, cfg, torch.zeros(1, 3, 2), stage1, variable, {0: 0},
                {18: torch.zeros(3, 3)}, torch.zeros(3, 3), {}, [1, 2], torch.ones(3)
            )
        finally:
            self._restore_state_flow_dependencies(old)
        self.assertEqual(old["starts"][0], [0.0, 0.0, 0.0])
        self.assertEqual(old["starts"][1], [0.0, 1.0, 0.0])
        self.assertTrue(torch.equal(final_z, x2))
        self.assertEqual(len(events), 2)

    def _patch_state_flow_dependencies(self, candidate_sequence, patches=None, uncertain=None):
        patches = patches or [[1]]
        uncertain = uncertain or {1}
        old = {
            "build_patch_plan": final.build_patch_plan,
            "optimize_sparse_candidate": final.optimize_sparse_candidate,
            "global_cut_loss": final.global_cut_loss,
            "accept_candidate": final.accept_candidate,
            "starts": [],
        }
        candidates = list(candidate_sequence)

        def fake_plan(*_args, **_kwargs):
            rows = []
            for idx, patch in enumerate(patches):
                rows.append({"patch_id": idx, "positions": patch, "uncertain_positions": [p for p in patch if p in uncertain], "uncertainty_score": 1.0, "relation_score": 1.0, "priority": 1.0})
            return {"priority_rows": rows, "ordered_rows": rows, "partition_uses_relational_graph": False, "ordering_uses_relational_graph": False}

        def fake_optimize(_model, _cfg, z_current, *_args, **_kwargs):
            old["starts"].append([float(x) for x in z_current[0, :, 0].detach().cpu().tolist()])
            return candidates.pop(0), {"step": 1, "candidate_cut_loss": 0.0, "candidate_vocab_loss": 0.0, "candidate_total_loss": 0.0, "best_candidate_cut_loss": 0.0}

        final.build_patch_plan = fake_plan
        final.optimize_sparse_candidate = fake_optimize
        final.global_cut_loss = lambda *_args, **_kwargs: torch.tensor(0.0)
        final.accept_candidate = lambda *_args, **_kwargs: (True, "accepted")
        return old

    def _restore_state_flow_dependencies(self, old):
        final.build_patch_plan = old["build_patch_plan"]
        final.optimize_sparse_candidate = old["optimize_sparse_candidate"]
        final.global_cut_loss = old["global_cut_loss"]
        final.accept_candidate = old["accept_candidate"]

    def cfg(self, **kwargs):
        data = dict(
            method="rap_final",
            run_name="test",
            output_dir="runs/rap_final/test",
            dataset_name="Skytrax",
            dataset_path="data/airline.json",
            dataset_len=1,
            seed=42,
            participant_number=4,
            attacker_position=4,
            inverted_block_count=18,
            target_layer=17,
            epoch=1,
            lr=0.08,
            lambda_vocab=0.1,
            top_k_embedding=1,
            top_y_semantic=0,
            max_token_len=896,
            grad_clip=1.0,
            uncertainty_fraction=0.2,
            patch_size=16,
            min_patch_size=16,
            max_patch_size=16,
            top_r=4,
            max_repair_passes=2,
            eps_cut=1e-6,
            eps_rel=0.0,
            eps_rel_ratio=None,
            uncertainty_detector="margin",
            adaptive_discretization=False,
            semantic_speculation=False,
            local_files_only=True,
        )
        data.update(kwargs)
        return final.RAPFinalConfig(**data)



class TinyEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(4, 2)

    def get_input_embeddings(self):
        return self.embed


if __name__ == "__main__":
    unittest.main()
