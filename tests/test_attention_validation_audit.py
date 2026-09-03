import unittest

import torch

import pia_attention_validation_audit as audit


class AttentionValidationAuditTests(unittest.TestCase):
    def test_patch_attention_mass_is_not_normalized_total(self):
        obs = [(18, torch.zeros(1, 4, 4))]
        cand = [(18, torch.zeros(1, 4, 4))]
        obs[0][1][0, 3, 1] = 0.8
        obs[0][1][0, 3, 2] = 0.1
        cand[0][1][0, 3, 1] = 0.2
        cand[0][1][0, 3, 2] = 0.1
        rows = [{"layer": 18, "head": 0, "q": 3}]
        self.assertGreater(audit.patch_attention_mass_delta(cand, obs, rows, [1]), 0.5)

    def test_hash_agreement_fields_can_be_true_for_same_candidate(self):
        row = {"candidate_hash_agreement": True, "before_hash_agreement": True, "selected_rows_same_true_shuffled": True}
        summary = audit.summarize_candidates([{**row, "gt_label": "neutral", "gt_gain": 0, "delta_js_true": 0.0, "delta_js_shuffled": 0.0, "delta_mass": 0.1, "delta_mass_shuffled": 0.1, "delta_laer": 0.1, "delta_laer_shuffled": 0.1, "changed_positions_match_candidate": True, "local_attention_used_for_query_selection": False}])
        self.assertEqual(summary["overall"]["candidate_hash_agreement_rate"], 1.0)
        self.assertEqual(summary["overall"]["selected_rows_same_rate"], 1.0)

    def test_no_misspelled_variable_audit_helper(self):
        import inspect
        self.assertNotIn("variable_audit_payload", inspect.getsource(audit.audit_prompt))
        self.assertNotIn("set_seed(args.seed)", inspect.getsource(audit.run_audit))
        self.assertNotIn("gold_ids", str(inspect.signature(audit.audit_prompt)))

    def test_gate_replay_uses_fixed_threshold(self):
        rows = [
            {"delta_laer": -0.1, "delta_laer_shuffled": 0.1, "gt_label": "positive", "gt_gain": 1},
            {"delta_laer": 0.1, "delta_laer_shuffled": -0.1, "gt_label": "harmful", "gt_gain": -1},
        ]
        true_gate = audit.gate_replay(rows, "LAER_GATE")
        self.assertEqual(true_gate["accepted_candidate_count"], 1)
        self.assertEqual(true_gate["positive_retained"], 1)
        self.assertEqual(true_gate["harmful_rejected"], 1)

    def test_changed_positions_are_real_top1_changes(self):
        variable_mask = torch.tensor([False, True, True, True])
        changed = audit.changed_top1_positions([1, 2, 3, 4], [1, 9, 3, 8], variable_mask, {3: 8})
        self.assertEqual(changed, [1])

    def test_laer_uses_downstream_edges_and_excludes_fixed_public(self):
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

    def test_delta_mass_is_candidate_minus_before_improvement(self):
        obs = [(18, torch.zeros(1, 4, 4))]
        before = [(18, torch.zeros(1, 4, 4))]
        cand = [(18, torch.zeros(1, 4, 4))]
        obs[0][1][0, 3, 1] = 0.8
        before[0][1][0, 3, 1] = 0.2
        cand[0][1][0, 3, 1] = 0.7
        rows = [{"layer": 18, "head": 0, "q": 3}]
        variable_mask = torch.tensor([False, True, True, True])
        scores = audit.local_relation_scores(before, cand, obs, rows, [1], variable_mask, {}, False, 1)
        self.assertLess(scores["delta_mass"], 0.0)
        self.assertAlmostEqual(scores["mass_before"], 0.6, places=6)
        self.assertAlmostEqual(scores["mass_candidate"], 0.1, places=6)

    def test_ground_truth_labeling_happens_after_score_trace_append(self):
        import inspect
        src = inspect.getsource(audit.run_audit)
        self.assertLess(src.index("jsonl_append(score_path"), src.index("gt_gain_for_candidate"))
        self.assertIn("ground_truth_labeling_after_score_trace_written", src)

    def test_state_persistence_is_sparse_candidate_state(self):
        import inspect
        src = inspect.getsource(audit.audit_prompt)
        self.assertIn('z = sparse["z"].detach().clone()', src)

    def test_fixed_causal_query_rows_do_not_use_attention_selection(self):
        attn = [(18, torch.zeros(2, 6, 6))]
        variable_mask = torch.tensor([False, True, True, True, True, True])
        rows, info = audit.fixed_causal_query_rows(attn, variable_mask, {}, [2], type("Cfg", (), {"query_window": 3})())
        self.assertFalse(info["attention_used_for_query_selection"])
        self.assertTrue(rows)
        self.assertTrue(all(row["q"] > 2 for row in rows))


if __name__ == "__main__":
    unittest.main()
