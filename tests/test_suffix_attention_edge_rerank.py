
import inspect
import unittest

import torch

import pia_suffix_attention_edge_rerank as saer


class SuffixAttentionEdgeRerankPureTests(unittest.TestCase):
    def _attn(self):
        h0 = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.6, 0.4, 0.0],
            [0.0, 0.2, 0.5, 0.3],
        ])
        h1 = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.0, 0.55, 0.45, 0.0],
            [0.0, 0.25, 0.45, 0.30],
        ])
        return [(18, torch.stack([h0, h1], dim=0))]

    def test_edge_graph_excludes_fixed_and_invalid_positions(self):
        variable_mask = torch.tensor([False, True, True, False])
        graph = saer.build_attention_edge_graph(
            self._attn(),
            variable_mask,
            top_r=2,
            sink_threshold=0.95,
            entropy_min=0.0,
            min_head_support=2,
            min_layer_support=2,
        )

        self.assertGreater(graph.edge_count, 0)
        self.assertEqual(graph.bos_or_fixed_edge_count, 0)
        for edge in graph.edges:
            self.assertNotIn(edge.query, {0, 3})
            self.assertNotIn(edge.key, {0, 3})
            self.assertLessEqual(edge.key, edge.query)

    def test_empty_graph_degrades_e2_e3_to_e0(self):
        candidates = [
            saer.CandidateScore(index=0, ids=[1, 10, 11], calibration_score=0.0, tail_loss=2.0, edge_loss=5.0),
            saer.CandidateScore(index=1, ids=[1, 20, 11], calibration_score=0.01, tail_loss=0.0, edge_loss=0.0),
        ]
        cfg = saer.RerankConfig(method='E3', lambda_tail_rank=1.0, lambda_edge_rank=1.0, margin_threshold=1.0)

        result = saer.gated_rerank_candidates(candidates, cfg, graph_is_empty=True)

        self.assertEqual(result.selected_index, 0)
        self.assertFalse(result.override)
        self.assertIn('empty_edge_graph', result.reason)

    def test_disabling_edge_lambda_makes_e2_equal_e0_and_e3_equal_e1(self):
        candidates = [
            saer.CandidateScore(index=0, ids=[1, 10, 11], calibration_score=0.0, tail_loss=2.0, edge_loss=5.0),
            saer.CandidateScore(index=1, ids=[1, 20, 11], calibration_score=0.005, tail_loss=0.0, edge_loss=0.0),
        ]
        e0 = saer.gated_rerank_candidates(candidates, saer.RerankConfig(method='E0', margin_threshold=1.0), graph_is_empty=False)
        e1 = saer.gated_rerank_candidates(candidates, saer.RerankConfig(method='E1', lambda_tail_rank=1.0, margin_threshold=1.0), graph_is_empty=False)
        e2 = saer.gated_rerank_candidates(candidates, saer.RerankConfig(method='E2', lambda_edge_rank=0.0, margin_threshold=1.0), graph_is_empty=False)
        e3 = saer.gated_rerank_candidates(candidates, saer.RerankConfig(method='E3', lambda_tail_rank=1.0, lambda_edge_rank=0.0, margin_threshold=1.0), graph_is_empty=False)

        self.assertEqual(e2.selected_index, e0.selected_index)
        self.assertEqual(e3.selected_index, e1.selected_index)

    def test_edge_loss_zero_for_matching_attention_and_changes_when_perturbed(self):
        variable_mask = torch.tensor([False, True, True, True])
        graph = saer.build_attention_edge_graph(
            self._attn(), variable_mask, top_r=2, sink_threshold=0.95, entropy_min=0.0, min_head_support=2, min_layer_support=2
        )
        ref = self._attn()
        pred = [(layer, attn.clone().requires_grad_(True)) for layer, attn in ref]

        loss, stats = saer.edge_consistency_loss(ref, pred, graph, variable_mask)
        loss.backward()

        self.assertLess(float(loss), 1e-8)
        self.assertGreater(stats['used_edges'], 0)
        self.assertIsNotNone(pred[0][1].grad)

        perturbed = [(layer, attn.clone()) for layer, attn in ref]
        perturbed[0][1][0, 2, 1] += 0.05
        changed, _ = saer.edge_consistency_loss(ref, perturbed, graph, variable_mask)
        self.assertGreater(float(changed), float(loss))

    def test_random_wrong_attention_has_larger_edge_loss_than_reference(self):
        variable_mask = torch.tensor([False, True, True, True])
        ref = self._attn()
        graph = saer.build_attention_edge_graph(ref, variable_mask, top_r=2, sink_threshold=0.95, entropy_min=0.0, min_head_support=2, min_layer_support=2)
        same, _ = saer.edge_consistency_loss(ref, ref, graph, variable_mask)
        wrong_tensor = torch.rand_like(ref[0][1]).tril()
        wrong_tensor = wrong_tensor / wrong_tensor.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        diff, _ = saer.edge_consistency_loss(ref, [(18, wrong_tensor)], graph, variable_mask)

        self.assertGreater(float(diff), float(same))

    def test_rerank_never_introduces_new_candidate_sequence(self):
        candidates = [
            saer.CandidateScore(index=0, ids=[1, 10, 11], calibration_score=0.0, tail_loss=3.0, edge_loss=3.0),
            saer.CandidateScore(index=1, ids=[1, 20, 11], calibration_score=0.001, tail_loss=0.0, edge_loss=0.0),
        ]
        result = saer.gated_rerank_candidates(
            candidates,
            saer.RerankConfig(method='E3', lambda_tail_rank=0.2, lambda_edge_rank=0.2, margin_threshold=1.0),
            graph_is_empty=False,
        )

        self.assertIn(result.selected_ids, [c.ids for c in candidates])

    def test_attack_api_signature_has_no_ground_truth_or_dummy_inputs(self):
        sig = inspect.signature(saer.invert_observed)
        banned = ['prompt', 'original', 'input_ids', 'token_ids', 'ground', 'dummy']
        hits = [name for name in sig.parameters if any(piece in name for piece in banned)]

        self.assertEqual(hits, [])
        self.assertTrue(saer.validate_attack_api()['passes'])


if __name__ == '__main__':
    unittest.main()
