import inspect
import unittest

import torch

import pia_b0_attention_consistency_repair as acdr


class ACDRTests(unittest.TestCase):
    def test_attack_api_has_no_ground_truth_parameters(self):
        names = set(inspect.signature(acdr.invert_observed).parameters)
        banned = {"prompt", "original_ids", "original_tokens", "input_ids", "token_ids", "ground_truth", "reference"}
        self.assertTrue(names.isdisjoint(banned))

    def test_strict_top1_required(self):
        cfg = acdr.ACDRConfig(
            method="b0_acdr",
            run_name="x",
            output_dir="x",
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
            top_k_embedding=10,
            top_y_semantic=0,
            max_token_len=896,
            grad_clip=1.0,
            uncertainty_fraction=0.2,
            patch_size=16,
            max_repair_passes=2,
            eps_cut=1e-6,
            eps_attn=0.0,
            query_window=64,
            attn_row_fraction=0.2,
            max_attn_rows=128,
            entropy_middle_fraction=0.8,
            public_sink_threshold=0.5,
            uncertainty_detector="margin",
            adaptive_discretization=False,
            semantic_speculation=False,
            local_files_only=True,
        )
        with self.assertRaises(ValueError):
            acdr.require_strict_top1(cfg)

    def test_js_zero_for_identical_rows(self):
        p = torch.tensor([0.2, 0.3, 0.5])
        self.assertLess(float(acdr.js_divergence(p, p)), 1e-8)

    def test_attention_row_normalization_masks_fixed_key(self):
        row = torch.tensor([10.0, 1.0, 1.0, 0.0])
        mask = torch.tensor([False, True, True, False])
        norm = acdr.normalize_attention_row(row, mask)
        self.assertAlmostEqual(float(norm[0]), 0.0, places=6)
        self.assertAlmostEqual(float(norm[1] + norm[2]), 1.0, places=6)

    def test_state_hash_changes_with_token_change(self):
        self.assertNotEqual(acdr.state_hash([1, 2, 3]), acdr.state_hash([1, 2, 4]))

    def test_method_integrity_no_forbidden_attention_loss(self):
        self.assertTrue(acdr.validate_method_integrity()["passes"])


if __name__ == "__main__":
    unittest.main()
