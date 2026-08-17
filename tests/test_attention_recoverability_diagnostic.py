import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import attention_recoverability_diagnostic as diag


class AttentionRecoverabilityStatsTests(unittest.TestCase):
    def test_spearman_positive_with_binary_correctness(self):
        rho = diag.spearman([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1])
        self.assertGreater(rho, 0.85)

    def test_spearman_returns_nan_for_constant_input(self):
        rho = diag.spearman([1.0, 1.0, 1.0], [0, 1, 1])
        self.assertTrue(math.isnan(rho))

    def test_quartiles_are_monotone_and_cover_rows(self):
        rows = [{"attention_importance": float(i), "position": i} for i in range(8)]
        grouped = diag.assign_quartiles(rows)
        counts = {name: 0 for name in ("Q1", "Q2", "Q3", "Q4")}
        for row in grouped:
            counts[row["attention_quartile"]] += 1
        self.assertEqual(counts, {"Q1": 2, "Q2": 2, "Q3": 2, "Q4": 2})
        self.assertEqual(grouped[0]["attention_quartile"], "Q1")
        self.assertEqual(grouped[-1]["attention_quartile"], "Q4")

    def test_inverted_weights_keep_mean_one_and_reverse_order(self):
        weights = [0.5, 1.0, 1.5]
        inverted = diag.invert_mean_one_weights(weights)
        self.assertAlmostEqual(sum(inverted) / len(inverted), 1.0, places=7)
        self.assertGreater(inverted[0], inverted[1])
        self.assertGreater(inverted[1], inverted[2])

    def test_counterfactual_objectives_keep_residuals_fixed(self):
        residuals = [1.0, 2.0, 4.0, 8.0]
        weights = [0.5, 0.75, 1.25, 1.5]
        out = diag.counterfactual_objectives(residuals, weights, random.Random(7))
        self.assertAlmostEqual(out["uniform_objective"], 3.75)
        self.assertAlmostEqual(out["original_attention_objective"], 4.75)
        self.assertAlmostEqual(sum(out["shuffled_weights"]) / 4, 1.0)
        self.assertAlmostEqual(sum(out["inverted_weights"]) / 4, 1.0)
        self.assertCountEqual(out["shuffled_weights"], weights)


if __name__ == "__main__":
    unittest.main()
