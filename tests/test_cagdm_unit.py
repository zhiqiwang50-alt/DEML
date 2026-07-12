import math
import unittest

import torch

from pia_cagdm import (
    CAGDMDiagnosticConfig,
    candidate_entropy,
    gradient_direction_scores,
    resolve_calibration_margin_threshold,
    should_open_gradient_gate,
)


class CAGDMGateTests(unittest.TestCase):
    def test_top_m_one_and_force_closed_disable_gate(self):
        cfg = CAGDMDiagnosticConfig(top_m=1, force_gate_closed=False)
        decision = should_open_gradient_gate(
            cfg,
            kappa=0.9,
            grad_margin=0.2,
            cal_margin=0.0,
            entropy=1.0,
            distinct_token_count=2,
        )
        self.assertFalse(decision.gate_open)
        self.assertIn('top_m_le_1', decision.closed_reasons)

        cfg = CAGDMDiagnosticConfig(top_m=3, force_gate_closed=True)
        decision = should_open_gradient_gate(
            cfg,
            kappa=0.9,
            grad_margin=0.2,
            cal_margin=0.0,
            entropy=1.0,
            distinct_token_count=2,
        )
        self.assertFalse(decision.gate_open)
        self.assertIn('force_gate_closed', decision.closed_reasons)

    def test_gate_requires_consensus_margin_uncertainty_entropy_and_diversity(self):
        cfg = CAGDMDiagnosticConfig(
            top_m=3,
            tau_consensus=0.6,
            tau_grad_margin=0.05,
            tau_entropy=0.3,
            threshold_cal=0.01,
        )
        open_decision = should_open_gradient_gate(
            cfg,
            kappa=0.8,
            grad_margin=0.2,
            cal_margin=0.005,
            entropy=0.8,
            distinct_token_count=2,
        )
        self.assertTrue(open_decision.gate_open)

        low_entropy = should_open_gradient_gate(
            cfg,
            kappa=0.8,
            grad_margin=0.2,
            cal_margin=0.005,
            entropy=0.1,
            distinct_token_count=2,
        )
        self.assertFalse(low_entropy.gate_open)
        self.assertIn('entropy_below_tau', low_entropy.closed_reasons)

    def test_calibration_threshold_supports_quantile_and_fixed_modes(self):
        margins = [0.001, 0.003, 0.010, 0.100]
        q = resolve_calibration_margin_threshold('prompt_quantile_50', margins)
        self.assertAlmostEqual(q, 0.0065)
        self.assertAlmostEqual(resolve_calibration_margin_threshold('fixed_0.005', margins), 0.005)
        self.assertAlmostEqual(resolve_calibration_margin_threshold('0.01', margins), 0.01)

    def test_gradient_direction_scores_cosine_and_dot(self):
        candidate_dirs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        grad_dir = torch.tensor([1.0, 0.0])
        cosine = gradient_direction_scores(candidate_dirs, grad_dir, 'cosine')
        dot = gradient_direction_scores(candidate_dirs, grad_dir, 'dot')
        self.assertGreater(float(cosine[0]), float(cosine[1]))
        self.assertGreater(float(dot[0]), float(dot[1]))
        self.assertTrue(math.isclose(float(candidate_entropy(torch.tensor([0.0, 0.0]))), math.log(2), rel_tol=1e-6))


if __name__ == '__main__':
    unittest.main()
