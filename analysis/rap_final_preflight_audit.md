# RAP-FINAL Preflight Audit

```json
{
  "title": "RAP-FINAL: Top-1 Relational Sparse Repair with Discrete Acceptance",
  "dataset_meta": {
    "dataset_name": "Skytrax",
    "dataset_path": "data/airline.json",
    "available_valid_prompts": 28,
    "requested_prompts": 1,
    "selected_prompts": 1,
    "selection_seed": 42,
    "filtering": "ASCII English prompts, non-empty, normalized whitespace, length >= 20 characters"
  },
  "target_layer": 17,
  "h_obs_semantics": "H_obs is output of 0-based block 17; server starts at block 18",
  "server_start": 18,
  "server_layers": [
    18,
    19,
    20,
    21
  ],
  "manual_h_obs_boundary_vs_full_max_abs_diff": 0.0,
  "manual_boundary_pass": true,
  "attack_api_check": {
    "signature": "(model: torch.nn.modules.module.Module, tokenizer: Any, cfg: __main__.RAPFinalConfig, observed_activation: torch.Tensor, seq_len: int, device: torch.device) -> Tuple[List[int], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]",
    "parameter_names": [
      "model",
      "tokenizer",
      "cfg",
      "observed_activation",
      "seq_len",
      "device"
    ],
    "banned_signature_hits": [],
    "banned_global_reference_hits": [],
    "passes": true
  },
  "method_integrity_check": {
    "checked_functions": [
      "invert_observed",
      "run_sparse_repair",
      "optimize_sparse_candidate",
      "build_relational_graph",
      "top_r_relational_destinations"
    ],
    "dummy_reference_hits": [],
    "scalar_attention_call_hits": [],
    "passes": true
  },
  "strict_top1": {
    "enabled": true,
    "K": 1,
    "Y": 0,
    "semantic": false,
    "calibration": false
  },
  "no_ground_truth_in_attack": true,
  "no_dummy_attention": true,
  "no_scalar_attention_weighting_loss": true,
  "updated_positions_subset_of_uncertain": true,
  "token_id_based_special_mask_available": false,
  "variable_mask_audit": {
    "variable_count": 134,
    "known_public_special_positions": [
      0
    ],
    "token_id_based_special_mask_available": false,
    "last_valid_position": 134,
    "semantic_note": "fixed_public positions such as position 0 are public framing positions available to the attacker. When attack API token ids are unavailable, this audit cannot claim EOS detection by token id. The last valid position is only a sequence boundary; exclude it as EOS only if the public protocol explicitly fixes the final token as EOS.",
    "rows": [
      {
        "position": 0,
        "valid": true,
        "fixed_public": true,
        "special": true,
        "token_id_special": false,
        "known_public_special": true,
        "variable_mask": false,
        "exclusion_reason": "fixed_public"
      },
      {
        "position": 1,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 2,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 3,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 4,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 5,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 6,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 7,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 8,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 9,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 10,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 11,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 12,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 13,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 14,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 15,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 16,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 17,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 18,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 19,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 20,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 21,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 22,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 23,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 24,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 25,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 26,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 27,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 28,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 29,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 30,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 31,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 32,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 33,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 34,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 35,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 36,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 37,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 38,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 39,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 40,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 41,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 42,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 43,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 44,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 45,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 46,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 47,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 48,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 49,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 50,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 51,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 52,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 53,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 54,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 55,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 56,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 57,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 58,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 59,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 60,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 61,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 62,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 63,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 64,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 65,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 66,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 67,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 68,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 69,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 70,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 71,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 72,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 73,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 74,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 75,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 76,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 77,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 78,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 79,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 80,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 81,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 82,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 83,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 84,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 85,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 86,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 87,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 88,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 89,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 90,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 91,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 92,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 93,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 94,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 95,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 96,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 97,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 98,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 99,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 100,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 101,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 102,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 103,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 104,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 105,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 106,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 107,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 108,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 109,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 110,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 111,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 112,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 113,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 114,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 115,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 116,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 117,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 118,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 119,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 120,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 121,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 122,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 123,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 124,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 125,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 126,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 127,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 128,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 129,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 130,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 131,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 132,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 133,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      },
      {
        "position": 134,
        "valid": true,
        "fixed_public": false,
        "special": false,
        "token_id_special": false,
        "known_public_special": false,
        "variable_mask": true,
        "exclusion_reason": "variable"
      }
    ]
  },
  "relational_graph_stats": {
    "formula": "S_l[q,k] = mean_h A_lh[q,k] * ||V_lh[k]||_2; S_norm_l = S_l / (mean_legal_variable_causal_edges(S_l) + eps); S = mean_l S_norm_l",
    "server_layers": [
      18,
      19,
      20,
      21
    ],
    "layer_count": 4,
    "layer_stats": [
      {
        "layer": 18,
        "shape": [
          135,
          135
        ],
        "upper_triangular_max": 0.0,
        "mean": 2.7832038402557373,
        "max": 456.7337646484375,
        "raw_mean": 0.009687336161732674,
        "normalized_mean": 1.0,
        "raw_max": 0.6025124788284302,
        "scale_factor": 0.009687336161732674,
        "legal_variable_causal_edge_count": 8911
      },
      {
        "layer": 19,
        "shape": [
          135,
          135
        ],
        "upper_triangular_max": 0.0,
        "mean": 2.9093425273895264,
        "max": 446.3321838378906,
        "raw_mean": 0.011587986722588539,
        "normalized_mean": 1.0,
        "raw_max": 0.43269866704940796,
        "scale_factor": 0.011587986722588539,
        "legal_variable_causal_edge_count": 8911
      },
      {
        "layer": 20,
        "shape": [
          135,
          135
        ],
        "upper_triangular_max": 0.0,
        "mean": 1.845890760421753,
        "max": 234.21241760253906,
        "raw_mean": 0.015030056238174438,
        "normalized_mean": 1.0,
        "raw_max": 0.6984728574752808,
        "scale_factor": 0.015030056238174438,
        "legal_variable_causal_edge_count": 8911
      },
      {
        "layer": 21,
        "shape": [
          135,
          135
        ],
        "upper_triangular_max": 0.0,
        "mean": 2.6050050258636475,
        "max": 433.3163146972656,
        "raw_mean": 0.02300436794757843,
        "normalized_mean": 1.0,
        "raw_max": 0.8468368053436279,
        "scale_factor": 0.02300436794757843,
        "legal_variable_causal_edge_count": 8911
      }
    ],
    "uses_layerwise_relational_scale_normalization": true,
    "uses_scalar_attention_weighted_activation_loss": false
  },
  "aggregate_shape": [
    135,
    135
  ],
  "passes": true
}
```
