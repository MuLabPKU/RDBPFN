from .activations import get_activations


DEFAULT_FIXED_HP = {
    # SCMPrior
    # "mix_probs": (1.0, 0.0),
    "mix_probs": (0.7, 0.3),
    # TreeSCM
    "tree_model": "xgboost",
    "tree_depth_lambda": 0.5,
    "tree_n_estimators_lambda": 0.5,
    # Reg2Cls
    "balanced": False,
    "multiclass_ordered_prob": 0.0,
    # "cat_prob": 0.5,
    "cat_prob": 0.5,
    "max_categories": 10,
    # "max_categories": float("inf"),
    "scale_by_max_features": False,
    "permute_features": True,
    "permute_labels": True,
}

DEFAULT_SAMPLED_HP = {
    # Reg2Cls
    "multiclass_type": {"distribution": "meta_choice", "choice_values": ["value", "rank"]},
    # MLPSCM
    "mlp_activations": {
        "distribution": "meta_choice_mixed",
        "choice_values": get_activations(random=True, scale=True, diverse=True),
    },
    "block_wise_dropout": {"distribution": "meta_choice", "choice_values": [True, False]},
    "mlp_dropout_prob": {"distribution": "meta_beta", "scale": 0.6, "b_min": 0.5, "b_max": 4.5, "k_min": 0.1, "k_max": 5.0},
    # MLPSCM and TreeSCM
    # "is_causal": {"distribution": "meta_choice", "choice_values": [True, False]},
    "is_causal": {"distribution": "meta_choice", "choice_values": [True]},
    "num_causes": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10,
        "min_mean": 3,
        "round": True,
        "lower_bound": 1,
    },
    "y_is_effect": {"distribution": "meta_choice", "choice_values": [True, False]},
    "in_clique": {"distribution": "meta_choice", "choice_values": [True, False]},
    # "in_clique": {"distribution": "meta_choice", "choice_values": [True]},
    "sort_features": {"distribution": "meta_choice", "choice_values": [True, False]},
    "num_layers": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10,
        "min_mean": 3,
        "round": True,
        "lower_bound": 2,
    },
    "hidden_dim": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 30,
        "min_mean": 6,
        "round": True,
        "lower_bound": 4,
    },
    "init_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
    },
    "noise_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 0.3,
        "min_mean": 0.0001,
        "round": False,
        "lower_bound": 0.0,
    },
    "sampling": {"distribution": "meta_choice", "choice_values": ["normal", "mixed", "uniform"]},
    "pre_sample_cause_stats": {"distribution": "meta_choice", "choice_values": [True, False]},
    "pre_sample_noise_std": {"distribution": "meta_choice", "choice_values": [True, False]},
}
