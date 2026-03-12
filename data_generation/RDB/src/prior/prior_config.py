import torch.nn as nn
from src.prior.activations import get_activations


"""
    MLP SCM Config
    --------------
    mlp_activations : default=nn.Tanh
        The activation function to be used after each linear transformation
        in the MLP layers (except the first).

    init_std : float, default=1.0
        The standard deviation of the normal distribution used for initializing
        the weights of the MLP's linear layers.

    block_wise_dropout : bool, default=True
        Specifies the weight initialization strategy.
        - If `True`, uses a 'block-wise dropout' initialization where only random
          blocks within the weight matrix are initialized with values drawn from
          a normal distribution (scaled by `init_std` and potentially dropout),
          while the rest are zero. This encourages sparsity.
        - If `False`, uses standard normal initialization for all weights, followed
          by applying dropout mask based on `mlp_dropout_prob`.

    mlp_dropout_prob : float, default=0.1
        The dropout probability applied to weights during *standard* initialization
        (i.e., when `block_wise_dropout=False`). Ignored if
        `block_wise_dropout=True`. The probability is clamped between 0 and 0.99.

    scale_init_std_by_dropout : bool, default=True
        Whether to scale the `init_std` during weight initialization to compensate
        for the variance reduction caused by dropout. If `True`, `init_std` is
        divided by `sqrt(1 - dropout_prob)` or `sqrt(keep_prob)` depending on the
        initialization method.

    sampling : str, default="normal"
        The method used by `XSampler` to generate the initial 'cause' variables.
        Options:
        - "normal": Standard normal distribution (potentially with pre-sampled stats).
        - "uniform": Uniform distribution between 0 and 1.
        - "mixed": A random combination of normal, multinomial (categorical),
          Zipf (power-law), and uniform distributions across different cause variables.

    pre_sample_cause_stats : bool, default=False
        If `True` and `sampling="normal"`, the mean and standard deviation for
        each initial cause variable are pre-sampled. Passed to `XSampler`.

    noise_std : float, default=0.01
        The base standard deviation for the Gaussian noise added after each MLP
        layer's linear transformation (except the first layer).

    pre_sample_noise_std : bool, default=False
        Controls how the standard deviation for the `GaussianNoise` layers is determined.

"""

DEFAULT_MLP_SCM_CONFIG = {
    "mlp_activations": nn.Tanh,
    "init_std": 1.0,
    "block_wise_dropout": True,
    "mlp_dropout_prob": 0.1,
    "scale_init_std_by_dropout": True,
    "sampling": "normal",
    "pre_sample_cause_stats": False,
    "noise_std": 0.01,
    "pre_sample_noise_std": False,
}


DEFAULT_FIXED_HP = {
    # SCMPrior
    "mix_probs": (0.7, 0.3),
    # TreeSCM
    "tree_model": "xgboost",
    "tree_depth_lambda": 0.5,
    "tree_n_estimators_lambda": 0.5,
    # Reg2Cls
    "balanced": False,
    "multiclass_ordered_prob": 0.0,
    "cat_prob": 0.2,
    "max_categories": float("inf"),
    "scale_by_max_features": False,
    "permute_features": True,
    "permute_labels": True,
}

DEFAULT_SAMPLED_HP = {
    # # Reg2Cls
    # "multiclass_type": {
    #     "distribution": "meta_choice",
    #     "choice_values": ["value", "rank"],
    # },
    # MLPSCM
    "mlp_activations": {
        "distribution": "meta_choice_mixed",
        "choice_values": get_activations(random=True, scale=True, diverse=True),
    },
    "block_wise_dropout": {
        "distribution": "meta_choice",
        "choice_values": [True, False],
    },
    "mlp_dropout_prob": {
        "distribution": "meta_beta",
        "scale": 0.6,
        "b_min": 0.5,
        "b_max": 4.5,
        "k_min": 0.1,
        "k_max": 5.0,
    },
    # MLPSCM and TreeSCM
    # "is_causal": {"distribution": "meta_choice", "choice_values": [True, False]},
    "is_causal": {
        "distribution": "meta_choice",
        "choice_values": [True],
    },  # Always True now
    "num_causes": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 12,
        "min_mean": 3,
        "round": True,
        "lower_bound": 1,
    },
    "num_outputs": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 12,
        "min_mean": 3,
        "round": True,
        "lower_bound": 1,
    },
    # "y_is_effect": {"distribution": "meta_choice", "choice_values": [True, False]},
    "y_is_effect": {
        "distribution": "meta_choice",
        "choice_values": [True],
    },  # Always True now
    "in_clique": {"distribution": "meta_choice", "choice_values": [True, False]},
    # "sort_features": {"distribution": "meta_choice", "choice_values": [True, False]},
    "sort_features": {
        "distribution": "meta_choice",
        "choice_values": [True],
    },  # Always True now
    "num_layers": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 12,
        "min_mean": 3,
        "round": True,
        "lower_bound": 2,
    },
    "hidden_dim": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 48,
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
    "sampling": {
        "distribution": "meta_choice",
        "choice_values": ["normal", "mixed", "uniform"],
        # "choice_values": ["normal", "uniform"],
        # "choice_values": ["normal"],
    },
    "pre_sample_cause_stats": {
        "distribution": "meta_choice",
        "choice_values": [False],  # Always False now
    },
    "pre_sample_noise_std": {
        "distribution": "meta_choice",
        "choice_values": [False],  # Always False now
    },
    "eta": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 5.0,
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
    },
    "parent_sampling_dist": {
        "distribution": "meta_choice",
        "choice_values": ["uniform", "zipf"],
    },
    "parent_sampling_alpha": {
        "distribution": "uniform",
        "max": 2.0,
        "min": 1.1,
    },
}
