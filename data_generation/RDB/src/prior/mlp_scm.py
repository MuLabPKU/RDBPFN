from __future__ import annotations

import math
import random
from enum import Enum
from typing import Dict, Any, List

import numpy as np
import torch
from torch import nn

from .utils import GaussianNoise, XSampler, MASK_TYPE, SCM_OUTPUT
from .prior_config import DEFAULT_MLP_SCM_CONFIG
from .temporal_vocab import TemporalVocab


class MLPSCM(nn.Module):
    """Generates synthetic tabular datasets using a Multi-Layer Perceptron (MLP) based Structural Causal Model (SCM).

    Parameters
    ----------
    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.

    num_features : int, default=100
        The number of features.

    num_outputs : int, default=1
        The number of outputs.

    is_causal : bool, default=True
        - If `True`, simulates a causal graph: `X` and `y` are sampled from the
          intermediate hidden states of the MLP transformation applied to initial causes.
          The `num_causes` parameter controls the number of initial root variables.
        - If `False`, simulates a direct predictive mapping: Initial causes are used
          directly as `X`, and the final output of the MLP becomes `y`. `num_causes`
          is effectively ignored and set equal to `num_features`.

    num_causes : int, default=10
        The number of initial root 'cause' variables sampled by `XSampler`.
        Only relevant when `is_causal=True`. If `is_causal=False`, this is internally
        set to `num_features`.

    other_causes : int, default=0
        The number of additional features from parent tables.

    sampling_ratio : float, default=1.0
        If parent tables are used, we sample `sampling_ratio` * `seq_len` samples,
        and only accept `seq_len` samples with high existence probability.

    y_is_effect : bool, default=True
        Specifies how the target `y` is selected when `is_causal=True`.
        - If `True`, `y` is sampled from the outputs of the final MLP layer(s),
          representing terminal effects in the causal chain.
        - If `False`, `y` is sampled from the earlier intermediate outputs (after
          permutation), representing variables closer to the initial causes.

    in_clique : bool, default=False
        Controls how features `X` and targets `y` are sampled from the flattened
        intermediate MLP outputs when `is_causal=True`.
        - If `True`, `X` and `y` are selected from a contiguous block of the
          intermediate outputs, potentially creating denser dependencies among them.
        - If `False`, `X` and `y` indices are chosen randomly and independently
          from all available intermediate outputs.

    sort_features : bool, default=True
        Determines whether to sort the features based on their original indices from
        the intermediate MLP outputs. Only relevant when `is_causal=True`.

    num_layers : int, default=10
        The total number of layers in the MLP transformation network. Must be >= 2.
        Includes the initial linear layer and subsequent blocks of
        (Activation -> Linear -> Noise).

    hidden_dim : int, default=20
        The dimensionality of the hidden representations within the MLP layers.
        If `is_causal=True`, this is automatically increased if it's smaller than
        `num_outputs + 2 * num_features` to ensure enough intermediate variables
        are generated for sampling `X` and `y`.

    device : str, default="cpu"
        The computing device ('cpu' or 'cuda') where tensors will be allocated.

    **kwargs : dict
        Unused hyperparameters passed from parent configurations.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        # num_features: int = 100,  # Deprecated, now merge to masks
        num_outputs: int = 10,
        num_causes: int = 10,  # Meaning changed to additional noise
        other_causes: int = 0,
        sampling_ratio: float = 1.0,
        is_causal: bool = True,  # Always True now
        in_clique: bool = False,  # Used now
        sort_features: bool = True,  # No need to sort features now
        num_layers: int = 10,
        hidden_dim: int = 20,
        # config: Dict[str, Any] = DEFAULT_MLP_SCM_CONFIG,
        mlp_activations: nn.Module = nn.Tanh,
        init_std: float = 1.0,
        block_wise_dropout: bool = True,
        mlp_dropout_prob: float = 0.1,
        scale_init_std_by_dropout: bool = True,
        sampling: str = "normal",
        pre_sample_cause_stats: bool = False,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        device: str = "cpu",
        masks: Dict[MASK_TYPE, int] = {},
        # New parameters for timestamp-based sampling
        use_timestamp_sampling: bool = False,
        embedding_dim: int = 10,
        batch_size: int = 32,
        parent_sampling_dist: str = "uniform",  # "uniform" or "zipf"
        parent_sampling_alpha: float = 1.0,
        **kwargs: Dict[str, Any],
    ):
        super(MLPSCM, self).__init__()
        self.seq_len = seq_len
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.other_causes = other_causes
        self.sampling_ratio = sampling_ratio
        self.in_clique = in_clique
        self.sort_features = sort_features

        assert is_causal, "is_causal is not implemented"
        assert num_layers >= 2, "Number of layers must be at least 2."
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.mlp_activations = mlp_activations
        self.init_std = init_std
        self.block_wise_dropout = block_wise_dropout
        self.mlp_dropout_prob = mlp_dropout_prob
        self.scale_init_std_by_dropout = scale_init_std_by_dropout
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std

        self.device = device

        # New parameters for timestamp-based sampling
        self.use_timestamp_sampling = use_timestamp_sampling
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.parent_sampling_dist = parent_sampling_dist
        self.parent_sampling_alpha = parent_sampling_alpha

        if self.use_timestamp_sampling:
            self.temporal_vocab = TemporalVocab()

        self.eta = kwargs.get("eta", 1.0)  # Controls influence of embedding affinity

        if self.is_causal:
            total_features = masks[MASK_TYPE.X]
            # Ensure enough intermediate variables for sampling X and y
            self.hidden_dim = max(
                self.hidden_dim,
                int(
                    np.ceil(
                        (self.num_outputs + total_features + 1) / (self.num_layers - 1)
                    )
                ),
            )
        else:
            # In non-causal mode, features are the causes
            raise ValueError("Non-causal mode is not implemented")
            # total_features = sum(masks.values()) if masks else 100
            # self.num_causes = total_features

        # Define the input sampler
        self.xsampler = XSampler(
            int(self.seq_len * self.sampling_ratio),
            self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )

        # Build layers
        layers = [nn.Linear(self.num_causes + self.other_causes, self.hidden_dim)]
        for _ in range(self.num_layers - 1):
            layers.append(self.generate_layer_modules())
        if not self.is_causal:
            layers.append(self.generate_layer_modules(is_output_layer=True))
        self.layers = nn.Sequential(*layers).to(device)

        # Initialize layers
        self.initialize_parameters()

        # Generate the masks, defining what position are used
        self.generate_masks(masks)

    def generate_layer_modules(self, is_output_layer=False):
        """Generates a layer module with activation, linear transformation, and noise."""
        out_dim = self.num_outputs if is_output_layer else self.hidden_dim
        activation = self.mlp_activations()
        linear_layer = nn.Linear(self.hidden_dim, out_dim)

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(
                    torch.zeros(size=(1, out_dim), device=self.device),
                    float(self.noise_std),
                )
            )
        else:
            noise_std = self.noise_std
        noise_layer = GaussianNoise(noise_std)

        return nn.Sequential(activation, linear_layer, noise_layer)

    def initialize_parameters(self):
        """Initializes parameters using block-wise dropout or normal initialization."""
        with torch.no_grad():
            for i, (_, param) in enumerate(self.layers.named_parameters()):
                if self.block_wise_dropout and param.dim() == 2:
                    self.initialize_with_block_dropout(param, i)
                else:
                    self.initialize_normally(param, i)

    def initialize_with_block_dropout(self, param, index):
        """Initializes parameters using block-wise dropout."""
        nn.init.zeros_(param)
        n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
        block_size = [dim // n_blocks for dim in param.shape]
        keep_prob = (n_blocks * block_size[0] * block_size[1]) / param.numel()
        for block in range(n_blocks):
            block_slice = tuple(
                slice(dim * block, dim * (block + 1)) for dim in block_size
            )
            nn.init.normal_(
                param[block_slice],
                std=self.init_std
                / (keep_prob**0.5 if self.scale_init_std_by_dropout else 1),
            )

    def initialize_normally(self, param, index):
        """Initializes parameters using normal distribution."""
        if param.dim() == 2:  # Applies only to weights, not biases
            dropout_prob = (
                self.mlp_dropout_prob if index > 0 else 0
            )  # No dropout for the first layer's weights
            dropout_prob = min(dropout_prob, 0.99)
            std = self.init_std / (
                (1 - dropout_prob) ** 0.5 if self.scale_init_std_by_dropout else 1
            )
            nn.init.normal_(param, std=std)
            param *= torch.bernoulli(torch.full_like(param, 1 - dropout_prob))

    def generate_masks(self, masks: Dict[MASK_TYPE, int]):
        """Generates the masks, defining what position are used"""
        self.masks = {}
        total_features = (self.num_layers - 1) * self.hidden_dim
        for key, value in masks.items():
            # each key is a string, each value is an integer representing the number of features to use
            if self.in_clique:
                # sample a consecutive block of features
                idx = torch.arange(value, device=self.device)
                idx += random.randint(0, total_features - value)
            else:
                # sample random features
                idx = torch.tensor(
                    random.sample(range(total_features), value),
                    device=self.device,
                )
            self.masks[key] = idx

        if self.in_clique:
            # sample a consecutive block of features
            idx = torch.arange(self.num_outputs, device=self.device)
            idx += random.randint(0, total_features - self.num_outputs)
        else:
            # sample random features
            idx = torch.tensor(
                random.sample(
                    range(total_features - self.num_outputs, total_features),
                    self.num_outputs,
                ),
                device=self.device,
            )
        self.masks[MASK_TYPE.CAUSAL_OUTPUT] = idx
        self.masks[MASK_TYPE.FULL] = torch.arange(total_features, device=self.device)

        return

    def sample_zipf_indices(
        self, n_parent_samples: int, num_samples: int, alpha: float = 2.0
    ):
        """
        Sample indices from a Zipf distribution with random remapping.

        This method uses np.random.zipf to generate samples following a Zipf
        distribution, then remaps them through a random permutation. This ensures
        the long-tail property is preserved, but the "popular" indices are randomly
        distributed rather than always being the first few indices.

        For example, instead of index 0 being most common, a random index like 47
        might be most common, preserving the Zipf distribution shape.

        Parameters
        ----------
        n_parent_samples : int
            Number of samples available in the parent table (max index)
        num_samples : int
            Number of indices to sample
        alpha : float
            Zipf distribution parameter (must be > 1). Higher values create
            heavier tails (more concentration on small indices).

        Returns
        -------
        list
            List of sampled indices in range [0, n_parent_samples) following
            a Zipf distribution remapped to random positions
        """
        # Edge case: empty parent table
        if n_parent_samples <= 0:
            raise ValueError(
                f"Parent table has {n_parent_samples} samples, cannot sample from empty table"
            )

        # Edge case: single row in parent table
        if n_parent_samples == 1:
            return [0] * num_samples

        # Ensure alpha is valid for Zipf distribution (must be > 1)
        alpha = max(alpha, 1.01)

        # Create a random permutation for remapping
        # This determines which actual indices will be "popular"
        index_permutation = np.random.permutation(n_parent_samples)

        # Sample from Zipf distribution
        # np.random.zipf generates values >= 1, so we subtract 1 to get 0-indexed
        zipf_samples = np.random.zipf(alpha, size=num_samples) - 1

        # Clip samples to valid range [0, n_parent_samples)
        # This handles the case where zipf samples exceed parent table size
        clipped_indices = np.clip(zipf_samples, 0, n_parent_samples - 1)

        # Remap through the permutation
        # This makes the long-tail distribution apply to random indices
        # rather than always favoring index 0
        remapped_indices = index_permutation[clipped_indices]

        return remapped_indices.tolist()

    def forward_without_input(self):
        """
        This case is for generate tables without parent tables.
        Therefore, we do not need to sample the parent tables.
        """
        causes = self.xsampler.sample()  # (seq_len, num_causes)

        # Generate outputs through MLP layers
        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[
            2:
        ]  # Start from 2 because the first layer is only linear without activation

        # Handle outputs based on causality
        X, outputs_flat = self.handle_outputs(outputs, self.masks)

        # Check for NaNs and handle them by setting to default values
        for _, value in X.items():
            if torch.any(torch.isnan(value)):
                value[:] = 0.0

        # Return both masked outputs and full outputs for TableGenerator
        return X, outputs_flat

    def forward_with_input(self, parent_data_list: List[torch.Tensor]):
        """
        This case is for generate tables with parent tables.
        Therefore, we need to sample the parent tables.

        Parameters
        ----------
        *args : list of torch.Tensor
            List of parent tables.
            Each parent table is a tensor of shape (seq_len_i, num_features_i).

        **kwargs : dict
            Unused hyperparameters passed from parent configurations.
        """
        causes = self.xsampler.sample()  # (seq_len * sampling_ratio, num_causes)
        parent_idxes = []
        for parent_table_data in parent_data_list:
            num_samples = int(self.seq_len * self.sampling_ratio)
            n_parent_samples = parent_table_data[MASK_TYPE.FULL].shape[0]

            if self.parent_sampling_dist == "uniform":
                # Original uniform sampling
                parent_causes_idx = random.choices(
                    range(n_parent_samples),
                    k=num_samples,
                )
            elif self.parent_sampling_dist == "zipf":
                # Zipf distribution sampling using direct method
                # This is more efficient and avoids multinomial edge cases
                try:
                    parent_causes_idx = self.sample_zipf_indices(
                        n_parent_samples=n_parent_samples,
                        num_samples=num_samples,
                        alpha=self.parent_sampling_alpha,
                    )
                except ValueError as e:
                    # Fallback to uniform if parent table is empty
                    print(
                        f"Warning: Zipf sampling failed ({e}), falling back to uniform"
                    )
                    if n_parent_samples > 0:
                        parent_causes_idx = random.choices(
                            range(n_parent_samples), k=num_samples
                        )
                    else:
                        raise ValueError("Cannot sample from empty parent table")
            else:
                # Fallback to uniform for any other distribution type
                print(
                    f"Warning: Unknown distribution '{self.parent_sampling_dist}', using uniform"
                )
                parent_causes_idx = random.choices(
                    range(n_parent_samples),
                    k=num_samples,
                )

            parent_idxes.append(parent_causes_idx)
            parent_causes = parent_table_data[MASK_TYPE.CAUSAL_OUTPUT][
                parent_causes_idx
            ]
            causes = torch.cat([causes, parent_causes], dim=-1)
        parent_idxes = torch.tensor(parent_idxes, device=self.device).long()
        # convert idxes to a 2-dim, and transpose it
        if parent_idxes.ndim == 1:
            parent_idxes = parent_idxes.unsqueeze(1)
        parent_idxes = parent_idxes.transpose(0, 1)

        assert causes.shape[0] == int(
            self.seq_len * self.sampling_ratio
        ), "The number of samples should be the same"
        assert (
            causes.shape[1] == self.num_causes + self.other_causes
        ), "The number of causes should be the same"

        # Generate outputs through MLP layers
        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[
            2:
        ]  # Start from 2 because the first layer is only linear without activation

        X, outputs_flat = self.handle_outputs(outputs, self.masks)

        # Check for NaNs and handle them by setting to default values
        for _, value in X.items():
            if torch.any(torch.isnan(value)):
                value[:] = 0.0

        # Use the edge probability to sample the final output
        X, parent_idxes = self.sample_final_output(X, parent_idxes)

        return X, parent_idxes, outputs_flat

    def forward_with_enhanced_temporal_sampling(
        self, parent_data_list: List[Dict[str, torch.Tensor]]
    ):
        """
        Enhanced temporal sampling method implementing the 5-step process:
        1. Draw Timestamps from temporal function Λ(t)
        2. Initialize Objects (mass vectors and edge kernel)
        3. Nested Sampling Loop (for each timestamp)
        4. Reinforcer Update (update masses and edge kernel)
        5. Return completed child table

        Parameters
        ----------
        parent_data_list : List[torch.Tensor]
            List of exactly 2 parent tables.
            Each parent table is a tensor of shape (seq_len_i, num_features_i).

        Returns
        -------
        X : Dict[str, torch.Tensor]
            Generated features for the child table
        parent_idxes : torch.Tensor
            Indices of selected parent samples
        outputs_flat : torch.Tensor
            Full flattened outputs from all MLP layers
        """
        assert (
            len(parent_data_list) == 2
        ), "This method requires exactly 2 parent tables"

        parent_P_data, parent_Q_data = parent_data_list

        # Step 1: Draw Timestamps from temporal function Λ(t)
        timestamps = self.temporal_vocab.sample(
            num_samples=self.seq_len,
            time_range=(0, 10),
        )

        # Step 2: Initialize Objects
        aug_emb_P, aug_emb_Q = self._generate_parent_embeddings(
            parent_P_data, parent_Q_data
        )
        self.temporal_vocab_p = TemporalVocab()
        self.temporal_vocab_q = TemporalVocab()
        self.temporal_vocab_p.generate(time_range=(0, 10))
        self.temporal_vocab_q.generate(time_range=(0, 10))
        self.temporal_vocab_p.norm_intensity()
        self.temporal_vocab_q.norm_intensity()

        # Initialize edge kernel E_ij = <e^P_i, e^Q_j>
        E = torch.matmul(aug_emb_P, aug_emb_Q.T)  # (n_P, n_Q)
        E = E.clamp(min=-10000.0, max=10000.0)

        if E.max() - E.min() != 0:
            E = (E - E.min()) / (E.max() - E.min())
        else:
            E = torch.ones_like(E)

        # Initialize mass vectors based on the degree of E.
        # Norm to mean 1, std 0.2. Clip to be non-negative and less than 10.
        m_P = E.sum(dim=1)
        m_Q = E.sum(dim=0)
        if m_P.max() - m_P.min() != 0:
            m_P = torch.clamp((m_P - m_P.mean()) / m_P.std() * 0.2 + 1, min=0.1, max=10)
        else:
            m_P = torch.ones_like(m_P)
        if m_Q.max() - m_Q.min() != 0:
            m_Q = torch.clamp((m_Q - m_Q.mean()) / m_Q.std() * 0.2 + 1, min=0.1, max=10)
        else:
            m_Q = torch.ones_like(m_Q)

        # Storage for results
        selected_P_indices = []
        selected_Q_indices = []
        all_child_features = []

        # Step 3: Nested Sampling Loop. Each with a batch of samples
        for batch_idx in range(0, self.seq_len, self.batch_size):
            if batch_idx + self.batch_size > self.seq_len:
                batch_timestamps = timestamps[batch_idx:]
            else:
                batch_timestamps = timestamps[batch_idx : batch_idx + self.batch_size]
            batch_size = batch_timestamps.shape[0]

            # 3.1: Pick a row from parent P: i ~ Cat(m^P)
            if m_P.sum() <= 0:
                P_probs = torch.full_like(m_P, 1.0 / m_P.numel())
            else:
                P_probs = m_P / m_P.sum()
            i = torch.multinomial(P_probs, num_samples=batch_size, replacement=True)

            # 3.2: Pick a row from parent Q conditioned on i: j ~ Cat(m^Q + η * E_{i•})
            if m_Q.max() - m_Q.min() != 0:
                norm_m_Q = (m_Q - m_Q.min()) / (m_Q.max() - m_Q.min())
            else:
                norm_m_Q = torch.full_like(m_Q, 1.0 / m_Q.numel())
            Q_weights = norm_m_Q + self.eta * E[i, :]
            row_sums = Q_weights.sum(dim=1, keepdim=True)
            zero_rows = row_sums.squeeze(1) <= 0
            safe_row_sums = torch.where(
                row_sums <= 0, torch.ones_like(row_sums), row_sums
            )
            Q_probs = Q_weights / safe_row_sums
            if zero_rows.any():
                Q_probs[zero_rows] = 1.0 / Q_weights.size(1)
            j = torch.multinomial(Q_probs, num_samples=1, replacement=True).squeeze(1)

            # 3.3: Generate child features via the child-table SCM
            # Create input causes for this sample
            cause_sample = self.xsampler.sample_batch(batch_size)
            parent_P_sample = parent_P_data[MASK_TYPE.CAUSAL_OUTPUT][i]
            parent_Q_sample = parent_Q_data[MASK_TYPE.CAUSAL_OUTPUT][j]

            # Concatenate all inputs
            combined_input = torch.cat(
                [cause_sample, parent_P_sample, parent_Q_sample], dim=-1
            )

            # Generate child features through SCM
            child_features = self._generate_batch_child_features(combined_input)

            # 3.4: Append (i, j, t_k, features) as a new child row
            selected_P_indices.extend(i.tolist())
            selected_Q_indices.extend(j.tolist())
            all_child_features.append(child_features)

            # Step 4: Reinforcer Update
            # Update masses with temporal decay factors
            beta_P_t = self._get_temporal_reinforcement(
                batch_timestamps, self.temporal_vocab_p.get_intensity
            )
            beta_Q_t = self._get_temporal_reinforcement(
                batch_timestamps, self.temporal_vocab_q.get_intensity
            )
            # beta_pair_t = self._get_temporal_reinforcement(t_k, self.beta_pair)

            if self.eta > 1:
                m_P[i] += (beta_P_t / self.eta).to(self.device)
                m_Q[j] += (beta_Q_t / self.eta).to(self.device)
            else:
                m_P[i] += (beta_P_t * (-4 * self.eta + 5)).to(self.device)
                m_Q[j] += (beta_Q_t * (-4 * self.eta + 5)).to(self.device)
            # E[i, j] += beta_pair_t

        # Step 5: Return completed child table
        # Combine all results
        all_child_features = torch.cat(all_child_features, dim=0)
        final_P_indices = torch.tensor(selected_P_indices, device=self.device)
        final_Q_indices = torch.tensor(selected_Q_indices, device=self.device)
        final_parent_idxes = torch.stack([final_P_indices, final_Q_indices], dim=1)

        # Create output dictionary
        X, outputs_flat = self.handle_outputs(
            all_child_features, self.masks, skip_concat=True
        )

        # Store the sampled timestamps if TIMESTAMP mask exists
        if MASK_TYPE.TIMESTAMP in self.masks:
            # Normalize timestamps to [0, 1] range for consistency with other features
            if timestamps.max() - timestamps.min() > 0:
                normalized_timestamps = (timestamps - timestamps.min()) / (
                    timestamps.max() - timestamps.min()
                )
            else:
                normalized_timestamps = torch.zeros_like(timestamps)

            # Store as 2D tensor (seq_len, 1)
            X[MASK_TYPE.TIMESTAMP] = normalized_timestamps.unsqueeze(1).to(self.device)

        # Check for NaNs and handle them by setting to default values
        for _, value in X.items():
            if torch.any(torch.isnan(value)):
                value[:] = 0.0

        return X, final_parent_idxes, outputs_flat

    def sample_final_output(self, X, parent_idxes):
        """
        Samples the final output using the edge probability.
        """
        assert MASK_TYPE.EDGE_PROB in X, "edge_prob is not in the X"
        edge_prob = X[MASK_TYPE.EDGE_PROB]
        # norm the edge_prob to be between 0 and 1
        if edge_prob.max() - edge_prob.min() != 0:
            edge_prob = (edge_prob - edge_prob.min()) / (
                edge_prob.max() - edge_prob.min()
            )
        else:
            edge_prob = torch.ones_like(edge_prob)
        # based on the edge_prob, sample the final output, eventually only keep the seq_len samples
        idx = torch.multinomial(
            edge_prob.squeeze(), num_samples=self.seq_len, replacement=False
        )

        for key, value in X.items():
            X[key] = value[idx]

        return X, parent_idxes[idx]

    def handle_outputs(self, outputs, masks, skip_concat=False):
        """
        Handles outputs from the MLP layers.

        Parameters
        ----------
        outputs : list of torch.Tensor
            List of output tensors from MLP layers

        masks : dict of str -> list of int
            Dictionary of masks, each key is a string, each value is a list of integers representing the features to use

        Returns
        -------
        X : Dict of str -> torch.Tensor
            Input features (seq_len, num_features)
        outputs_flat : torch.Tensor
            Full flattened outputs from all MLP layers
        """
        X = {}
        if skip_concat:
            outputs_flat = outputs
        else:
            outputs_flat = torch.cat(outputs, dim=-1)
        for key, value in masks.items():
            X[key] = outputs_flat[:, value]

        return X, outputs_flat

    def _generate_parent_embeddings(self, *args):
        """
        Generate mass and augmentation embeddings for a parent table.

        Parameters
        ----------
        parent_data : torch.Tensor
            Parent table data of shape (n_samples, n_features)

        Returns
        -------
        mass_emb : torch.Tensor
            Mass embeddings of shape (n_samples, embedding_dim)
        aug_emb : torch.Tensor
            Augmentation embeddings of shape (n_samples, embedding_dim)
        """
        # Simple linear transformation to generate embeddings
        aug_emb = []
        for parent_data in args:
            # random pick number of self.embedding_dim idx from the parent_data[FULL]
            aug_idx = torch.randperm(parent_data[MASK_TYPE.FULL].shape[1])[
                : self.embedding_dim
            ]
            aug_emb.append(parent_data[MASK_TYPE.FULL][:, aug_idx])

        return aug_emb

    def _get_temporal_reinforcement(self, t: torch.Tensor, intensity: torch.Tensor):
        """
        Get temporal reinforcement factor β(t).

        Parameters
        ----------
        t : torch.Tensor
            Current timestamp
        intensity : torch.Tensor
            Intensity of the temporal distribution

        Returns
        -------
        beta_t : torch.Tensor
            Time-dependent reinforcement factor
        """
        beta_t = intensity[t]
        return beta_t

    def _generate_batch_child_features(self, combined_input: torch.Tensor):
        """
        Generate child features for a batch of samples using the SCM.

        Parameters
        ----------
        combined_input : torch.Tensor
            Combined input tensor with causes and parent data

        Returns
        -------
        features : torch.Tensor
            Generated child features for one batch of samples
        """
        # Generate outputs through MLP layers
        outputs = [combined_input]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[2:]  # Skip first two layers

        # Flatten and extract features
        outputs_flat = torch.cat(outputs, dim=-1)

        return outputs_flat


if __name__ == "__main__":
    # Example with uniform sampling (default behavior)
    model_uniform = MLPSCM(
        seq_len=16,
        num_outputs=10,
        is_causal=True,
        num_causes=10,
        other_causes=0,
        sampling_ratio=1.0,
        masks={
            MASK_TYPE.X: 10,
            MASK_TYPE.EDGE_PROB: 1,  # Add edge_prob for sampling
        },
        parent_sampling_dist="uniform",  # Uniform distribution
    )

    # Example with Zipf distribution (long-tail distribution)
    # Higher alpha values create heavier tails (more concentration on small indices)
    model_zipf = MLPSCM(
        seq_len=16,
        num_outputs=10,
        is_causal=True,
        num_causes=10,
        other_causes=0,
        sampling_ratio=1.0,
        masks={
            MASK_TYPE.X: 10,
            MASK_TYPE.EDGE_PROB: 1,
        },
        parent_sampling_dist="zipf",  # Zipf distribution for long-tail sampling
        parent_sampling_alpha=2.0,  # Alpha > 1 required; typical range: 1.5-4.0
    )
