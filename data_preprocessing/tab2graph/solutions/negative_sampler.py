from typing import Dict
import torch

__all__ = ['negative_sampling']

def negative_sampling(
    minibatch : Dict[str, torch.Tensor],
    negative_sampling_ratio,
    target_column_name,
    target_column_capacity,
    key_prediction_label_column,
    key_prediction_query_idx_column,
    shuffle_rest_columns=False,
) -> Dict[str, torch.Tensor]:
    """Augment the minibatch with negative samples.

    All negative samples are appended to the positive samples.

    Parameters
    ----------

    Returns
    -------
    minibatch : Dict[str, torch.Tensor]
        New minibatch with negative samples, labels and query index.
    """
    assert negative_sampling_ratio > 0

    target_column = minibatch[target_column_name]
    size = len(target_column)
    neg_samples = torch.randint(0, target_column_capacity, (size * negative_sampling_ratio,))
    target_column = torch.cat([target_column, neg_samples])

    # Corrupt features
    for key in list(minibatch.keys()):
        val = minibatch[key]
        if shuffle_rest_columns:
            val_corrupted = [val[torch.randperm(size)] for _ in range(negative_sampling_ratio)]
            val_augmented = torch.cat([val] + val_corrupted, 0)
        else:
            val_augmented = val.repeat(negative_sampling_ratio + 1)
        minibatch[key] = val_augmented

    minibatch[target_column_name] = target_column
    key_prediction_label = torch.zeros(size * (negative_sampling_ratio + 1), dtype=torch.int64)
    key_prediction_label[:size] = 1

    minibatch[key_prediction_label_column] = key_prediction_label
    minibatch[key_prediction_query_idx_column] = (
        torch.arange(size).repeat(negative_sampling_ratio + 1)
    )

    return minibatch
