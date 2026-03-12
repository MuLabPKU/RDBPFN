from typing import Tuple
from collections.abc import Mapping
import torch
from torch.utils.data import functional_datapipe

import dgl.graphbolt as gb
from dgl.graphbolt import MiniBatchTransformer

__all__ = ['DBBNegativeSampler']

@functional_datapipe("tgif_sample_negative")
class DBBNegativeSampler(MiniBatchTransformer):
    def __init__(
        self,
        datapipe,
        graph : gb.SamplingGraph,
        negative_ratio : int,
        target_seed_idx : int,
        key_prediction_label_column : str,
        key_prediction_query_idx_column : str,
    ):
        super().__init__(datapipe, self._sample)
        self.graph = graph
        self.negative_ratio = negative_ratio
        self.target_seed_idx = target_seed_idx
        self.key_prediction_label_column = key_prediction_label_column
        self.key_prediction_query_idx_column = key_prediction_query_idx_column

    def _sample(self, minibatch : gb.MiniBatch) -> gb.MiniBatch:
        node_pairs = minibatch.node_pairs
        assert node_pairs is not None
        assert isinstance(node_pairs, Mapping)
        negative_src_dict = {}
        negative_dst_dict = {}
        query_idx_dict = {}
        labels_dict = {}
        for etype, pos_pairs in node_pairs.items():
            neg_src, neg_dst, query_idx, labels = self._sample_one_type(pos_pairs, etype)
            negative_src_dict[etype] = neg_src
            negative_dst_dict[etype] = neg_dst
            query_idx_dict[etype] = query_idx
            labels_dict[etype] = labels
        minibatch.negative_srcs = negative_src_dict
        minibatch.negative_dsts = negative_dst_dict
        minibatch.query_idx = query_idx_dict
        minibatch.labels = labels_dict
        return minibatch

    def _sample_one_type(
        self,
        node_pairs : Tuple[torch.Tensor, torch.Tensor],
        etype : str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample negatives for one edge type.

        Parameters
        ----------
        node_pairs : (torch.Tensor, torch.Tensor)
            Source and desintation nodes of shape (N,)
        etype: str

        Returns
        -------
        negative_src: torch.Tensor
            Shape: (N * negative_ratio,)
        negative_dst: torch.Tensor
            Shape: (N * negative_ratio,)
        query_idx : torch.Tensor
            Index tensor of shape (N * (negative_ratio + 1),) indicating which input node_pair.
            the negative sample belongs to.
        """
        N = len(node_pairs[0])
        minibatch = {
            '__src__' : node_pairs[0],
            '__dst__' : node_pairs[1],
        }
        if self.target_seed_idx == 0:
            target_column_name = '__src__'
        elif self.target_seed_idx == 1:
            target_column_name = '__dst__'
        else:
            raise ValueError(
                "Only support target seed idx to be 0 or 1, "
                f"but got {self.target_seed_idx}.")

        src_type, _, dst_type = etype.split(':')
        corrupt_type = [src_type, dst_type][self.target_seed_idx]
        num_nodes = self.graph.num_nodes[corrupt_type]

        minibatch = negative_sampling(
            minibatch,
            self.negative_ratio,
            target_column_name=target_column_name,
            target_column_capacity=num_nodes,
            key_prediction_label_column=self.key_prediction_label_column,
            key_prediction_query_idx_column=self.key_prediction_query_idx_column,
        )

        new_src = minibatch['__src__']
        new_dst = minibatch['__dst__']
        labels = minibatch[self.key_prediction_label_column]
        query_idx = minibatch[self.key_prediction_query_idx_column]

        # TODO(minjie): A hack to workaround current limitation of GraphBolt pipeline.
        # GB expects neg_src and neg_dst to have shape (N, K) so that timestamp
        # can be properly propagated to negative samples.
        # Only keep negatives.
        negative_src = new_src[N:]
        negative_dst = new_dst[N:]
        negative_src = negative_src.view(self.negative_ratio, N).T.contiguous()
        negative_dst = negative_dst.view(self.negative_ratio, N).T.contiguous()

        # TODO(minjie): GB's current design separates positive and negative node pairs
        # into separate fields. Its internal logic also heavily couples with this
        # design. Ideally, positive and negative node pairs should be treated
        # the same after the negative sampling stage. To workaround this, we follow
        # GB's current design with an exception to labels & query_idx, for which we include
        # also the info for positive samples so the negative sampling logic
        # will not "leak" to the later pipeline.
        labels = labels.view((self.negative_ratio + 1), N).T.reshape(-1)
        query_idx = query_idx.view((self.negative_ratio + 1), N).T.reshape(-1)

        return negative_src, negative_dst, query_idx, labels