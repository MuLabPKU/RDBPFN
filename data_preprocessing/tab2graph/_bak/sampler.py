from typing import Dict
import dgl
from dgl.dataloading.base import BlockSampler, Sampler, set_edge_lazy_features, \
    set_node_lazy_features
from dgl.sampling.utils import EidExcluder
import numpy as np
import torch

class GlobalTemporalNeighborSampler(BlockSampler):
    def __init__(
        self,
        fanouts,
        node_timestamps : Dict[str, np.ndarray],
        timestamp_threshold : int,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        # NaT converts to FLOAT_MIN in numpy so I compute a separate mask for NaT.
        self.node_nat_mask = {
            k: torch.from_numpy(np.isnan(v))
            for k, v in node_timestamps.items()
        }
        self.node_timestamps = {
            k: torch.from_numpy(v.astype('float64'))
            for k, v in node_timestamps.items()
        }
        self.timestamp_threshold = timestamp_threshold

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )

            filtered_edges = {}
            filtered_edge_ids = {}
            for utype, etype, vtype in frontier.canonical_etypes:
                u, v, eid = frontier.edges(etype=(utype, etype, vtype), form='all')
                mask = None
                if utype in self.node_timestamps:
                    mask = self.node_nat_mask[utype][u]
                    mask |= (self.node_timestamps[utype][u] <= self.timestamp_threshold)
                if vtype in self.node_timestamps:
                    vmask = self.node_nat_mask[vtype][v]
                    vmask |= (self.node_timestamps[vtype][v] <= self.timestamp_threshold)
                    mask = vmask if mask is None else (mask & vmask)

                if mask is not None:
                    u = u[mask]
                    v = v[mask]
                    eid = eid[mask]
                filtered_edges[(utype, etype, vtype)] = (u, v)
                filtered_edge_ids[(utype, etype, vtype)] = eid

            new_frontier = dgl.heterograph(
                filtered_edges,
                num_nodes_dict={
                    ntype: frontier.num_nodes(ntype) for ntype in frontier.ntypes
                }
            )
            for k, v in filtered_edge_ids.items():
                new_frontier.edges[k].data[dgl.EID] = v

            eid = new_frontier.edata[dgl.EID]
            block = dgl.to_block(new_frontier, seed_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

class GlobalTemporalShaDowKHopSampler(Sampler):
    def __init__(
        self,
        fanouts,
        node_timestamps : Dict[str, np.ndarray],
        timestamp_threshold : int,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_edge_feats=None,
        output_device=None,
    ):
        super().__init__()
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        # NaT converts to FLOAT_MIN in numpy so I compute a separate mask for NaT.
        self.node_nat_mask = {
            k: torch.from_numpy(np.isnan(v))
            for k, v in node_timestamps.items()
        }
        self.node_timestamps = {
            k: torch.from_numpy(v.astype('float64'))
            for k, v in node_timestamps.items()
        }
        self.timestamp_threshold = timestamp_threshold
        self.prefetch_node_feats = prefetch_node_feats
        self.prefetch_edge_feats = prefetch_edge_feats
        self.output_device = output_device

    def sample(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )

            filtered_edges = {}
            filtered_edge_ids = {}
            for utype, etype, vtype in frontier.canonical_etypes:
                u, v, eid = frontier.edges(etype=(utype, etype, vtype), form='all')
                mask = None
                if utype in self.node_timestamps:
                    mask = self.node_nat_mask[utype][u]
                    mask |= (self.node_timestamps[utype][u] <= self.timestamp_threshold)
                if vtype in self.node_timestamps:
                    vmask = self.node_nat_mask[vtype][v]
                    vmask |= (self.node_timestamps[vtype][v] <= self.timestamp_threshold)
                    mask = vmask if mask is None else (mask & vmask)

                if mask is not None:
                    u = u[mask]
                    v = v[mask]
                    eid = eid[mask]
                filtered_edges[(utype, etype, vtype)] = (u, v)
                filtered_edge_ids[(utype, etype, vtype)] = eid

            new_frontier = dgl.heterograph(
                filtered_edges,
                num_nodes_dict={
                    ntype: frontier.num_nodes(ntype) for ntype in frontier.ntypes
                }
            )
            for k, v in filtered_edge_ids.items():
                new_frontier.edges[k].data[dgl.EID] = v

            eid = new_frontier.edata[dgl.EID]
            block = dgl.to_block(new_frontier, seed_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        subg = g.subgraph(
            seed_nodes, relabel_nodes=True, output_device=self.output_device
        )
        if exclude_eids is not None:
            subg = EidExcluder(exclude_eids)(subg)

        set_node_lazy_features(subg, self.prefetch_node_feats)
        set_edge_lazy_features(subg, self.prefetch_edge_feats)

        return seed_nodes, output_nodes, [subg]
