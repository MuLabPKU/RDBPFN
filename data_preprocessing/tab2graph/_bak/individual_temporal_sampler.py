from typing import Dict
import dgl
from dgl.dataloading.base import (
    BlockSampler,
)
import numpy as np
import torch
from collections import defaultdict
from scipy.sparse import coo_matrix

class TorchSparseSamplingGraph:
    def __init__(self, g):
        self.g = g
        self.cscs = {}
        for utype, etype, vtype in g.canonical_etypes:
            u, v = g.edges(etype=(utype, etype, vtype), form="uv", order="srcdst")
            u = u.numpy()
            v = v.numpy()
            data = np.ones(u.shape[0])
            self.cscs[(utype, etype, vtype)] = coo_matrix(
                (data, (u, v)), shape=(g.num_nodes(utype), g.num_nodes(vtype))
            ).tocsc()

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.g, name)

    def slice(self, timestamp, seeds, seeds_timestamp):
        sliced = {}
        bias = {}
        for (utype, etype, vtype), csc in self.cscs.items():
            if vtype in seeds:
                sliced_coo = csc[:, seeds[vtype].numpy()].tocoo()
                row = sliced_coo.row
                col = sliced_coo.col
                idtype = seeds[vtype].numpy().dtype
                if utype in timestamp:
                    neighbor_timestamp = timestamp[utype][
                        torch.from_numpy(row).long()
                    ]
                    seed_broadcast_timestamp = seeds_timestamp[vtype][
                        torch.from_numpy(col).long()
                    ]
                    bias[(utype, etype, vtype)] = (
                        neighbor_timestamp <= seed_broadcast_timestamp
                    )
                    # Need to cast the dtype to the same as the node ID tensor.
                    # This is because Scipy automatically casts the idtype to int32 by default.
                    sliced[(utype, etype, vtype)] = (row.astype(idtype), col.astype(idtype))
                else:
                    bias[(utype, etype, vtype)] = torch.ones(row.shape[0], dtype=seeds_timestamp[vtype].dtype)
                    sliced[(utype, etype, vtype)] = (row.astype(idtype), col.astype(idtype))
            else:
                sliced[(utype, etype, vtype)] = None
                bias[(utype, etype, vtype)] = None
        
        hetero_dict = {}
        for etype, coo in sliced.items():
            if coo is not None:
                hetero_dict[etype] = (torch.from_numpy(coo[0]), torch.from_numpy(coo[1]))
        g = dgl.heterograph(hetero_dict)
        for etype, b in bias.items():
            if b is not None:
                g.edges[etype].data["bias"] = b
        return g


def _to_block(sampled_subgraph, relabeled_seeds, seeds, seeds_timestamp):
    new_edges = {}
    node_count = defaultdict(int, {k: 0 for k in sampled_subgraph.ntypes})
    inverse_node_ids = defaultdict(list, {k: [] for k in sampled_subgraph.ntypes})
    new_timestamp = defaultdict(list)
    for vtype, ids in seeds.items():
        node_count[vtype] = ids.numel()
        inverse_node_ids[vtype].append(ids)
        new_timestamp[vtype].append(seeds_timestamp[vtype])
    for utype, etype, vtype in sampled_subgraph.canonical_etypes:
        if sampled_subgraph.num_edges(etype=(utype, etype, vtype)) > 0:
            u, v = sampled_subgraph.edges(etype=(utype, etype, vtype), form="uv")
            new_u = node_count[utype] + torch.arange(0, u.numel(), dtype=u.dtype)
            node_count[utype] += u.numel()
            inverse_node_ids[utype].append(u)
            new_timestamp[utype].append(seeds_timestamp[vtype][v])
            new_edges[(utype, etype, vtype)] = (new_u, v)
        else:
            new_edges[(utype, etype, vtype)] = (
                torch.tensor([], dtype=torch.int64),
                torch.tensor([], dtype=torch.int64),
            )
    new_g = dgl.heterograph(new_edges, num_nodes_dict=node_count)
    new_g = dgl.to_block(new_g, relabeled_seeds)

    for ntype, timestamp in new_timestamp.items():
        new_timestamp[ntype] = torch.cat(timestamp)
    for ntype, ids in inverse_node_ids.items():
        new_g.srcnodes[ntype].data[dgl.NID] = torch.cat(ids)
        if ntype in seeds:
            new_g.dstnodes[ntype].data[dgl.NID] = seeds[ntype]

    # Generate seeds for the next layer
    new_seeds = {}
    for ntype in new_g.ntypes:
        new_seeds[ntype] = new_g.srcnodes[ntype].data[dgl.NID]
    return new_g, new_seeds, new_timestamp


class IndividualTemporalNeighborSampler(BlockSampler):
    def __init__(
        self,
        fanouts,
        node_timestamps: Dict[str, np.ndarray],
        train_timestamp_threshold,
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
        self.replace = replace
        self.node_timestamps = {
            k: torch.from_numpy(v.astype("float64")) for k, v in node_timestamps.items()
        }

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        seeds_timestamp = {}
        for ntype, nid in seed_nodes.items():
            seeds_timestamp[ntype] = self.node_timestamps[ntype][nid]

        blocks = []
        for fanout in reversed(self.fanouts):
            # 1. Slicing: get the in-neighbor subgraph of the seed nodes, which could have duplication.
            #   The in_subgraph has the same source node as the original graph, but its destination nodes are rebeled.
            #   Edges in the subgraph have the sampling bias stored in the edge data.
            in_subgraph = g.slice(self.node_timestamps, seed_nodes, seeds_timestamp)
            # 2. Relabel seed nodes to the consecutive range starting from 0, which id
            # the destination node ID range of `in_subgraph`.
            relabeled_seeds = {}
            for ntype, nid in seed_nodes.items():
                relabeled_seeds[ntype] = torch.arange(
                    0,
                    nid.numel(),
                    dtype=seed_nodes[ntype].dtype,
                    device=seed_nodes[ntype].device,
                )
            # 3. Sampling neighbors for each seed node.
            sampled_subgraph = in_subgraph.sample_neighbors(
                relabeled_seeds,
                fanout,
                prob="bias",
                replace=self.replace,
            )
            # 4. Custom to_block.
            #   * It gives all sampled neighbor nodes a new ID starting from the maximum ID of seed nodes.
            #   * It calls `dgl.to_block` to generate a block.
            #   * It properly assigns both source and destination node IDs to the block, which are the IDs
            #     in the original graph.
            block, new_seeds, new_seeds_timestamp = _to_block(
                sampled_subgraph, relabeled_seeds, seed_nodes, seeds_timestamp
            )
            seeds_timestamp = new_seeds_timestamp
            seed_nodes = new_seeds
            blocks.insert(0, block)
        return seed_nodes, output_nodes, blocks
