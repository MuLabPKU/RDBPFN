from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, List

import torch
from torch import nn


class RowEdgeType(IntEnum):
    """Types of edges in the row-level relational graph."""

    PARENT_TO_CHILD = 0
    CHILD_TO_PARENT = 1


@dataclass
class EdgeBatch:
    """Describes a batch of edges between two tables."""

    src_table: str
    dst_table: str
    src_indices: torch.Tensor
    dst_indices: torch.Tensor


@dataclass
class RowGraphData:
    """Container for the row graph information without padding."""

    node_features: Dict[str, torch.Tensor]
    edges: Dict[RowEdgeType, List[EdgeBatch]]


class RowGraphBuilder:
    """Builds a row-level graph from cached SCM outputs."""

    def __init__(self, table_generators: Dict[str, object], device: str = "cpu"):
        self.table_generators = table_generators
        self.device = device

    def build(self) -> Optional[RowGraphData]:
        node_features: Dict[str, torch.Tensor] = {}
        edges: Dict[RowEdgeType, List[EdgeBatch]] = {
            edge_type: [] for edge_type in RowEdgeType
        }

        # Collect embeddings per table
        for table_name, generator in self.table_generators.items():
            row_emb = getattr(generator, "row_embeddings", None)
            if row_emb is None or row_emb.shape[0] == 0:
                continue
            node_features[table_name] = row_emb.to(self.device)

        if not node_features:
            return None

        # Build per-edge-type batches
        for child_table, generator in self.table_generators.items():
            fk_ids = getattr(generator, "pending_fk_ids", None)
            parent_tables = getattr(generator, "pending_parent_tables", None) or []
            if fk_ids is None or len(parent_tables) == 0:
                continue
            if child_table not in node_features:
                continue

            child_rows = torch.arange(
                node_features[child_table].shape[0],
                device=self.device,
                dtype=torch.long,
            )

            for parent_idx, parent_table in enumerate(parent_tables):
                if parent_table not in node_features:
                    continue
                parent_rows = fk_ids[:, parent_idx].to(
                    device=self.device, dtype=torch.long
                )

                edges[RowEdgeType.PARENT_TO_CHILD].append(
                    EdgeBatch(
                        src_table=parent_table,
                        dst_table=child_table,
                        src_indices=parent_rows,
                        dst_indices=child_rows,
                    )
                )
                edges[RowEdgeType.CHILD_TO_PARENT].append(
                    EdgeBatch(
                        src_table=child_table,
                        dst_table=parent_table,
                        src_indices=child_rows,
                        dst_indices=parent_rows,
                    )
                )

        return RowGraphData(node_features=node_features, edges=edges)


class RelationalGraphSAGELayer(nn.Module):
    """GraphSAGE layer operating on per-table feature dictionaries."""

    def __init__(
        self,
        activation: nn.Module = nn.SELU,
        update_keep_prob: float = 0.5,
        feature_keep_prob: float = 0.3,
    ):
        super().__init__()
        self.edge_transforms = nn.ModuleDict()
        self.self_transforms = nn.ModuleDict()
        self.layer_norms = nn.ModuleDict()
        self.residual_projs = nn.ModuleDict()
        self.activation = activation()
        self.update_keep_prob = update_keep_prob
        self.feature_keep_prob = feature_keep_prob
        self.feature_masks = nn.ParameterDict()

    def _edge_key(self, edge_type: RowEdgeType, src: str, dst: str) -> str:
        return f"{edge_type.name}|{src}|{dst}"

    def _get_edge_transform(
        self,
        key: str,
        in_dim: int,
        out_dim: int,
    ) -> nn.Linear:
        if key not in self.edge_transforms:
            self.edge_transforms[key] = nn.Linear(in_dim, out_dim)
        return self.edge_transforms[key]

    def _ensure_table_modules(self, table_name: str, dim: int) -> None:
        if table_name not in self.self_transforms:
            self.self_transforms[table_name] = nn.Linear(dim, dim)
            self.layer_norms[table_name] = nn.LayerNorm(dim)
            self.residual_projs[table_name] = nn.Identity()

    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edges: Dict[RowEdgeType, List[EdgeBatch]],
    ) -> Dict[str, torch.Tensor]:
        aggregated = {
            table: torch.zeros_like(feat) for table, feat in node_features.items()
        }
        degrees = {
            table: torch.zeros(feat.shape[0], device=feat.device)
            for table, feat in node_features.items()
        }

        for edge_type, batches in edges.items():
            if not batches:
                continue
            for batch in batches:
                if (
                    batch.src_table not in node_features
                    or batch.dst_table not in node_features
                ):
                    continue
                src_feat = node_features[batch.src_table]
                dst_feat = node_features[batch.dst_table]
                if src_feat.shape[0] == 0 or dst_feat.shape[0] == 0:
                    continue

                key = self._edge_key(edge_type, batch.src_table, batch.dst_table)
                transform = self._get_edge_transform(
                    key, src_feat.shape[1], dst_feat.shape[1]
                )
                messages = transform(src_feat[batch.src_indices])
                update_mask = torch.bernoulli(
                    torch.full_like(messages[:, :1], self.update_keep_prob)
                )
                messages = messages * update_mask
                aggregated[batch.dst_table].index_add_(0, batch.dst_indices, messages)
                degrees[batch.dst_table].index_add_(
                    0,
                    batch.dst_indices,
                    update_mask.squeeze(-1),
                )

        updated_features: Dict[str, torch.Tensor] = {}
        for table_name, feats in node_features.items():
            self._ensure_table_modules(table_name, feats.shape[1])
            degree = degrees[table_name].clamp(min=1.0).unsqueeze(-1)
            agg = aggregated[table_name] / degree
            mask_key = f"mask|{table_name}"
            if mask_key not in self.feature_masks:
                base_mask = torch.bernoulli(
                    torch.full(
                        (feats.shape[1],), self.feature_keep_prob, device=feats.device
                    )
                )
                if base_mask.sum() == 0:
                    base_mask[
                        torch.randint(0, feats.shape[1], (1,), device=feats.device)
                    ] = 1
                self.feature_masks[mask_key] = nn.Parameter(
                    base_mask, requires_grad=False
                )
            feature_mask = self.feature_masks[mask_key].to(feats.device)
            mask_bool = feature_mask > 0.5
            if not mask_bool.any():
                updated_features[table_name] = feats
                continue

            agg = agg * feature_mask

            combined = self.self_transforms[table_name](feats) + agg
            combined = self.layer_norms[table_name](combined)
            combined = combined + self.residual_projs[table_name](feats)
            combined = self.activation(combined)

            result = feats.clone()
            result[:, mask_bool] = combined[:, mask_bool]
            updated_features[table_name] = result

        return updated_features


class RelationalGraphSAGE(nn.Module):
    """Stack of relational GraphSAGE layers operating on dict features."""

    def __init__(self, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(
            [RelationalGraphSAGELayer() for _ in range(max(1, num_layers))]
        )

    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edges: Dict[RowEdgeType, List[EdgeBatch]],
    ) -> Dict[str, torch.Tensor]:
        h = node_features
        for layer in self.layers:
            h = layer(h, edges)
        return h


class RowGNNRunner:
    """Helper that owns the GNN module and runs refinement passes."""

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 1,
        num_steps: int = 1,
        device: str = "cpu",
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_steps
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but no CUDA device is available.")
        self.model: Optional[RelationalGraphSAGE] = None
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = RelationalGraphSAGE(num_layers=self.num_layers).to(self.device)

    def run(self, graph: Optional[RowGraphData]) -> Optional[Dict[str, torch.Tensor]]:
        if graph is None:
            return None

        features = {k: v.to(self.device) for k, v in graph.node_features.items()}
        edges = {
            edge_type: batches
            for edge_type, batches in graph.edges.items()
        }

        self._ensure_model()
        self.model.eval()

        with torch.no_grad():
            h = features
            for _ in range(max(1, self.num_steps)):
                h = self.model(h, edges)

        return {k: v.cpu() for k, v in h.items()}
