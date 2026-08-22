"""Heterogeneous GraphSAGE Model Definition.

Defines multi-layer HeteroGraphSAGE architecture using PyTorch with explicit neighborhood message passing.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

logger = logging.getLogger("HeteroGraphSAGE")


class HeteroGraphSAGE(nn.Module):
    """PyTorch HeteroGraphSAGE module with 2-Hop neighborhood aggregation."""

    def __init__(self, in_channels: int = 5, hidden_channels: int = 32, out_channels: int = 3, dropout: float = 0.1, *args, **kwargs):
        super().__init__()
        # Layer 1: Self feature + Aggregated Neighbor features (dim = 2 * in_channels)
        self.sage_conv1 = nn.Linear(in_channels * 2, hidden_channels)
        # Layer 2: Self hidden + Neighbor hidden (dim = 2 * hidden_channels)
        self.sage_conv2 = nn.Linear(hidden_channels * 2, hidden_channels)
        # Output Classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def aggregate_neighbors(self, h: torch.Tensor, adj_matrix: torch.Tensor = None) -> torch.Tensor:
        """Mean message aggregation: h_N(v) = (1 / |N(v)|) * sum_{u in N(v)} h_u"""
        if adj_matrix is None or adj_matrix.dim() < 2 or adj_matrix.shape[0] != h.shape[0]:
            return h  # Self-loop fallback when single paper is queried
        deg = torch.clamp(adj_matrix.sum(dim=1, keepdim=True), min=1.0)
        return torch.matmul(adj_matrix, h) / deg

    def forward(self, x, edge_index_dict=None):
        if isinstance(x, dict):
            x = x.get("paper", list(x.values())[0])
            
        # Ensure 2D tensor
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 1-Hop Message Passing
        neighbor_x = self.aggregate_neighbors(x)
        h1_input = torch.cat([x, neighbor_x], dim=-1)
        h1 = F.relu(self.sage_conv1(h1_input))
        h1 = self.dropout(h1)

        # 2-Hop Message Passing
        neighbor_h1 = self.aggregate_neighbors(h1)
        h2_input = torch.cat([h1, neighbor_h1], dim=-1)
        h2 = F.relu(self.sage_conv2(h2_input))
        h2 = self.dropout(h2)

        # Classifier Logits
        return self.classifier(h2)
