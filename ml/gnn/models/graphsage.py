"""Heterogeneous GraphSAGE Model Definition.

Defines multi-layer HeteroGraphSAGE architecture using PyTorch / PyTorch Geometric.
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


if HAS_PYG:
    class HeteroGraphSAGE(nn.Module):
        """Heterogeneous GraphSAGE neural network."""

        def __init__(self, metadata=None, in_channels: int = 5, hidden_channels: int = 32, out_channels: int = 3, num_layers: int = 2):
            super().__init__()
            self.convs = nn.ModuleList()
            if metadata and len(metadata) > 1 and len(metadata[1]) > 0:
                for _ in range(num_layers):
                    conv = HeteroConv({
                        edge_type: SAGEConv((-1, -1), hidden_channels)
                        for edge_type in metadata[1]
                    }, aggr='sum')
                    self.convs.append(conv)
            self.paper_classifier = nn.Linear(hidden_channels if len(self.convs) > 0 else in_channels, out_channels)

        def forward(self, x_dict, edge_index_dict=None):
            if isinstance(x_dict, dict):
                for conv in self.convs:
                    x_dict = conv(x_dict, edge_index_dict)
                    x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                if "paper" in x_dict:
                    return self.paper_classifier(x_dict["paper"])
            # Fallback linear forward pass
            return self.paper_classifier(x_dict)
else:
    class HeteroGraphSAGE(nn.Module):
        """Fallback PyTorch HeteroGraphSAGE module when PyG is not present."""

        def __init__(self, in_channels: int = 5, hidden_channels: int = 32, out_channels: int = 3, *args, **kwargs):
            super().__init__()
            logger.info("PyG not installed. HeteroGraphSAGE operating in PyTorch fallback mode.")
            self.fc1 = nn.Linear(in_channels, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, out_channels)

        def forward(self, x, edge_index_dict=None):
            if isinstance(x, dict):
                x = x.get("paper", list(x.values())[0])
            h = F.relu(self.fc1(x))
            return self.fc2(h)
