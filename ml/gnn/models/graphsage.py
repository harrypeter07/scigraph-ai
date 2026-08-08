"""Heterogeneous GraphSAGE Model Definition.

Defines multi-layer HeteroGraphSAGE architecture using PyTorch / PyTorch Geometric.
"""

import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

logger = logging.getLogger("HeteroGraphSAGE")


if HAS_PYG:
    class HeteroGraphSAGE(nn.Module):
        """Heterogeneous GraphSAGE neural network."""

        def __init__(self, metadata, hidden_channels: int = 64, out_channels: int = 3, num_layers: int = 2):
            super().__init__()
            self.convs = nn.ModuleList()

            # First HeteroConv layer
            conv1 = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            }, aggr='sum')
            self.convs.append(conv1)

            # Additional HeteroConv layers
            for _ in range(num_layers - 1):
                conv = HeteroConv({
                    edge_type: SAGEConv((-1, -1), hidden_channels)
                    for edge_type in metadata[1]
                }, aggr='sum')
                self.convs.append(conv)

            # Paper classification head
            self.paper_classifier = Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}

            # Output predictions for target paper nodes
            if "paper" in x_dict:
                out = self.paper_classifier(x_dict["paper"])
                return out
            return None
else:
    class HeteroGraphSAGE:
        """Dummy fallback class when PyG is not installed."""
        def __init__(self, *args, **kwargs):
            logger.info("PyG not installed. HeteroGraphSAGE operating in fallback mode.")
