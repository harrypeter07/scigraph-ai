"""Authentic Heterogeneous GraphSAGE Implementation for SciGraph AI.

Uses PyG HeteroConv + SAGEConv with authentic 2-hop message passing across:
- ('paper', 'cites', 'paper')
- ('paper', 'cited_by', 'paper')
- ('author', 'authored', 'paper')
- ('paper', 'authored_by', 'author')
- ('paper', 'published_in', 'venue')
- ('venue', 'publishes', 'paper')
- ('paper', 'has_topic', 'topic')
- ('topic', 'topic_of', 'paper')

Dual-head output:
1. 3-class classification logits (Low, Medium, High impact tier) with Temperature Scaling calibration.
2. Log-citation regression estimate.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class AuthenticHeteroSAGE(nn.Module):
    """2-hop Heterogeneous GraphSAGE network with multi-relational message passing."""

    def __init__(
        self,
        node_in_channels: Dict[str, int],
        hidden_channels: int = 64,
        out_channels: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Input projection layers to align heterogeneous node feature dimensions
        self.in_proj = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hidden_channels)
            for node_type, in_dim in node_in_channels.items()
        })

        # Conv 1: First Hop of message passing
        self.conv1 = HeteroConv({
            ("paper", "cites", "paper"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("paper", "cited_by", "paper"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("author", "authored", "paper"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("paper", "authored_by", "author"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("paper", "published_in", "venue"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("venue", "publishes", "paper"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("paper", "has_topic", "topic"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("topic", "topic_of", "paper"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
        }, aggr="mean")

        # Conv 2: Second Hop of message passing
        self.conv2 = HeteroConv({
            ("paper", "cites", "paper"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("paper", "cited_by", "paper"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("author", "authored", "paper"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("paper", "authored_by", "author"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("paper", "published_in", "venue"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("venue", "publishes", "paper"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("paper", "has_topic", "topic"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("topic", "topic_of", "paper"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
        }, aggr="mean")

        # Classification & Regression Heads for target 'paper' nodes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

        # Learnable Temperature Scaling parameter for probability calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # 1. Project heterogeneous inputs into common hidden space
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.in_proj:
                h_dict[node_type] = F.relu(self.in_proj[node_type](x))

        # 2. Hop 1 Message Passing
        # Filter edge_index_dict to only include edges supported by conv1
        valid_edges = {e: edge_index_dict[e] for e in self.conv1.convs.keys() if e in edge_index_dict}
        h_dict = self.conv1(h_dict, valid_edges)
        for k in h_dict:
            h_dict[k] = F.dropout(F.relu(h_dict[k]), p=self.dropout, training=self.training)

        # 3. Hop 2 Message Passing
        h_dict = self.conv2(h_dict, valid_edges)
        for k in h_dict:
            h_dict[k] = F.relu(h_dict[k])

        # 4. Target representations for 'paper' nodes
        paper_emb = h_dict["paper"]
        logits = self.classifier(paper_emb)
        calibrated_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        log_citations = self.regressor(paper_emb).squeeze(-1)

        return {
            "logits": logits,
            "calibrated_logits": calibrated_logits,
            "probabilities": F.softmax(calibrated_logits, dim=-1),
            "predicted_log_citations": log_citations,
            "embeddings": paper_emb
        }
