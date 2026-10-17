"""Train AuthenticHeteroSAGE on the updated 47-dim feature graph and update checkpoint."""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import settings
from ml.gnn.graph_builder import build_heterogeneous_graph
from ml.gnn.models.hetero_gnn import AuthenticHeteroSAGE


def main():
    print("Building heterogeneous graph with central 47-dim features...")
    graph = build_heterogeneous_graph()
    paper_dim = graph["paper"].x.shape[1]
    print(f"Paper node feature dimension: {paper_dim}")
    assert paper_dim == 51, f"Expected 51 features, got {paper_dim}"

    node_in_channels = {
        "paper": paper_dim,
        "author": graph["author"].x.shape[1],
        "institution": graph["institution"].x.shape[1],
        "venue": graph["venue"].x.shape[1],
        "topic": graph["topic"].x.shape[1],
    }

    model = AuthenticHeteroSAGE(
        node_in_channels=node_in_channels,
        hidden_channels=64,
        out_channels=3,
        dropout=0.2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_reg = torch.nn.MSELoss()

    model.train()
    x_dict = graph.x_dict
    edge_index_dict = graph.edge_index_dict
    train_mask = graph["paper"].train_mask
    y_cls = graph["paper"].y[train_mask]
    y_reg = graph["paper"].y_reg[train_mask]

    print("Training AuthenticHeteroSAGE for 80 epochs...")
    for epoch in range(80):
        optimizer.zero_grad()
        out = model(x_dict, edge_index_dict)
        loss_cls = criterion_cls(out["logits"][train_mask], y_cls)
        loss_reg = criterion_reg(out["predicted_log_citations"][train_mask], y_reg)
        loss = loss_cls + 0.2 * loss_reg
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:2d}/80 - Loss: {loss.item():.4f} (cls: {loss_cls.item():.4f}, reg: {loss_reg.item():.4f})")

    ckpt_path = settings.CHECKPOINTS_DIR / "hetero_graphsage.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Updated checkpoint saved to: {ckpt_path}")
    print(f"in_proj.paper.weight shape: {model.in_proj['paper'].weight.shape}")


if __name__ == "__main__":
    main()
