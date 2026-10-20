"""One-hop relational GraphSAGE on real citation neighbors.

Incoming and outgoing citations have separate learned message functions. Every
neighbor feature is computed at the TARGET cutoff, not the neighbor's cutoff.
No pretrained legacy graph checkpoint is reused with this new architecture.
"""
import hashlib
import numpy as np
from ml.features.forecast import clean_id, field, number


class CitationNeighborhoods:
    neighbor_dim = 8

    def __init__(self, records, edges, max_neighbors=16):
        self.max_neighbors = max_neighbors
        self.records = {clean_id(r["work_id"]): {key: r.get(key) for key in
                       ["work_id", "title", "publication_year", "primary_field_name", "num_authors",
                        "referenced_works_count", "counts_by_year"]} for r in records
                        if np.isfinite(number(r.get("publication_year"), np.nan))}
        self.incoming, self.outgoing = {}, {}
        for source, target in edges:
            source, target = clean_id(source), clean_id(target)
            if source != target and source in self.records and target in self.records:
                self.outgoing.setdefault(source, set()).add(target)
                self.incoming.setdefault(target, set()).add(source)

    def build(self, row):
        wid, year = clean_id(row.get("work_id") or row.get("id")), int(row["publication_year"])
        cutoff = year + 2
        values = np.zeros((2, self.max_neighbors, self.neighbor_dim), dtype=np.float32)
        mask = np.zeros((2, self.max_neighbors), dtype=np.float32)
        context = []
        for relation, lookup in enumerate([self.incoming, self.outgoing]):
            ids = sorted(lookup.get(wid, []), key=lambda value: hashlib.sha256(value.encode()).digest())
            valid = [i for i in ids if int(self.records[i]["publication_year"]) <= (cutoff if relation == 0 else year)]
            for slot, nid in enumerate(valid[:self.max_neighbors]):
                neighbor = self.records[nid]
                annual = {int(c["year"]): int(c["cited_by_count"]) for c in neighbor.get("counts_by_year") or []}
                total = sum(count for y, count in annual.items() if y <= cutoff)
                values[relation, slot] = [np.log1p(max(0, total)),
                    *[np.log1p(max(0, annual.get(cutoff-offset, 0))) for offset in [2, 1, 0]],
                    min(50, cutoff-int(neighbor["publication_year"]))/10,
                    float(field(row) == field(neighbor)),
                    np.log1p(max(0, number(neighbor.get("referenced_works_count")))),
                    np.log1p(max(0, number(neighbor.get("num_authors"))))]
                mask[relation, slot] = 1
                context.append({"work_id": nid, "relation": "cited_by" if relation == 0 else "cites",
                                "title": neighbor.get("title") or nid,
                                "url": "https://openalex.org/"+nid,
                                "publication_year": int(neighbor["publication_year"]), "feature_cutoff": cutoff})
        return values, mask, context


def make_model(root_dim, hidden=32):
    import torch
    from torch import nn
    class CitationSAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.root = nn.Linear(root_dim, hidden)
            self.messages = nn.ModuleList([nn.Linear(8, hidden) for _ in range(2)])
            self.head = nn.Sequential(nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(.2), nn.Linear(hidden, 3))

        def forward(self, roots, neighbors, mask):
            h = [torch.relu(self.root(roots))]
            for r, layer in enumerate(self.messages):
                messages = torch.relu(layer(neighbors[:, r])) * mask[:, r, :, None]
                h.append(messages.sum(dim=1) / mask[:, r].sum(dim=1, keepdim=True).clamp(min=1))
            return self.head(torch.cat(h, dim=1))
    return CitationSAGE()


class CitationPredictor:
    """CPU serving for the exact validation-selected graph ensemble."""
    def __init__(self, graph_bundle, tabular_bundle):
        import torch
        torch.set_num_threads(4)
        self.graph, self.tabular = graph_bundle, tabular_bundle
        self.models = []
        for state in graph_bundle["states"]:
            model = make_model(graph_bundle["root_dim"])
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

    def _graph_probabilities(self, records, removed_relation=None):
        import torch
        features = self.tabular["features"].transform(records)
        roots = self.graph["root_scaler"].transform(features).astype(np.float32)
        neighborhoods = [self.graph["neighborhoods"].build(row) for row in records]
        neighbors = np.stack([item[0] for item in neighborhoods])
        mask = np.stack([item[1] for item in neighborhoods])
        if self.graph["report"].get("without_edges_control"):
            mask[:] = 0
        if removed_relation == "all":
            mask[:] = 0
        elif removed_relation is not None:
            mask[:, int(removed_relation)] = 0
        neighbors = self.graph["neighbor_scaler"].transform(neighbors.reshape(-1, 8)).reshape(neighbors.shape).astype(np.float32)
        with torch.no_grad():
            p = np.mean([torch.softmax(model(torch.from_numpy(roots), torch.from_numpy(neighbors), torch.from_numpy(mask)), dim=1).numpy()
                         for model in self.models], axis=0)
        return p

    def predict(self, records, removed_relation=None):
        from scipy.special import softmax
        from ml.features.forecast import probabilities
        weight = self.graph["graph_weight"]
        if weight == 0:
            return probabilities(self.tabular, records)
        p = self._graph_probabilities(records, removed_relation)
        combined = (1-weight)*probabilities(self.tabular, records)+weight*p
        return softmax(np.log(np.clip(combined, 1e-12, 1))/self.graph["temperature"], axis=1)

    def evidence(self, record):
        original = self.predict([record])[0]
        predicted_class = int(original.argmax())
        attributions = []
        for relation, name in [(0, "Incoming citation messages"), (1, "Outgoing reference messages")]:
            removed = self.predict([record], removed_relation=relation)[0]
            delta = float(original[predicted_class]-removed[predicted_class])
            attributions.append(dict(feature_name=name, raw_value=None, importance_score=abs(delta),
                                     delta_probability=abs(delta), delta_percentage_points=abs(delta)*100,
                                     direction="positive" if delta>=0 else "negative",
                                     provenance="served_graph_model | relation_removal_sensitivity",
                                     explained_class=predicted_class))
        total = max(sum(item["importance_score"] for item in attributions), 1e-12)
        for item in attributions:
            item["importance_score"] /= total
        without_graph = self.predict([record], removed_relation="all")[0]
        experimental = self._graph_probabilities([record])[0]
        experimental_removed = self._graph_probabilities([record], removed_relation="all")[0]
        experimental_class = int(experimental.argmax())
        return dict(neighbors=self.graph["neighborhoods"].build(record)[2],
                    uncalibrated_experimental_gnn_probabilities=experimental.tolist(),
                    experimental_gnn_message_delta_percentage_points=float((experimental[experimental_class]-experimental_removed[experimental_class])*100),
                    attributions=attributions, without_graph_probabilities=without_graph.tolist(),
                    predicted_class_delta_percentage_points=float((original[predicted_class]-without_graph[predicted_class])*100))
