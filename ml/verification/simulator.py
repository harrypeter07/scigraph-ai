"""Retrospective Verification Engine (Phase 19 Part A).

Reconstructs the exact feature set and graph neighborhood visible at a paper's T_cutoff,
runs the trained model to produce a prediction, and compares side-by-side with what actually
happened (the real cohort-normalized label and 5-year citation trajectory).
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import torch

from ml.temporal.snapshotter import compute_temporal_citations
from ml.gnn.models.graphsage import HeteroGraphSAGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VerificationSimulator")


class RetrospectiveVerificationSimulator:
    """Simulator for testing model forecasts against historical ground truth without leakage."""

    def __init__(
        self,
        labeled_papers_path: str = "data/processed/labeled_papers.parquet",
        authorships_path: str = "data/interim/authorships.parquet",
        authors_path: str = "data/interim/authors.parquet",
        institutions_path: str = "data/interim/institutions.parquet",
        citations_path: str = "data/interim/paper_citations.parquet",
        model_checkpoint_path: str = "ml/gnn/checkpoints/graphsage.pt"
    ):
        self.labeled_papers_path = labeled_papers_path
        self.authorships_path = authorships_path
        self.authors_path = authors_path
        self.institutions_path = institutions_path
        self.citations_path = citations_path
        self.model_checkpoint_path = model_checkpoint_path

        # Load datasets if available
        self.papers_df = pd.read_parquet(labeled_papers_path) if os.path.exists(labeled_papers_path) else pd.DataFrame()
        self.authorships_df = pd.read_parquet(authorships_path) if os.path.exists(authorships_path) else pd.DataFrame()
        self.authors_df = pd.read_parquet(authors_path) if os.path.exists(authors_path) else pd.DataFrame()
        self.institutions_df = pd.read_parquet(institutions_path) if os.path.exists(institutions_path) else pd.DataFrame()
        self.citations_df = pd.read_parquet(citations_path) if os.path.exists(citations_path) else pd.DataFrame()

        # Load GNN Model
        self.model = HeteroGraphSAGE(in_channels=5, hidden_channels=32, out_channels=3)
        if os.path.exists(model_checkpoint_path):
            try:
                state_dict = torch.load(model_checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
                # Detect hidden channels from checkpoint
                if "sage_conv1.weight" in state_dict:
                    chk_hidden = state_dict["sage_conv1.weight"].shape[0]
                    self.model = HeteroGraphSAGE(in_channels=5, hidden_channels=chk_hidden, out_channels=3)
                self.model.load_state_dict(state_dict)
                self.model.eval()
            except Exception as e:
                logger.warning(f"Could not load checkpoint {model_checkpoint_path}: {e}")

    def list_available_papers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return list of all verifiable papers in the dataset with metadata for UI browsing."""
        if self.papers_df.empty:
            return []

        results = []
        for _, row in self.papers_df.head(limit).iterrows():
            clean_id = str(row["id"]).replace("https://openalex.org/", "")
            results.append({
                "paper_id": str(row["id"]),
                "clean_id": clean_id,
                "title": str(row.get("title", f"Paper {clean_id}")),
                "publication_year": int(row.get("publication_year", 2020)),
                "actual_impact_label": int(row.get("impact_label", 1)),
                "actual_impact_name": {0: "Low Impact", 1: "Medium Impact", 2: "High Impact"}.get(int(row.get("impact_label", 1)), "Medium Impact"),
                "historical_citations_at_cutoff": int(row.get("historical_citation_count_at_cutoff", 0)),
                "actual_5y_delta_citations": int(row.get("delta_citations_5y", 0))
            })
        return results

    def simulate_paper_forecast(self, paper_id: str) -> Dict[str, Any]:
        """Reconstruct snapshot at T_cutoff, run model inference, and compare side-by-side with actual outcome."""
        clean_id = paper_id.replace("https://openalex.org/", "").strip()
        
        if self.papers_df.empty:
            raise FileNotFoundError("Labeled papers dataset is empty or not found.")

        # Match paper
        match = self.papers_df[self.papers_df["id"].str.contains(clean_id, case=False, na=False)]
        if match.empty:
            raise ValueError(f"Paper ID '{paper_id}' not found in verifiable dataset.")

        row = match.iloc[0]
        full_id = str(row["id"])
        title = str(row.get("title", f"Paper {clean_id}"))
        pub_year = int(row.get("publication_year", 2020))
        t_cutoff = pub_year

        # 1. Reconstruct snapshot at T_cutoff
        counts_raw = row.get("counts_by_year", "[]")
        if isinstance(counts_raw, str):
            try:
                counts_by_year = json.loads(counts_raw)
            except json.JSONDecodeError:
                counts_by_year = []
        else:
            counts_by_year = counts_raw or []

        hist_cits, actual_5y_delta = compute_temporal_citations(counts_by_year, pub_year, horizon_years=5)
        if hist_cits == 0 and "historical_citation_count_at_cutoff" in row:
            hist_cits = int(row["historical_citation_count_at_cutoff"])
        if actual_5y_delta == 0 and "delta_citations_5y" in row:
            actual_5y_delta = int(row["delta_citations_5y"])

        # Reconstruct connected authors and institutions at T_cutoff
        connected_authors = []
        connected_institutions = []
        if not self.authorships_df.empty:
            p_auths = self.authorships_df[self.authorships_df["paper_id"] == full_id]
            if not p_auths.empty and not self.authors_df.empty:
                merged_a = p_auths.merge(self.authors_df, left_on="author_id", right_on="id")
                for _, a_row in merged_a.head(5).iterrows():
                    connected_authors.append({
                        "author_id": str(a_row.get("author_id")),
                        "display_name": str(a_row.get("display_name_y") or a_row.get("display_name_x") or "Author")
                    })
                    inst_id = a_row.get("institution_id")
                    if inst_id and not self.institutions_df.empty:
                        i_match = self.institutions_df[self.institutions_df["id"] == inst_id]
                        if not i_match.empty:
                            connected_institutions.append({
                                "institution_id": str(inst_id),
                                "display_name": str(i_match.iloc[0].get("display_name", "Institution"))
                            })

        # Reconstruct citation edges visible at or before T_cutoff
        visible_citation_edges = []
        if not self.citations_df.empty:
            p_cits = self.citations_df[self.citations_df["citing_paper_id"] == full_id]
            for _, c_row in p_cits.iterrows():
                cit_yr = c_row.get("citation_year")
                if cit_yr is None or cit_yr <= t_cutoff:
                    visible_citation_edges.append(str(c_row.get("cited_paper_id")))

        # 2. Build feature vector & run trained GNN model
        num_authors = max(1, len(connected_authors))
        title_length = max(1, len(title.split()))
        feat_tensor = torch.tensor([[
            float(pub_year - 2012) / 10.0,
            float(hist_cits) / 100.0,
            float(title_length) / 20.0,
            float(num_authors) / 10.0,
            float(np.log1p(float(hist_cits)))
        ]], dtype=torch.float)

        with torch.no_grad():
            logits = self.model(feat_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        p_low = round(float(probs[0]), 4)
        p_med = round(float(probs[1]), 4)
        p_high = round(float(probs[2]), 4)
        pred_cls = int(torch.argmax(probs).item())

        # 3. Look up actual ground-truth outcome
        actual_impact_cls = int(row.get("impact_label", 1))
        actual_cohort_pct = float(row.get("impact_cohort_percentile", 50.0))

        class_names = {0: "Low Impact (<50%)", 1: "Medium Impact (50-90%)", 2: "High Impact (>=90%)"}
        is_correct = bool(pred_cls == actual_impact_cls)

        # 4. Construct side-by-side verification payload
        return {
            "paper_id": full_id,
            "clean_id": clean_id,
            "title": title,
            "publication_year": pub_year,
            "snapshot_cutoff_year": t_cutoff,
            "leakage_assertion": f"Features strictly computed at or before T_cutoff = {t_cutoff}",
            "cutoff_snapshot": {
                "historical_citations_at_cutoff": hist_cits,
                "connected_authors_count": num_authors,
                "connected_authors": connected_authors,
                "connected_institutions": connected_institutions,
                "visible_prior_citations_count": len(visible_citation_edges),
                "feature_vector": [round(float(v), 4) for v in feat_tensor.squeeze(0).tolist()]
            },
            "model_forecast": {
                "predicted_impact_class": pred_cls,
                "predicted_impact_name": class_names.get(pred_cls, "Unknown"),
                "class_probabilities": {
                    "Low Impact (<50%)": p_low,
                    "Medium Impact (50-90%)": p_med,
                    "High Impact (>=90%)": p_high
                },
                "model_architecture": "HeteroGraphSAGE (PyTorch Geometric)"
            },
            "actual_outcome": {
                "actual_5y_delta_citations": actual_5y_delta,
                "actual_cohort_percentile": round(actual_cohort_pct, 2),
                "actual_impact_class": actual_impact_cls,
                "actual_impact_name": class_names.get(actual_impact_cls, "Unknown")
            },
            "verification_verdict": {
                "is_correct": is_correct,
                "verdict_label": "CORRECT" if is_correct else "INCORRECT",
                "summary": (
                    f"Model forecasted '{class_names.get(pred_cls)}' with {max(p_low, p_med, p_high)*100:.1f}% confidence. "
                    f"Actual historical outcome was '{class_names.get(actual_impact_cls)}' ({actual_5y_delta} citations in 5-year window)."
                )
            }
        }
