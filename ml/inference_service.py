"""Authentic Model Inference Service for SciGraph AI.

Features:
- Loads trained PyG AuthenticHeteroSAGE from ml/gnn/checkpoints/hetero_graphsage.pt
- Loud failure on startup if checkpoint is missing or corrupted (Rule R2)
- Zero fabricated features, zero random fallbacks, zero hash() (Rule R1)
- Strict temporal cutoff (T_cutoff = pub_year + 2) (Rule R3)
- Calibrated class probabilities via learned temperature parameter
- Ground-truth backtest comparison for papers published <= 2020
- Transparent feature attribution with explicit provenance
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from api.config import settings
from ml.gnn.models.hetero_gnn import AuthenticHeteroSAGE
from ml.scientometrics.venue_resolver import venue_resolver
from ml.scientometrics.access_economics import access_service
from ml.scientometrics.lineage_engine import lineage_engine
from ml.features.builder import (
    build_paper_features,
    get_text_encoder,
    DEFAULT_TEXT_ENCODER_PATH,
    TOTAL_FEATURE_DIM,
    FEATURE_NAMES,
)
from ml.gnn.ego_graph import ego_graph_builder
from ml.explain.attribution import compute_local_occlusion_attributions

logger = logging.getLogger("scigraph.inference")


class ModelInferenceService:
    """Production ML inference engine for citation impact trajectory."""

    def __init__(self):
        self.model: Optional[AuthenticHeteroSAGE] = None
        self.primary_payload: Optional[Dict[str, Any]] = None
        self.primary_model = None
        self.primary_scaler = None
        self.primary_temperature: float = 1.0
        self.model_card: Optional[Dict[str, Any]] = None
        self.df_works: Optional[pd.DataFrame] = None
        self.df_counts: Optional[pd.DataFrame] = None
        self.cohort_reference: Dict[str, Any] = {}
        self.node_in_channels = {
            "paper": TOTAL_FEATURE_DIM,
            "author": 8,
            "institution": 8,
            "venue": 8,
            "topic": 8
        }
        self.cohort_thresholds: Dict[int, Dict[str, float]] = {}

    def load(self):
        """Loads model weights and reference metadata with strict loud failures."""
        ckpt_path = settings.CHECKPOINTS_DIR / "hetero_graphsage.pt"
        card_path = settings.CHECKPOINTS_DIR / "MODEL_CARD.json"

        if not ckpt_path.exists():
            raise RuntimeError(
                f"CRITICAL: Model checkpoint missing at '{ckpt_path}'. "
                "Per Rule R2, server will not boot with uninitialized weights."
            )

        if not card_path.exists():
            raise RuntimeError(
                f"CRITICAL: Model card missing at '{card_path}'. "
                "Per Rule R2, server requires valid model metadata."
            )

        # 0. Load Primary Served Model: M1_LogisticRegression (Best empirical performer)
        primary_ckpt = settings.CHECKPOINTS_DIR / "primary_model.joblib"
        if not primary_ckpt.exists():
            raise RuntimeError(
                f"CRITICAL: Primary model checkpoint missing at '{primary_ckpt}'. "
                "Run 'python ml/evaluation/fit_primary_model.py' to fit and calibrate it."
            )
        import joblib
        self.primary_payload = joblib.load(primary_ckpt)
        self.primary_model = self.primary_payload["model"]
        self.primary_scaler = self.primary_payload["scaler"]
        self.primary_temperature = float(self.primary_payload.get("temperature", 1.0))
        logger.info(f"Successfully loaded primary model (M1_LogisticRegression) with T={self.primary_temperature:.4f}")

        # 1. Initialize HeteroSAGE architecture (for graph relational comparative analysis)
        self.model = AuthenticHeteroSAGE(
            node_in_channels=self.node_in_channels,
            hidden_channels=64,
            out_channels=3,
            dropout=0.0
        )

        try:
            state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info("Successfully loaded AuthenticHeteroSAGE checkpoint.")
        except Exception as exc:
            raise RuntimeError(f"CRITICAL: Failed to load model weights from '{ckpt_path}': {exc}")

        # 2. Load Model Card
        with open(card_path, "r", encoding="utf-8") as f:
            self.model_card = json.load(f)

        # 3. Load persistent text encoder (strictly fitted on TRAIN titles only)
        if not DEFAULT_TEXT_ENCODER_PATH.exists():
            raise RuntimeError(
                f"CRITICAL: Text encoder checkpoint missing at '{DEFAULT_TEXT_ENCODER_PATH}'. "
                "Per WP1, server requires valid text encoder."
            )
        get_text_encoder(DEFAULT_TEXT_ENCODER_PATH)
        logger.info("Successfully loaded persistent text encoder checkpoint.")

        # 4. Load reference tables
        works_path = settings.DATA_DIR / "works.parquet"
        counts_path = settings.DATA_DIR / "counts_by_year.parquet"

        if not works_path.exists():
            raise RuntimeError(f"CRITICAL: Reference dataset missing at '{works_path}'.")
        self.df_works = pd.read_parquet(works_path)
        logger.info(f"Loaded {len(self.df_works):,} reference works for corpus lookups.")

        # Compute empirical cohort percentiles (50th, 90th) by pub_year for backtesting
        for year, group in self.df_works.groupby("publication_year"):
            fut_cits = group["future_citations_5y"].dropna().values
            if len(fut_cits) > 0:
                self.cohort_thresholds[int(year)] = {
                    "p50": float(np.percentile(fut_cits, 50)),
                    "p90": float(np.percentile(fut_cits, 90)),
                }

        if not counts_path.exists():
            raise RuntimeError(f"CRITICAL: Reference yearly citation records missing at '{counts_path}'.")
        self.df_counts = pd.read_parquet(counts_path)
        logger.info(f"Loaded {len(self.df_counts):,} citation yearly count records.")

        # 5. Load empirical cohort thresholds reference (single source of truth: label_thresholds.json)
        cohort_ref_path = settings.DATA_DIR / "reference" / "label_thresholds.json"
        if cohort_ref_path.exists():
            with open(cohort_ref_path, "r", encoding="utf-8") as f:
                self.cohort_reference = json.load(f)
            logger.info("Successfully loaded empirical cohort reference thresholds.")
        else:
            self.cohort_reference = {}

    def get_cohort_threshold(self, field_name: Optional[str], pub_year: int) -> Dict[str, Any]:
        """Reads empirical p50/p90 from cohort reference with strict hierarchical fallback."""
        if not self.cohort_reference:
            return {"cohort_level": "global", "p50": 330.0, "p90": 882.0}

        by_field_year = self.cohort_reference.get("by_field_year", {})
        key = f"{field_name}::{pub_year}"
        if key in by_field_year:
            info = by_field_year[key]
            return {
                "cohort_level": info.get("cohort_level", "field_year"),
                "p50": float(info["p50"]),
                "p90": float(info["p90"])
            }

        by_field = self.cohort_reference.get("by_field", {})
        if field_name and str(field_name) in by_field:
            info = by_field[str(field_name)]
            return {
                "cohort_level": "field",
                "p50": float(info["p50"]),
                "p90": float(info["p90"])
            }

        glob = self.cohort_reference.get("global", {"p50": 330.0, "p90": 882.0})
        return {
            "cohort_level": "global",
            "p50": float(glob["p50"]),
            "p90": float(glob["p90"])
        }

    def predict(
        self,
        raw_work: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs full end-to-end scientometric inference and ground truth backtest."""
        if self.model is None:
            raise RuntimeError("ModelInferenceService is not initialized. Call load() first.")

        work_id = str(raw_work.get("id") or raw_work.get("work_id") or "").split("/")[-1]
        title = raw_work.get("title") or "Untitled Academic Work"

        pub_year_raw = raw_work.get("publication_year")
        if pub_year_raw is None or (isinstance(pub_year_raw, float) and np.isnan(pub_year_raw)):
            raise ValueError("Missing required publication_year field (Rule F2).")
        pub_year = int(pub_year_raw)
        cutoff_year = pub_year + settings.CUTOFF_OFFSET_YEARS

        # 1. Historical citations at cutoff
        counts_by_year = raw_work.get("counts_by_year", [])
        cits_at_cutoff = sum(
            c.get("cited_by_count", 0)
            for c in counts_by_year
            if c.get("year", 9999) <= cutoff_year
        )
        total_citations = int(raw_work.get("cited_by_count", 0))

        # 2. Extract author information
        authorships = raw_work.get("authorships", [])
        author_names = []
        for a in authorships:
            name = (a.get("author") or {}).get("display_name")
            if name:
                author_names.append(name)
        num_authors = max(1, len(author_names))

        # 3. Resolve Venue Ranking (Phase 2 Authenticity: CORE / SCImago)
        primary_loc = raw_work.get("primary_location") or {}
        source = primary_loc.get("source") or {}
        source_name = source.get("display_name") or ""
        issn_l = source.get("issn_l") or ((source.get("issn") or [""])[0] if isinstance(source.get("issn"), list) else None)
        
        venue_info = venue_resolver.resolve(
            issn_l=issn_l,
            venue_name=source_name,
            pub_year=pub_year
        )

        is_top_tier = False
        venue_sjr = 0.0
        if venue_info:
            venue_sjr = venue_info.sjr or 0.0
            if venue_info.quartile == "Q1" or venue_info.core_rank in ["A*", "A"]:
                is_top_tier = True

        # 4. Extract Topic & Field
        topic_obj = raw_work.get("primary_topic") or {}
        primary_topic = topic_obj.get("display_name") or raw_work.get("primary_topic_name")
        primary_field = (topic_obj.get("field") or {}).get("display_name") or raw_work.get("primary_field_name")

        # 5. Access Economics
        open_access = raw_work.get("open_access") or {}
        econ_info = access_service.compute(open_access, primary_loc, field_name=primary_field)

        # 6. Lineage Engine (constrained to cutoff_year)
        lineage_info = lineage_engine.compute(raw_work, cutoff_year=cutoff_year)

        # 6. Build Cutoff Feature Tensor via Universal Central Builder
        ref_count = len(raw_work.get("referenced_works", [])) if "referenced_works" in raw_work else int(raw_work.get("referenced_works_count", 0))
        feats, _ = build_paper_features(raw_work, cutoff_offset=settings.CUTOFF_OFFSET_YEARS)
        x_paper = torch.from_numpy(feats).unsqueeze(0).float()

        # 7. Model Forward Pass: Authentic Ego-Graph Message Passing
        x_dict, edge_index_dict, graph_context = ego_graph_builder.build_ego_graph(
            raw_work=raw_work,
            cutoff_year=cutoff_year
        )
        with torch.no_grad():
            out = self.model(x_dict, edge_index_dict)

        pred_log_cits = float(out["predicted_log_citations"][0].cpu().item())
        pred_future_cits = float(np.expm1(max(0.0, pred_log_cits)))

        # Primary Model: Logistic Regression (Empirical Best on held-out test set)
        if self.primary_model is not None and self.primary_scaler is not None:
            feats_scaled = self.primary_scaler.transform(feats.reshape(1, -1))
            logits = self.primary_model.decision_function(feats_scaled)
            scaled_logits = logits / max(self.primary_temperature, 1e-4)
            exp_z = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = (exp_z / np.sum(exp_z, axis=1, keepdims=True))[0]
            pred_cls = int(np.argmax(probs))
        else:
            probs = out["probabilities"][0].cpu().numpy()
            pred_cls = int(np.argmax(probs))

        label_map = {0: "Low (<50th)", 1: "Medium (50-90th)", 2: "High (>=90th)"}
        pred_label = label_map[pred_cls]

        calibrated_probs = {
            "Low (<50th)": round(float(probs[0]), 4),
            "Medium (50-90th)": round(float(probs[1]), 4),
            "High (>=90th)": round(float(probs[2]), 4),
        }

        # 8. Trajectory Points
        # Build chronological trajectory from publication year to latest available year
        trajectory_points = []
        cits_accum = 0
        sorted_counts = sorted(counts_by_year, key=lambda x: x.get("year", 0))
        for item in sorted_counts:
            yr = item.get("year")
            cnt = item.get("cited_by_count", 0)
            if yr:
                trajectory_points.append({
                    "year": int(yr),
                    "cited_by_count": int(cnt),
                    "is_cutoff": (int(yr) == cutoff_year),
                    "is_future_horizon": (int(yr) == pub_year + settings.TRAJECTORY_HORIZON_YEARS)
                })

        # 9. Authentic Local Occlusion Feature Attributions
        attributions = compute_local_occlusion_attributions(
            model=self.model,
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            pred_cls=pred_cls,
            original_prob=float(probs[pred_cls]),
            raw_work=raw_work
        )

        # 10. Ground Truth Backtest Comparison (strictly window_start..window_end)
        window_start = cutoff_year + 1
        window_end = pub_year + 5
        current_year = datetime.now(timezone.utc).year
        complete_through = current_year - 1
        has_gt = (window_end <= complete_through)

        threshold_info = self.get_cohort_threshold(primary_field, pub_year)
        p50 = threshold_info["p50"]
        p90 = threshold_info["p90"]

        if has_gt:
            actual_window = sum(
                c.get("cited_by_count", 0)
                for c in counts_by_year
                if isinstance(c, dict) and window_start <= c.get("year", 0) <= window_end
            )
            actual_log = float(np.log1p(actual_window))
            pred_log = float(pred_log_cits)
            err_abs = float(abs(pred_future_cits - actual_window))
            err_log = float(abs(pred_log - actual_log))

            clean_wid = str(work_id).split("/")[-1]
            stored_rows = self.df_works[self.df_works["work_id"] == clean_wid] if self.df_works is not None else None
            if stored_rows is not None and len(stored_rows) > 0 and "impact_class" in stored_rows.columns and pd.notnull(stored_rows.iloc[0]["impact_class"]):
                actual_cls = int(stored_rows.iloc[0]["impact_class"])
                actual_lbl = label_map[actual_cls]
                provenance_str = "ground truth from corpus (works.parquet)"
                if "future_citations_5y" in stored_rows.columns and pd.notnull(stored_rows.iloc[0]["future_citations_5y"]):
                    actual_window = int(stored_rows.iloc[0]["future_citations_5y"])
                    actual_log = float(np.log1p(actual_window))
            else:
                if actual_window < p50:
                    actual_cls = 0
                    actual_lbl = "Low (<50th)"
                elif actual_window < p90:
                    actual_cls = 1
                    actual_lbl = "Medium (50-90th)"
                else:
                    actual_cls = 2
                    actual_lbl = "High (>=90th)"
                provenance_str = "estimated using cohort thresholds from 2012-2017"

            is_correct = bool(pred_cls == actual_cls)

            # Empirical percentile rank from cohort distribution
            if self.df_works is not None and "future_citations_5y" in self.df_works.columns:
                cohort_vals = self.df_works[
                    self.df_works["publication_year"] == pub_year
                ]["future_citations_5y"].dropna().values
                if len(cohort_vals) > 0:
                    pct = float(np.mean(cohort_vals <= actual_window)) * 100.0
                else:
                    pct = 50.0
            else:
                pct = 50.0

            backtest = {
                "has_ground_truth": True,
                "window_complete": True,
                "window_start": window_start,
                "window_end": window_end,
                "actual_window": int(actual_window),
                "actual_future_citations_5y": int(actual_window),
                "actual_citations_window": int(actual_window),
                "actual_impact_class": actual_cls,
                "actual_label": actual_lbl,
                "predicted_impact_class": pred_cls,
                "predicted_label": pred_label,
                "is_correct_class": is_correct,
                "actual_log_citations": round(actual_log, 4),
                "predicted_log_citations": round(pred_log, 4),
                "error_abs": round(err_abs, 2),
                "error_log": round(err_log, 4),
                "absolute_error_log_citations": round(err_log, 4),
                "warning": None,
                "threshold_provenance": threshold_info,
                "cohort_percentile": round(pct, 1),
                "provenance": provenance_str,
                "class_provenance": provenance_str,
            }
        else:
            backtest = {
                "has_ground_truth": False,
                "window_complete": False,
                "window_start": window_start,
                "window_end": window_end,
                "actual_window": None,
                "actual_future_citations_5y": None,
                "actual_citations_window": None,
                "actual_impact_class": None,
                "actual_label": None,
                "predicted_impact_class": pred_cls,
                "predicted_label": pred_label,
                "is_correct_class": None,
                "actual_log_citations": None,
                "predicted_log_citations": round(float(pred_log_cits), 4),
                "error_abs": None,
                "error_log": None,
                "absolute_error_log_citations": None,
                "warning": f"Evaluation window ({window_start}..{window_end}) has not closed yet (complete through {complete_through})",
                "threshold_provenance": threshold_info,
                "cohort_percentile": None,
                "provenance": "openalex | model | computed",
            }

        import math
        def sanitize_nans(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: sanitize_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_nans(v) for v in obj]
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            elif hasattr(obj, "item") and callable(obj.item):
                val = obj.item()
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    return None
                return val
            elif pd.isna(obj):
                return None
            return obj

        doi_val = raw_work.get("doi")
        if pd.isna(doi_val):
            doi_val = None

        # Temporal split and training set membership
        if pub_year <= 2016:
            split_name = "train"
            in_train = True
        elif pub_year == 2017:
            split_name = "val"
            in_train = False
        elif pub_year >= 2018:
            split_name = "test"
            in_train = False
        else:
            split_name = "external"
            in_train = False

        author_details = []
        for a in authorships[:10]:
            aname = (a.get("author") or {}).get("display_name")
            apos = a.get("author_position") or "middle"
            inst_list = [i.get("display_name") for i in (a.get("institutions") or []) if isinstance(i, dict) and i.get("display_name")]
            if aname:
                author_details.append({
                    "name": aname,
                    "position": apos,
                    "institutions": inst_list
                })

        result_payload = {
            "paper_id": work_id,
            "doi": doi_val,
            "title": title,
            "publication_year": pub_year,
            "primary_field": primary_field if not pd.isna(primary_field) else None,
            "primary_topic": primary_topic if not pd.isna(primary_topic) else None,
            "authors": author_names[:10],
            "author_details": author_details,
            "cutoff_year": cutoff_year,
            "citations_at_cutoff": cits_at_cutoff,
            "predicted_impact_class": pred_cls,
            "predicted_impact_label": pred_label,
            "calibrated_probabilities": calibrated_probs,
            "predicted_future_citations_5y": round(pred_future_cits, 1),
            "citations_in_window": round(pred_future_cits, 1),
            "citations_window": round(pred_future_cits, 1),
            "split": split_name,
            "in_training_data": in_train,
            "venue": venue_info.model_dump() if venue_info else None,
            "access_economics": econ_info.model_dump(),
            "lineage": lineage_info.model_dump(),
            "trajectory": trajectory_points,
            "backtest": backtest,
            "attributions": attributions,
            "graph_context": graph_context,
            "model_version": "M1_LogisticRegression (Primary) + HeteroSAGE v2.0",
            "provenance": "openalex | scimago | core | model | computed"
        }

        return sanitize_nans(result_payload)


inference_service = ModelInferenceService()
