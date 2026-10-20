"""Citation-impact serving with a versioned training/inference feature contract.

V3 optionally loads a validation-selected citation GraphSAGE ensemble. Legacy
checkpoints remain supported. API evidence identifies the classifier actually
used; absent regression estimates remain null. Temporal citation features use
publication year through year+2, not an exact rolling 24-month window.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from api.config import settings
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

logger = logging.getLogger("scigraph.inference")


class ModelInferenceService:
    """Production ML inference engine for citation impact trajectory."""

    def __init__(self):
        self.model = None  # Optional GNN; served predictions use primary_model
        self.primary_payload: Optional[Dict[str, Any]] = None
        self.primary_model = None
        self.primary_scaler = None
        self.primary_temperature: float = 1.0
        self.model_card: Optional[Dict[str, Any]] = None
        self.df_works: Optional[pd.DataFrame] = None
        self.df_counts: Optional[pd.DataFrame] = None
        self.cohort_reference: Dict[str, Any] = {}
        self.forecast_bundle = None
        self.forecast_bundle_sha256 = None
        self.graph_predictor = None
        self.annual_histories = {}
        self.node_in_channels = {
            "paper": TOTAL_FEATURE_DIM,
            "author": 8,
            "institution": 8,
            "venue": 8,
            "topic": 8
        }
        self.cohort_thresholds: Dict[int, Dict[str, float]] = {}

    def load(self):
        """Loads model weights and reference metadata with strict loud failures for served models."""
        card_path = settings.CHECKPOINTS_DIR / "MODEL_CARD.json"
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
        forecast_path = settings.CHECKPOINTS_DIR / "forecast_v3.joblib"
        if forecast_path.exists():
            import hashlib
            import io
            forecast_bytes = forecast_path.read_bytes()
            self.forecast_bundle = joblib.load(io.BytesIO(forecast_bytes))
            self.forecast_bundle_sha256 = hashlib.sha256(forecast_bytes).hexdigest()
            if self.forecast_bundle.get("schema_version") != 3:
                raise RuntimeError("Incompatible forecast feature bundle")
        graph_forecast_path = settings.CHECKPOINTS_DIR / "citation_forecast.joblib"
        if graph_forecast_path.exists():
            import hashlib
            from ml.gnn.citation_forecast import CitationPredictor
            graph_bundle = joblib.load(graph_forecast_path)
            if self.forecast_bundle is None or graph_bundle["tabular_sha256"] != hashlib.sha256(forecast_path.read_bytes()).hexdigest():
                raise RuntimeError("Graph checkpoint does not match the tabular feature bundle")
            self.graph_predictor = CitationPredictor(graph_bundle, self.forecast_bundle)
        logger.info(f"Successfully loaded primary model (M1_LogisticRegression) with T={self.primary_temperature:.4f}")

        # 1. Optionally load HeteroSAGE architecture (for graph relational comparative analysis)
        ckpt_path = settings.CHECKPOINTS_DIR / "hetero_graphsage.pt"
        self.model = None
        try:
            import torch
            from ml.gnn.models.hetero_gnn import AuthenticHeteroSAGE
            if ckpt_path.exists():
                gnn = AuthenticHeteroSAGE(
                    node_in_channels=self.node_in_channels,
                    hidden_channels=64,
                    out_channels=3,
                    dropout=0.0
                )
                state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)
                gnn.load_state_dict(state_dict)
                gnn.eval()
                self.model = gnn
                logger.info("Successfully loaded AuthenticHeteroSAGE checkpoint.")
        except ImportError:
            logger.info("Lightweight serve mode: PyTorch is not installed; serving calibrated M1 logistic regression only.")
            self.model = None
        except Exception as exc:
            logger.info("Optional HeteroSAGE checkpoint skipped: %s", exc)
            self.model = None

        # 2. Load Model Card
        with open(card_path, "r", encoding="utf-8") as f:
            self.model_card = json.load(f)
        forecast_card = settings.CHECKPOINTS_DIR / "FORECAST_CARD.json"
        if self.forecast_bundle is not None and forecast_card.exists():
            self.model_card = json.loads(forecast_card.read_text())

        # 3. Load persistent text encoder (strictly fitted on TRAIN titles only)
        if not DEFAULT_TEXT_ENCODER_PATH.exists():
            raise RuntimeError(
                f"CRITICAL: Text encoder checkpoint missing at '{DEFAULT_TEXT_ENCODER_PATH}'. "
                "Per WP1, server requires valid text encoder."
            )
        get_text_encoder(DEFAULT_TEXT_ENCODER_PATH)
        logger.info("Successfully loaded persistent text encoder checkpoint.")

        # 4. Load reference tables (lean column selection to prevent OOM)
        works_path = settings.DATA_DIR / "works.parquet"
        counts_path = settings.DATA_DIR / "counts_by_year.parquet"

        if not works_path.exists():
            raise RuntimeError(f"CRITICAL: Reference dataset missing at '{works_path}'.")
        self.df_works = pd.read_parquet(
            works_path,
            columns=[
                "work_id", "title", "publication_year", "citations_at_cutoff",
                "future_citations_5y", "impact_class", "primary_field_name",
                "venue_name", "oa_status"
            ]
        )
        if self.forecast_bundle is not None and self.forecast_bundle.get("evaluation_reference"):
            self.df_works = pd.DataFrame(self.forecast_bundle["evaluation_reference"])
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
        self.df_counts = None
        if self.forecast_bundle is not None:
            from ml.features.forecast import clean_id
            annual = pd.read_parquet(counts_path)
            for item in annual.to_dict("records"):
                self.annual_histories.setdefault(clean_id(item["work_id"]), []).append(
                    {"year": int(item["year"]), "cited_by_count": int(item["cited_by_count"])})
            self.annual_histories.update(self.forecast_bundle.get("early_history_reference", {}))
        logger.info("Verified citation yearly records table exists on disk.")

        # 5. Load empirical cohort thresholds reference (single source of truth: label_thresholds.json)
        cohort_ref_path = settings.DATA_DIR / "reference" / "label_thresholds.json"
        if cohort_ref_path.exists():
            with open(cohort_ref_path, "r", encoding="utf-8") as f:
                self.cohort_reference = json.load(f)
            logger.info("Successfully loaded empirical cohort reference thresholds.")
        else:
            self.cohort_reference = {}
        if self.forecast_bundle is not None and self.forecast_bundle.get("evaluation_reference"):
            # Backtest cohort cutpoints must describe the model's own sampled
            # population. They are never supplied to the forecast features.
            self.cohort_reference = {"by_field_year": {}}
            for (field_name, year), group in self.df_works.groupby(["primary_field_name", "publication_year"]):
                vals = group["future_citations_5y"].dropna().to_numpy()
                self.cohort_reference["by_field_year"][f"{field_name}::{year}"] = {
                    "p50": float(np.percentile(vals, 50)), "p90": float(np.percentile(vals, 90)),
                    "cohort_level": "model_sample_field_year", "n": int(len(vals))
                }

    def get_cohort_threshold(self, field_name: Optional[str], pub_year: int) -> Dict[str, Any]:
        """Reads empirical p50/p90 from cohort reference with strict hierarchical fallback."""
        if self.cohort_reference:
            by_field_year = self.cohort_reference.get("by_field_year", {})
            key = f"{field_name}::{pub_year}"
            if key in by_field_year:
                info = by_field_year[key]
                return {
                    "cohort_level": info.get("cohort_level", "field_year"),
                    "p50": float(info["p50"]),
                    "p90": float(info["p90"]), "n": info.get("n")
                }

            by_field = self.cohort_reference.get("by_field", {})
            if field_name and str(field_name) in by_field:
                info = by_field[str(field_name)]
                return {
                    "cohort_level": "field",
                    "p50": float(info["p50"]),
                    "p90": float(info["p90"])
                }

            glob = self.cohort_reference.get("global")
            if glob and glob.get("p50") is not None:
                return {
                    "cohort_level": "global",
                    "p50": float(glob["p50"]),
                    "p90": float(glob["p90"])
                }

        year_info = self.cohort_thresholds.get(int(pub_year))
        if year_info:
            return {"cohort_level": "year", "p50": float(year_info["p50"]), "p90": float(year_info["p90"])}

        if self.cohort_thresholds:
            p50s = [v["p50"] for v in self.cohort_thresholds.values()]
            p90s = [v["p90"] for v in self.cohort_thresholds.values()]
            return {
                "cohort_level": "empirical_mean",
                "p50": float(np.mean(p50s)),
                "p90": float(np.mean(p90s)),
            }

        raise RuntimeError(
            "Cohort thresholds are not loaded. Place data/reference/label_thresholds.json "
            "or load works.parquet before serving predictions."
        )

    def predict(
        self,
        raw_work: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs full end-to-end scientometric inference and ground truth backtest."""
        if self.primary_model is None or self.primary_scaler is None:
            raise RuntimeError(
                "Primary served model is not initialized. Call load() first "
                "(GNN weights are optional and not required for live predictions)."
            )

        work_id = str(raw_work.get("id") or raw_work.get("work_id") or "").split("/")[-1]
        title = raw_work.get("title") or "Untitled Academic Work"

        pub_year_raw = raw_work.get("publication_year")
        if pub_year_raw is None or (isinstance(pub_year_raw, float) and np.isnan(pub_year_raw)):
            raise ValueError("Missing required publication_year field (Rule F2).")
        pub_year = int(pub_year_raw)
        cutoff_year = pub_year + settings.CUTOFF_OFFSET_YEARS

        # 1. Historical citations at cutoff
        counts_by_year = raw_work.get("counts_by_year", []) or []
        cits_at_cutoff = sum(
            int(c.get("cited_by_count", 0) or 0)
            for c in counts_by_year
            if isinstance(c, dict) and int(c.get("year", 9999) or 9999) <= cutoff_year
        )
        if cits_at_cutoff == 0 and self.df_works is not None:
            stored_rows = self.df_works[self.df_works["work_id"].astype(str).str.endswith(work_id)]
            if len(stored_rows) > 0 and pd.notnull(stored_rows.iloc[0].get("citations_at_cutoff")):
                cits_at_cutoff = int(stored_rows.iloc[0]["citations_at_cutoff"])
        total_citations = int(raw_work.get("cited_by_count", 0) or 0)

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

        # Primary Model: Logistic Regression (temperature-calibrated on validation)
        feats_scaled = self.primary_scaler.transform(feats.reshape(1, -1))
        logits = self.primary_model.decision_function(feats_scaled)
        scaled_logits = logits / max(self.primary_temperature, 1e-4)
        exp_z = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = (exp_z / np.sum(exp_z, axis=1, keepdims=True))[0]
        forecast_record = dict(raw_work)
        model_insights = {}
        if self.forecast_bundle is not None:
            from ml.features.forecast import annual_counts, probabilities
            if annual_counts(forecast_record) is None and work_id in self.annual_histories:
                forecast_record["counts_by_year"] = self.annual_histories[work_id]
            probs = probabilities(self.forecast_bundle, [forecast_record])[0]
            observed = annual_counts(forecast_record)
            if observed is not None:
                cits_at_cutoff = sum(observed)
            feature_values = self.forecast_bundle["features"].transform([forecast_record])[0]
            model_insights = {
                "early_citation_percentile": float(feature_values[10] * 100),
                "annual_citations": observed,
                "annual_history_missing": bool(feature_values[8]),
                "cohort_reference_missing": bool(feature_values[12]),
                "cohort_reference": "field/publication-year early-count peers; training-field fallback if unavailable",
                "trajectory": "unavailable" if observed is None else ("accelerating" if observed[2]-2*observed[1]+observed[0] > 0 else "steady_or_decelerating")
            }
            if self.graph_predictor is not None:
                probs = self.graph_predictor.predict([forecast_record])[0]
        pred_cls = int(np.argmax(probs))

        label_map = {0: "Low (<50th)", 1: "Medium (50-90th)", 2: "High (>=90th)"}
        pred_label = label_map[pred_cls]

        calibrated_probs = {
            "Low (<50th)": round(float(probs[0]), 4),
            "Medium (50-90th)": round(float(probs[1]), 4),
            "High (>=90th)": round(float(probs[2]), 4),
        }

        # 7. Model Forward Pass: Optional GNN or High-Performance Linear Regression
        pred_log_cits = None
        pred_future_cits = None
        graph_context = None

        if self.model is not None and self.forecast_bundle is None:
            try:
                import torch
                from ml.gnn.ego_graph import ego_graph_builder
                x_dict, edge_index_dict, graph_context = ego_graph_builder.build_ego_graph(
                    raw_work=raw_work,
                    cutoff_year=cutoff_year
                )
                with torch.no_grad():
                    out = self.model(x_dict, edge_index_dict)
                pred_log_cits = float(out["predicted_log_citations"][0].cpu().item())
                pred_future_cits = float(np.expm1(max(0.0, pred_log_cits)))
            except Exception as e:
                logger.debug(f"GNN forward pass fallback: {e}")
                self.model = None

        # Classification cutpoints do not identify an expected citation count.
        # Leave the numeric forecast unavailable without a fitted regressor.

        if not isinstance(graph_context, dict) or "nodes" not in graph_context:
            graph_nodes = [{"data": {"id": work_id, "label": title[:30], "type": "paper"}}]
            graph_edges = []
            for i, auth in enumerate(author_names[:4]):
                a_id = f"author_{i}"
                graph_nodes.append({"data": {"id": a_id, "label": auth, "type": "author"}})
                graph_edges.append({"data": {"source": a_id, "target": work_id, "label": "authored"}})
            if source_name:
                graph_nodes.append({"data": {"id": "venue_0", "label": source_name[:25], "type": "venue"}})
                graph_edges.append({"data": {"source": work_id, "target": "venue_0", "label": "published_in"}})
            if primary_topic:
                graph_nodes.append({"data": {"id": "topic_0", "label": primary_topic[:25], "type": "topic"}})
                graph_edges.append({"data": {"source": work_id, "target": "topic_0", "label": "has_topic"}})
            graph_context = {"nodes": graph_nodes, "edges": graph_edges}

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

        # 9. Authentic Feature Attributions (Local Occlusion or Linear Model Attribution)
        attributions = []
        # Explain the classifier that actually produced the served tier.
        graph_evidence = None
        if self.graph_predictor is not None:
            graph_evidence = self.graph_predictor.evidence(forecast_record)
            if self.graph_predictor.graph["graph_weight"] > 0:
                attributions = graph_evidence["attributions"]
            else:
                from ml.features.forecast import explain
                attributions = explain(self.forecast_bundle, forecast_record)
        elif self.forecast_bundle is not None:
            from ml.features.forecast import explain
            attributions = explain(self.forecast_bundle, forecast_record)
        if not attributions and self.primary_model is not None and self.primary_scaler is not None:
            from ml.features.builder import FEATURE_GROUPS
            from ml.explain.attribution import FEATURE_DISPLAY_NAMES
            feats_scaled = self.primary_scaler.transform(feats.reshape(1, -1))
            group_deltas = []
            group_names = []
            directions = []
            for grp_name, sl in FEATURE_GROUPS.items():
                occluded = feats_scaled.copy()
                occluded[0, sl] = 0.0  # Standardized training mean.
                occ_logits = self.primary_model.decision_function(occluded) / max(self.primary_temperature, 1e-4)
                from scipy.special import softmax
                grp_val = float(probs[pred_cls] - softmax(occ_logits, axis=1)[0, pred_cls])
                group_deltas.append(abs(grp_val))
                directions.append("positive" if grp_val >= 0 else "negative")
                group_names.append(grp_name)
            sum_d = sum(group_deltas)
            scores = [d / sum_d for d in group_deltas] if sum_d > 0 else [1.0 / len(group_deltas)] * len(group_deltas)
            for grp, sc, dl, direct in zip(group_names, scores, group_deltas, directions):
                attributions.append({
                    "feature_name": FEATURE_DISPLAY_NAMES.get(grp, grp),
                    "raw_value": None,
                    "importance_score": round(float(sc), 4),
                    "delta_probability": round(float(dl), 4),
                    "delta_percentage_points": round(float(dl * 100.0), 2),
                    "direction": direct,
                    "provenance": "served_model | training_mean_occlusion",
                    "explained_class": pred_cls
                })
            attributions.sort(key=lambda a: a["importance_score"], reverse=True)

        # 10. Ground Truth Backtest Comparison (strictly window_start..window_end)
        window_start = cutoff_year + 1
        window_end = pub_year + 5
        current_year = datetime.now(timezone.utc).year
        complete_through = current_year - 1
        has_gt = (window_end <= complete_through)
        if self.forecast_bundle is not None and self.forecast_bundle.get("evaluation_reference"):
            known_outcome = bool(self.df_works["work_id"].astype(str).str.endswith(work_id).any())
            has_gt = has_gt and known_outcome

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
            pred_log = float(pred_log_cits) if pred_log_cits is not None else None
            err_abs = float(abs(pred_future_cits - actual_window)) if pred_future_cits is not None else None
            err_log = float(abs(pred_log - actual_log)) if pred_log is not None else None

            clean_wid = str(work_id).split("/")[-1]
            stored_rows = self.df_works[self.df_works["work_id"].astype(str).str.endswith(clean_wid)] if self.df_works is not None else None
            if stored_rows is not None and len(stored_rows) > 0 and "impact_class" in stored_rows.columns and pd.notnull(stored_rows.iloc[0]["impact_class"]):
                actual_cls = int(stored_rows.iloc[0]["impact_class"])
                actual_lbl = label_map[actual_cls]
                provenance_str = "stored outcome from selected model evaluation corpus" if self.forecast_bundle is not None else "stored outcome from original works.parquet corpus"
                if pd.notnull(stored_rows.iloc[0].get("future_citations_5y")):
                    actual_window = int(stored_rows.iloc[0]["future_citations_5y"])
                    actual_log = float(np.log1p(actual_window))
                    err_abs = abs(pred_future_cits-actual_window) if pred_future_cits is not None else None
                    err_log = abs(pred_log-actual_log) if pred_log is not None else None
                if not counts_by_year and "future_citations_5y" in stored_rows.columns and pd.notnull(stored_rows.iloc[0]["future_citations_5y"]):
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
                "predicted_log_citations": round(pred_log, 4) if pred_log is not None else None,
                "error_abs": round(err_abs, 2) if err_abs is not None else None,
                "error_log": round(err_log, 4) if err_log is not None else None,
                "absolute_error_log_citations": round(err_log, 4) if err_log is not None else None,
                "warning": None,
                "threshold_provenance": threshold_info,
                "cohort_percentile": round(pct, 1),
                "provenance": provenance_str,
                "class_provenance": provenance_str,
            }
        else:
            backtest = {
                "has_ground_truth": False,
                "window_complete": bool(window_end <= complete_through),
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
                "predicted_log_citations": round(float(pred_log_cits), 4) if pred_log_cits is not None else None,
                "error_abs": None,
                "error_log": None,
                "absolute_error_log_citations": None,
                "warning": "No verified cohort outcome in the model's evaluation corpus" if window_end <= complete_through else f"Evaluation window ({window_start}..{window_end}) has not closed yet (complete through {complete_through})",
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
        known_member = bool(work_id and self.df_works is not None and self.df_works["work_id"].astype(str).str.endswith(work_id).any())
        if known_member and 2012 <= pub_year <= 2016:
            split_name = "train"
            in_train = True
        elif known_member and pub_year == 2017:
            split_name = "val"
            in_train = False
        elif known_member and 2018 <= pub_year <= 2020:
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

        reliability_reasons = []
        if cutoff_year >= current_year:
            reliability_reasons.append("early_observation_window_incomplete")
        if self.forecast_bundle and cutoff_year < self.forecast_bundle.get("label_availability_year", 2022):
            reliability_reasons.append("retrospective_only_model_labels_matured_after_paper_cutoff")
        if model_insights.get("annual_history_missing"):
            reliability_reasons.append("annual_citation_history_unavailable")
        if model_insights.get("cohort_reference_missing"):
            reliability_reasons.append("matching_field_year_reference_unavailable")
        if graph_evidence is not None and not graph_evidence["neighbors"]:
            reliability_reasons.append("citation_neighborhood_unavailable")
        if float(np.max(probs)) < .8:
            reliability_reasons.append("uncertain_classification")
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
            "cutoff_citations": cits_at_cutoff,
            "predicted_impact_class": pred_cls,
            "predicted_impact_label": pred_label,
            "predicted_label": pred_label,
            "calibrated_probabilities": calibrated_probs,
            "predicted_future_citations_5y": round(pred_future_cits, 1) if pred_future_cits is not None else None,
            "citations_in_window": round(pred_future_cits, 1) if pred_future_cits is not None else None,
            "citations_window": round(pred_future_cits, 1) if pred_future_cits is not None else None,
            "split": split_name,
            "in_training_data": in_train,
            "in_train_split": in_train,
            "venue": venue_info.model_dump() if venue_info else None,
            "access_economics": econ_info.model_dump(),
            "lineage": lineage_info.model_dump(),
            "trajectory": trajectory_points,
            "backtest": backtest,
            "attributions": attributions,
            "graph_context": graph_context,
            "model_version": "CitationSAGE_validation_selected_fusion" if self.graph_predictor is not None and self.graph_predictor.graph["graph_weight"] > 0 else (self.forecast_bundle["model_name"] if self.forecast_bundle else "M1_LogisticRegression"),
            "model_evidence": {
                "tier_source": "citation_graph_fusion" if self.graph_predictor and self.graph_predictor.graph["graph_weight"] > 0 else "tabular_classifier",
                "gnn_used_for_tier": bool(self.graph_predictor and self.graph_predictor.graph["graph_weight"] > 0 and graph_evidence and graph_evidence["neighbors"]),
                "graph_predictive_lift": self.graph_predictor.graph["report"].get("accuracy_lift_over_tabular") if self.graph_predictor else None,
                "graph_status": ("experimental_measured_messages" if self.graph_predictor.graph["graph_weight"] > 0 else "evaluated_not_selected") if self.graph_predictor else "descriptive_context_only",
                "citation_estimate_method": "experimental_gnn" if pred_future_cits is not None else "unavailable_no_validated_regressor",
                "evaluation_status": "development_benchmark; fresh temporal holdout required"
            },
            "model_insights": model_insights,
            "graph_evidence": graph_evidence,
            "prediction_reliability": {
                "status": "limited_evidence" if reliability_reasons else "higher_model_confidence",
                "reasons": reliability_reasons,
                "notice": "Model confidence is not a guarantee of correctness or scientific validity"
            },
            "provenance": "openalex | scimago | core | model | computed"
        }

        return sanitize_nans(result_payload)


inference_service = ModelInferenceService()
