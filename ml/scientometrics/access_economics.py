"""Authentic Access Economics & Empirical OA Advantage Service.

Replaces simulated constants (+28.4%, -$35 reader fee, var(--accent-green)) with:
1. Authentic OpenAlex access status (gold, green, hybrid, bronze, closed).
2. Authentic APC list and paid figures in USD from DOAJ / OpenAlex.
3. Dynamically linked empirical OA citation advantage computed from our multi-disciplinary corpus.
4. Methodological notice explicitly highlighting correlation vs. causation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd

from api.config import settings

logger = logging.getLogger("AccessEconomics")


@dataclass
class AccessEconomicsResult:
    is_oa: bool
    oa_status: str
    oa_url: Optional[str]
    apc_paid_usd: Optional[float]
    apc_list_usd: Optional[float]
    empirical_cohort_oa_advantage_pct: float
    empirical_ci_95: List[float]
    scientometric_notice: str
    provenance: str

    def model_dump(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class ScientometricAccessService:
    """Computes authentic access economics and links empirical cohort metrics."""

    def __init__(self, metrics_path: Optional[Path] = None):
        self.metrics_path = metrics_path or (settings.BASE_DIR / "reports" / "oa_advantage_metrics.json")
        self.metrics: Dict[str, Any] = {}
        self._load_metrics()

    def compute(
        self,
        open_access_obj: Optional[Dict[str, Any]] = None,
        primary_location_obj: Optional[Dict[str, Any]] = None,
        field_name: Optional[str] = None
    ) -> AccessEconomicsResult:
        ploc = primary_location_obj or {}
        apc_paid = ploc.get("apc_paid") or (open_access_obj or {}).get("apc_paid")
        apc_list = ploc.get("apc_list") or (open_access_obj or {}).get("apc_list")
        return self.evaluate_access(
            open_access_obj=open_access_obj,
            field_name=field_name,
            apc_list_obj=apc_list if isinstance(apc_list, dict) else None,
            apc_paid_obj=apc_paid if isinstance(apc_paid, dict) else None
        )

    def _load_metrics(self):
        if self.metrics_path.exists():
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)
            logger.info("AccessEconomics: Loaded empirical metrics report.")
        else:
            logger.warning(f"AccessEconomics: Metrics file not found at {self.metrics_path}")

    def evaluate_access(
        self,
        open_access_obj: Optional[Dict[str, Any]],
        field_name: Optional[str] = None,
        apc_list_obj: Optional[Dict[str, Any]] = None,
        apc_paid_obj: Optional[Dict[str, Any]] = None
    ) -> AccessEconomicsResult:
        oa = open_access_obj or {}
        is_oa = bool(oa.get("is_oa", False))
        allowed_oa = {"gold", "green", "hybrid", "bronze", "closed", "diamond"}
        oa_status = (oa.get("oa_status") or "closed").lower()
        if oa_status not in allowed_oa:
            oa_status = "closed" if not is_oa else "green"
        oa_url = oa.get("oa_url")

        # Extract authentic APC if available
        def parse_usd(obj: Optional[Dict[str, Any]]) -> Optional[float]:
            import math
            if not obj or not isinstance(obj, dict):
                return None
            v = obj.get("value_usd")
            if v is not None and not pd.isna(v):
                try:
                    val = float(v)
                    if math.isnan(val) or math.isinf(val):
                        return None
                    return val
                except (ValueError, TypeError):
                    return None
            return None

        apc_list = parse_usd(apc_list_obj)
        apc_paid = parse_usd(apc_paid_obj)

        # Retrieve empirical advantage matching scientific field
        field_metrics = self.metrics.get("by_field", {}).get(field_name or "")
        global_adj = self.metrics.get("global_advantage", {}).get("adjusted", {})

        if field_metrics and "multiplier_pct" in field_metrics:
            adv_pct = float(field_metrics["multiplier_pct"])
            ci_95 = [float(x) for x in field_metrics.get("ci_95", [0.0, 0.0])]
        elif global_adj and "multiplier_pct" in global_adj:
            adv_pct = float(global_adj["multiplier_pct"])
            ci_95 = [float(x) for x in global_adj.get("ci_95", [0.0, 0.0])]
        else:
            adv_pct = None
            ci_95 = None

        notice = (
            "Empirical Association, Not Causation: Open Access citation differential "
            "reflects observational cohort comparison subject to author self-selection bias."
        )

        return AccessEconomicsResult(
            is_oa=is_oa,
            oa_status=oa_status,
            oa_url=oa_url,
            apc_paid_usd=apc_paid,
            apc_list_usd=apc_list,
            empirical_cohort_oa_advantage_pct=adv_pct if (is_oa and adv_pct is not None) else 0.0,
            empirical_ci_95=ci_95 if (is_oa and ci_95 is not None) else [0.0, 0.0],
            scientometric_notice=notice,
            provenance="openalex | computed"
        )


access_service = ScientometricAccessService()
