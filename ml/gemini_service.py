"""SciGraph AI — Gemini Model Intelligence Service (Mode G)
==========================================================
Provides LLM-driven scientometric impact forecasting with multi-key failover rotation.

Key Design:
1. Multi-key pool rotation for resilient fallback on 429 quota exhaustion.
2. Scientometric prompting grounded in empirical field-year cohort thresholds (P50/P90).
3. Structured JSON generation returning predicted tier, calibrated probabilities, and reasoning.
4. Graceful fallback to primary M1 model if Gemini service is unreachable or unconfigured.
"""

import os
import re
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("scigraph.gemini")

# Try importing google.genai
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    logger.warning("google-genai package not found; Gemini service will use REST or fallback.")


class GeminiService:
    def __init__(self):
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._current_key_idx = 0
        self._model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self._load_keys()

    def _load_keys(self):
        """Loads Gemini API keys from environment or .env file."""
        keys = []
        # Check GEMINI_API_KEYS (comma separated)
        pool = os.getenv("GEMINI_API_KEYS", "")
        if pool:
            for k in pool.split(","):
                k_clean = k.strip()
                if k_clean and k_clean not in keys:
                    keys.append(k_clean)

        # Check individual GEMINI_API_KEY
        single_key = os.getenv("GEMINI_API_KEY", "").strip()
        if single_key and single_key not in keys:
            keys.insert(0, single_key)

        # Also check numbered keys
        for i in range(1, 10):
            k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
            if k and k not in keys:
                keys.append(k)

        # Also check local sibling directory if running in local environment without .env
        if not keys:
            sibling_env = Path(__file__).resolve().parent.parent.parent / "Blind_But_Smart" / ".env"
            if sibling_env.exists():
                try:
                    from dotenv import dotenv_values
                    s_vals = dotenv_values(sibling_env)
                    s_key = s_vals.get("GEMINI_API_KEY", "").strip()
                    if s_key and s_key not in keys:
                        keys.append(s_key)
                    s_pool = s_vals.get("GEMINI_API_KEYS", "").strip()
                    if s_pool:
                        for pk in s_pool.split(","):
                            pk = pk.strip()
                            if pk and pk not in keys:
                                keys.append(pk)
                except Exception:
                    pass

        self._keys = keys
        logger.info(f"GeminiService initialized with {len(self._keys)} API key(s). Model: {self._model_name}")

    def get_active_key(self) -> Optional[str]:
        if not self._keys:
            return None
        with self._lock:
            return self._keys[self._current_key_idx % len(self._keys)]

    def rotate_key(self):
        """Advances to the next API key in the rotation pool."""
        with self._lock:
            if len(self._keys) > 1:
                self._current_key_idx = (self._current_key_idx + 1) % len(self._keys)
                logger.info(f"Rotated to Gemini API key index {self._current_key_idx}")

    def is_available(self) -> bool:
        """Returns True if Gemini can be called."""
        return HAS_GENAI and len(self._keys) > 0

    def _build_prompt(
        self,
        paper_context: Dict[str, Any],
        cohort_info: Dict[str, Any],
        m1_prediction: Optional[Dict[str, Any]] = None
    ) -> str:
        """Builds a rich, scientometrically rigorous prompt for Gemini."""
        title = paper_context.get("title", "Untitled Academic Paper")
        pub_year = paper_context.get("publication_year", 2018)
        cutoff_year = paper_context.get("cutoff_year", pub_year + 2)
        early_cits = paper_context.get("citations_at_cutoff", 0)
        field = paper_context.get("primary_field") or "General Science"
        topic = paper_context.get("primary_topic") or "Interdisciplinary Research"
        
        venue_info = paper_context.get("venue") or {}
        v_name = venue_info.get("venue_name", "Academic Publication")
        v_quartile = venue_info.get("quartile") or venue_info.get("core_rank") or "Unranked"

        authors = paper_context.get("authors", [])
        author_str = ", ".join(authors[:5]) if authors else "Academic Research Team"

        trajectory = paper_context.get("trajectory", [])
        early_traj = [
            f"Year {t.get('year')}: {t.get('cited_by_count', 0)} citations"
            for t in trajectory
            if int(t.get("year", 9999)) <= cutoff_year
        ]
        traj_str = "; ".join(early_traj) if early_traj else f"Total early citations at cutoff: {early_cits}"

        p50 = cohort_info.get("p50", 12.0)
        p90 = cohort_info.get("p90", 55.0)

        m1_context = ""
        if m1_prediction:
            m1_label = m1_prediction.get("predicted_label") or m1_prediction.get("predicted_impact_label", "Unknown")
            m1_cits = m1_prediction.get("predicted_future_citations_5y", 0)
            m1_context = f"\n- Local Statistical Model (M1 Prior): Predicted Tier: {m1_label}, Projected window citations: {m1_cits}"

        prompt = f"""You are an elite scientometrics and citation forecasting intelligence system.
Your mission is to evaluate this research paper and predict its citation impact tier strictly during the 3-year evaluation window (years {cutoff_year + 1} to {pub_year + 5}, i.e., years 3 to 5 after publication).

=== PAPER EVIDENCE FROZEN AT CUTOFF (T_cutoff = {cutoff_year}) ===
- Title: "{title}"
- Publication Year: {pub_year}
- Observation Cutoff Year: {cutoff_year} (Strict boundary: no data past this year is known)
- Primary Field: {field}
- Primary Topic: {topic}
- Authors: {author_str}
- Publication Venue: {v_name} (Quartile / Rank: {v_quartile})
- Early Citations Accumulated (up to {cutoff_year}): {early_cits} citations
- Early Annual Trajectory: {traj_str}{m1_context}

=== EMPIRICAL FIELD-YEAR COHORT BENCHMARKS (Discipline: {field}, Year: {pub_year}) ===
- 50th Percentile (Median) threshold: {p50:.1f} citations
- 90th Percentile (Top 10%) threshold: {p90:.1f} citations

Impact Tier Definitions:
1. Low (<50th): Window citations < {p50:.1f} (Paper is below field median)
2. Medium (50-90th): Window citations between {p50:.1f} and {p90:.1f} (Paper is above median but below top 10%)
3. High (>=90th): Window citations >= {p90:.1f} (Paper is in the top 10% elite scientific tier)

=== SCIENTOMETRIC HEURISTICS TO ENSURE ACCURACY ===
1. Citation Velocity Multiplier: In academic literature, post-cutoff citations (years 3-5) typically scale at 2.5x to 4.5x of early 2-year citations for papers in active disciplines.
2. Venue Lift: Q1 / Top-tier venues (CORE A*, Nature, IEEE, ACM) sustain high compound citation velocity. Unranked or low-quartile venues have decelerating trajectories.
3. If early citations are already near or above {p50:.1f}, the paper will almost certainly achieve at least Medium tier, and if early citations exceed 0.5 * {p90:.1f} with a top venue, High tier is probable.
4. If early citations are very low (0 to 3) in a mature discipline and venue is unranked, it is Low tier with high probability.

=== RESPONSE INSTRUCTIONS ===
You MUST return ONLY a valid JSON object with the following exact keys:
{{
  "predicted_tier": "Low" | "Medium" | "High",
  "predicted_impact_class": 0 | 1 | 2,
  "predicted_impact_label": "Low (<50th)" | "Medium (50-90th)" | "High (>=90th)",
  "predicted_future_citations": <float number estimating window citations>,
  "probabilities": {{
    "Low": <float between 0.0 and 1.0>,
    "Medium": <float between 0.0 and 1.0>,
    "High": <float between 0.0 and 1.0>
  }},
  "confidence": <float between 0.0 and 1.0>,
  "gemini_reasoning": "<2-3 sentence rigorous scientometric explanation synthesizing venue rank, early velocity, and cohort percentile thresholds>"
}}
Ensure the probabilities sum to approximately 1.0. Do NOT return any markdown wrapping other than ```json or plain text.
"""
        return prompt

    def predict(
        self,
        paper_context: Dict[str, Any],
        cohort_info: Dict[str, Any],
        m1_prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Runs Gemini prediction with auto-rotation across API keys and fallback."""
        if not HAS_GENAI or not self._keys:
            logger.warning("Gemini unavailable; returning M1 prediction as fallback.")
            return self._fallback_prediction(m1_prediction, reason="Gemini client or keys not available")

        prompt = self._build_prompt(paper_context, cohort_info, m1_prediction)
        
        # Try up to len(self._keys) times to rotate through keys if quota or rate limit hits
        attempts = min(3, len(self._keys))
        last_error = None

        for attempt in range(attempts):
            api_key = self.get_active_key()
            try:
                client = genai.Client(api_key=api_key)
                
                # Execute generation
                response = client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,  # Low temperature for deterministic scientometric accuracy
                        response_mime_type="application/json"
                    ) if hasattr(types, "GenerateContentConfig") else None
                )

                raw_text = response.text if hasattr(response, "text") else str(response)
                parsed = self._parse_gemini_json(raw_text)
                
                if parsed:
                    # Successfully parsed
                    parsed["mode"] = "G"
                    parsed["model_version"] = f"Gemini ({self._model_name} Intelligence Mode G)"
                    parsed["provenance"] = "openalex | gemini_ai | cohort_grounded"
                    return parsed
                else:
                    logger.warning(f"Failed to parse JSON from Gemini response: {raw_text[:200]}")
                    last_error = "JSON parse error from Gemini output"
            except Exception as exc:
                err_msg = str(exc)
                logger.warning(f"Gemini API attempt {attempt + 1} with key index {self._current_key_idx} failed: {err_msg}")
                last_error = err_msg
                # If quota/rate limit error, rotate to next key
                if "429" in err_msg or "quota" in err_msg.lower() or "resource_exhausted" in err_msg.lower():
                    self.rotate_key()
                elif "403" in err_msg or "api_key" in err_msg.lower():
                    self.rotate_key()

        # If all attempts failed, fallback gracefully to M1
        logger.error(f"All Gemini attempts exhausted. Falling back to primary model. Last error: {last_error}")
        return self._fallback_prediction(m1_prediction, reason=f"Gemini API error: {last_error}")

    def _parse_gemini_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Cleans and extracts JSON object from Gemini response text."""
        cleaned = raw_text.strip()
        # Remove markdown fences if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except Exception:
            # Fallback regex search for JSON object
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    return None
            else:
                return None

        # Validate and sanitize fields
        tier = data.get("predicted_tier", "Medium")
        if tier not in ["Low", "Medium", "High"]:
            # Normalize
            if "low" in str(tier).lower():
                tier = "Low"
            elif "high" in str(tier).lower():
                tier = "High"
            else:
                tier = "Medium"

        class_map = {"Low": 0, "Medium": 1, "High": 2}
        label_map = {0: "Low (<50th)", 1: "Medium (50-90th)", 2: "High (>=90th)"}
        cls_idx = class_map.get(tier, 1)

        probs = data.get("probabilities", {})
        low_p = float(probs.get("Low", 0.15))
        med_p = float(probs.get("Medium", 0.70))
        high_p = float(probs.get("High", 0.15))
        total_p = low_p + med_p + high_p
        if total_p > 0:
            low_p /= total_p
            med_p /= total_p
            high_p /= total_p
        else:
            low_p, med_p, high_p = (0.2, 0.6, 0.2)

        probs_normalized = {
            "Low": round(low_p, 4),
            "Medium": round(med_p, 4),
            "High": round(high_p, 4),
        }

        reasoning = data.get("gemini_reasoning") or data.get("reasoning") or (
            f"Gemini predicted {tier} impact tier based on early citation growth rate and field cohort baseline."
        )

        future_cits = data.get("predicted_future_citations")
        try:
            future_cits = float(future_cits) if future_cits is not None else None
        except (ValueError, TypeError):
            future_cits = None

        return {
            "predicted_tier": tier,
            "predicted_impact_class": cls_idx,
            "predicted_impact_label": label_map[cls_idx],
            "predicted_label": label_map[cls_idx],
            "calibrated_probabilities": probs_normalized,
            "predicted_future_citations_5y": future_cits,
            "confidence": float(data.get("confidence", 0.85)),
            "gemini_reasoning": reasoning
        }

    def _fallback_prediction(
        self,
        m1_prediction: Optional[Dict[str, Any]],
        reason: str = "Fallback"
    ) -> Dict[str, Any]:
        """Provides a safe, non-crashing fallback from M1 local prediction."""
        if m1_prediction:
            res = dict(m1_prediction)
            res["mode"] = "M"
            res["gemini_reasoning"] = f"Operating in local model mode ({reason})."
            return res

        return {
            "mode": "M",
            "predicted_tier": "Medium",
            "predicted_impact_class": 1,
            "predicted_impact_label": "Medium (50-90th)",
            "predicted_label": "Medium (50-90th)",
            "calibrated_probabilities": {"Low": 0.25, "Medium": 0.55, "High": 0.20},
            "predicted_future_citations_5y": 25.0,
            "confidence": 0.70,
            "gemini_reasoning": f"Local baseline inference ({reason}).",
            "model_version": "M1_LogisticRegression (Fallback)",
            "provenance": "model | fallback"
        }


# Global singleton instance
gemini_service = GeminiService()
