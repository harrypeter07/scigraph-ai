"""FastAPI Router for Topic Feasibility & Research Success Rate Exploration (WP10).

Provides endpoints to:
1. Explore any keyword or research field across live OpenAlex, SCImago, and CORE data.
2. Calculate composite scientometric success rate, 3-year publication CAGR, and venue accessibility.
3. Deliver deep Gemini Intelligence (Mode G) with concrete novel research gaps, methodologies, and common pitfalls.
4. Retrieve curated preset topics for instant one-click analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Literal
from fastapi import APIRouter, Query, HTTPException

from ml.scientometrics.topic_explorer import topic_explorer

logger = logging.getLogger("scigraph.api.topic")

router = APIRouter(prefix="/api/v1/topic", tags=["Topic Feasibility"])

PRESET_TOPICS = [
    {
        "id": "gnn_drug",
        "title": "Graph Neural Networks for Drug Discovery",
        "field": "Computer Science / Bioengineering",
        "badge": "High Viability"
    },
    {
        "id": "multimodal_llm",
        "title": "Multimodal Large Language Models",
        "field": "Artificial Intelligence",
        "badge": "Emerging Frontier"
    },
    {
        "id": "fed_health",
        "title": "Federated Learning in Healthcare",
        "field": "Computer Science / Medicine",
        "badge": "High Viability"
    },
    {
        "id": "quantum_qec",
        "title": "Quantum Error Correction",
        "field": "Quantum Physics / Computer Science",
        "badge": "Emerging Frontier"
    },
    {
        "id": "zk_crypt",
        "title": "Zero-Knowledge Cryptography",
        "field": "Computer Science / Mathematics",
        "badge": "High Growth"
    },
    {
        "id": "perovskite",
        "title": "Perovskite Solar Cells Degradation",
        "field": "Materials Science / Energy",
        "badge": "High Citation Yield"
    },
    {
        "id": "rlhf",
        "title": "Reinforcement Learning from Human Feedback",
        "field": "Artificial Intelligence",
        "badge": "High Demand"
    },
    {
        "id": "bci_decoding",
        "title": "Brain-Computer Interface Neural Decoding",
        "field": "Neuroscience / Machine Learning",
        "badge": "Emerging Frontier"
    }
]


@router.get("/presets")
def get_preset_topics() -> Dict[str, Any]:
    """Returns curated high-interest research topics for quick one-click exploration."""
    return {
        "count": len(PRESET_TOPICS),
        "presets": PRESET_TOPICS
    }


@router.get("/explore")
async def explore_topic(
    query: str = Query(..., min_length=2, description="Research keyword, topic, or field to evaluate"),
    mode: Literal["M", "G"] = Query("M", description="Engine mode: M (Scientometrics) or G (Gemini Intelligence)")
) -> Dict[str, Any]:
    """Explores multi-source research viability, success rate, and strategic gaps for a keyword."""
    clean_q = query.strip()
    if not clean_q:
        raise HTTPException(status_code=400, detail="Query keyword cannot be empty.")

    try:
        result = await topic_explorer.explore(clean_q, mode=mode)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error exploring topic '{clean_q}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Topic exploration failed: {str(exc)}")
