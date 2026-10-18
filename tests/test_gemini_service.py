import pytest
from ml.gemini_service import gemini_service

def test_gemini_service_initialization():
    assert gemini_service is not None
    assert len(gemini_service._keys) > 0
    assert gemini_service.is_available() is True
    assert gemini_service.get_active_key() is not None

def test_gemini_fallback_prediction():
    m1_sample = {
        "predicted_tier": "Low",
        "predicted_impact_class": 0,
        "predicted_impact_label": "Low (<50th)",
        "calibrated_probabilities": {"Low": 0.8, "Medium": 0.15, "High": 0.05},
        "predicted_future_citations_5y": 4.0
    }
    fallback = gemini_service._fallback_prediction(m1_sample, reason="Testing fallback")
    assert fallback["mode"] == "M"
    assert fallback["predicted_impact_class"] == 0
    assert "Testing fallback" in fallback["gemini_reasoning"]

def test_gemini_parse_json():
    sample_json = '''```json
    {
        "predicted_tier": "High",
        "predicted_impact_class": 2,
        "predicted_impact_label": "High (>=90th)",
        "predicted_future_citations": 125.0,
        "probabilities": {
            "Low": 0.05,
            "Medium": 0.15,
            "High": 0.80
        },
        "confidence": 0.92,
        "gemini_reasoning": "High venue prestige and rapid early citation velocity."
    }
    ```'''
    parsed = gemini_service._parse_gemini_json(sample_json)
    assert parsed is not None
    assert parsed["predicted_tier"] == "High"
    assert parsed["predicted_impact_class"] == 2
    assert parsed["calibrated_probabilities"]["High"] == 0.8
    assert "venue prestige" in parsed["gemini_reasoning"]
