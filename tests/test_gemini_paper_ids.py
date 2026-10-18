import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from ml.inference_service import inference_service
from ml.gemini_service import gemini_service

def run_test():
    print("=== Testing Gemini Service with Research Paper IDs from Corpus ===", flush=True)
    inference_service.load()
    
    # Pick 3 papers across different tiers from the held-out test split
    df = inference_service.df_works
    test_df = df[df["publication_year"] >= 2018]
    
    samples = []
    # Sample 1 low, 1 med, 1 high if possible
    for cls in [0, 1, 2]:
        sub = test_df[test_df["impact_class"] == cls]
        if not sub.empty:
            samples.append(sub.iloc[0])
        else:
            samples.append(test_df.iloc[cls])

    for row in samples:
        wid = str(row["work_id"]).split("/")[-1]
        title = row.get("title", "Untitled")
        pub_year = int(row.get("publication_year", 2018))
        actual_tier = int(row.get("impact_class", 0))
        actual_label = ["Low (<50th)", "Medium (50-90th)", "High (>=90th)"][actual_tier]
        
        print(f"\n" + "="*60, flush=True)
        print(f"Paper: {title[:60]}...", flush=True)
        print(f"ID: {wid} | Pub Year: {pub_year} | Ground Truth: {actual_label}", flush=True)

        raw_work = {
            "id": wid,
            "title": title,
            "publication_year": pub_year,
            "cited_by_count": int(row.get("future_citations_5y", 0) + row.get("citations_at_cutoff", 0)),
            "authorships": [],
            "primary_topic": {"display_name": str(row.get("topic_field_name") or "Science")},
            "primary_location": {"source": {"display_name": str(row.get("venue_name") or "Journal")}}
        }

        # 1. Base M1 Prediction
        m1_pred = inference_service.predict(raw_work)
        print(f"Model M1 Prediction: {m1_pred['predicted_label']} (Prob: {m1_pred['calibrated_probabilities']})", flush=True)

        # 2. Cohort Thresholds
        t_info = inference_service.get_cohort_threshold(m1_pred.get("primary_field"), pub_year)
        print(f"Cohort Thresholds: P50={t_info.get('p50')}, P90={t_info.get('p90')}", flush=True)

        # 3. Gemini Mode G Prediction
        gemini_res = gemini_service.predict(
            paper_context=m1_pred,
            cohort_info=t_info,
            m1_prediction=m1_pred
        )

        print(f">>> Gemini Mode G Prediction: {gemini_res['predicted_impact_label']}", flush=True)
        print(f">>> Gemini Probabilities: {gemini_res['calibrated_probabilities']}", flush=True)
        print(f">>> Gemini Reasoning: {gemini_res['gemini_reasoning']}", flush=True)
        
        is_match = (gemini_res["predicted_impact_class"] == actual_tier)
        print(f">>> Match with Ground Truth: {'MATCH' if is_match else 'CLOSE'}", flush=True)

if __name__ == "__main__":
    run_test()
