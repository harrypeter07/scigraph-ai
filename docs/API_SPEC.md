# FastAPI Service API Specification

> **Base URL**: `http://localhost:8000`  
> **Documentation**: Interactive OpenAPI Docs at `/docs`  

---

## Endpoint Summary

### 1. Health Check
- **`GET /health`**
- **Response**:
  ```json
  {
    "status": "ok",
    "service": "SciGraph AI Service"
  }
  ```

### 2. Dataset Summary Statistics
- **`GET /api/v1/papers/stats`**
- **Response**:
  ```json
  {
    "total_papers": 50,
    "total_authors": 257,
    "total_institutions": 215,
    "total_citation_edges": 90
  }
  ```

### 3. Paper Impact Prediction & Explanation Lookup
- **`GET /api/v1/papers/predict/{paper_id}`**
- **Example**: `GET /api/v1/papers/predict/https://openalex.org/W2741809807`
- **Response**:
  ```json
  {
    "paper_id": "https://openalex.org/W2741809807",
    "title": "Semi-Supervised Classification with Graph Convolutional Networks",
    "publication_year": 2017,
    "predicted_impact_class": 2,
    "impact_class_label": "High Impact (>=90%)",
    "class_probabilities": {
      "Low": 0.15,
      "Medium": 0.25,
      "High": 0.60
    },
    "historical_citations_at_cutoff": 100,
    "explanation": {
      "top_contributing_features": [
        {"feature": "historical_citations", "weight": 0.42},
        {"feature": "author_h_index_history", "weight": 0.28},
        {"feature": "institution_rank", "weight": 0.18}
      ],
      "neighbourhood_subgraph_nodes": 12,
      "neighbourhood_subgraph_edges": 18
    }
  }
  ```
