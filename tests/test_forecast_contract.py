"""Temporal leakage, serving parity and real graph-neighbor regression tests."""
import copy
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml.features.forecast import ForecastFeatures, annual_counts, probabilities, explain
from ml.gnn.citation_forecast import CitationNeighborhoods


def row(i=0, year=2016):
    return dict(work_id=f"W{i}", title="Example scientific paper", publication_year=year,
                primary_field_name="Computer Science", num_authors=2,
                referenced_works_count=10, counts_by_year=[
                    dict(year=year+j, cited_by_count=i+j+1) for j in range(3)])


def fitted_bundle():
    records = [row(i) for i in range(60)]
    features = ForecastFeatures().fit(records)
    x = features.transform(records)
    model = LogisticRegression(max_iter=1000).fit(x, np.repeat([0, 1, 2], 20))
    return dict(features=features, model=model, temperature=1.2, feature_medians=np.median(x, axis=0))


def test_future_values_and_labels_cannot_change_features():
    builder = ForecastFeatures().fit([row(i) for i in range(60)])
    original = row(1)
    mutated = copy.deepcopy(original)
    mutated["counts_by_year"].append(dict(year=2025, cited_by_count=999999))
    mutated.update(future_citations_5y=999999, impact_class=2, cohort_percentile=1., fwci=1000, is_oa=True)
    np.testing.assert_array_equal(builder.transform([original]), builder.transform([mutated]))


def test_annual_history_is_keyed_by_year_and_missing_is_explicit():
    original = row()
    original["counts_by_year"].reverse()
    assert annual_counts(original) == [1, 2, 3]
    original["counts_by_year"] = []
    assert annual_counts(original) is None
    builder = ForecastFeatures().fit([row(i) for i in range(60)])
    assert builder.transform([original])[0, 8] == 1


def test_training_and_serialized_serving_probabilities_match(tmp_path):
    bundle = fitted_bundle()
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    p = probabilities(bundle, [row(7)])
    np.testing.assert_array_equal(p, probabilities(restored, [row(7)]))
    np.testing.assert_allclose(p.sum(axis=1), 1.)
    for effect in explain(restored, row(7)):
        assert 0 <= effect["delta_probability"] <= 1
        assert effect["explained_class"] == p.argmax()


def test_graph_uses_real_neighbors_and_target_cutoff():
    root, prior, future = row(0, 2016), row(1, 2015), row(2, 2020)
    edges = [("W1", "W0"), ("W2", "W0"), ("W0", "W1")]
    builder = CitationNeighborhoods([root, prior, future], edges)
    original, mask, context = builder.build(root)
    assert mask.sum() == 2
    assert all(item["work_id"] == "W1" for item in context)
    assert all(item["feature_cutoff"] == 2018 for item in context)
    assert np.any(original[0] != 0)
    mutated = copy.deepcopy(prior)
    mutated["counts_by_year"].append(dict(year=2025, cited_by_count=999999))
    changed = CitationNeighborhoods([root, mutated, future], edges)
    np.testing.assert_array_equal(original, changed.build(root)[0])


def test_missing_graph_has_no_fake_edges():
    values, mask, context = CitationNeighborhoods([row()], []).build(row())
    assert mask.sum() == 0
    assert context == []
    assert not values.any()


def test_unlabelled_cohort_references_ignore_future_targets():
    peers = [row(i, 2019) for i in range(60)]
    altered = copy.deepcopy(peers)
    for record in altered:
        record.update(impact_class=2, future_citations_5y=999999)
    a = ForecastFeatures().fit([row(i) for i in range(60)], peers)
    b = ForecastFeatures().fit([row(i) for i in range(60)], altered)
    np.testing.assert_array_equal(a.transform(peers), b.transform(peers))


def test_sparse_author_metadata_cannot_change_live_predictions():
    train = [row(i) for i in range(60)]
    for record in train:
        record["num_authors"] = None
    features = ForecastFeatures().fit(train)
    original = row()
    changed = dict(original, num_authors=100)
    np.testing.assert_array_equal(features.transform([original]), features.transform([changed]))


def test_gemini_cannot_replace_statistical_probabilities(monkeypatch):
    import ml.gemini_service as module
    service = module.GeminiService()
    monkeypatch.setattr(module, "HAS_GENAI", True)
    service._keys = ["test-placeholder"]
    monkeypatch.setattr(service, "_generate_with_retry", lambda *a, **k:
                        '{"predicted_tier":"High","gemini_reasoning":"An uncertain forecast."}')
    model = dict(predicted_impact_class=0, predicted_label="Low (<50th)",
                 calibrated_probabilities={"Low (<50th)": .8, "Medium (50-90th)": .15, "High (>=90th)": .05},
                 model_version="test-classifier")
    result = service.predict({}, {}, model)
    assert result["predicted_impact_class"] == 0
    assert result["calibrated_probabilities"] == model["calibrated_probabilities"]
    assert result["model_version"] == "test-classifier"


def test_retrospective_labels_never_enter_forecasts():
    bundle = fitted_bundle()
    expected = probabilities(bundle, [row(7)])
    bundle["evaluation_reference"] = [{"work_id": "W7", "impact_class": 2, "future_citations_5y": 999999}]
    np.testing.assert_array_equal(expected, probabilities(bundle, [row(7)]))


def test_graph_serving_matches_training_forward(tmp_path):
    import pytest
    torch = pytest.importorskip("torch")
    from scipy.special import softmax
    from sklearn.preprocessing import StandardScaler
    from ml.gnn.citation_forecast import make_model, CitationPredictor
    tabular = fitted_bundle()
    records = [row(i) for i in range(60)]
    roots = tabular["features"].transform(records)
    root_scaler = StandardScaler().fit(roots)
    neighbor_scaler = StandardScaler().fit(np.vstack([np.zeros(8), np.ones(8)]))
    neighborhoods = CitationNeighborhoods(records, [("W1", "W7"), ("W7", "W2")])
    model = make_model(roots.shape[1]); model.eval()
    graph = dict(states=[model.state_dict()], root_dim=roots.shape[1], root_scaler=root_scaler,
                 neighbor_scaler=neighbor_scaler, neighborhoods=neighborhoods,
                 graph_weight=.25, temperature=1.3, report={})
    path = tmp_path / "graph.joblib"
    joblib.dump(graph, path)
    graph = joblib.load(path)
    record = row(7)
    neighbors, mask, _ = neighborhoods.build(record)
    values = neighbor_scaler.transform(neighbors.reshape(-1, 8)).reshape(1, *neighbors.shape).astype(np.float32)
    root = root_scaler.transform(tabular["features"].transform([record])).astype(np.float32)
    with torch.no_grad():
        graph_p = torch.softmax(model(torch.tensor(root), torch.tensor(values), torch.tensor(mask[None])), dim=1).numpy()
    expected = softmax(np.log(.75*probabilities(tabular, [record])+.25*graph_p)/1.3, axis=1)
    runtime = CitationPredictor(graph, tabular)
    np.testing.assert_allclose(expected, runtime.predict([record]), atol=1e-7)
    evidence = runtime.evidence(record)
    assert len(evidence["neighbors"]) == 2
    experimental = runtime._graph_probabilities([record])[0]
    cls = int(experimental.argmax())
    for neighbor in evidence["neighbors"]:
        relation = 0 if neighbor["relation"] == "cited_by" else 1
        removed = runtime._graph_probabilities([record], removed_neighbor=(relation, 0))[0]
        assert neighbor["explained_experimental_class"] == cls
        np.testing.assert_allclose(neighbor["experimental_node_delta_percentage_points"], (experimental[cls]-removed[cls])*100, atol=1e-5)
    graph["graph_weight"] = 0
    graph["temperature"] = 3
    np.testing.assert_array_equal(probabilities(tabular, [record]), runtime.predict([record]))


def test_label_maturity_excludes_too_early_forecast_dates():
    from scripts.train_forecast import strict_temporal_evaluation
    bundle = fitted_bundle()
    bundle["label_availability_year"] = 2022
    records = [dict(row(i, year), impact_class=0) for i, year in enumerate([2018, 2019, 2020])]
    result = strict_temporal_evaluation(bundle, records)
    assert result["n"] == 1
    assert result["publication_years"] == [2020]
