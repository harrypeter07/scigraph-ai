"""Versioned, portable early-impact features shared by training and serving.

Cohort references use unlabelled, cutoff-frozen peers with a TRAIN-field fallback. Missing annual history is
explicit and never replaced with an invented trajectory. Calendar-year contract:
publication year through publication year + 2 inclusive.
"""
import numpy as np


def clean_id(value):
    return str(value or "").rstrip("/").split("/")[-1]


def annual_counts(record):
    year = int(record["publication_year"])
    history = record.get("counts_by_year")
    if history is None:
        history = record.get("historical_citation_trajectory_to_cutoff")
    if not isinstance(history, (list, tuple, np.ndarray)):
        return None
    counts = {}
    for item in history:
        y = int(item["year"])
        if year <= y <= year + 2:
            counts[y] = counts.get(y, 0) + max(0, int(item.get("cited_by_count", 0)))
    return [counts.get(year + i, 0) for i in range(3)] if counts else None


def number(value, default=0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def field(record):
    topic = record.get("primary_topic") or {}
    return str(record.get("primary_field_name") or
               (topic.get("field") or {}).get("display_name") or "Unknown")


class ForecastFeatures:
    version = "early-impact-v3-calendar"
    names = ["log_early_citations", "log_y0", "log_y1", "log_y2",
             "velocity", "acceleration", "growth_ratio", "last_year_share",
             "annual_history_missing", "early_count_missing",
             "field_early_percentile", "field_early_zscore", "cohort_missing",
             "log_references", "author_count", "log_title_words"]

    def fit(self, records, cohort_records=None):
        # If a source omits authorships, do not let partial metadata coverage
        # become a predictive signal that disappears on live OpenAlex records.
        self.use_author_count = np.mean([np.isfinite(number(r.get("num_authors"), np.nan)) for r in records]) >= .95
        self.references = {}
        values = {}
        for row in records:
            counts = annual_counts(row)
            total = sum(counts) if counts is not None else number(row.get("citations_at_cutoff"), np.nan)
            if np.isfinite(total):
                values.setdefault(field(row), []).append(np.log1p(max(0, total)))
        self.references = {key: np.sort(val) for key, val in values.items()}
        self.global_reference = np.sort(np.concatenate(list(self.references.values())))
        self.fields = sorted(self.references)
        # Unlabelled peers from the SAME publication year have the same calendar
        # cutoff. Store early counts only; never store their future targets here.
        self.year_references = {}
        groups = {}
        for row in cohort_records or records:
            counts = annual_counts(row)
            if counts is not None:
                groups.setdefault((field(row), int(row["publication_year"])), []).append(np.log1p(sum(counts)))
        self.year_references = {key: np.sort(values) for key, values in groups.items() if len(values) >= 30}
        self.feature_names = self.names + ["field=" + name for name in self.fields]
        return self

    def transform(self, records):
        features = []
        for row in records:
            counts = annual_counts(row)
            missing = counts is None
            total_raw = row.get("citations_at_cutoff")
            total = sum(counts) if counts is not None else number(total_raw)
            y0, y1, y2 = counts if counts is not None else (0, 0, 0)
            log_total = np.log1p(max(0, total))
            reference_key = (field(row), int(row["publication_year"]))
            reference = self.year_references.get(reference_key, self.references.get(field(row), self.global_reference))
            percentile = (np.searchsorted(reference, log_total, side="left") +
                          np.searchsorted(reference, log_total, side="right")) / (2 * len(reference))
            z = (log_total - reference.mean()) / max(reference.std(), 1e-6)
            refs = row.get("referenced_works_count", len(row.get("referenced_works") or []))
            authors = row.get("num_authors", len(row.get("authorships") or []))
            values = [log_total, np.log1p(y0), np.log1p(y1), np.log1p(y2),
                      y2-y1, y2-2*y1+y0, (y2+1)/(y0+y1+1), y2/max(total, 1),
                      int(missing), int(missing and not np.isfinite(number(total_raw, np.nan))),
                      percentile, z, int(reference_key not in self.year_references),
                      np.log1p(max(0, number(refs))), min(100, max(0, number(authors))) if getattr(self, "use_author_count", True) else 0,
                      np.log1p(len(str(row.get("title") or "").split()))]
            features.append(values + [int(field(row) == name) for name in self.fields])
        return np.asarray(features, dtype=np.float64)


def probabilities(bundle, records):
    """Same function called by training evaluation and the API."""
    from scipy.special import softmax
    x = bundle["features"].transform(records)
    models = bundle.get("members") or [bundle["model"]]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        raw = np.mean([model.predict_proba(x) for model in models], axis=0)
    return softmax(np.log(np.clip(raw, 1e-12, 1)) / bundle.get("temperature", 1), axis=1)


def explain(bundle, record):
    """Model sensitivity to replacing each feature with its training median."""
    x = bundle["features"].transform([record])
    models = bundle.get("members") or [bundle["model"]]
    from scipy.special import softmax
    def predict(values):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):
            raw = np.mean([model.predict_proba(values) for model in models], axis=0)
        return softmax(np.log(np.clip(raw, 1e-12, 1)) / bundle.get("temperature", 1), axis=1)
    original = predict(x)[0]
    cls = int(original.argmax())
    variants = np.repeat(x, x.shape[1], axis=0)
    np.fill_diagonal(variants, bundle["feature_medians"])
    deltas = original[cls] - predict(variants)[:, cls]
    scale = max(float(np.abs(deltas).sum()), 1e-12)
    return sorted([dict(feature_name=name, raw_value=float(x[0, i]),
                        importance_score=float(abs(deltas[i])/scale),
                        delta_probability=float(abs(deltas[i])),
                        delta_percentage_points=float(abs(deltas[i])*100),
                        direction="positive" if deltas[i] >= 0 else "negative",
                        provenance="served_model | training_median_occlusion",
                        explained_class=cls)
                   for i, name in enumerate(bundle["features"].feature_names)],
                  key=lambda item: item["importance_score"], reverse=True)
