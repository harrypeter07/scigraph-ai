"""Reproducible development benchmark; never promises an accuracy threshold.

python scripts/train_forecast.py --data-dir data --output-dir reports/forecast_v3
Existing 2018-2020 test is explicitly a development benchmark, not fresh evidence.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.features.forecast import ForecastFeatures, annual_counts, clean_id, probabilities


def load_records(data_dir):
    works = pd.read_parquet(data_dir / "works.parquet")
    counts = pd.read_parquet(data_dir / "counts_by_year.parquet")
    if works.work_id.map(clean_id).duplicated().any():
        raise ValueError("Duplicate work IDs")
    if counts.duplicated(["work_id", "year"]).any():
        raise ValueError("Duplicate annual citation rows")
    if (counts.cited_by_count < 0).any():
        raise ValueError("Negative citation counts")
    histories = {}
    for row in counts.to_dict("records"):
        histories.setdefault(clean_id(row["work_id"]), []).append(
            {"year": int(row["year"]), "cited_by_count": int(row["cited_by_count"])})
    records = works.where(pd.notnull(works), None).to_dict("records")
    for row in records:
        row["counts_by_year"] = histories.get(clean_id(row["work_id"]), [])
    return records


def metrics(y, p):
    p = np.asarray(p, dtype=np.float64)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            "per_class": classification_report(y, pred, labels=[0, 1, 2], output_dict=True, zero_division=0)}


def strict_temporal_evaluation(bundle, records):
    """Enforce training/calibration LABEL maturity, not only feature cutoffs."""
    ready_year = bundle.get("label_availability_year", 2022)
    eligible = [r for r in records if 2018 <= int(r["publication_year"]) <= 2020
                and int(r["publication_year"])+2 >= ready_year and r.get("impact_class") in (0, 1, 2)]
    y = np.array([int(r["impact_class"]) for r in eligible])
    p = probabilities(bundle, eligible)
    result = metrics(y, p)
    correct = p.argmax(axis=1) == y
    rng = np.random.default_rng(42)
    result.update(n=len(eligible), model_label_availability_year=ready_year,
                  majority_baseline_accuracy=float(np.mean(y == 0)),
                  by_field={name: metrics(y[mask], p[mask]) for name in sorted(set(r["primary_field_name"] for r in eligible))
                            if (mask := np.array([r["primary_field_name"] == name for r in eligible])).any()},
                  publication_years=sorted(set(int(r["publication_year"]) for r in eligible)),
                  accuracy_ci95=np.quantile([correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(1000)], [.025, .975]).tolist(),
                  notice="Calendar-year labels from training, selection and calibration mature by the target cutoff. Retrospective bibliographic snapshots still require a point-in-time metadata audit.")
    return result


def train(data_dir, output_dir, quick=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(data_dir)
    def select(lo, hi):
        return [row for row in records if lo <= int(row["publication_year"]) <= hi
                and row.get("impact_class") in (0, 1, 2)]
    train_rows, validation, test_rows = select(2012, 2016), select(2017, 2017), select(2018, 2020)
    if not all([train_rows, validation, test_rows]):
        raise ValueError("Need mature labeled temporal train, validation and development-test years")
    selection, calibration = train_test_split(validation, test_size=0.5, random_state=42,
                                              stratify=[r["impact_class"] for r in validation])
    features = ForecastFeatures().fit(train_rows, cohort_records=records)
    xt, xv, xc = [features.transform(rows) for rows in (train_rows, selection, calibration)]
    yt, yv, yc = [np.array([int(r["impact_class"]) for r in rows]) for rows in (train_rows, selection, calibration)]
    audit = {"sizes": dict(train=len(train_rows), selection=len(selection), calibration=len(calibration), development_test=len(test_rows)),
             "annual_history_missing": {name: sum(annual_counts(r) is None for r in rows)
                                        for name, rows in [("train", train_rows), ("validation", validation), ("development_test", test_rows)]},
             "feature_contract": features.version,
             "excluded_mutable_metadata": ["current_OA", "current_author_citations", "current_venue_rank"],
             "reference": "Unlabelled field/publication-year peers, each frozen at the shared year+2 cutoff; historical train-field fallback for unknown cohorts. Evaluation is cohort-assisted, not a new-cohort guarantee.",
             "test_status": "Previously inspected development benchmark; a fresh mature temporal holdout is required for confirmation",
             "data_sha256": {name: hashlib.sha256((data_dir/name).read_bytes()).hexdigest()
                             for name in ["works.parquet", "counts_by_year.parquet"]}}
    (output_dir / "input_audit.json").write_text(json.dumps(audit, indent=2))
    candidates = []
    for cw in [None, "balanced"]:
        for c in ([1.] if quick else [.01, .1, 1., 10., 100.]):
            candidates.append((f"logistic_C={c}_weight={cw}", make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight=cw, max_iter=1500, random_state=42))))
    for leaves in ([15] if quick else [7, 15, 31]):
        for cw in [None, "balanced"]:
            candidates.append((f"hist_leaves={leaves}_weight={cw}", HistGradientBoostingClassifier(max_leaf_nodes=leaves, max_iter=200, learning_rate=.05, l2_regularization=5, class_weight=cw, random_state=42)))
    try:
        from lightgbm import LGBMClassifier
        for leaves in ([15] if quick else [7, 15, 31]):
            for cw in [None, "balanced"]:
                candidates.append((f"lightgbm_leaves={leaves}_weight={cw}", LGBMClassifier(n_estimators=300, num_leaves=leaves, learning_rate=.03, reg_lambda=5, class_weight=cw, n_jobs=4, verbosity=-1, random_state=42)))
    except ImportError:
        pass
    leaderboard, fitted = [], []
    for name, model in candidates:
        model.fit(xt, yt)
        result = metrics(yv, model.predict_proba(xv))
        fitted.append((result["accuracy"], result["macro_f1"], name, model))
        leaderboard.append(dict(name=name, selection=result))
        print(name, round(result["accuracy"], 4), round(result["macro_f1"], 4), flush=True)
    fitted.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = fitted[0]
    ensemble = [item[3] for item in fitted[:3]]
    ensemble_metrics = metrics(yv, np.mean([model.predict_proba(xv) for model in ensemble], axis=0))
    leaderboard.append(dict(name="top3_average", selection=ensemble_metrics))
    use_ensemble = (ensemble_metrics["accuracy"], ensemble_metrics["macro_f1"]) > best[:2]
    members = ensemble if use_ensemble else [best[3]]
    logits = np.log(np.clip(np.mean([m.predict_proba(xc) for m in members], axis=0), 1e-12, 1))
    temperature = minimize_scalar(lambda t: log_loss(yc, softmax(logits/t, axis=1), labels=[0, 1, 2]), bounds=(.2, 5), method="bounded").x
    bundle = dict(schema_version=3, features=features, model=members[0], members=members,
                  label_availability_year=max(int(r["publication_year"])+5 for r in train_rows+validation),
                  temperature=float(temperature), feature_medians=np.median(xt, axis=0),
                  model_name="top3_average" if use_ensemble else best[2], audit=audit,
                  early_history_reference={clean_id(row["work_id"]): [
                      {"year": int(row["publication_year"])+i, "cited_by_count": value}
                      for i, value in enumerate(annual_counts(row))]
                      for row in records if annual_counts(row) is not None},
                  # Isolated outcome records are for retrospective display only.
                  # probabilities() and feature extraction never read this key.
                  evaluation_reference=[{key: row.get(key) for key in
                      ["work_id", "title", "publication_year", "primary_field_name", "citations_at_cutoff",
                       "future_citations_5y", "impact_class", "venue_name", "oa_status"]} for row in records])
    joblib.dump(bundle, output_dir / "forecast_v3.joblib")
    # Test data is used only after selection and calibration have completed.
    ytest = np.array([int(r["impact_class"]) for r in test_rows])
    p = probabilities(bundle, test_rows)
    result = metrics(ytest, p)
    rng = np.random.default_rng(42)
    correct = p.argmax(axis=1) == ytest
    result["accuracy_ci95"] = np.quantile([correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(1000)], [.025, .975]).tolist()
    result["majority_baseline_accuracy"] = float(np.mean(ytest == np.bincount(yt).argmax()))
    result["above_80_percent"] = result["accuracy"] > .8
    result["fresh_holdout_confirmed"] = False
    result["selective_accuracy"] = [{"minimum_probability": threshold, "coverage": float((p.max(axis=1)>=threshold).mean()),
                                      "accuracy": float(correct[p.max(axis=1)>=threshold].mean()) if (p.max(axis=1)>=threshold).any() else None}
                                     for threshold in [.5, .6, .7, .8, .9]]
    result["by_year"] = {str(year): metrics(ytest[mask], p[mask]) for year in [2018, 2019, 2020]
                         if (mask := np.array([r["publication_year"] == year for r in test_rows])).any()}
    result["by_field"] = {name: metrics(ytest[mask], p[mask]) for name in features.fields
                          if (mask := np.array([r["primary_field_name"] == name for r in test_rows])).any()}
    from sklearn.base import clone
    learning_curve = []
    for size in sorted(set([1000, 2500, 5000, 7500, 15000, 25000, len(train_rows)])):
        if size > len(train_rows):
            continue
        indices = np.arange(len(train_rows)) if size == len(train_rows) else train_test_split(np.arange(len(train_rows)), train_size=size, stratify=yt, random_state=42)[0]
        probe = clone(best[3]).fit(xt[indices], yt[indices])
        learning_curve.append(dict(training_size=len(indices), selection=metrics(yv, probe.predict_proba(xv))))
    report = dict(selected_model=bundle["model_name"], temperature=float(temperature), candidates=leaderboard, learning_curve=learning_curve, development_test=result,
                  strict_temporal_test=strict_temporal_evaluation(bundle, records), audit=audit)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"selected": bundle["model_name"], "development_accuracy": result["accuracy"], "ci95": result["accuracy_ci95"], "macro_f1": result["macro_f1"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "forecast_v3")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.quick)
