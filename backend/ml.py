import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

DATA = Path(__file__).resolve().parent / "data"
ZONES = ("North", "Central", "Hilltop")
FEATURES = ["temperature_c", "rainfall_mm", "weekend"]


def write_csv(name, header, rows):
    DATA.mkdir(exist_ok=True)
    with (DATA / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def forecast_metrics(actual, prediction):
    return {
        "mae_kl": float(mean_absolute_error(actual, prediction)),
        "rmse_kl": float(np.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)),
    }


def make_meter_data(rng, count, anomaly_count):
    labels = np.zeros(count, dtype=int)
    labels[rng.choice(count, anomaly_count, replace=False)] = 1
    readings = rng.normal(0.035, 0.012, count)
    # Mild incidents overlap with normal readings so the test includes hard cases.
    readings[labels == 1] += rng.uniform(0.015, 0.12, anomaly_count)
    return readings.clip(0, 1).reshape(-1, 1), labels


@lru_cache(maxsize=1)
def train_models():
    rng = np.random.default_rng(42)
    days = np.arange(365)
    temperature = (
        30 + 5 * np.sin(2 * np.pi * days / 365)
        + 2 * np.sin(days / 17) + rng.normal(0, 2, len(days))
    )
    rainfall = np.where(rng.random(len(days)) < 0.55, 0, rng.uniform(0, 20, len(days)))
    weekend = (days % 7 >= 5).astype(int)
    features = np.column_stack([temperature, rainfall, weekend])
    demand = (
        np.array([8.0, 12.0, 8.0])
        + np.maximum(0, temperature[:, None] - 26) * np.array([0.23, 0.30, 0.25])
        - np.minimum(12, rainfall[:, None]) * 0.10
        + weekend[:, None] * np.array([0.65, 0.90, 0.50])
        + rng.normal(0, 0.45, (len(days), len(ZONES)))
    )

    # Days 1-245: train. Days 246-305: validation. Days 306-365: test.
    train_end, test_start = 245, 305
    split = np.repeat(["train", "validation", "test"], [245, 60, 60])
    write_csv(
        "synthetic_demand.csv",
        ["day", "split", *FEATURES, "north_kl", "central_kl", "hilltop_kl"],
        zip(days + 1, split, *features.T, *demand.T),
    )

    candidates = []
    for depth, leaf in [(5, 4), (10, 3), (None, 2)]:
        model = RandomForestRegressor(
            n_estimators=120, max_depth=depth, min_samples_leaf=leaf,
            random_state=42, n_jobs=1,
        )
        model.fit(features[:train_end], demand[:train_end])
        validation_mae = mean_absolute_error(
            demand[train_end:test_start],
            model.predict(features[train_end:test_start]),
        )
        candidates.append((float(validation_mae), depth, leaf, model))

    validation_mae, depth, leaf, model = min(candidates, key=lambda item: item[0])
    model.fit(features[:test_start], demand[:test_start])
    actual = demand[test_start:]
    prediction = model.predict(features[test_start:])
    # Yesterday's observed demand is available for each one-day-ahead prediction.
    baseline = demand[test_start - 1:-1]

    normal_readings, normal_labels = make_meter_data(rng, 400, 0)
    validation_readings, validation_labels = make_meter_data(rng, 180, 25)
    test_readings, test_labels = make_meter_data(rng, 180, 25)

    detector = IsolationForest(n_estimators=120, random_state=42, n_jobs=1)
    detector.fit(normal_readings)
    validation_scores = -detector.score_samples(validation_readings)
    thresholds = np.unique(validation_scores)
    threshold = float(max(
        thresholds,
        key=lambda value: f1_score(
            validation_labels, validation_scores >= value, zero_division=0
        ),
    ))
    test_scores = -detector.score_samples(test_readings)
    test_flags = test_scores >= threshold
    tn, fp, fn, tp = confusion_matrix(test_labels, test_flags, labels=[0, 1]).ravel()

    meter_rows = [
        (split_name, float(reading), int(label))
        for split_name, readings, labels in [
            ("train", normal_readings, normal_labels),
            ("validation", validation_readings, validation_labels),
            ("test", test_readings, test_labels),
        ]
        for reading, label in zip(readings[:, 0], labels)
    ]
    write_csv(
        "synthetic_meter_readings.csv",
        ["split", "meter_imbalance_fraction", "injected_anomaly"],
        meter_rows,
    )

    report = {
        "data_source": "Synthetic demo data, generator seed 42",
        "forecast": {
            "model": "Random Forest Regressor",
            "features": FEATURES,
            "days": {"train": 245, "validation": 60, "test": 60},
            "selected_settings": {"max_depth": depth, "min_samples_leaf": leaf},
            "validation_mae_kl": validation_mae,
            "test": {
                "random_forest": forecast_metrics(actual.ravel(), prediction.ravel()),
                "previous_day": forecast_metrics(actual.ravel(), baseline.ravel()),
                "by_zone": {
                    zone: {
                        "random_forest": forecast_metrics(actual[:, i], prediction[:, i]),
                        "previous_day": forecast_metrics(actual[:, i], baseline[:, i]),
                    }
                    for i, zone in enumerate(ZONES)
                },
            },
        },
        "anomaly": {
            "model": "Isolation Forest",
            "threshold": threshold,
            "test_readings": len(test_labels),
            "injected_test_anomalies": int(test_labels.sum()),
            "precision": float(precision_score(test_labels, test_flags, zero_division=0)),
            "recall": float(recall_score(test_labels, test_flags, zero_division=0)),
            "f1": float(f1_score(test_labels, test_flags, zero_division=0)),
            "confusion_matrix": {
                "true_negative": int(tn), "false_positive": int(fp),
                "false_negative": int(fn), "true_positive": int(tp),
            },
        },
    }

    (DATA / "model_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return model, detector, threshold, report


if __name__ == "__main__":
    print(json.dumps(train_models()[3], indent=2))
