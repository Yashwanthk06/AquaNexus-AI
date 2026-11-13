import csv
import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from random import Random

import numpy as np
from fastapi import APIRouter, Query
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

router = APIRouter()
DATA = Path(__file__).resolve().parent / "data" / "real"


@lru_cache(maxsize=1)
def train_real_forecast():
    readings = {}
    with (DATA / "clean_daily_demand.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            key = (row["building"], date.fromisoformat(row["date_utc"]))
            readings[key] = float(row["demand_litres"])

    buildings = sorted({building for building, _ in readings})
    rows = []
    for (building, day), actual in sorted(readings.items()):
        history = [
            readings.get((building, day - timedelta(days=lag)))
            for lag in range(1, 8)
        ]
        if any(value is None for value in history):
            continue
        building_flags = [int(building == name) for name in buildings]
        weekday_flags = [int(day.weekday() == number) for number in range(7)]
        features = [*history, sum(history) / 7, *weekday_flags, *building_flags]
        rows.append((building, day, actual, features))

    # Every home uses the same date boundaries to avoid mixing future
    # readings from one home into training for another home's test dates.
    days = sorted({row[1] for row in rows})
    if len(days) < 60:
        raise ValueError("At least 60 usable dates are required.")
    validation_start = days[int(len(days) * 0.65)]
    test_start = days[int(len(days) * 0.80)]
    training = [row for row in rows if row[1] < validation_start]
    validation = [row for row in rows if validation_start <= row[1] < test_start]
    test = [row for row in rows if row[1] >= test_start]

    def inputs(group):
        return [row[3] for row in group]

    def targets(group):
        return [row[2] for row in group]

    candidates = []
    for depth, leaf in [(6, 5), (12, 5), (None, 12)]:
        model = RandomForestRegressor(
            n_estimators=120, max_depth=depth, min_samples_leaf=leaf,
            random_state=42, n_jobs=1,
        )
        model.fit(inputs(training), targets(training))
        error = mean_absolute_error(targets(validation), model.predict(inputs(validation)))
        candidates.append((error, model))

    validation_mae, model = min(candidates, key=lambda candidate: candidate[0])
    fit_rows = training + validation
    model.fit(inputs(fit_rows), targets(fit_rows))
    actual = targets(test)
    predicted = model.predict(inputs(test))

    def metrics(estimate):
        return {
            "mae_litres": round(float(mean_absolute_error(actual, estimate)), 2),
            "rmse_litres": round(float(np.sqrt(mean_squared_error(actual, estimate))), 2),
        }

    report = {
        "data_source": "Aalborg University household water meter readings",
        "unit": "litres per building per UTC day",
        "evaluation": "One-day-ahead forecast using observed past readings",
        "buildings": len(buildings),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_start": test_start.isoformat(),
        "test_end": days[-1].isoformat(),
        "validation_mae_litres": round(float(validation_mae), 2),
        "selected_settings": {
            "max_depth": model.max_depth,
            "min_samples_leaf": model.min_samples_leaf,
        },
        "test_scores": {
            "random_forest": metrics(predicted),
            "yesterday": metrics([row[3][0] for row in test]),
            "same_day_last_week": metrics([row[3][6] for row in test]),
        },
    }
    examples = [
        {
            "building": building,
            "date_utc": day.isoformat(),
            "actual_litres": observed,
            "predicted_litres": round(float(estimate), 2),
            "absolute_error_litres": round(abs(float(estimate) - observed), 2),
        }
        for (building, day, observed, _), estimate in zip(test, predicted)
    ]
    (DATA / "real_forecast_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report, examples


@router.get("/api/real-forecast-report")
def real_forecast_report(seed: int = Query(default=42, ge=0, le=999999999)):
    report, examples = train_real_forecast()
    dates = sorted({row["date_utc"] for row in examples})
    selected = Random(seed).choice(dates)
    return {
        **report,
        "seed": seed,
        "selected_test_date": selected,
        "predicted_vs_actual": [row for row in examples if row["date_utc"] == selected],
    }


if __name__ == "__main__":
    print(json.dumps(real_forecast_report(42), indent=2))
