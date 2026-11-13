
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.optimize import linprog

from ml import train_models

from real_forecast import router as real_forecast_router

app = FastAPI(title="AquaNexus AI")
from historical_replay import router as historical_replay_router
app.include_router(historical_replay_router)

app.include_router(real_forecast_router)

ZONES = ["North", "Central", "Hilltop"]
CAPACITY = np.array([12.0, 15.0, 8.5])
START = np.array([2.0, 3.0, 1.0])
FLOW = np.array([0.53, 0.65, 0.42])
LEAK = np.array([0.02, 0.03, 0.28])


class Scenario(BaseModel):
    minutes: int = Field(default=60, ge=0, le=60)
    rainfall: float = Field(default=8, ge=0, le=20)
    repaired: bool = False



def train():
    model, detector, threshold, report = train_models()
    meter_readings = np.array([[0.034], [0.038], [0.28]])
    flagged = -detector.score_samples(meter_readings) >= threshold
    forecast = report["forecast"]["test"]
    return model, forecast["random_forest"]["mae_kl"], forecast["previous_day"]["mae_kl"], flagged


@app.get("/api/model-report")
def model_report():
    return train_models()[3]


def allocate(demand, minutes, rates):
    # Variables: minutes per zone, then the lowest demand coverage across zones.
    # Maximize minimum coverage while respecting tank space and one shared pump.
    constraints = np.vstack([
        [1, 1, 1, 0],
        np.column_stack([-np.diag(rates), demand]),
    ])
    time_limits = np.minimum(CAPACITY - START, np.maximum(0, demand - START)) / rates
    bounds = [(0, limit) for limit in time_limits] + [(0, 1)]
    fair = linprog(
        [0, 0, 0, -1], A_ub=constraints,
        b_ub=np.r_[minutes, START], bounds=bounds, method="highs",
    )
    if not fair.success:
        raise HTTPException(500, "Could not calculate a fair supply schedule.")

    # Keep that coverage floor, then maximize useful water delivered.
    bounds[-1] = (max(0, fair.x[-1] - 1e-8), 1)
    efficient = linprog(
        np.r_[-rates, 0], A_ub=constraints,
        b_ub=np.r_[minutes, START], bounds=bounds, method="highs",
    )
    if not efficient.success:
        raise HTTPException(500, "Could not calculate water allocation.")
    return efficient.x[:3]


@app.post("/api/simulate")
def simulate(scenario: Scenario):
    model, forecast_mae, baseline_mae, flagged = train()
    demand = model.predict([[34, scenario.rainfall, 0]])[0]

    # Drier conditions reduce source throughput in this fictional village.
    source_factor = 0.75 + 0.25 * scenario.rainfall / 20
    losses = LEAK.copy()
    if scenario.repaired:
        losses[2] = 0.03
    rates = FLOW * source_factor * (1 - losses)

    plan = allocate(demand, scenario.minutes, rates)
    after_fill = START + plan * rates
    shortage = np.maximum(0, demand - after_fill)
    ending = np.maximum(0, after_fill - demand)
    worst = int(np.argmax(shortage))

    if shortage.sum() < 0.01:
        recommendation = "Use this schedule. Starting storage and the planned refill cover forecast demand in every zone."
    else:
        recommendation = f"{ZONES[worst]} has the largest gap ({shortage[worst]:.2f} kL)."
        if CAPACITY[worst] < demand[worst]:
            recommendation += " Its tank cannot hold a full day of demand. Test a second refill or more storage."
        else:
            recommendation += " Test a longer supply window or more starting storage."
        if flagged[2] and not scenario.repaired:
            recommendation += " Inspect the Hilltop line and compare the simulated repair."

    return {
        "data_source": "Synthetic demo data",
        "demand": float(demand.sum()),
        "shortage": float(shortage.sum()),
        "forecast_mae": forecast_mae,
        "baseline_mae": baseline_mae,
        "recommendation": recommendation,
        "zones": [
            {
                "name": zone,
                "demand": float(demand[i]),
                "capacity": float(CAPACITY[i]),
                "start": float(START[i]),
                "after_fill": float(after_fill[i]),
                "end": float(ending[i]),
                "minutes": float(plan[i]),
                "shortage": float(shortage[i]),
                "anomaly": bool(flagged[i]),
            }
            for i, zone in enumerate(ZONES)
        ],
    }



@app.post("/api/digital-twin")
def digital_twin(scenario: Scenario):
    from twin import simulate_day

    result = simulate_day(simulate(scenario))
    result["scenario"] = scenario.model_dump()
    return result


@app.get("/api/fresh-test")
def fresh_test(seed: int):
    from ml import make_meter_data

    if not 0 <= seed <= 999999999:
        raise HTTPException(422, "Seed must be between 0 and 999999999.")

    model, detector, threshold, _ = train_models()
    rng = np.random.default_rng(seed)
    count = 60

    # New synthetic inputs and known demand, using the training data assumptions.
    temperature = rng.uniform(23, 37, count)
    rainfall = np.where(rng.random(count) < 0.55, 0, rng.uniform(0, 20, count))
    weekend = (np.arange(count) % 7 >= 5).astype(int)
    features = np.column_stack([temperature, rainfall, weekend])
    actual = (
        np.array([8.0, 12.0, 8.0])
        + np.maximum(0, temperature[:, None] - 26) * np.array([0.23, 0.30, 0.25])
        - np.minimum(12, rainfall[:, None]) * 0.10
        + weekend[:, None] * np.array([0.65, 0.90, 0.50])
        + rng.normal(0, 0.45, (count, 3))
    )
    predicted = model.predict(features)

    readings, labels = make_meter_data(rng, 180, 25)
    flags = -detector.score_samples(readings) >= threshold
    true_positive = int(np.sum(flags & (labels == 1)))
    false_positive = int(np.sum(flags & (labels == 0)))
    false_negative = int(np.sum(~flags & (labels == 1)))

    return {
        "seed": seed,
        "data_source": "Fresh synthetic test data",
        "forecast_days": count,
        "mae_litres": float(np.abs(actual - predicted).mean() * 1000),
        "accuracy": float(np.mean(flags == labels)),
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, true_positive + false_negative),
        "anomalies_caught": true_positive,
        "false_alarms": false_positive,
        "injected_anomalies": int(labels.sum()),
        "meter_readings": len(labels),
        "example": {
            "zone": "Hilltop",
            "temperature_c": float(temperature[0]),
            "rainfall_mm": float(rainfall[0]),
            "actual_kl": float(actual[0, 2]),
            "predicted_kl": float(predicted[0, 2]),
        },
    }


