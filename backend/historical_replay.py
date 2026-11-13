import numpy as np
from fastapi import APIRouter, HTTPException, Query

from real_forecast import real_forecast_report
from twin import simulate_day

router = APIRouter()

GROUPS = (
    ("North", (1, 2, 3)),
    ("Central", (4, 5, 6, 7)),
    ("Hilltop", (8, 9, 10)),
)
DEMAND_SCALE = 10


@router.get("/api/historical-replay")
def historical_replay(
    seed: int = Query(default=4721, ge=0, le=999999999),
    minutes: int = Query(default=30, ge=0, le=60),
):
    from main import CAPACITY, FLOW, LEAK, START, allocate

    report = real_forecast_report(seed)
    readings = {row["building"]: row for row in report["predicted_vs_actual"]}
    zone_inputs = []

    for name, numbers in GROUPS:
        buildings = [f"BUILD-D-{number:02d}" for number in numbers]
        if any(building not in readings for building in buildings):
            raise HTTPException(422, "This test date is missing a building. Try another seed.")

        forecast_litres = sum(readings[b]["predicted_litres"] for b in buildings)
        actual_litres = sum(readings[b]["actual_litres"] for b in buildings)
        zone_inputs.append({
            "name": name,
            "buildings": buildings,
            "forecast_demand_kl": forecast_litres * DEMAND_SCALE / 1000,
            "actual_demand_kl": actual_litres * DEMAND_SCALE / 1000,
        })

    forecast = np.array([zone["forecast_demand_kl"] for zone in zone_inputs])

    # Same simulated pipe losses and source conditions for both plans.
    source_factor = 0.75 + 0.25 * 8 / 20
    rates = FLOW * source_factor * (1 - LEAK)

    # Schedule using only the forecast. Actual demand enters during replay.
    optimized_slots = allocate(forecast, minutes, rates)
    equal_slots = np.full(len(GROUPS), minutes / len(GROUPS))

    def replay(slots):
        plan = {"zones": []}
        for i, zone in enumerate(zone_inputs):
            plan["zones"].append({
                "name": zone["name"],
                "start": float(START[i]),
                "capacity": float(CAPACITY[i]),
                "minutes": float(slots[i]),
                # Pass the scheduled inflow. The twin applies tank limits
                # minute by minute and accounts for concurrent consumption.
                "after_fill": float(START[i] + rates[i] * slots[i]),
                "demand": zone["actual_demand_kl"],
            })

        result = simulate_day(plan)
        for zone in result["zones"]:
            demand = zone["demand_kl"]
            zone["coverage_percent"] = (
                round(100 * (1 - zone["shortage_kl"] / demand), 2)
                if demand > 0 else None
            )

        active = [z for z in result["zones"] if z["coverage_percent"] is not None]
        worst = min(active, key=lambda z: z["coverage_percent"]) if active else None
        return {
            "total_shortage_kl": round(result["total_shortage_kl"], 3),
            "worst_served_zone": worst["name"] if worst else None,
            "worst_zone_coverage_percent": worst["coverage_percent"] if worst else None,
            "worst_zone_first_unserved_time": worst["first_unserved_time"] if worst else None,
            "zones": result["zones"],
        }

    equal = replay(equal_slots)
    optimized = replay(optimized_slots)
    saved = round(equal["total_shortage_kl"] - optimized["total_shortage_kl"], 3)

    return {
        "seed": seed,
        "selected_test_date": report["selected_test_date"],
        "supply_window_minutes": minutes,
        "data_source": report["data_source"],
        "simulation_assumptions": (
            "Real held-out household daily totals, grouped and scaled 10x. "
            "Tank capacities, pipe losses, pump rates and hourly demand shape are simulated."
        ),
        "zone_inputs": zone_inputs,
        "equal_time": equal,
        "optimized": optimized,
        "shortage_reduction_kl": saved,
    }
