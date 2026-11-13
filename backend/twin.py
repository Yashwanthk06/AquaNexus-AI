import numpy as np

# Relative demand in each clock hour, starting at midnight.
HOURLY_WEIGHTS = np.array([
    1, 1, 1, 1, 2, 3, 7, 9, 8, 5, 4, 4,
    5, 4, 4, 4, 5, 7, 9, 8, 6, 4, 3, 2,
], dtype=float)
HOURLY_WEIGHTS /= HOURLY_WEIGHTS.sum()
START_HOUR = 6


def clock(elapsed_minutes):
    total = START_HOUR * 60 + elapsed_minutes
    day, minute_of_day = divmod(total, 1440)
    hour, minute = divmod(minute_of_day, 60)
    suffix = " next day" if day else ""
    return f"{hour:02d}:{minute:02d}{suffix}"


def simulate_day(plan):
    zones = plan["zones"]
    storage = np.array([zone["start"] for zone in zones])
    capacity = np.array([zone["capacity"] for zone in zones])
    demand = np.array([zone["demand"] for zone in zones])
    minutes = np.array([zone["minutes"] for zone in zones])
    planned_refill = np.array([
        zone["after_fill"] - zone["start"] for zone in zones
    ])
    refill_rate = np.divide(
        planned_refill, minutes,
        out=np.zeros_like(planned_refill),
        where=minutes > 0,
    )

    slot_end = np.cumsum(minutes)
    slot_start = slot_end - minutes
    shortage = np.zeros(len(zones))
    received = np.zeros(len(zones))
    first_unserved = [None] * len(zones)
    timeline = [{
        "elapsed_minutes": 0,
        "time": clock(0),
        "tank_kl": storage.tolist(),
        "cumulative_shortage_kl": shortage.tolist(),
    }]

    for minute in range(1440):
        # Fraction of this minute spent pumping to each zone.
        overlap = np.maximum(
            0, np.minimum(minute + 1, slot_end) - np.maximum(minute, slot_start)
        )
        inflow = np.minimum(refill_rate * overlap, capacity - storage)
        storage += inflow
        received += inflow

        hour = (START_HOUR + minute // 60) % 24
        need = demand * HOURLY_WEIGHTS[hour] / 60
        served = np.minimum(storage, need)
        unserved = need - served
        storage -= served
        shortage += unserved

        for i, gap in enumerate(unserved):
            if gap > 1e-9 and first_unserved[i] is None:
                first_unserved[i] = clock(minute)

        if (minute + 1) % 60 == 0:
            timeline.append({
                "elapsed_minutes": minute + 1,
                "time": clock(minute + 1),
                "tank_kl": storage.tolist(),
                "cumulative_shortage_kl": shortage.tolist(),
            })

    return {
        "data_source": "Synthetic demo data",
        "total_shortage_kl": float(shortage.sum()),
        "zones": [
            {
                "name": zone["name"],
                "tank_capacity_kl": float(capacity[i]),
                "starting_storage_kl": zone["start"],
                "scheduled_pump_minutes": zone["minutes"],
                "received_kl": float(received[i]),
                "demand_kl": zone["demand"],
                "shortage_kl": float(shortage[i]),
                "ending_storage_kl": float(storage[i]),
                "first_unserved_time": first_unserved[i],
            }
            for i, zone in enumerate(zones)
        ],
        "timeline": timeline,
    }
