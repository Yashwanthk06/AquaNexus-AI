import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

folder = Path(__file__).resolve().parent / "data" / "real"
source = folder / "final_data" / "daily_data.csv"
output = folder / "clean_daily_demand.csv"
report_path = folder / "cleaning_report.json"

stats = Counter({
    "input_rows": 0,
    "invalid_building_or_timestamp_rows": 0,
    "missing_water_rows": 0,
    "invalid_water_rows": 0,
    "duplicate_reading_rows": 0,
    "conflicting_hours": 0,
    "incomplete_days_excluded": 0,
    "complete_days_kept": 0,
    "zero_use_hours_kept": 0,
})
readings = defaultdict(set)
days_seen = set()

with source.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    required = {"building", "date_time", "water_use_l"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Expected these CSV columns: {sorted(required)}")

    for row in reader:
        stats["input_rows"] += 1
        building = (row["building"] or "").strip()
        try:
            timestamp = datetime.fromisoformat(
                (row["date_time"] or "").strip().replace("Z", "+00:00")
            )
            if not building or timestamp.utcoffset() is None:
                raise ValueError
            timestamp = timestamp.astimezone(timezone.utc)
            if timestamp.minute or timestamp.second or timestamp.microsecond:
                raise ValueError
        except ValueError:
            stats["invalid_building_or_timestamp_rows"] += 1
            continue

        day = timestamp.date().isoformat()
        days_seen.add((building, day))
        raw = (row["water_use_l"] or "").strip()
        if raw.lower() in {"", "na", "nan", "null"}:
            stats["missing_water_rows"] += 1
            continue
        try:
            litres = float(raw)
            if not math.isfinite(litres) or litres < 0:
                raise ValueError
        except ValueError:
            stats["invalid_water_rows"] += 1
            continue

        values = readings[(building, day, timestamp.hour)]
        if litres in values:
            stats["duplicate_reading_rows"] += 1
        values.add(litres)

daily = defaultdict(list)
for (building, day, _hour), values in readings.items():
    if len(values) != 1:
        stats["conflicting_hours"] += 1
        continue
    daily[(building, day)].append(next(iter(values)))

clean_rows = []
per_building = Counter()
for building, day in sorted(days_seen):
    hours = daily[(building, day)]
    if len(hours) != 24:
        stats["incomplete_days_excluded"] += 1
        continue
    clean_rows.append((building, day, round(math.fsum(hours), 6), 24))
    per_building[building] += 1
    stats["zero_use_hours_kept"] += hours.count(0)

stats["complete_days_kept"] = len(clean_rows)
report = {
    "source": str(source),
    "day_boundary": "UTC",
    "unit": "litres per building per day",
    "counts": dict(stats),
    "complete_days_by_building": dict(per_building),
}
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

if not clean_rows:
    raise SystemExit("No complete days found. Check cleaning_report.json before training.")

with output.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["building", "date_utc", "demand_litres", "observed_hours"])
    writer.writerows(clean_rows)

print(f"\nCleaned dataset saved to {output}")
