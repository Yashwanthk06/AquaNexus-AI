import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const normal = { minutes: 60, rainfall: 8, repaired: false };
const fmt = value => value.toFixed(2);

function TankTimeline({ twin }) {
  const colors = ["#08786c", "#2864b0", "#b34d20"];
  const maxTank = Math.max(...twin.zones.map(zone => zone.tank_capacity_kl));
  const x = hour => 40 + 550 * hour / 24;
  const y = level => 180 - 150 * level / maxTank;

  return (
    <section>
      <h2>24-hour digital twin</h2>
      <p>Hourly tank levels in kL. Hover over a point to see its value.</p>
      <svg viewBox="0 0 630 215" style={{ width: "100%", maxHeight: 300 }}
        role="img" aria-label="Hourly tank levels for all three zones">
        {[0, 5, 10, 15].map(level => (
          <g key={level}>
            <line x1="40" x2="590" y1={y(level)} y2={y(level)} stroke="#d8e5e3" />
            <text x="30" y={y(level) + 4} textAnchor="end" fontSize="12">{level}</text>
          </g>
        ))}
        {[0, 6, 12, 18, 24].map(hour => (
          <text key={hour} x={x(hour)} y="205" textAnchor="middle" fontSize="12">
            {twin.timeline[hour].time.slice(0, 5)}
          </text>
        ))}
        {twin.zones.map((zone, index) => (
          <g key={zone.name}>
            <polyline fill="none" stroke={colors[index]} strokeWidth="3"
              points={twin.timeline.map((point, hour) =>
                `${x(hour)},${y(point.tank_kl[index])}`
              ).join(" ")} />
            {twin.timeline.map((point, hour) => (
              <circle key={hour} cx={x(hour)} cy={y(point.tank_kl[index])}
                r="3" fill={colors[index]}>
                <title>{`${zone.name}, ${point.time}: ${fmt(point.tank_kl[index])} kL`}</title>
              </circle>
            ))}
          </g>
        ))}
      </svg>
      <div className="grid">
        {twin.zones.map((zone, index) => (
          <div key={zone.name}>
            <strong style={{ color: colors[index] }}>{zone.name}</strong>
            <p>First unmet demand: {zone.first_unserved_time ?? "None"}<br />
              Daily shortage: {fmt(zone.shortage_kl)} kL</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function ModelEvaluation() {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const percent = value => `${(100 * value).toFixed(1)}%`;

  async function runTest() {
    if (loading) return;
    setLoading(true);
    setError("");
    try {
      const seed = 1000 + Math.floor(Math.random() * 999999000);
      const response = await fetch(`/api/fresh-test?seed=${seed}`);
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      setReport(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section>
      <h2>Model evaluation</h2>
      <p>Generate new synthetic test data and score the trained models against known answers.</p>
      <button onClick={runTest} disabled={loading}>
        Run synthetic model test
      </button>
      {error && <p role="alert">{error}</p>}
      {report && <div aria-live="polite">
        <p>Synthetic test · Seed {report.seed} · {report.forecast_days} days · {report.meter_readings} meter readings</p>
        <div className="grid">
          <p><strong>{Math.round(report.mae_litres)} L</strong><br />Average forecast error per zone per day</p>
          <p><strong>{percent(report.accuracy)}</strong><br />Anomaly classification accuracy</p>
          <p><strong>{percent(report.precision)}</strong><br />Alert precision</p>
          <p><strong>{percent(report.recall)}</strong><br />Anomaly recall</p>
        </div>
        <p>Caught {report.anomalies_caught} of {report.injected_anomalies} injected anomalies, with {report.false_alarms} false alarms.</p>
        <p>Example for {report.example.zone}: {report.example.temperature_c.toFixed(1)} °C,
          {" "}{report.example.rainfall_mm.toFixed(1)} mm rain.
          Predicted {report.example.predicted_kl.toFixed(2)} kL.
          Synthetic actual {report.example.actual_kl.toFixed(2)} kL.</p>
      </div>}
    </section>
  );
}

function WaterAlerts({ result, busy, onRepair }) {
  const leakAlerts = result.zones.filter(zone => zone.anomaly);
  const shortages = result.twin.zones
    .filter(zone => zone.shortage_kl >= 0.01)
    .sort((a, b) => b.shortage_kl - a.shortage_kl);

  function openTwin() {
    scrollToTankChart();
  }

  return (
    <section aria-labelledby="water-alerts-heading">
      <h2 id="water-alerts-heading">Water alerts</h2>
      <p className="note">
        Alerts for the last run: {result.scenario.minutes} min supply,
        {" "}{result.scenario.rainfall} mm rain,
        {" "}{result.scenario.repaired ? "Hilltop repair simulated" : "current modeled losses"}.
      </p>

      {leakAlerts.length === 0 && shortages.length === 0 && (
        <p>No unusual meter readings or projected shortages for this run.</p>
      )}

      <div className="grid">
        {leakAlerts.map(zone => (
          <article key={`leak-${zone.name}`} style={{ border: "1px solid #e5c391" }}>
            <h3>Possible leak · {zone.name}</h3>
            <p>
              The anomaly model flagged an unusual difference between water
              sent and water received in the synthetic meter data.
              Inspect the line to confirm the cause.
            </p>
            {zone.name === "Hilltop" && !result.scenario.repaired && (
              <button type="button" disabled={busy} onClick={onRepair}>
                Test Hilltop repair in digital twin
              </button>
            )}
            {zone.name === "Hilltop" && result.scenario.repaired && (
              <p>The repair is simulated. The original meter alert is shown for reference.</p>
            )}
          </article>
        ))}

        {shortages.map(zone => (
          <article key={`shortage-${zone.name}`} style={{ border: "1px solid #e5b4a7" }}>
            <h3 className="danger">Projected shortage · {zone.name}</h3>
            <p>
              {fmt(zone.shortage_kl)} kL of demand goes unserved.
              {" "}First unmet demand: <strong>{zone.first_unserved_time}</strong>.
            </p>
            <button type="button" onClick={openTwin}>View tank timeline</button>
          </article>
        ))}
      </div>
    </section>
  );
}

function ScheduleComparison({ minutes }) {
  const [seed, setSeed] = useState("4721");
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function compare(event) {
    event.preventDefault();
    if (loading) return;
    setLoading(true);
    setError("");
    // Keep the last comparison visible while the new one runs.
    try {
      const query = new URLSearchParams({ seed, minutes: String(minutes) });
      const response = await fetch(`/api/historical-replay?${query}`);
      if (response.status === 404) {
        throw new Error("The historical replay endpoint is missing. Check for GET /api/historical-replay in FastAPI Docs.");
      }
      const data = await response.json();
      if (!response.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : "Check the seed and try again.");
      }
      setReport(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section>
      <h2>Compare pump schedules</h2>
      <p className="note">Compare two schedules with the same pump time. Both are tested against actual household demand from the selected day.</p>
      <form onSubmit={compare}>
        <label>
          Test date seed{" "}
          <input type="number" min="0" max="999999999" step="1" required disabled={loading}
            value={seed} onChange={event => setSeed(event.target.value)} />
        </label>
        <button type="submit" disabled={loading}>
          {`Compare schedules at ${minutes} min`}
        </button>
      </form>
      {error && <p role="alert">{error}</p>}
      {report && (
        <div aria-live="polite">
          <p><strong>{report.selected_test_date}</strong> · {report.supply_window_minutes} min pump window · seed {report.seed}</p>
          <div className="grid">
            {[["Equal time per zone", report.equal_time], ["Prioritize the least-served zone", report.optimized]].map(([name, plan]) => (
              <article key={name}>
                <h3>{name}</h3>
                <p><strong>{fmt(plan.total_shortage_kl)} kL</strong> total shortage</p>
                <p>Least-served zone: {plan.worst_served_zone ?? "None"}<br />
                  Demand met: {plan.worst_zone_coverage_percent == null
                    ? "Unavailable" : `${plan.worst_zone_coverage_percent}%`}<br />
                  First unmet demand: {plan.worst_zone_first_unserved_time ?? "None"}
                </p>
                <ul>
                  {plan.zones.map(zone => (
                    <li key={zone.name}>{zone.name}: {fmt(zone.scheduled_pump_minutes)} min</li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
          <p>
            {Math.abs(report.shortage_reduction_kl) < 0.005
              ? "Both schedules produced the same total shortage."
              : `Prioritizing the least-served zone resulted in ${fmt(Math.abs(report.shortage_reduction_kl))} kL ${
                  report.shortage_reduction_kl > 0 ? "less" : "more"
                } total shortage.`}
          </p>
          <p className="note">{report.simulation_assumptions}</p>
        </div>
      )}
    </section>
  );
}

function scrollToTankChart() {
  const chart = document.getElementById("digital-twin");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  chart?.focus({ preventScroll: true });
  chart?.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}

function TwinPumpInput({ result, busy, onRun }) {
  const [minutes, setMinutes] = useState("");
  const scenario = result.scenario;

  async function submit(event) {
    event.preventDefault();
    if (busy || minutes.trim() === "") return;
    const value = Number(minutes);
    if (!Number.isInteger(value) || value < 0 || value > 60) return;
    await onRun({ ...scenario, minutes: value });
  }

  return (
    <section aria-labelledby="twin-input-heading">
      <h2 id="twin-input-heading">Try your own pump time</h2>
      <p>Pick how many minutes the three zones can share the pump. Then see when their tanks run short.</p>
      <form onSubmit={submit}>
        <label>
          Pump time in minutes (0 to 60)
          <input
            type="number"
            min="0"
            max="60"
            step="1"
            required
            placeholder={String(scenario.minutes)}
            value={minutes}
            onChange={event => setMinutes(event.target.value)}
            disabled={busy}
          />
        </label>
        <button type="submit" disabled={busy} aria-busy={busy}>
          Run digital twin
        </button>
      </form>
      <p aria-live="polite">
        Chart shows <strong>{scenario.minutes} min</strong> of pump time,
        {" "}{scenario.rainfall} mm of rain, and
        {" "}{scenario.repaired ? "a simulated Hilltop repair" : "current modeled pipe losses"}.
        {" "}Projected shortage: <strong>{fmt(result.twin.total_shortage_kl)} kL</strong>.
      </p>
    </section>
  );
}

function App() {
  const [settings, setSettings] = useState(normal);
  const [result, setResult] = useState(null);
  const [baseline, setBaseline] = useState(null);
  const [repairComparison, setRepairComparison] = useState(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");

  const running = useRef(false);
  const [pendingAction, setPendingAction] = useState("Loading baseline");

  async function run(scenario, action = "Simulating next 24 hours") {
    if (running.current) return;
    running.current = true;
    setPendingAction(action);
    setSettings(scenario);
    setError("");
    setBusy(true);
    try {
      const response = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(scenario),
      });
      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      const data = await response.json();
      const twinResponse = await fetch("/api/digital-twin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(scenario),
      });
      if (!twinResponse.ok) {
        throw new Error(`Digital twin returned ${twinResponse.status}`);
      }
      data.twin = await twinResponse.json();
      setRepairComparison(null);
      setResult({ ...data, scenario });
      if (scenario.minutes === 60 && scenario.rainfall === 8 && !scenario.repaired) {
        setBaseline(data);
      }
      return data;
    } catch (err) {
      setError(`Simulation failed. Check that the Python server is running on port 8000. ${err.message}`);
    } finally {
      running.current = false;
      setBusy(false);
    }
  }

  useEffect(() => { run(normal, "Loading baseline"); }, []);

  const delta = result && baseline ? result.twin.total_shortage_kl - baseline.twin.total_shortage_kl : null;

  return (
    <main>
      <header>
        <small>SYNTHETIC DEMO DATA</small>
        <h1>AquaNexus AI</h1>
        <p>Plan tomorrow's water supply for a village with three zones and one shared pump.</p>
      </header>

      <section>
        <h2>Supply scenario</h2>
        <p className="note">Set the pump window and rainfall, then simulate the next 24 hours. Use a preset to test a supply cut or a dry day.</p>
        <form onSubmit={event => { event.preventDefault(); run(settings); }}>
          <fieldset disabled={busy} style={{border: 0, padding: 0, margin: 0}} className="scenario-form">
            <div className="grid scenario-controls">
              <label>
                Supply window: {settings.minutes} min
                <input type="range" min="0" max="60" value={settings.minutes}
                  onChange={e => setSettings({...settings, minutes: Number(e.target.value)})} />
              </label>
              <label>
                Rainfall: {settings.rainfall} mm
                <input type="range" min="0" max="20" value={settings.rainfall}
                  onChange={e => setSettings({...settings, rainfall: Number(e.target.value)})} />
              </label>
              <label>
                <input type="checkbox" checked={settings.repaired}
                  onChange={e => setSettings({...settings, repaired: e.target.checked})} />
                Simulate Hilltop leak repair
              </label>
            </div>
            <button type="submit">Simulate next 24 hours</button>
            <button type="button" onClick={() => run({...settings, minutes: 30}, "Simulating 30-minute supply")}>Simulate 30-min supply cut</button>
            <button type="button" onClick={() => run({...settings, rainfall: 0}, "Simulating dry day")}>Simulate dry day</button>
            <button type="button" onClick={() => run(normal, "Restoring baseline")}>Restore 60-min baseline</button>
          </fieldset>
        </form>
        <span role="status" style={{
          position: "absolute", width: 1, height: 1, overflow: "hidden",
          clipPath: "inset(50%)", whiteSpace: "nowrap"
        }}>
          {busy ? pendingAction : !error && result ? "Simulation results updated" : ""}
        </span>
        {error && <p role="alert" className="danger">{error}</p>}
      </section>

      {result && <>
        <section aria-live="polite">
          <h2>Results for {result.scenario.minutes} min / {result.scenario.rainfall} mm rain / {result.scenario.repaired ? "repair simulated" : "current losses"}</h2>
          <div className="grid">
            <div><strong className="metric">{fmt(result.demand)} kL</strong>Forecast demand</div>
            <div><strong className="metric">{fmt(result.twin.total_shortage_kl)} kL</strong>Projected shortage</div>
            <div>
              <strong className="metric">{delta === null ? "Unavailable" : `${delta > 0.005 ? "+" : ""}${fmt(delta)} kL`}</strong>
              Shortage change vs. 60-minute baseline
            </div>
          </div>
        </section>

        <WaterAlerts
          result={result}
          busy={busy}
          onRepair={async () => {
            const before = result.twin.total_shortage_kl;
            const repaired = await run({ ...result.scenario, repaired: true }, "Testing Hilltop repair");
            if (repaired) {
              setRepairComparison({ before, after: repaired.twin.total_shortage_kl });
              requestAnimationFrame(scrollToTankChart);
            }
          }}
        />
        <div id="digital-twin" tabIndex={-1}>
          {repairComparison && (
            <section aria-live="polite">
              <h2>Hilltop repair comparison</h2>
              <p>
                Same pump window and rainfall. Village shortage before repair:
                {" "}<strong>{fmt(repairComparison.before)} kL</strong>.
                After repair: <strong>{fmt(repairComparison.after)} kL</strong>.
              </p>
            </section>
          )}
          <TwinPumpInput
            result={result}
            busy={busy}
            onRun={run}
          />
          <TankTimeline twin={result.twin} />
        </div>
        <div className="grid">
          {result.zones.map(zone => (
            <article key={zone.name}>
              <h2>{zone.name}</h2>
              <p><strong>{fmt(zone.minutes)} min</strong> pump time</p>
              <label>
                Planned tank after refill: {fmt(zone.after_fill)} / {zone.capacity} kL
                <meter min="0" max={zone.capacity} value={zone.after_fill} />
              </label>
              <p>Starting water: {fmt(zone.start)} kL<br />
                Forecast demand: {fmt(zone.demand)} kL<br />
                End-of-day water: {fmt(zone.end)} kL</p>
              <p className={zone.shortage > 0.01 ? "danger" : ""}>Shortage: {fmt(zone.shortage)} kL</p>
              {zone.anomaly && <p className="danger">Unusual historical meter imbalance. Inspect for a possible leak.</p>}
            </article>
          ))}
        </div>

        <section>
          <h2>Recommendation from the daily supply plan</h2>
          <p>{result.recommendation}</p>
          <p className="note">The plan raises the coverage of the least-served zone as far as possible, then delivers the most useful water within the remaining limits. The twin simulates one refill before daily demand, with fixed tank capacities and flow rates adjusted for rainfall and pipe loss.</p>
          <p className="note">Forecast error on the 60 held-out synthetic days: Random Forest {fmt(result.forecast_mae)} kL, previous-day baseline {fmt(result.baseline_mae)} kL. Lower is better. Isolation Forest flags unusual meter readings for inspection.</p>
        </section>
      </>}
      <ModelEvaluation />
      {result && <ScheduleComparison minutes={result.scenario.minutes} />}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);











