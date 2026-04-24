import { useEffect, useMemo, useState, useTransition } from "react";

import Plot from "react-plotly.js";

import { fetchMap2D, fetchOverview, fetchRuns, fetchScene3D } from "./api";

function valueOrPlaceholder(value, fallback = "n/a") {
  return value === undefined || value === null ? fallback : String(value);
}

function buildHeatmapFigure(mapPayload, title) {
  if (!mapPayload) {
    return null;
  }
  return {
    data: [
      {
        type: "heatmap",
        x: mapPayload.xs.map((value) => value * 1000.0),
        y: mapPayload.ys.map((value) => value * 1000.0),
        z: mapPayload.ppm,
        zmid: 0,
        colorscale: "RdBu",
        colorbar: { title: "ppm" },
      },
    ],
    layout: {
      title,
      margin: { l: 40, r: 10, b: 40, t: 40 },
      xaxis: { title: "x (mm)" },
      yaxis: { title: "y (mm)" },
    },
  };
}

function buildCrossSectionFigure(mapPayload, title) {
  if (!mapPayload) {
    return null;
  }
  const yIndex = mapPayload.ys.reduce(
    (bestIndex, value, index, values) =>
      Math.abs(value) < Math.abs(values[bestIndex]) ? index : bestIndex,
    0,
  );
  return {
    data: [
      {
        type: "scatter",
        mode: "lines",
        x: mapPayload.xs.map((value) => value * 1000.0),
        y: mapPayload.ppm[yIndex],
        line: { color: "#1f4f99", width: 2 },
      },
    ],
    layout: {
      title,
      margin: { l: 40, r: 10, b: 40, t: 40 },
      xaxis: { title: "x (mm)" },
      yaxis: { title: "ppm" },
    },
  };
}

function buildSceneFigure(scenePayload, title) {
  if (!scenePayload) {
    return null;
  }

  const traces = [
    {
      type: "scatter3d",
      mode: "markers",
      name: scenePayload.primary.run_name,
      x: scenePayload.primary.centers.map((item) => item[0]),
      y: scenePayload.primary.centers.map((item) => item[1]),
      z: scenePayload.primary.centers.map((item) => item[2]),
      marker: { color: "#1f77b4", size: 3 },
    },
  ];

  if (scenePayload.secondary) {
    traces.push({
      type: "scatter3d",
      mode: "markers",
      name: scenePayload.secondary.run_name,
      x: scenePayload.secondary.centers.map((item) => item[0]),
      y: scenePayload.secondary.centers.map((item) => item[1]),
      z: scenePayload.secondary.centers.map((item) => item[2]),
      marker: { color: "#ff7f0e", size: 3 },
    });
  }

  return {
    data: traces,
    layout: {
      title,
      margin: { l: 0, r: 0, b: 0, t: 40 },
      scene: {
        aspectmode: "data",
        xaxis: { title: "x", range: scenePayload.scene_ranges[0] },
        yaxis: { title: "y", range: scenePayload.scene_ranges[1] },
        zaxis: { title: "z", range: scenePayload.scene_ranges[2] },
      },
      legend: { orientation: "h" },
    },
  };
}

function SummaryCard({ title, summary }) {
  return (
    <section className="summary-card">
      <h3>{title}</h3>
      {summary ? (
        <>
          <p>
            <strong>Name:</strong> {summary.name}
          </p>
          <p>
            <strong>Framework:</strong> {summary.framework}
          </p>
          <p>
            <strong>B0_T:</strong> {valueOrPlaceholder(summary.key_stats?.B0_T)}
          </p>
          <p>
            <strong>R/K/N:</strong>{" "}
            {[
              valueOrPlaceholder(summary.geometry_summary?.R),
              valueOrPlaceholder(summary.geometry_summary?.K),
              valueOrPlaceholder(summary.geometry_summary?.N),
            ].join(" / ")}
          </p>
        </>
      ) : (
        <p className="muted">Select a run to load overview data.</p>
      )}
    </section>
  );
}

function PlotPanel({ title, figure, className = "" }) {
  return (
    <section className={`panel ${className}`.trim()}>
      <h3>{title}</h3>
      {figure ? (
        <Plot
          data={figure.data}
          layout={figure.layout}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
          config={{ displayModeBar: false }}
        />
      ) : (
        <p className="muted">No data loaded.</p>
      )}
    </section>
  );
}

export default function App() {
  const [runs, setRuns] = useState([]);
  const [initRunPath, setInitRunPath] = useState("");
  const [optRunPath, setOptRunPath] = useState("");
  const [singleTarget, setSingleTarget] = useState("optimized");
  const [roiRadius, setRoiRadius] = useState(0.14);
  const [step, setStep] = useState(0.002);
  const [stride, setStride] = useState(2);
  const [hideXNegative, setHideXNegative] = useState(false);
  const [magModelEval, setMagModelEval] = useState("auto");
  const [overview, setOverview] = useState({ initial: null, optimized: null });
  const [maps, setMaps] = useState({ initial: null, optimized: null });
  const [scenes, setScenes] = useState({ initial: null, optimized: null, compare: null });
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    let active = true;
    fetchRuns()
      .then((payload) => {
        if (!active) {
          return;
        }
        const nextRuns = payload.runs || [];
        setRuns(nextRuns);
        if (nextRuns.length > 0) {
          setInitRunPath(nextRuns[0].path);
          setOptRunPath(nextRuns[0].path);
        }
      })
      .catch((loadError) => {
        if (!active) {
          return;
        }
        setError(loadError.message);
      });
    return () => {
      active = false;
    };
  }, []);

  const settings = useMemo(
    () => ({
      roiRadius,
      step,
      stride,
      hideXNegative,
      magModelEval,
    }),
    [hideXNegative, magModelEval, roiRadius, step, stride],
  );

  async function refreshReadOnlyData() {
    setError("");
    try {
      const [
        initialOverview,
        optimizedOverview,
        initialMap,
        optimizedMap,
        initialScene,
        optimizedScene,
        compareScene,
      ] = await Promise.all([
        initRunPath ? fetchOverview(initRunPath) : Promise.resolve(null),
        optRunPath ? fetchOverview(optRunPath) : Promise.resolve(null),
        initRunPath ? fetchMap2D(initRunPath, settings) : Promise.resolve(null),
        optRunPath ? fetchMap2D(optRunPath, settings) : Promise.resolve(null),
        initRunPath ? fetchScene3D(initRunPath, settings) : Promise.resolve(null),
        optRunPath ? fetchScene3D(optRunPath, settings) : Promise.resolve(null),
        initRunPath && optRunPath
          ? fetchScene3D(initRunPath, settings, optRunPath)
          : Promise.resolve(null),
      ]);

      startTransition(() => {
        setOverview({ initial: initialOverview, optimized: optimizedOverview });
        setMaps({ initial: initialMap, optimized: optimizedMap });
        setScenes({
          initial: initialScene,
          optimized: optimizedScene,
          compare: compareScene,
        });
      });
    } catch (loadError) {
      setError(loadError.message);
    }
  }

  const singleScene = singleTarget === "initial" ? scenes.initial : scenes.optimized;

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Phase 4 Vertical Slice</p>
          <h1>Halbach Read-only GUI</h1>
          <p className="muted">
            Streamlit と並行運用する、React + FastAPI の最初の移植面です。
          </p>
        </div>
        <button className="primary-button" onClick={refreshReadOnlyData} type="button">
          {isPending ? "Refreshing..." : "Refresh Read-only Data"}
        </button>
      </section>

      {error ? <p className="error-banner">{error}</p> : null}

      <section className="panel controls">
        <h2>Run Selection</h2>
        <div className="control-grid">
          <label>
            Initial run
            <select value={initRunPath} onChange={(event) => setInitRunPath(event.target.value)}>
              <option value="">(none)</option>
              {runs.map((run) => (
                <option key={`init-${run.path}`} value={run.path}>
                  {run.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Optimized run
            <select value={optRunPath} onChange={(event) => setOptRunPath(event.target.value)}>
              <option value="">(none)</option>
              {runs.map((run) => (
                <option key={`opt-${run.path}`} value={run.path}>
                  {run.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            ROI radius (m)
            <input
              type="number"
              step="0.001"
              min="0.001"
              value={roiRadius}
              onChange={(event) => setRoiRadius(Number(event.target.value))}
            />
          </label>
          <label>
            Step (m)
            <input
              type="number"
              step="0.001"
              min="0.001"
              value={step}
              onChange={(event) => setStep(Number(event.target.value))}
            />
          </label>
          <label>
            3D stride
            <input
              type="number"
              step="1"
              min="1"
              value={stride}
              onChange={(event) => setStride(Number(event.target.value))}
            />
          </label>
          <label>
            2D magnetization
            <select value={magModelEval} onChange={(event) => setMagModelEval(event.target.value)}>
              <option value="auto">auto</option>
              <option value="fixed">fixed</option>
              <option value="self-consistent-easy-axis">self-consistent-easy-axis</option>
            </select>
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={hideXNegative}
              onChange={(event) => setHideXNegative(event.target.checked)}
            />
            Hide x &lt; 0
          </label>
          <label>
            3D single target
            <select value={singleTarget} onChange={(event) => setSingleTarget(event.target.value)}>
              <option value="optimized">optimized</option>
              <option value="initial">initial</option>
            </select>
          </label>
        </div>
      </section>

      <section className="summary-grid">
        <SummaryCard title="Initial Overview" summary={overview.initial} />
        <SummaryCard title="Optimized Overview" summary={overview.optimized} />
      </section>

      <section className="plot-grid">
        <PlotPanel
          title="Initial 2D Heatmap"
          figure={buildHeatmapFigure(maps.initial, "Initial 2D Heatmap")}
        />
        <PlotPanel
          title="Optimized 2D Heatmap"
          figure={buildHeatmapFigure(maps.optimized, "Optimized 2D Heatmap")}
        />
        <PlotPanel
          title="Initial Cross Section"
          figure={buildCrossSectionFigure(maps.initial, "Initial Cross Section")}
        />
        <PlotPanel
          title="Optimized Cross Section"
          figure={buildCrossSectionFigure(maps.optimized, "Optimized Cross Section")}
        />
      </section>

      <section className="plot-grid plot-grid-3d">
        <PlotPanel
          title="3D Single View"
          figure={buildSceneFigure(singleScene, "3D Single View")}
          className="tall-panel"
        />
        <PlotPanel
          title="3D Compare View"
          figure={buildSceneFigure(scenes.compare, "3D Compare View")}
          className="tall-panel"
        />
      </section>
    </main>
  );
}
