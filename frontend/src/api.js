const API_PREFIX = "/api";

async function readJson(response) {
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed");
  }
  return payload;
}

export async function fetchRuns() {
  const response = await fetch(`${API_PREFIX}/runs`);
  return readJson(response);
}

export async function fetchOverview(runPath) {
  const response = await fetch(`${API_PREFIX}/run/overview`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ run_path: runPath }),
  });
  return readJson(response);
}

export async function fetchMap2D(runPath, settings) {
  const response = await fetch(`${API_PREFIX}/run/map2d`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      run_path: runPath,
      roi_r: settings.roiRadius,
      step: settings.step,
      mag_model_eval: settings.magModelEval,
    }),
  });
  return readJson(response);
}

export async function fetchScene3D(primaryPath, settings, secondaryPath = null) {
  const response = await fetch(`${API_PREFIX}/run/scene3d`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      primary_path: primaryPath,
      secondary_path: secondaryPath,
      stride: settings.stride,
      hide_x_negative: settings.hideXNegative,
    }),
  });
  return readJson(response);
}
