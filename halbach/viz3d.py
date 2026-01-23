from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from halbach.constants import phi0
from halbach.run_types import RunBundle
from halbach.types import FloatArray

PlotlyFigure: TypeAlias = Any
PlotlyMode = Literal["fast", "pretty"]


@dataclass(frozen=True)
class MagnetGeometry:
    centers: FloatArray
    phi: FloatArray
    ring_id: NDArray[np.int_]
    layer_id: NDArray[np.int_]


def enumerate_magnets(
    run: RunBundle,
    *,
    stride: int = 1,
    hide_x_negative: bool = False,
) -> tuple[FloatArray, FloatArray, NDArray[np.int_], NDArray[np.int_]]:
    """
    Enumerate magnet centers and magnetization directions for the run.
    """
    if stride <= 0:
        raise ValueError("stride must be >= 1")

    theta = run.geometry.theta
    sin2 = run.geometry.sin2
    cth = run.geometry.cth
    sth = run.geometry.sth
    z_layers = run.geometry.z_layers
    ring_offsets = run.geometry.ring_offsets
    r_bases = run.results.r_bases
    alphas = run.results.alphas

    R, K = alphas.shape
    idx = np.arange(0, theta.size, stride)
    if idx.size == 0:
        raise ValueError("stride is too large for the theta grid")

    th = theta[idx]
    s2 = sin2[idx]
    c = cth[idx]
    s = sth[idx]

    centers_list: list[FloatArray] = []
    phi_list: list[FloatArray] = []
    ring_list: list[NDArray[np.int_]] = []
    layer_list: list[NDArray[np.int_]] = []

    for r in range(R):
        ring_offset = ring_offsets[r]
        for k in range(K):
            rho = r_bases[k] + ring_offset
            x = rho * c
            y = rho * s
            z = np.full_like(x, z_layers[k], dtype=np.float64)
            centers_list.append(np.column_stack([x, y, z]).astype(np.float64))

            phi = 2.0 * th + phi0 + alphas[r, k] * s2
            phi_list.append(np.asarray(phi, dtype=np.float64))

            ring_list.append(np.full(th.shape, r, dtype=np.int_))
            layer_list.append(np.full(th.shape, k, dtype=np.int_))

    centers = np.concatenate(centers_list, axis=0)
    phi = np.concatenate(phi_list)
    ring_id = np.concatenate(ring_list)
    layer_id = np.concatenate(layer_list)

    if hide_x_negative:
        mask = centers[:, 0] >= 0.0
        centers = centers[mask]
        phi = phi[mask]
        ring_id = ring_id[mask]
        layer_id = layer_id[mask]

    return centers, phi, ring_id, layer_id


def _require_plotly() -> Any:
    import importlib

    try:
        return importlib.import_module("plotly.graph_objects")
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("plotly is required for viz3d (pip install plotly)") from exc


def _magnetization_lines(
    centers: FloatArray, phi: FloatArray, scale: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    dx = np.cos(phi) * scale
    dy = np.sin(phi) * scale
    dz = np.zeros_like(dx)

    start = centers
    end = centers + np.column_stack([dx, dy, dz]).astype(np.float64)

    n = centers.shape[0]
    xs = np.empty(3 * n, dtype=np.float64)
    ys = np.empty(3 * n, dtype=np.float64)
    zs = np.empty(3 * n, dtype=np.float64)
    xs[0::3] = start[:, 0]
    xs[1::3] = end[:, 0]
    xs[2::3] = np.nan
    ys[0::3] = start[:, 1]
    ys[1::3] = end[:, 1]
    ys[2::3] = np.nan
    zs[0::3] = start[:, 2]
    zs[1::3] = end[:, 2]
    zs[2::3] = np.nan
    return xs, ys, zs


def _estimate_arrow_scale(centers: FloatArray) -> float:
    if centers.size == 0:
        return 0.01
    radii = np.linalg.norm(centers[:, :2], axis=1)
    max_r = float(np.max(radii)) if radii.size > 0 else 0.0
    return max(0.01, 0.1 * max_r)


def build_magnet_figure(
    run: RunBundle,
    *,
    stride: int = 1,
    hide_x_negative: bool = False,
    mode: PlotlyMode = "fast",
    magnet_size_m: float | None = None,
    magnet_thickness_m: float | None = None,
    compare: RunBundle | None = None,
) -> PlotlyFigure:
    """
    Build a Plotly figure with magnet centers and optional magnetization vectors.
    """
    go = _require_plotly()
    fig = go.Figure()

    def add_run(run_item: RunBundle, name: str, color: str | None) -> None:
        centers, phi, _ring_id, layer_id = enumerate_magnets(
            run_item, stride=stride, hide_x_negative=hide_x_negative
        )
        if centers.size == 0:
            return
        if compare is None:
            marker_color = layer_id.astype(float)
            marker_kwargs = {
                "color": marker_color,
                "colorscale": "Viridis",
                "showscale": True,
            }
        else:
            marker_kwargs = {"color": color or "#1f77b4"}

        base_marker_size = 3 if mode == "fast" else 4
        marker_size = base_marker_size
        if magnet_size_m is not None:
            marker_size = max(base_marker_size, int(round(magnet_size_m * 400.0)))

        fig.add_trace(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode="markers",
                name=name,
                marker={"size": marker_size, **marker_kwargs},
            )
        )

        if mode == "pretty":
            scale = _estimate_arrow_scale(centers)
            if magnet_size_m is not None:
                scale = max(scale, magnet_size_m)
            xs, ys, zs = _magnetization_lines(centers, phi, scale)
            line_width = 2
            if magnet_thickness_m is not None:
                line_width = max(1, int(round(magnet_thickness_m * 2000.0)))
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    name=f"{name} phi",
                    line={"width": line_width, "color": color or "#444"},
                    showlegend=False,
                )
            )

    add_run(run, run.name, "#1f77b4")
    if compare is not None:
        add_run(compare, compare.name, "#ff7f0e")

    fig.update_layout(
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x"},
            "yaxis": {"title": "y"},
            "zaxis": {"title": "z"},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 30},
        legend={"orientation": "h"},
    )
    return fig


__all__ = ["MagnetGeometry", "enumerate_magnets", "build_magnet_figure"]
