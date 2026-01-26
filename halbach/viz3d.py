from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from halbach.constants import phi0
from halbach.run_types import RunBundle
from halbach.types import FloatArray

PlotlyFigure: TypeAlias = Any
PlotlyMode = Literal["fast", "pretty", "cubes", "cubes_arrows"]
SceneRanges: TypeAlias = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


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


def compute_scene_ranges(
    centers_list: Sequence[FloatArray], *, margin: float = 0.05
) -> SceneRanges:
    mins: list[FloatArray] = []
    maxs: list[FloatArray] = []
    for centers in centers_list:
        if centers.size == 0:
            continue
        mins.append(np.min(centers, axis=0))
        maxs.append(np.max(centers, axis=0))
    if not mins:
        return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    min_all = np.min(np.stack(mins, axis=0), axis=0)
    max_all = np.max(np.stack(maxs, axis=0), axis=0)

    ranges: list[tuple[float, float]] = []
    fallback_pad = 0.01
    for axis in range(3):
        min_val = float(min_all[axis])
        max_val = float(max_all[axis])
        span = max_val - min_val
        if span <= 0.0:
            pad = fallback_pad
        else:
            pad = float(span * margin)
        ranges.append((min_val - pad, max_val + pad))
    return (ranges[0], ranges[1], ranges[2])


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


def build_magnetization_arrows(
    centers: FloatArray,
    phi: FloatArray,
    length_m: float,
    head_angle_deg: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if centers.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    length = float(max(length_m, 0.0))
    if length == 0.0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    head_length = max(1e-6, 0.25 * length)
    half_angle = 0.5 * np.deg2rad(head_angle_deg)

    dir_x = np.cos(phi)
    dir_y = np.sin(phi)
    tip = centers + np.column_stack([dir_x, dir_y, np.zeros_like(dir_x)]) * length

    back_angle = phi + np.pi
    a1 = back_angle - half_angle
    a2 = back_angle + half_angle
    hx1 = np.cos(a1) * head_length
    hy1 = np.sin(a1) * head_length
    hx2 = np.cos(a2) * head_length
    hy2 = np.sin(a2) * head_length

    head1 = tip + np.column_stack([hx1, hy1, np.zeros_like(hx1)])
    head2 = tip + np.column_stack([hx2, hy2, np.zeros_like(hx2)])

    n = centers.shape[0]
    xs = np.empty(n * 9, dtype=np.float64)
    ys = np.empty_like(xs)
    zs = np.empty_like(xs)
    cursor = 0
    for idx in range(n):
        xs[cursor] = centers[idx, 0]
        xs[cursor + 1] = tip[idx, 0]
        xs[cursor + 2] = np.nan
        ys[cursor] = centers[idx, 1]
        ys[cursor + 1] = tip[idx, 1]
        ys[cursor + 2] = np.nan
        zs[cursor] = centers[idx, 2]
        zs[cursor + 1] = tip[idx, 2]
        zs[cursor + 2] = np.nan
        cursor += 3

        xs[cursor] = tip[idx, 0]
        xs[cursor + 1] = head1[idx, 0]
        xs[cursor + 2] = np.nan
        ys[cursor] = tip[idx, 1]
        ys[cursor + 1] = head1[idx, 1]
        ys[cursor + 2] = np.nan
        zs[cursor] = tip[idx, 2]
        zs[cursor + 1] = head1[idx, 2]
        zs[cursor + 2] = np.nan
        cursor += 3

        xs[cursor] = tip[idx, 0]
        xs[cursor + 1] = head2[idx, 0]
        xs[cursor + 2] = np.nan
        ys[cursor] = tip[idx, 1]
        ys[cursor + 1] = head2[idx, 1]
        ys[cursor + 2] = np.nan
        zs[cursor] = tip[idx, 2]
        zs[cursor + 1] = head2[idx, 2]
        zs[cursor + 2] = np.nan
        cursor += 3

    return xs, ys, zs


def build_cubes_mesh(
    centers: FloatArray, phi: FloatArray, size_m: float, thickness_m: float
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    NDArray[np.int_],
    NDArray[np.int_],
    NDArray[np.int_],
    FloatArray,
    FloatArray,
    FloatArray,
]:
    if centers.size == 0:
        empty_f = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int_)
        return empty_f, empty_f, empty_f, empty_i, empty_i, empty_i, empty_f, empty_f, empty_f

    hx = 0.5 * size_m
    hy = 0.5 * size_m
    hz = 0.5 * thickness_m
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    lx = local[:, 0]
    ly = local[:, 1]
    lz = local[:, 2]

    c = np.cos(phi)
    s = np.sin(phi)
    x = c[:, None] * lx[None, :] - s[:, None] * ly[None, :]
    y = s[:, None] * lx[None, :] + c[:, None] * ly[None, :]
    z = np.broadcast_to(lz, x.shape).astype(np.float64, copy=True)

    x += centers[:, 0, None]
    y += centers[:, 1, None]
    z += centers[:, 2, None]

    vx = x.reshape(-1)
    vy = y.reshape(-1)
    vz = z.reshape(-1)

    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int_,
    )
    offsets = (np.arange(centers.shape[0], dtype=np.int_) * 8)[:, None, None]
    faces_full = faces[None, :, :] + offsets
    i = faces_full[:, :, 0].reshape(-1)
    j = faces_full[:, :, 1].reshape(-1)
    k = faces_full[:, :, 2].reshape(-1)

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int_,
    )
    n_edges = edges.shape[0]
    ex = np.empty(centers.shape[0] * n_edges * 3, dtype=np.float64)
    ey = np.empty_like(ex)
    ez = np.empty_like(ex)
    cursor = 0
    for m in range(centers.shape[0]):
        for a, b in edges:
            ex[cursor] = x[m, a]
            ex[cursor + 1] = x[m, b]
            ex[cursor + 2] = np.nan
            ey[cursor] = y[m, a]
            ey[cursor + 1] = y[m, b]
            ey[cursor + 2] = np.nan
            ez[cursor] = z[m, a]
            ez[cursor + 1] = z[m, b]
            ez[cursor + 2] = np.nan
            cursor += 3

    return vx, vy, vz, i, j, k, ex, ey, ez


def build_magnet_figure(
    run: RunBundle,
    *,
    stride: int = 1,
    hide_x_negative: bool = False,
    mode: PlotlyMode = "fast",
    magnet_size_m: float | None = None,
    magnet_thickness_m: float | None = None,
    arrow_length_m: float | None = None,
    arrow_head_angle_deg: float = 30.0,
    scene_camera: dict[str, Any] | None = None,
    scene_ranges: SceneRanges | None = None,
    height: int | None = None,
    compare: RunBundle | None = None,
) -> PlotlyFigure:
    """
    Build a Plotly figure with magnet centers, optional magnetization vectors, or cubes.
    """
    go = _require_plotly()
    fig = go.Figure()

    def add_run(run_item: RunBundle, name: str, color: str | None) -> None:
        centers, phi, _ring_id, layer_id = enumerate_magnets(
            run_item, stride=stride, hide_x_negative=hide_x_negative
        )
        if centers.size == 0:
            return
        if mode in ("cubes", "cubes_arrows"):
            size_m = 0.02 if magnet_size_m is None else magnet_size_m
            thickness_m = size_m
            edge_width = max(1, int(round((size_m * 1000.0) / 7.5)))
            size_mm = size_m * 1000.0
            ambient = 0.3
            if size_mm < 15.0:
                ambient = 0.3 + 0.7 * (15.0 - size_mm) / 15.0
            vx, vy, vz, i, j, k, ex, ey, ez = build_cubes_mesh(centers, phi, size_m, thickness_m)
            fig.add_trace(
                go.Mesh3d(
                    x=vx,
                    y=vy,
                    z=vz,
                    i=i,
                    j=j,
                    k=k,
                    name=name,
                    color="rgb(229,229,229)",
                    opacity=1.0,
                    flatshading=True,
                    lighting={
                        "ambient": ambient,
                        "diffuse": 0.7,
                        "specular": 0.2,
                        "roughness": 0.9,
                        "fresnel": 0.2,
                    },
                    lightposition={"x": 0.0, "y": 0.0, "z": 2.0},
                    showscale=False,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=ex,
                    y=ey,
                    z=ez,
                    mode="lines",
                    name=f"{name} edges",
                    line={"color": "black", "width": edge_width},
                    showlegend=False,
                )
            )
            if mode == "cubes_arrows":
                length = arrow_length_m
                if length is None:
                    length = max(0.01, 1.5 * size_m)
                ax, ay, az = build_magnetization_arrows(
                    centers,
                    phi,
                    length_m=length,
                    head_angle_deg=arrow_head_angle_deg,
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=ax,
                        y=ay,
                        z=az,
                        mode="lines",
                        name=f"{name} arrows",
                        line={"color": "red", "width": 4},
                        showlegend=False,
                    )
                )
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

    layout: dict[str, Any] = {
        "scene": {
            "aspectmode": "data",
            "xaxis": {"title": "x"},
            "yaxis": {"title": "y"},
            "zaxis": {"title": "z"},
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 30},
        "legend": {"orientation": "h"},
    }
    if scene_ranges is not None:
        layout["scene"]["xaxis"]["range"] = list(scene_ranges[0])
        layout["scene"]["yaxis"]["range"] = list(scene_ranges[1])
        layout["scene"]["zaxis"]["range"] = list(scene_ranges[2])
    if scene_camera is not None:
        layout["scene_camera"] = scene_camera
    if height is not None:
        layout["height"] = height
    fig.update_layout(
        **layout,
    )
    return fig


__all__ = [
    "MagnetGeometry",
    "enumerate_magnets",
    "compute_scene_ranges",
    "build_cubes_mesh",
    "build_magnetization_arrows",
    "build_magnet_figure",
]
