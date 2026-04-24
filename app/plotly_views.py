from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go

from halbach.app_services import Map2DPayload, Scene3DPayload
from halbach.coil_overlay import build_plotly_polyline_groups
from halbach.viz3d import build_cubes_mesh, build_magnetization_arrows

if TYPE_CHECKING:
    from halbach.coil_overlay import CoilPolylineSet
    from halbach.obj_mesh import ObjMesh

PlotlyMode = Literal["fast", "pretty", "cubes", "cubes_arrows"]


def _sample_colorscale(name: str, levels: int) -> list[tuple[float, str]]:
    import plotly.colors as pc

    steps = max(2, int(levels))
    scale = pc.get_colorscale(name)
    values = [index / (steps - 1) for index in range(steps)]
    colors = pc.sample_colorscale(scale, values)
    return list(zip(values, colors, strict=True))


def common_ppm_limits(
    maps: list[Map2DPayload],
    *,
    limit_ppm: float | None = 5000.0,
    symmetric: bool = True,
) -> tuple[float, float]:
    if limit_ppm is not None:
        limit = float(limit_ppm)
        return -limit, limit
    if not maps:
        raise ValueError("maps must be non-empty when limit_ppm is None")

    mins = [float(np.nanmin(payload.ppm)) for payload in maps]
    maxs = [float(np.nanmax(payload.ppm)) for payload in maps]
    vmin = float(np.nanmin(np.asarray(mins, dtype=np.float64)))
    vmax = float(np.nanmax(np.asarray(maxs, dtype=np.float64)))
    if symmetric:
        max_abs = max(abs(vmin), abs(vmax))
        return -max_abs, max_abs
    return vmin, vmax


def combine_scene_ranges(
    sources: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]],
    *,
    margin: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    maxs: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    for centers in sources:
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
        pad = fallback_pad if span <= 0.0 else float(span * margin)
        ranges.append((min_val - pad, max_val + pad))
    return (ranges[0], ranges[1], ranges[2])


def plot_error_map(
    payload: Map2DPayload,
    *,
    vmin: float,
    vmax: float,
    contour_level: float,
    title: str,
) -> go.Figure:
    xs_mm = np.asarray(payload.xs, dtype=np.float64) * 1000.0
    ys_mm = np.asarray(payload.ys, dtype=np.float64) * 1000.0
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=xs_mm,
            y=ys_mm,
            z=payload.ppm,
            zmin=vmin,
            zmax=vmax,
            zmid=0.0,
            colorscale=_sample_colorscale("RdBu", 50),
            colorbar={"title": "ppm"},
        )
    )
    contour_start = -float(contour_level)
    contour_end = float(contour_level)
    contour_steps = 50
    contour_size = (contour_end - contour_start) / contour_steps
    fig.add_trace(
        go.Contour(
            x=xs_mm,
            y=ys_mm,
            z=payload.ppm,
            contours={"start": contour_start, "end": contour_end, "size": contour_size},
            line={"color": "black", "width": 1},
            showscale=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        height=420,
        margin={"l": 40, "r": 10, "b": 40, "t": 40},
    )
    return fig


def plot_cross_section(
    payload: Map2DPayload,
    *,
    vmin: float,
    vmax: float,
    title: str,
) -> go.Figure:
    idx = int(np.argmin(np.abs(payload.ys)))
    y0 = float(payload.ys[idx])
    ppm_line = np.asarray(payload.ppm[idx, :], dtype=np.float64)
    x_mm = np.asarray(payload.xs, dtype=np.float64) * 1000.0
    fig = go.Figure(go.Scatter(x=x_mm, y=ppm_line, mode="lines", line={"color": "#1f77b4"}))
    fig.update_layout(
        title=f"{title} (y0={y0 * 1000.0:.2f} mm)",
        xaxis_title="x (mm)",
        yaxis_title="ppm",
        yaxis={"range": [vmin, vmax]},
        height=240,
        margin={"l": 40, "r": 10, "b": 40, "t": 40},
    )
    return fig


def _magnetization_lines(
    centers: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    directions: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    scale: float,
) -> tuple[
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
]:
    start = centers
    end = centers + directions * float(scale)
    count = centers.shape[0]
    xs = np.empty(3 * count, dtype=np.float64)
    ys = np.empty(3 * count, dtype=np.float64)
    zs = np.empty(3 * count, dtype=np.float64)
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


def _estimate_arrow_scale(
    centers: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> float:
    if centers.size == 0:
        return 0.01
    radii = np.linalg.norm(centers[:, :2], axis=1)
    max_r = float(np.max(radii)) if radii.size > 0 else 0.0
    return max(0.01, 0.1 * max_r)


def _add_scene_run(
    fig: go.Figure,
    *,
    run_name: str,
    centers: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    directions: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    layer_ids: np.ndarray[tuple[int, ...], np.dtype[np.int_]],
    mode: PlotlyMode,
    compare_mode: bool,
    color: str | None,
    magnet_size_m: float | None,
    magnet_thickness_m: float | None,
    arrow_length_m: float | None,
    arrow_head_angle_deg: float,
    magnet_surface_color: str,
) -> None:
    if centers.size == 0:
        return

    phi = np.arctan2(directions[:, 1], directions[:, 0])
    if mode in ("cubes", "cubes_arrows"):
        size_m = 0.02 if magnet_size_m is None else magnet_size_m
        thickness_m = size_m if magnet_thickness_m is None else magnet_thickness_m
        edge_width = max(1, int(round((size_m * 1000.0) / 7.5)))
        size_mm = size_m * 1000.0
        ambient = 0.3
        if size_mm < 15.0:
            ambient = 0.3 + 0.7 * (15.0 - size_mm) / 15.0
        ambient = min(ambient + 0.3, 1.0)
        vx, vy, vz, i, j, k, ex, ey, ez = build_cubes_mesh(centers, phi, size_m, thickness_m)
        fig.add_trace(
            go.Mesh3d(
                x=vx,
                y=vy,
                z=vz,
                i=i,
                j=j,
                k=k,
                name=run_name,
                color=magnet_surface_color,
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
                name=f"{run_name} edges",
                line={"color": "black", "width": edge_width},
                showlegend=False,
            )
        )
        if mode == "cubes_arrows":
            length = max(0.01, 1.5 * size_m) if arrow_length_m is None else arrow_length_m
            ax, ay, az = build_magnetization_arrows(
                centers,
                directions,
                length_m=length,
                head_angle_deg=arrow_head_angle_deg,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=ax,
                    y=ay,
                    z=az,
                    mode="lines",
                    name=f"{run_name} arrows",
                    line={"color": "red", "width": 4},
                    showlegend=False,
                )
            )
        return

    if compare_mode:
        marker_kwargs: dict[str, object] = {"color": color or "#1f77b4"}
    else:
        marker_kwargs = {
            "color": layer_ids.astype(float),
            "colorscale": "Viridis",
            "showscale": True,
        }

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
            name=run_name,
            marker={"size": marker_size, **marker_kwargs},
        )
    )

    if mode == "pretty":
        scale = _estimate_arrow_scale(centers)
        if magnet_size_m is not None:
            scale = max(scale, magnet_size_m)
        xs, ys, zs = _magnetization_lines(centers, directions, scale)
        line_width = 2
        if magnet_thickness_m is not None:
            line_width = max(1, int(round(magnet_thickness_m * 2000.0)))
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=f"{run_name} phi",
                line={"width": line_width, "color": color or "#444"},
                showlegend=False,
            )
        )


def build_scene_figure(
    payload: Scene3DPayload,
    *,
    mode: PlotlyMode,
    magnet_size_m: float | None = None,
    magnet_thickness_m: float | None = None,
    arrow_length_m: float | None = None,
    arrow_head_angle_deg: float = 30.0,
    scene_camera: dict[str, Any] | None = None,
    height: int | None = None,
    overlay_mesh: ObjMesh | None = None,
    overlay_name: str = "Human model",
    overlay_color: str = "rgb(210,180,170)",
    overlay_opacity: float = 0.35,
    coil_overlay: CoilPolylineSet | None = None,
    coil_line_width: float = 4.0,
    magnet_surface_color: str = "rgb(229,229,229)",
) -> go.Figure:
    fig = go.Figure()

    if overlay_mesh is not None:
        fig.add_trace(
            go.Mesh3d(
                x=overlay_mesh.vertices[:, 0],
                y=overlay_mesh.vertices[:, 1],
                z=overlay_mesh.vertices[:, 2],
                i=overlay_mesh.faces[:, 0],
                j=overlay_mesh.faces[:, 1],
                k=overlay_mesh.faces[:, 2],
                name=overlay_name,
                color=overlay_color,
                opacity=float(overlay_opacity),
                flatshading=True,
                lighting={
                    "ambient": 0.45,
                    "diffuse": 0.55,
                    "specular": 0.08,
                    "roughness": 0.95,
                    "fresnel": 0.05,
                },
                lightposition={"x": 0.0, "y": 0.0, "z": 2.0},
                showscale=False,
            )
        )

    if coil_overlay is not None:
        for group in build_plotly_polyline_groups(coil_overlay):
            fig.add_trace(
                go.Scatter3d(
                    x=group.x,
                    y=group.y,
                    z=group.z,
                    mode="lines",
                    name=f"Coil {group.color_css}",
                    line={"color": group.color_css, "width": float(coil_line_width)},
                )
            )

    compare_mode = payload.secondary is not None
    _add_scene_run(
        fig,
        run_name=payload.primary.run_name,
        centers=np.asarray(payload.primary.centers, dtype=np.float64),
        directions=np.asarray(payload.primary.direction_vectors, dtype=np.float64),
        layer_ids=np.asarray(payload.primary.layer_ids, dtype=np.int_),
        mode=mode,
        compare_mode=compare_mode,
        color="#1f77b4",
        magnet_size_m=magnet_size_m,
        magnet_thickness_m=magnet_thickness_m,
        arrow_length_m=arrow_length_m,
        arrow_head_angle_deg=arrow_head_angle_deg,
        magnet_surface_color=magnet_surface_color,
    )
    if payload.secondary is not None:
        _add_scene_run(
            fig,
            run_name=payload.secondary.run_name,
            centers=np.asarray(payload.secondary.centers, dtype=np.float64),
            directions=np.asarray(payload.secondary.direction_vectors, dtype=np.float64),
            layer_ids=np.asarray(payload.secondary.layer_ids, dtype=np.int_),
            mode=mode,
            compare_mode=True,
            color="#ff7f0e",
            magnet_size_m=magnet_size_m,
            magnet_thickness_m=magnet_thickness_m,
            arrow_length_m=arrow_length_m,
            arrow_head_angle_deg=arrow_head_angle_deg,
            magnet_surface_color=magnet_surface_color,
        )

    layout: dict[str, Any] = {
        "scene": {
            "aspectmode": "data",
            "xaxis": {"title": "x", "range": list(payload.scene_ranges[0])},
            "yaxis": {"title": "y", "range": list(payload.scene_ranges[1])},
            "zaxis": {"title": "z", "range": list(payload.scene_ranges[2])},
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 30},
        "legend": {"orientation": "h"},
    }
    if scene_camera is not None:
        layout["scene_camera"] = scene_camera
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig


__all__ = [
    "PlotlyMode",
    "build_scene_figure",
    "combine_scene_ranges",
    "common_ppm_limits",
    "plot_cross_section",
    "plot_error_map",
]
