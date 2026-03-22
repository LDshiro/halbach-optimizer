from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from halbach.types import FloatArray

FaceArray: TypeAlias = NDArray[np.int32]


@dataclass(frozen=True)
class ObjMesh:
    vertices: FloatArray
    faces: FaceArray
    bbox_min: FloatArray
    bbox_max: FloatArray

    @property
    def bbox_size(self) -> FloatArray:
        return np.asarray(self.bbox_max - self.bbox_min, dtype=np.float64)


def _compute_bbox(vertices: FloatArray) -> tuple[FloatArray, FloatArray]:
    bbox_min = np.min(vertices, axis=0).astype(np.float64, copy=False)
    bbox_max = np.max(vertices, axis=0).astype(np.float64, copy=False)
    return bbox_min, bbox_max


def _parse_vertex_index(token: str, vertex_count: int) -> int:
    head = token.split("/", 1)[0]
    if not head:
        raise ValueError(f"Unsupported OBJ face token: {token!r}")
    raw_idx = int(head)
    index = raw_idx - 1 if raw_idx > 0 else vertex_count + raw_idx
    if index < 0 or index >= vertex_count:
        raise ValueError(f"OBJ face index out of range: {raw_idx}")
    return index


def load_obj_mesh(path: Path) -> ObjMesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid OBJ vertex line: {line!r}")
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    raise ValueError(f"Invalid OBJ face line: {line!r}")
                idx = [_parse_vertex_index(token, len(vertices)) for token in parts]
                for pos in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[pos], idx[pos + 1]))

    if not vertices:
        raise ValueError(f"OBJ file has no vertices: {path}")
    if not faces:
        raise ValueError(f"OBJ file has no faces: {path}")

    vertices_arr = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int32)
    bbox_min, bbox_max = _compute_bbox(vertices_arr)
    return ObjMesh(
        vertices=vertices_arr,
        faces=faces_arr,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def transform_obj_mesh(
    mesh: ObjMesh,
    *,
    unit_scale_m: float,
    uniform_scale: float = 1.0,
    rotation_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    anchor_mode: Literal["bbox_center"] = "bbox_center",
) -> ObjMesh:
    if anchor_mode != "bbox_center":
        raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")
    if unit_scale_m <= 0.0:
        raise ValueError("unit_scale_m must be > 0")
    if uniform_scale <= 0.0:
        raise ValueError("uniform_scale must be > 0")

    bbox_center = 0.5 * (mesh.bbox_min + mesh.bbox_max)
    vertices = np.asarray(mesh.vertices - bbox_center, dtype=np.float64)
    vertices = vertices * (float(unit_scale_m) * float(uniform_scale))

    rx_deg, ry_deg, rz_deg = rotation_xyz_deg
    rx = np.deg2rad(float(rx_deg))
    ry = np.deg2rad(float(ry_deg))
    rz = np.deg2rad(float(rz_deg))

    cos_rx = float(np.cos(rx))
    sin_rx = float(np.sin(rx))
    cos_ry = float(np.cos(ry))
    sin_ry = float(np.sin(ry))
    cos_rz = float(np.cos(rz))
    sin_rz = float(np.sin(rz))

    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_rx, -sin_rx],
            [0.0, sin_rx, cos_rx],
        ],
        dtype=np.float64,
    )
    rot_y = np.array(
        [
            [cos_ry, 0.0, sin_ry],
            [0.0, 1.0, 0.0],
            [-sin_ry, 0.0, cos_ry],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [cos_rz, -sin_rz, 0.0],
            [sin_rz, cos_rz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    transformed = vertices @ rot_x.T
    transformed = transformed @ rot_y.T
    transformed = transformed @ rot_z.T
    transformed = transformed + np.asarray(translation_xyz_m, dtype=np.float64)

    bbox_min, bbox_max = _compute_bbox(transformed)
    return ObjMesh(
        vertices=np.asarray(transformed, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


__all__ = ["ObjMesh", "load_obj_mesh", "transform_obj_mesh"]
