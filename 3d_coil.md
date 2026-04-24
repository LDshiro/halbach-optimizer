# 3D Coil NPZ Format

This document defines the `gradientcoil_3d_coil` export format used by the coil evaluation tab.

## Overview

- Container: `.npz` created with `np.savez_compressed`
- Purpose: export coil polylines already reconstructed in world `(x, y, z)` coordinates
- Coordinate unit: meters
- Color encoding: `srgba_u8`
- Scope of v1: coil geometry only

ROI guides, aperture guides, Bz profiles, and field data are not included in v1.

## Required keys

- `format_name`: scalar string, must be `"gradientcoil_3d_coil"`
- `format_version`: scalar string, currently `"1.0"`
- `points_xyz`: `float64`, shape `(N, 3)`
- `polyline_start`: `int64`, shape `(L,)`
- `polyline_count`: `int64`, shape `(L,)`
- `polyline_surface`: `int64`, shape `(L,)`
- `polyline_sign`: `float64`, shape `(L,)`
- `polyline_closed`: `bool`, shape `(L,)`
- `polyline_periodic_closed`: `bool`, shape `(L,)`
- `polyline_color_rgba_u8`: `uint8`, shape `(L, 4)`
- `source_contour_npz`: scalar string
- `source_config_json`: scalar string
- `input_kind`: scalar string
- `coordinate_unit`: scalar string, must be `"m"`
- `color_space`: scalar string, must be `"srgba_u8"`

## Geometry layout

The polyline data uses a flat point buffer with offsets.

- `points_xyz[k]` is the `k`-th 3D point in world coordinates.
- Polyline `i` is reconstructed as:

```python
start = polyline_start[i]
count = polyline_count[i]
pts_i = points_xyz[start : start + count]
```

- `sum(polyline_count) == len(points_xyz)` must hold.
- Point order is the same as the displayed coil path.

## Semantics

- `polyline_surface[i]`: source surface index
- `polyline_sign[i]`: current sign carried from the contour record
- `polyline_closed[i]`:
  - `True` for explicitly closed polylines
  - `False` for open polylines
- `polyline_periodic_closed[i]`:
  - `True` when the source contour is open in unwrap space but physically closes across a cylinder seam
  - `False` otherwise

## Color rule

The exported color matches the current GUI rule.

- `polyline_sign > 0`: red = `[255, 0, 0, 255]`
- `polyline_sign <= 0`: blue = `[0, 0, 255, 255]`

## Minimal loader example

```python
from pathlib import Path

import numpy as np

path = Path("runs/coil3d_example.npz")
with np.load(path, allow_pickle=False) as npz:
    points_xyz = np.asarray(npz["points_xyz"], dtype=float)
    starts = np.asarray(npz["polyline_start"], dtype=np.int64)
    counts = np.asarray(npz["polyline_count"], dtype=np.int64)
    colors = np.asarray(npz["polyline_color_rgba_u8"], dtype=np.uint8)

    polylines = [
        points_xyz[start : start + count]
        for start, count in zip(starts, counts, strict=False)
    ]
```

## Reserved extensions

Future versions may add optional keys without breaking v1 readers.

- Point attributes: `point_attr_*`
- Polyline attributes: `polyline_attr_*`
- Field data: `field_*`

Readers should ignore unknown keys.
