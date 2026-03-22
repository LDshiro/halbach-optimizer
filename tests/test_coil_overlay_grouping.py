import numpy as np

from halbach.coil_overlay import (
    CoilPolylineSet,
    build_plotly_polyline_groups,
    rotate_coil_polyline_x90,
)


def test_build_plotly_polyline_groups_groups_by_rgba_and_inserts_nan_separators() -> None:
    coil = CoilPolylineSet(
        points_xyz=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        polyline_start=np.asarray([0, 2, 3], dtype=np.int64),
        polyline_count=np.asarray([2, 1, 2], dtype=np.int64),
        polyline_color_rgba_u8=np.asarray(
            [
                [255, 0, 0, 255],
                [0, 0, 255, 255],
                [255, 0, 0, 255],
            ],
            dtype=np.uint8,
        ),
        bbox_min=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        bbox_max=np.asarray([2.0, 1.0, 0.0], dtype=np.float64),
    )

    groups = build_plotly_polyline_groups(coil)

    assert len(groups) == 2
    by_color = {group.color_css: group for group in groups}
    red = by_color["rgba(255,0,0,1.000)"]
    blue = by_color["rgba(0,0,255,1.000)"]
    assert red.polyline_count == 2
    assert blue.polyline_count == 1
    assert np.isnan(red.x[-1])
    assert np.isnan(red.y[-1])
    assert np.isnan(red.z[-1])
    assert np.isnan(blue.x[-1])


def test_rotate_coil_polyline_x90_rotates_about_x_axis_in_quarter_turns() -> None:
    coil = CoilPolylineSet(
        points_xyz=np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float64,
        ),
        polyline_start=np.asarray([0], dtype=np.int64),
        polyline_count=np.asarray([2], dtype=np.int64),
        polyline_color_rgba_u8=np.asarray([[255, 0, 0, 255]], dtype=np.uint8),
        bbox_min=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        bbox_max=np.asarray([4.0, 5.0, 6.0], dtype=np.float64),
    )

    rotated = rotate_coil_polyline_x90(coil, 1)

    assert np.allclose(rotated.points_xyz[0], np.array([1.0, -3.0, 2.0]))
    assert np.allclose(rotated.points_xyz[1], np.array([4.0, -6.0, 5.0]))
    assert np.allclose(rotated.bbox_min, np.array([1.0, -6.0, 2.0]))
    assert np.allclose(rotated.bbox_max, np.array([4.0, -3.0, 5.0]))
