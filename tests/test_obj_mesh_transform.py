import numpy as np

from halbach.obj_mesh import ObjMesh, transform_obj_mesh


def _box_mesh() -> ObjMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 6.0],
            [2.0, 0.0, 6.0],
            [2.0, 4.0, 6.0],
            [0.0, 4.0, 6.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return ObjMesh(
        vertices=vertices,
        faces=faces,
        bbox_min=np.min(vertices, axis=0),
        bbox_max=np.max(vertices, axis=0),
    )


def test_transform_obj_mesh_applies_scale_rotation_and_translation() -> None:
    mesh = _box_mesh()

    transformed = transform_obj_mesh(
        mesh,
        unit_scale_m=0.01,
        uniform_scale=1.0,
        rotation_xyz_deg=(90.0, 0.0, 0.0),
        translation_xyz_m=(1.0, 2.0, 3.0),
        anchor_mode="bbox_center",
    )

    bbox_center = 0.5 * (transformed.bbox_min + transformed.bbox_max)
    assert np.allclose(bbox_center, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(transformed.bbox_size, np.array([0.02, 0.06, 0.04]), atol=1e-12)


def test_transform_obj_mesh_applies_uniform_scale_after_unit_conversion() -> None:
    mesh = _box_mesh()

    transformed = transform_obj_mesh(
        mesh,
        unit_scale_m=0.01,
        uniform_scale=2.0,
        rotation_xyz_deg=(0.0, 0.0, 0.0),
        translation_xyz_m=(0.0, 0.0, 0.0),
        anchor_mode="bbox_center",
    )

    assert np.allclose(transformed.bbox_size, np.array([0.04, 0.08, 0.12]), atol=1e-12)
