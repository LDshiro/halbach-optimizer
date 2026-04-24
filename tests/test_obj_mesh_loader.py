from pathlib import Path

import numpy as np

from halbach.obj_mesh import load_obj_mesh


def test_load_obj_mesh_triangulates_quads_and_ignores_material_lines(tmp_path: Path) -> None:
    obj_path = tmp_path / "body.obj"
    obj_path.write_text(
        "\n".join(
            [
                "# test mesh",
                "mtllib body.mtl",
                "o Human",
                "g Torso",
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 1 1",
                "vt 0 1",
                "vn 0 0 1",
                "usemtl Skin",
                "f 1/1/1 2/2/1 3/3/1 4/4/1",
            ]
        ),
        encoding="utf-8",
    )

    mesh = load_obj_mesh(obj_path)

    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (2, 3)
    assert np.array_equal(mesh.faces[0], np.array([0, 1, 2], dtype=np.int32))
    assert np.array_equal(mesh.faces[1], np.array([0, 2, 3], dtype=np.int32))
    assert np.allclose(mesh.bbox_min, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(mesh.bbox_max, np.array([1.0, 1.0, 0.0]))


def test_load_obj_mesh_supports_face_slash_syntax_and_negative_indices(tmp_path: Path) -> None:
    obj_path = tmp_path / "slash.obj"
    obj_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 2 0 0",
                "v 2 1 0",
                "f -3//1 -2//1 -1//1",
            ]
        ),
        encoding="utf-8",
    )

    mesh = load_obj_mesh(obj_path)

    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)
    assert np.array_equal(mesh.faces[0], np.array([0, 1, 2], dtype=np.int32))
