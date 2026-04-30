from __future__ import annotations

import math

from halbach.assembly.types import MagnetError, OrientationCandidate


def default_orientations() -> tuple[OrientationCandidate, ...]:
    """Return the standard discrete4 Plan C insertion orientations."""
    return (
        OrientationCandidate(
            id="O0",
            angle_deg=0.0,
            instruction="align marking arrow with the slot reference mark",
        ),
        OrientationCandidate(
            id="O90",
            angle_deg=90.0,
            instruction="rotate marking arrow 90 degrees counterclockwise from the slot reference mark",
        ),
        OrientationCandidate(
            id="O180",
            angle_deg=180.0,
            instruction="point marking arrow opposite the slot reference mark",
        ),
        OrientationCandidate(
            id="O270",
            angle_deg=270.0,
            instruction="rotate marking arrow 270 degrees counterclockwise from the slot reference mark",
        ),
    )


def _orientation_from_id(orientation_id: str) -> OrientationCandidate:
    for orientation in default_orientations():
        if orientation.id == orientation_id:
            return orientation
    raise ValueError(f"Unsupported orientation id: {orientation_id}")


def rotate_error_for_orientation(
    error: MagnetError,
    orientation: OrientationCandidate | str,
) -> MagnetError:
    """
    Rotate transverse magnet error by the insertion orientation.

    The rotation is counterclockwise in the local transverse plane:
    d1' = cos(psi) d1 - sin(psi) d2
    d2' = sin(psi) d1 + cos(psi) d2
    """
    orientation_item = _orientation_from_id(orientation) if isinstance(orientation, str) else orientation
    psi = math.radians(float(orientation_item.angle_deg))
    c = math.cos(psi)
    s = math.sin(psi)
    d1 = float(error.delta_perp_1)
    d2 = float(error.delta_perp_2)
    return MagnetError(
        epsilon_parallel=float(error.epsilon_parallel),
        delta_perp_1=c * d1 - s * d2,
        delta_perp_2=s * d1 + c * d2,
    )


__all__ = ["default_orientations", "rotate_error_for_orientation"]
