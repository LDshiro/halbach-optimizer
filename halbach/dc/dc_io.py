from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def write_dc_run(
    out_dir: Path,
    *,
    meta: dict[str, Any],
    trace: dict[str, Any],
    results: dict[str, Any],
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with (out_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    with (out_path / "trace.json").open("w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, sort_keys=True)

    npz_payload: dict[str, NDArray[np.generic]] = {}
    for key, value in results.items():
        npz_payload[str(key)] = np.asarray(value)
    np.savez_compressed(out_path / "results.npz", **npz_payload)


__all__ = ["write_dc_run"]
