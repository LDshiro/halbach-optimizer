"""Halbach package."""

from halbach.near import (
    NearGraph,
    NearWindow,
    build_near_graph,
    flatten_index,
    get_near_graph_from_geom,
    unflatten_index,
)

__all__ = [
    "NearGraph",
    "NearWindow",
    "build_near_graph",
    "flatten_index",
    "get_near_graph_from_geom",
    "unflatten_index",
]
