import numpy as np

from halbach.near import NearWindow, build_near_graph, flatten_index


def test_near_graph_shapes_and_validity() -> None:
    R, K, N = 1, 4, 8
    window = NearWindow(wr=0, wz=1, wphi=1)
    graph = build_near_graph(R, K, N, window)

    assert graph.deg_max == 8
    M = R * K * N
    assert graph.nbr_idx.shape == (M, graph.deg_max)
    assert graph.nbr_mask.shape == (M, graph.deg_max)
    assert graph.nbr_idx.dtype == np.int32
    assert graph.nbr_mask.dtype == np.bool_

    valid = graph.nbr_idx[graph.nbr_mask]
    assert valid.size > 0
    assert int(valid.min()) >= 0
    assert int(valid.max()) < M

    for i in range(M):
        nbrs = graph.nbr_idx[i][graph.nbr_mask[i]]
        assert not np.any(nbrs == i)


def test_near_graph_wrap_phi() -> None:
    R, K, N = 1, 2, 8
    window = NearWindow(wr=0, wz=0, wphi=1)
    graph = build_near_graph(R, K, N, window)
    src = flatten_index(0, 0, 0, R, K, N)
    dst = flatten_index(0, 0, N - 1, R, K, N)
    nbrs = graph.nbr_idx[src][graph.nbr_mask[src]]
    assert dst in nbrs


def test_near_graph_cache() -> None:
    window = NearWindow(wr=0, wz=1, wphi=1)
    g1 = build_near_graph(1, 4, 8, window)
    g2 = build_near_graph(1, 4, 8, window)
    assert g1 is g2
