import numpy as np
from numpy.typing import NDArray

from halbach.types import Geometry
from robust_opt_halbach_gradnorm_minimal import (
    FACTOR,
    build_roi_points,
    m0,
    objective_only,
    objective_with_grads_fixed,
    phi0,
)


def fd_grad_yspace(
    alphas: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    geom: Geometry,
    pts: NDArray[np.float64],
    eps: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    y-space での中心差分FD勾配（テスト用）
    - alphas: (R,K)
    - r_bases: (K,)
    戻り値:
    - gA_fd: (R,K)
    - gR_fd: (K,)
    """
    R, K = alphas.shape
    gA = np.zeros((R, K), dtype=float)
    gR = np.zeros(K, dtype=float)

    # alphas
    for r in range(R):
        for k in range(K):
            ap = alphas.copy()
            am = alphas.copy()
            ap[r, k] += eps
            am[r, k] -= eps
            fp = objective_only(
                ap,
                r_bases,
                geom.theta,
                geom.sin2,
                geom.cth,
                geom.sth,
                geom.z_layers,
                geom.ring_offsets,
                pts,
                FACTOR,
                phi0,
                m0,
            )
            fm = objective_only(
                am,
                r_bases,
                geom.theta,
                geom.sin2,
                geom.cth,
                geom.sth,
                geom.z_layers,
                geom.ring_offsets,
                pts,
                FACTOR,
                phi0,
                m0,
            )
            gA[r, k] = (fp - fm) / (2.0 * eps)

    # r_bases
    for k in range(K):
        rp = r_bases.copy()
        rm = r_bases.copy()
        rp[k] += eps
        rm[k] -= eps
        fp = objective_only(
            alphas,
            rp,
            geom.theta,
            geom.sin2,
            geom.cth,
            geom.sth,
            geom.z_layers,
            geom.ring_offsets,
            pts,
            FACTOR,
            phi0,
            m0,
        )
        fm = objective_only(
            alphas,
            rm,
            geom.theta,
            geom.sin2,
            geom.cth,
            geom.sth,
            geom.z_layers,
            geom.ring_offsets,
            pts,
            FACTOR,
            phi0,
            m0,
        )
        gR[k] = (fp - fm) / (2.0 * eps)

    return gA, gR


def test_gradcheck_yspace_matches_fd() -> None:
    """
    解析勾配（objective_with_grads_fixed）と中心差分FDが一致することを確認する。
    目標精度：最大絶対誤差 ~1e-6〜1e-9 オーダ（超小モデルなので厳しめでもOK）
    """
    # tiny geometry
    N = 10
    K = 5
    R = 1

    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2 = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)

    z_layers = np.linspace(-0.04, 0.04, K)
    ring_offsets = np.array([0.0], dtype=float)

    rng = np.random.default_rng(1)
    alphas = 1e-3 * rng.standard_normal((R, K))
    r0 = 0.2
    r_bases = r0 + 1e-4 * rng.standard_normal(K)

    pts = build_roi_points(roi_r=0.03, roi_step=0.03)

    dzs = np.diff(z_layers)
    dz = float(np.median(np.abs(dzs))) if dzs.size > 0 else 0.01
    Lz = dz * K
    geom = Geometry(
        theta=theta,
        sin2=sin2,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
        N=N,
        K=K,
        R=R,
        dz=dz,
        Lz=Lz,
    )

    J, gA_ana, gR_ana, B0n = objective_with_grads_fixed(alphas, r_bases, geom, pts)
    assert np.isfinite(J)
    assert np.isfinite(B0n)

    eps = 1e-6
    gA_fd, gR_fd = fd_grad_yspace(alphas, r_bases, geom, pts, eps)

    # Compare
    errA = np.max(np.abs(gA_ana - gA_fd))
    errR = np.max(np.abs(gR_ana - gR_fd))

    # 許容値：小規模モデルなのでかなり厳しめでOKだが、環境差を考え少し余裕を持たせる
    assert errA < 1e-6
    assert errR < 1e-6
