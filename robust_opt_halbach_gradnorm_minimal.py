# -*- coding: utf-8 -*-
"""
robust_opt_halbach_gradnorm_minimal.py

Load nominal optimized Halbach array (.npz), run gradient-norm (y-space) robust optimization
with L-BFGS-B, perform Monte Carlo evaluation, and save results (.npz/.json/.mat + figures).

- Robust term: J_gn(x) = J(x) + 0.5 * rho_gn * || Sigma_y^{1/2} * grad_y J ||^2
- Gradient:    grad_x J_gn = grad_x J + rho_gn * (dy/dx)^T [ H_y(J) * (Sigma_y * grad_y J) ]
- H_y(J) v is computed via central differences in y-space using analytic grad_y J.

Author: your project

【用語と記号の対応】
- x-space（設計空間）: x = [alphas(:); r_vars]
    * alphas     : (R, K) … 角度の自由度（全層・全リング）
    * r_vars     : (K/2-2,) … Z対称性により「下半分」層のみを設計変数（中央4層は固定）。
- y-space（物理パラメタ空間）: y = [alphas (R, K); r_bases (K)]
    * r_bases    : (K,) … 各層の基準半径。x からの写像で上下対称にコピーされる。

【ロバスト項の狙い】
- 本スクリプトのロバスト正則化は「y-space 上の勾配ノルム」に基づく。
- Monte Carlo（MC）で乱数摂動を与えるのも y-space なので、評価空間とロバスト化空間が整合する。
- HVP（Hessian-vector product）は y-space で中心差分により計算する（解析勾配を再利用）。

【実行フロー】
1. 名目設計（.npz）を読み込み → 幾何情報と初期点を構築
2. y-space 勾配ノルム正則化のロバスト目的で L-BFGS-B を実行（半径は下限 bound）
3. 最適化履歴・結果（robust と viewer 互換の両方）・MAT ファイルを保存
4. 名目 vs ロバストの Monte Carlo 解析を同一サンプルで比較し、図とサマリを保存
"""

import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.optimize import minimize
from scipy.io import savemat

# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を定義・取得する。
    - in_npz : 名目設計（既存最適化済み）を保存した .npz のパス
    - out_dir: 出力先ディレクトリ
    - roi_*  : 目的関数評価に使う球状 ROI の半径・サンプリングピッチ（m）
    - sigma_*: Monte Carloで与える 1σ の公差（角度はdeg、半径はmm）
    - rho_gn : 勾配ノルム正則化の重み（y-space）
    - eps_hvp: y-space HVP 中心差分のベースステップ
    - maxiter, ftol, gtol: L-BFGS-B の収束パラメタ
    - min_radius_drop_mm : 半径下限（初期値 - この mm）を box 制約として与える
    - mc_samples, seed: Monte Carlo のサンプル数と乱数種
    """
    ap = argparse.ArgumentParser(
        description="Robust (gradient-norm) optimization from nominal .npz, with MC & saving")
    ap.add_argument("--in_npz", type=str, default="/mnt/data/diam_opt_saved_results.npz",
                    help="path to nominal .npz")
    ap.add_argument("--out_dir", type=str, default="/mnt/data",
                    help="output directory")

    # ROI for objective
    ap.add_argument("--roi_r", type=float, default=0.140,
                    help="ROI sphere radius [m]")
    ap.add_argument("--roi_step", type=float, default=0.020,
                    help="ROI grid step [m]")

    # Robust hyper-parameters (y-space)
    ap.add_argument("--sigma_alpha_deg", type=float, default=0.5,
                    help="1-sigma angle tol [deg]")
    ap.add_argument("--sigma_r_mm", type=float, default=0.20,
                    help="1-sigma radius tol [mm]")
    ap.add_argument("--rho_gn", type=float, default=1e-4,
                    help="gradient-norm weight (y-space)")
    ap.add_argument("--eps_hvp", type=float, default=1e-6,
                    help="HVP step base for y-space central difference")

    # Optimizer
    ap.add_argument("--maxiter", type=int, default=900,
                    help="L-BFGS-B iterations")
    ap.add_argument("--ftol", type=float, default=1e-12,
                    help="L-BFGS-B ftol")
    ap.add_argument("--gtol", type=float, default=1e-12,
                    help="L-BFGS-B gtol")
    ap.add_argument("--min_radius_drop_mm", type=float, default=20.0,
                    help="lower bound: r_vars >= r_init - this [mm]")

    # Monte Carlo
    ap.add_argument("--mc_samples", type=int, default=600,
                    help="Monte Carlo samples")
    ap.add_argument("--seed", type=int, default=20250926,
                    help="random seed")
    return ap.parse_args()


# ---------------- Physics constants ----------------
# 物理定数と双極子近似の係数。いずれも SI 単位系（m, T, A）に整合。
mu0 = 4*np.pi*1e-7
FACTOR = mu0/(4*np.pi)
phi0 = -np.pi/2  # ハルバッハ p=1 を +y 強化とする位相
m0 = 1.0         # 双極子の相対強度（正規化）


# ---------------- Utilities ----------------
def load_nominal_npz(path: str) -> dict:
    """
    名目設計の .npz を読み込むヘルパ。
    期待フィールド:
      - alphas_opt: (R, K)
      - r_bases_opt: (K,)
      - theta: (N,)  （角度テーブル）
      - sin2 or sin2th: (N,)  （sin(2θ)）
      - z_layers: (K,) （各層の z 位置）
      - ring_offsets: (R,) （同心リングの半径オフセット）
    返却: 幾何と角テーブル、三角関数、層位置などを 1 つの dict にまとめる。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Nominal NPZ not found: {path}")

    dat = np.load(path, allow_pickle=True)
    alphas = np.array(dat["alphas_opt"], dtype=float)                       # (R,K)
    r_bases = np.array(dat["r_bases_opt"], dtype=float).reshape(-1)         # (K,)
    theta = np.array(dat["theta"], dtype=float).reshape(-1)                 # (N,)

    # sin(2θ) は保存名が sin2 or sin2th のどちらかだった経緯があるので両対応
    if "sin2" in dat:
        sin2 = np.array(dat["sin2"], dtype=float).reshape(-1)
    elif "sin2th" in dat:
        sin2 = np.array(dat["sin2th"], dtype=float).reshape(-1)
    else:
        sin2 = np.sin(2.0*theta)

    z_layers = np.array(dat["z_layers"], dtype=float).reshape(-1)           # (K,)
    ring_offsets = np.array(dat["ring_offsets"], dtype=float).reshape(-1)   # (R,)

    # 派生量：N, R, K, cosθ, sinθ, および dz, Lz（情報提供用）
    N = theta.size
    R, K = alphas.shape
    cth, sth = np.cos(theta), np.sin(theta)
    dzs = np.diff(z_layers)
    dz = float(np.median(np.abs(dzs))) if dzs.size > 0 else 0.01
    Lz = dz * K

    return dict(
        alphas=alphas, r_bases=r_bases, theta=theta, sin2=sin2,
        z_layers=z_layers, ring_offsets=ring_offsets,
        N=N, R=R, K=K, cth=cth, sth=sth, dz=dz, Lz=Lz
    )


def build_symmetry_indices(K: int):
    """
    半径の z 対称設計を実装するためのインデックス集合を構築。
    - 中央 4 層は固定（文脈上の慣例）
    - 「下半分」の index 群 lower_var を変数、上半分 upper_var へミラーする
    """
    fixed_center = np.arange(K//2 - 2, K//2 + 2)  # [K/2-2, ..., K/2+1]
    lower_var = np.arange(0, K//2 - 2)            # 下半分（中央4層の手前まで）
    upper_var = (K - 1) - lower_var               # 上半分を鏡映
    return lower_var, upper_var, fixed_center


def build_roi_points(roi_r: float, roi_step: float) -> np.ndarray:
    """
    球状 ROI 内のグリッド点を生成（xy×z の三次元格子を球形マスクで切り抜き）。
    返却 shape: (M, 3)
    """
    xs = np.linspace(-roi_r, roi_r, 2*int(np.ceil(roi_r/roi_step)) + 1)
    ys = xs
    zs = xs
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='xy')
    mask = (X**2 + Y**2 + Z**2) <= roi_r**2 + 1e-15
    P = np.stack([X[mask], Y[mask], Z[mask]], axis=1).astype(np.float64)
    return P


def build_r_bases_from_vars(r_vars: np.ndarray, K: int, r0: float,
                            lower_var: np.ndarray, upper_var: np.ndarray) -> np.ndarray:
    """
    r_vars（下半分のみの可変半径）から、全層 K の r_bases を z 対称に構築。
    - 未指定の層は名目半径 r0。
    """
    r = np.full(K, r0, dtype=np.float64)
    for j, k_low in enumerate(lower_var):
        k_up = upper_var[j]
        r[k_low] = r_vars[j]
        r[k_up] = r_vars[j]
    return r


def pack_x(alphas: np.ndarray, r_vars: np.ndarray) -> np.ndarray:
    """x = [alphas(:); r_vars] に連結して返す。"""
    return np.concatenate([alphas.ravel(), r_vars])


def unpack_x(x: np.ndarray, R: int, K: int):
    """x から alphas (R,K) と r_vars を切り出す。"""
    P = R * K
    al = x[:P].reshape(R, K)
    rv = x[P:]
    return al, rv


# ---------------- Field & objective (Numba kernels) ----------------
@njit(cache=True)
def compute_B_and_B0(alphas, r_bases, theta, sin2, cth, sth, z_layers, ring_offsets, pts, factor, phi0, m0):
    """
    ROI 内の全点 p における B(p) と、中心点 B0 を（同一ループで）計算。
    - 双極子和モデル。ベクトル成分 (Bx, By, Bz) と中心 (B0x, B0y, B0z) を返す。
    - 数値安定化のため rmag, r2 分母に微小項 (+1e-30) を付加。
    """
    Rloc = alphas.shape[0]; Kloc = alphas.shape[1]; Nloc = theta.shape[0]; M = pts.shape[0]
    Bx = np.zeros(M, dtype=np.float64); By = np.zeros(M, dtype=np.float64); Bz = np.zeros(M, dtype=np.float64)
    B0x = 0.0; B0y = 0.0; B0z = 0.0

    for k in range(Kloc):
        z0 = z_layers[k]; rb = r_bases[k]
        for r in range(Rloc):
            ak = alphas[r, k]; rho = rb + ring_offsets[r]
            for i in range(Nloc):
                th = theta[i]; c = cth[i]; s = sth[i]; s2 = sin2[i]
                phi = 2.0*th + phi0 + ak * s2  # Halbach p=1 の向き + α 調整
                mx = m0 * math.cos(phi); my = m0 * math.sin(phi)
                px = rho * c; py = rho * s     # 磁石の位置（xy）
                # ROI 点での場
                for p in range(M):
                    dx = pts[p,0] - px; dy = pts[p,1] - py; dz = pts[p,2] - z0
                    r2 = dx*dx + dy*dy + dz*dz
                    rmag = math.sqrt(r2) + 1e-30
                    invr3 = 1.0/(rmag*r2 + 1e-30)
                    rx = dx/rmag; ry = dy/rmag; rz = dz/rmag
                    mdotr = mx*rx + my*ry
                    Bx[p] += factor * (3.0*mdotr*rx - mx) * invr3
                    By[p] += factor * (3.0*mdotr*ry - my) * invr3
                    Bz[p] += factor * (3.0*mdotr*rz - 0.0) * invr3
                # 中心場（p=0）を同時に加算
                dx0 = -px; dy0 = -py; dz0 = -z0
                r20 = dx0*dx0 + dy0*dy0 + dz0*dz0
                r0mag = math.sqrt(r20) + 1e-30
                invr30 = 1.0/(r0mag*r20 + 1e-30)
                rx0 = dx0/r0mag; ry0 = dy0/r0mag; rz0 = dz0/r0mag
                mdotr0 = mx*rx0 + my*ry0
                B0x += factor * (3.0*mdotr0*rx0 - mx) * invr30
                B0y += factor * (3.0*mdotr0*ry0 - my) * invr30
                B0z += factor * (3.0*mdotr0*rz0 - 0.0) * invr30
    return Bx, By, Bz, B0x, B0y, B0z


@njit(cache=True)
def objective_only(alphas, r_bases, theta, sin2, cth, sth, z_layers, ring_offsets, pts, factor, phi0, m0):
    """
    目的関数 J(x) = mean_p ||B(p) - B0||^2 を返す（ベクトル誤差）。
    （ベクトルの向き差も評価に含まれる。）
    """
    Bx, By, Bz, B0x, B0y, B0z = compute_B_and_B0(
        alphas, r_bases, theta, sin2, cth, sth, z_layers, ring_offsets, pts, factor, phi0, m0)
    DBx = Bx - B0x; DBy = By - B0y; DBz = Bz - B0z
    return np.mean(DBx*DBx + DBy*DBy + DBz*DBz)


@njit(cache=True)
def grad_alpha_and_radius_fixed(alphas, r_bases, theta, sin2, cth, sth, z_layers, ring_offsets,
                                pts, factor, phi0, m0, DBx, DBy, DBz, sumDBx, sumDBy, sumDBz):
    """
    J の解析勾配（y-space: alphas と r_bases）を返す。
    - 中心場 B0 の勾配も差し引く 2 パス構成。
    - ループ構造は compute_B_and_B0 に準ずる（Numba でJIT）。
    """
    Rloc = alphas.shape[0]; Kloc = alphas.shape[1]; Nloc = theta.shape[0]; M = DBx.shape[0]; invM = 1.0/M
    g_alpha = np.zeros((Rloc, Kloc), dtype=np.float64)
    g_rbase = np.zeros(Kloc, dtype=np.float64)

    # dB0/dα, dB0/drb を先に評価（中心のみ）
    dB0_dalpha_x = np.zeros((Rloc, Kloc), dtype=np.float64)
    dB0_dalpha_y = np.zeros((Rloc, Kloc), dtype=np.float64)
    dB0_dalpha_z = np.zeros((Rloc, Kloc), dtype=np.float64)
    dB0_drb_x = np.zeros(Kloc, dtype=np.float64)
    dB0_drb_y = np.zeros(Kloc, dtype=np.float64)
    dB0_drb_z = np.zeros(Kloc, dtype=np.float64)

    for k in range(Kloc):
        z0 = z_layers[k]; rb = r_bases[k]
        for r in range(Rloc):
            ak = alphas[r, k]; rho = rb + ring_offsets[r]
            for i in range(Nloc):
                th = theta[i]; c = cth[i]; s = sth[i]; s2i = sin2[i]

                # 磁化方向 m とその α 導関数（dm/dα）
                phi = 2.0*th + phi0 + ak * s2i
                mx = m0 * math.cos(phi); my = m0 * math.sin(phi)
                dmx = -m0 * math.sin(phi); dmy =  m0 * math.cos(phi)

                # 中心点での B0 導関数（dB0/dα, dB0/drb）
                px = rho * c; py = rho * s
                dx0 = -px; dy0 = -py; dz0 = -z0
                r20 = dx0*dx0 + dy0*dy0 + dz0*dz0
                r0mag = math.sqrt(r20) + 1e-30
                invr30 = 1.0/(r0mag*r20 + 1e-30)
                rx0 = dx0/r0mag; ry0 = dy0/r0mag; rz0 = dz0/r0mag

                dmdotr0 = dmx*rx0 + dmy*ry0
                dBx0 = factor * (3.0*dmdotr0*rx0 - dmx) * invr30
                dBy0 = factor * (3.0*dmdotr0*ry0 - dmy) * invr30
                dBz0 = factor * (3.0*dmdotr0*rz0 - 0.0) * invr30

                dB0_dalpha_x[r, k] += dBx0 * s2i
                dB0_dalpha_y[r, k] += dBy0 * s2i
                dB0_dalpha_z[r, k] += dBz0 * s2i

                # dB0/drb の位置導関数（J 行列）に -∂p/∂rb = -(c, s, 0) を掛ける形
                r2 = r20; rmag = r0mag
                r5 = rmag * r2 * r2
                r7 = r5 * r2
                S = mx*dx0 + my*dy0  # m・Δx

                # ∂B/∂x, ∂B/∂y, ∂B/∂z の要素（J）
                Jx0 = factor * ( (3.0*mx*dx0 + 3.0*mx*dx0)/r5 + 3.0*S/r5 - 15.0*S*dx0*dx0/r7 )
                Jx1 = factor * ( (3.0*mx*dy0 + 3.0*my*dx0)/r5 - 15.0*S*dy0*dx0/r7 )
                Jx2 = factor * ( (3.0*mx*dz0 + 3.0*0.0*dx0)/r5 - 15.0*S*dz0*dx0/r7 )
                Jy0 = factor * ( (3.0*my*dx0 + 3.0*mx*dy0)/r5 - 15.0*S*dx0*dy0/r7 )
                Jy1 = factor * ( (3.0*my*dy0 + 3.0*my*dy0)/r5 + 3.0*S/r5 - 15.0*S*dy0*dy0/r7 )
                Jy2 = factor * ( (3.0*my*dz0 + 3.0*0.0*dy0)/r5 - 15.0*S*dz0*dy0/r7 )

                dB0_drb_x[k] += -(Jx0*c + Jy0*s)
                dB0_drb_y[k] += -(Jx1*c + Jy1*s)
                dB0_drb_z[k] += -(Jx2*c + Jy2*s)

    # ROI 全点の寄与を足し合わせる（B-B0 の誤差に対する ∂/∂α, ∂/∂rb）
    for k in range(Kloc):
        z0 = z_layers[k]; rb = r_bases[k]
        for r in range(Rloc):
            ak = alphas[r, k]; rho = rb + ring_offsets[r]
            sum_dot_alpha = 0.0
            sum_dot_rbase = 0.0
            for i in range(Nloc):
                th = theta[i]; c = cth[i]; s = sth[i]; s2i = sin2[i]
                phi = 2.0*th + phi0 + ak * s2i
                mx = m0 * math.cos(phi); my = m0 * math.sin(phi)
                dmx = -m0 * math.sin(phi); dmy =  m0 * math.cos(phi)
                px = rho * c; py = rho * s
                for p in range(M):
                    dx = pts[p,0] - px; dy = pts[p,1] - py; dz = pts[p,2] - z0
                    r2 = dx*dx + dy*dy + dz*dz
                    rmag = math.sqrt(r2) + 1e-30
                    invr3 = 1.0/(rmag*r2 + 1e-30)
                    rx = dx/rmag; ry = dy/rmag; rz = dz/rmag
                    mdotr = mx*rx + my*ry
                    dmdotr = dmx*rx + dmy*ry

                    # dB/dα（m の向きのみが α に依存）
                    dBx = factor * (3.0*dmdotr*rx - dmx) * invr3
                    dBy = factor * (3.0*dmdotr*ry - dmy) * invr3
                    dBz = factor * (3.0*dmdotr*rz - 0.0) * invr3
                    sum_dot_alpha += (DBx[p]*dBx + DBy[p]*dBy + DBz[p]*dBz) * s2i

                    # dB/drb（位置の導関数）
                    r5 = rmag * r2 * r2
                    r7 = r5 * r2
                    S = mx*dx + my*dy
                    Jx0 = factor * ( (3.0*mx*dx + 3.0*mx*dx)/r5 + 3.0*S/r5 - 15.0*S*dx*dx/r7 )
                    Jx1 = factor * ( (3.0*mx*dy + 3.0*my*dx)/r5 - 15.0*S*dy*dx/r7 )
                    Jx2 = factor * ( (3.0*mx*dz + 3.0*0.0*dx)/r5 - 15.0*S*dz*dx/r7 )
                    Jy0 = factor * ( (3.0*my*dx + 3.0*mx*dy)/r5 - 15.0*S*dx*dy/r7 )
                    Jy1 = factor * ( (3.0*my*dy + 3.0*my*dy)/r5 + 3.0*S/r5 - 15.0*S*dy*dy/r7 )
                    Jy2 = factor * ( (3.0*my*dz + 3.0*0.0*dy)/r5 - 15.0*S*dz*dy/r7 )
                    dBrx = -Jx0*c - Jy0*s
                    dBry = -Jx1*c - Jy1*s
                    dBrz = -Jx2*c - Jy2*s
                    sum_dot_rbase += (DBx[p]*dBrx + DBy[p]*dBry + DBz[p]*dBrz)

            # 中心場の導関数を差し引く
            g_alpha[r, k] += 2.0*invM * (
                sum_dot_alpha - (sumDBx*dB0_dalpha_x[r, k] + sumDBy*dB0_dalpha_y[r, k] + sumDBz*dB0_dalpha_z[r, k])
            )
            g_rbase[k]    += 2.0*invM * (sum_dot_rbase)

        # dB0/drb の寄与（全 r で共通）を差し引く
        g_rbase[k] += - 2.0*invM * (sumDBx*dB0_drb_x[k] + sumDBy*dB0_drb_y[k] + sumDBz*dB0_drb_z[k])

    return g_alpha, g_rbase


def objective_with_grads_fixed(alphas: np.ndarray, r_bases: np.ndarray,
                               geom: dict, pts: np.ndarray):
    """
    目的関数 J と、その y-space 勾配（g_alpha, g_rbase）、中心 |B0| を返す。
    - geom: 角テーブル、z 層位置、リングオフセットなど一式
    - pts : ROI の三次元サンプル点（shape: (M,3)）
    """
    Bx, By, Bz, B0x, B0y, B0z = compute_B_and_B0(
        alphas, r_bases,
        geom["theta"], geom["sin2"], geom["cth"], geom["sth"],
        geom["z_layers"], geom["ring_offsets"], pts, FACTOR, phi0, m0
    )
    DBx = Bx - B0x; DBy = By - B0y; DBz = Bz - B0z
    J = np.mean(DBx*DBx + DBy*DBy + DBz*DBz)
    sumDBx = np.sum(DBx); sumDBy = np.sum(DBy); sumDBz = np.sum(DBz)

    g_alpha, g_rbase = grad_alpha_and_radius_fixed(
        alphas, r_bases,
        geom["theta"], geom["sin2"], geom["cth"], geom["sth"],
        geom["z_layers"], geom["ring_offsets"],
        pts, FACTOR, phi0, m0,
        DBx, DBy, DBz, sumDBx, sumDBy, sumDBz
    )

    B0n = float(np.sqrt(B0x*B0x + B0y*B0y + B0z*B0z))
    return J, g_alpha, g_rbase, B0n


# ------------- GN robust in y-space; gradient in x-space -------------
def hvp_y(alphas: np.ndarray, r_bases: np.ndarray, geom: dict, pts: np.ndarray,
          v_y: np.ndarray, eps_hvp: float):
    """
    y-space の HVP: H_y(J) * v_y を中心差分で近似。
    - v_y は [vA(:); vR] 連結ベクトル（alphas と r_bases の両方）
    - 数値安定化のため、ステップ幅 h = eps_hvp / ||v||_∞ とする
    """
    P_A = geom["R"] * geom["K"]
    vA = v_y[:P_A].reshape(geom["R"], geom["K"])
    vR = v_y[P_A:]

    vmax = float(np.max(np.abs(v_y)) + 1e-16)  # 機械ゼロ除け
    h = eps_hvp / vmax

    a_p = alphas + h * vA; r_p = r_bases + h * vR
    a_m = alphas - h * vA; r_m = r_bases - h * vR

    _, gA_p, gR_p, _ = objective_with_grads_fixed(a_p, r_p, geom, pts)
    _, gA_m, gR_m, _ = objective_with_grads_fixed(a_m, r_m, geom, pts)

    hvA = (gA_p - gA_m) / (2.0*h)
    hvR = (gR_p - gR_m) / (2.0*h)
    return hvA.ravel(), hvR


def fun_grad_gradnorm_fixed(x: np.ndarray, geom: dict, pts: np.ndarray,
                            sigma_alpha: float, sigma_r: float,
                            rho_gn: float, eps_hvp: float,
                            r0: float, lower_var: np.ndarray, upper_var: np.ndarray):
    """
    ロバスト目的 J_gn(x) と、その x-space 勾配を返す。
    - J_gn(x) = J(x) + 0.5*rho_gn*||Sigma_y^{1/2} grad_y J||^2  （y-space 定義）
    - 勾配は、y-space で HVP を計算後、(dy/dx)^T で x-space に写像。
      * alphas: 単位写像
      * r_vars: 下半分と上半分の対称ペア（和）で集約
    """
    # x → (alphas, r_vars) → y-space の r_bases
    alphas, r_vars = unpack_x(x, geom["R"], geom["K"])
    r_bases = build_r_bases_from_vars(r_vars, geom["K"], r0, lower_var, upper_var)

    # y-space の目的・勾配
    J, gA_y, gRb_y, B0n = objective_with_grads_fixed(alphas, r_bases, geom, pts)

    # ||Σ^{1/2} ∇_y J||^2 の構成
    g_y = np.concatenate([gA_y.ravel(), gRb_y])      # ∇_y J 連結
    P_A = geom["R"] * geom["K"]
    gn2 = (sigma_alpha**2)*np.dot(g_y[:P_A], g_y[:P_A]) + (sigma_r**2)*np.dot(g_y[P_A:], g_y[P_A:])
    v_y = np.concatenate([(sigma_alpha**2)*g_y[:P_A], (sigma_r**2)*g_y[P_A:]])  # Σ * ∇_y J

    # y-space で HVP を近似
    hvA, hvR = hvp_y(alphas, r_bases, geom, pts, v_y, eps_hvp)

    # ロバスト目的値
    Jgn = J + 0.5 * rho_gn * gn2

    # 勾配を x-space へ写像
    gA_x = gA_y.ravel() + rho_gn * hvA
    gR_x = np.zeros_like(r_vars)
    for j, k_low in enumerate(lower_var):
        k_up = upper_var[j]
        gR_x[j] = (gRb_y[k_low] + gRb_y[k_up]) + rho_gn * (hvR[k_low] + hvR[k_up])

    grad_x = np.concatenate([gA_x, gR_x])
    return float(Jgn), grad_x, float(B0n), float(J), float(gn2)


# ---------------- Optimization, MC & Saving ----------------
def run_robust_from_nominal(args: argparse.Namespace):
    """
    名目 .npz を読み込み → y-space 勾配ノルム正則化でロバスト最適化 → 保存 → MC 比較。
    - 返却値は不要（ファイル保存が主な成果物）。途中経過は標準出力へ。
    """
    # 1) 名目データの読取と幾何セットアップ
    nom = load_nominal_npz(args.in_npz)
    geom = dict(
        N=nom["N"], R=nom["R"], K=nom["K"], r0=float(np.median(nom["r_bases"])),
        theta=nom["theta"], sin2=nom["sin2"], cth=nom["cth"], sth=nom["sth"],
        z_layers=nom["z_layers"], ring_offsets=nom["ring_offsets"],
        dz=nom["dz"], Lz=nom["Lz"]
    )
    lower_var, upper_var, _ = build_symmetry_indices(geom["K"])

    # 2) ROI の離散点（目的関数評価用）
    pts = build_roi_points(args.roi_r, args.roi_step)

    # 3) 初期点の構成（x-space）：
    #    - alphas は名目のまま
    #    - r_vars は「下半分」層の r_bases を切り出す
    al0 = nom["alphas"].copy()
    rv0 = np.array([nom["r_bases"][k] for k in lower_var], dtype=np.float64)
    x0 = pack_x(al0, rv0)

    # 4) Box 制約（半径の下限: 初期値 - min_radius_drop_mm）
    delta_m = args.min_radius_drop_mm * 1e-3
    P = geom["R"] * geom["K"]
    bounds = [(None, None)] * P + [(rv0[j] - delta_m, None) for j in range(rv0.size)]

    # 5) ロバスト正則化の公差・重み（y-space）
    sigma_alpha = np.deg2rad(args.sigma_alpha_deg)  # [rad]
    sigma_r     = args.sigma_r_mm * 1e-3           # [m]

    # 6) 最適化履歴の記録
    J_hist, Jn_hist, B0_hist, gn2_hist = [], [], [], []

    def cb(xk: np.ndarray):
        """
        L-BFGS-B の反復ごとに呼ばれるコールバック。
        ロバスト目的・名目目的・中心場・勾配ノルムの履歴を記録。
        """
        Jk, gk, B0k, Jn, gn2 = fun_grad_gradnorm_fixed(
            xk, geom, pts, sigma_alpha, sigma_r, args.rho_gn, args.eps_hvp,
            geom["r0"], lower_var, upper_var)
        J_hist.append(Jk); Jn_hist.append(Jn); B0_hist.append(B0k); gn2_hist.append(gn2)
        if len(J_hist) % 10 == 0:
            print(f"[iter {len(J_hist):4d}] Jgn={Jk:.6e}  J={Jn:.6e}  ||Σ^1/2∇J||^2={gn2:.3e}  |B0|={B0k*1e3:.3f} mT")

    # 7) L-BFGS-B でロバスト最適化
    res = minimize(
        lambda x: fun_grad_gradnorm_fixed(x, geom, pts, sigma_alpha, sigma_r,
                                          args.rho_gn, args.eps_hvp,
                                          geom["r0"], lower_var, upper_var)[0],
        x0,
        jac=lambda x: fun_grad_gradnorm_fixed(x, geom, pts, sigma_alpha, sigma_r,
                                              args.rho_gn, args.eps_hvp,
                                              geom["r0"], lower_var, upper_var)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=args.maxiter, ftol=args.ftol, gtol=args.gtol, disp=True),
        callback=cb
    )

    # 8) 出力整形
    x_opt = res.x
    al_opt, rv_opt = unpack_x(x_opt, geom["R"], geom["K"])
    rb_opt = build_r_bases_from_vars(rv_opt, geom["K"], geom["r0"], lower_var, upper_var)

    Jgn_f, _, B0_f, Jn_f, gn2_f = fun_grad_gradnorm_fixed(
        x_opt, geom, pts, sigma_alpha, sigma_r, args.rho_gn, args.eps_hvp,
        geom["r0"], lower_var, upper_var)
    print(f"[done] success={res.success}, iters={res.nit}, Jgn={Jgn_f:.6e}, "
          f"J={Jn_f:.6e}, ||Σ^1/2∇J||^2={gn2_f:.3e}, |B0|={B0_f*1e3:.3f} mT")

    # 9) 保存（robust 専用 & viewer 互換）
    os.makedirs(args.out_dir, exist_ok=True)
    robust_npz  = os.path.join(args.out_dir, "diam_opt_saved_results_robust.npz")
    robust_meta = os.path.join(args.out_dir, "diam_opt_saved_meta_robust.json")

    np.savez_compressed(
        robust_npz,
        alphas_opt=al_opt, r_bases_opt=rb_opt,
        theta=geom["theta"], sin2=geom["sin2"],
        z_layers=geom["z_layers"], ring_offsets=geom["ring_offsets"],
        J_hist=np.array(J_hist), Jn_hist=np.array(Jn_hist),
        B0_hist=np.array(B0_hist), gn2_hist=np.array(gn2_hist)
    )

    meta = dict(
        N=int(geom["N"]), K=int(geom["K"]), R=int(geom["R"]), r0=float(geom["r0"]),
        ring_offsets=geom["ring_offsets"].tolist(), z_layers=geom["z_layers"].tolist(),
        Lz=float(geom["Lz"]), dz=float(geom["dz"]), phi0=float(phi0), m0=float(m0),
        roi_r=float(args.roi_r), roi_step=float(args.roi_step),
        robust=dict(type="grad-norm(y-space)", rho_gn=float(args.rho_gn), eps_hvp=float(args.eps_hvp),
                    sigma_alpha_deg=float(args.sigma_alpha_deg), sigma_r_mm=float(args.sigma_r_mm)),
        optimizer=dict(method="L-BFGS-B", maxiter=int(args.maxiter), ftol=float(args.ftol),
                       gtol=float(args.gtol), min_radius_drop_mm=float(args.min_radius_drop_mm)),
        scipy_result=dict(success=bool(res.success), status=int(res.status),
                          message=str(res.message), nit=int(res.nit),
                          nfev=int(getattr(res, "nfev", -1)), njev=-1)
    )
    with open(robust_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # viewer 互換ファイル（.npz/.json 名を上書きして 3D/2D ビューア群がそのまま読めるように）
    std_npz  = os.path.join(args.out_dir, "diam_opt_saved_results.npz")
    std_meta = os.path.join(args.out_dir, "diam_opt_saved_meta.json")
    np.savez_compressed(
        std_npz,
        alphas_opt=al_opt, r_bases_opt=rb_opt,
        theta=geom["theta"], sin2=geom["sin2"],
        z_layers=geom["z_layers"], ring_offsets=geom["ring_offsets"],
        J_hist=np.array(J_hist), Jn_hist=np.array(Jn_hist),
        B0_hist=np.array(B0_hist), gn2_hist=np.array(gn2_hist)
    )
    with open(std_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 10) 履歴図（最適化の進行状況）
    iters = np.arange(1, len(J_hist) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    ax[0].plot(iters, J_hist, lw=1.2); ax[0].set_xlabel("iter"); ax[0].set_ylabel("J_gn [T^2]"); ax[0].set_title("Robust Obj (GN)"); ax[0].grid(True, alpha=0.3)
    ax[1].plot(iters, Jn_hist, lw=1.2); ax[1].set_xlabel("iter"); ax[1].set_ylabel("J [T^2]");   ax[1].set_title("Nominal J");     ax[1].grid(True, alpha=0.3)
    ax[2].plot(iters, np.array(B0_hist)*1e3, lw=1.2); ax[2].set_xlabel("iter"); ax[2].set_ylabel("|B0| [mT]");   ax[2].set_title("Center field");  ax[2].grid(True, alpha=0.3)
    hist_png = os.path.join(args.out_dir, "histories_gradnorm.png")
    fig.savefig(hist_png, dpi=180); plt.close(fig)

    # 11) MATLAB 互換 .mat（3D 可視化ツールがそのまま読めるように）
    mat_path = os.path.join(args.out_dir, "diam_opt_saved_results.mat")
    savemat(mat_path, {
        "alphas_opt": al_opt,
        "r_bases_opt": rb_opt.reshape(-1, 1),
        "theta": geom["theta"].reshape(-1, 1),
        "sin2th": geom["sin2"].reshape(-1, 1),
        "z_layers": geom["z_layers"].reshape(-1, 1),
        "ring_offsets": geom["ring_offsets"].reshape(-1, 1),
        "J_hist": np.array(J_hist).reshape(-1, 1),
        "Jn_hist": np.array(Jn_hist).reshape(-1, 1),
        "B0_hist": np.array(B0_hist).reshape(-1, 1),
        "gn2_hist": np.array(gn2_hist).reshape(-1, 1),
    }, do_compression=True)

    print("Saved robust results:")
    print("  ", robust_npz);  print("  ", robust_meta)
    print("  ", std_npz);     print("  ", std_meta)
    print("  ", hist_png);    print("  ", mat_path)

    # 12) Monte Carlo（名目 vs ロバスト） … 同一サンプルの共通乱数で公平比較
    print("Running Monte Carlo evaluation ...")
    pts_mc = pts  # 同一 ROI サンプルをそのまま流用

    def mc_eval(alphas_base: np.ndarray, r_bases_base: np.ndarray, S: int, seed: int,
                sigma_alpha: float, sigma_r: float) -> np.ndarray:
        """
        y-space へ独立ガウス摂動（角度=rad, 半径=m）を加えて J をサンプリング。
        """
        rng = np.random.default_rng(seed)
        out = np.zeros(S, dtype=np.float64)
        for s in range(S):
            dA = rng.standard_normal(size=(geom["R"], geom["K"])) * sigma_alpha
            dR = rng.standard_normal(size=(geom["K"],)) * sigma_r
            out[s] = objective_only(
                alphas_base + dA, r_bases_base + dR,
                geom["theta"], geom["sin2"], geom["cth"], geom["sth"],
                geom["z_layers"], geom["ring_offsets"],
                pts_mc, FACTOR, phi0, m0
            )
        return out

    sigma_alpha = np.deg2rad(args.sigma_alpha_deg)
    sigma_r = args.sigma_r_mm * 1e-3

    # 名目とロバストの MC サンプリング（シードをずらして同一乱数を両者に適用）
    J_nom_mc = mc_eval(nom["alphas"], nom["r_bases"], args.mc_samples, args.seed,   sigma_alpha, sigma_r)
    J_rob_mc = mc_eval(al_opt,        rb_opt,          args.mc_samples, args.seed+1, sigma_alpha, sigma_r)

    def summarize(arr: np.ndarray, name: str) -> dict:
        q = np.quantile(arr, [0.50, 0.90, 0.95, 0.99])
        return dict(name=name, mean=float(arr.mean()), std=float(arr.std()),
                    median=float(q[0]), p90=float(q[1]), p95=float(q[2]), p99=float(q[3]))

    sum_nom = summarize(J_nom_mc, "nominal")
    sum_rob = summarize(J_rob_mc, "robust-GN")
    improv = dict(
        mean_ratio=float(sum_rob['mean']/sum_nom['mean']),
        p90_ratio=float(sum_rob['p90']/sum_nom['p90']),
        p95_ratio=float(sum_rob['p95']/sum_nom['p95']),
        p99_ratio=float(sum_rob['p99']/sum_nom['p99']),
    )
    summary = dict(
        settings=dict(in_npz=args.in_npz, roi_r=float(args.roi_r), roi_step=float(args.roi_step),
                      sigma_alpha_deg=float(args.sigma_alpha_deg), sigma_r_mm=float(args.sigma_r_mm),
                      rho_gn=float(args.rho_gn), eps_hvp=float(args.eps_hvp),
                      maxiter=int(args.maxiter), ftol=float(args.ftol), gtol=float(args.gtol),
                      min_radius_drop_mm=float(args.min_radius_drop_mm),
                      mc_samples=int(args.mc_samples), seed=int(args.seed)),
        nominal=sum_nom, robust=sum_rob, improvement=improv
    )

    # サマリ JSON とヒストグラム図を保存
    sum_json = os.path.join(args.out_dir, "robust_summary_gradnorm.json")
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    mc_png = os.path.join(args.out_dir, "robust_mc_compare_gradnorm.png")
    plt.figure(figsize=(10, 5))
    bins = 40
    plt.hist(J_nom_mc, bins=bins, alpha=0.55, label="Nominal", density=True)
    plt.hist(J_rob_mc, bins=bins, alpha=0.55, label="Robust (GN)", density=True)
    plt.xlabel("Objective J under perturbations [T^2]"); plt.ylabel("Density")
    plt.title(f"Monte Carlo robustness (σ_α={args.sigma_alpha_deg:.2f}°, σ_r={args.sigma_r_mm:.2f} mm)")
    plt.legend(); plt.yscale("linear"); plt.xscale("log"); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    plt.savefig(mc_png, dpi=180); plt.close()

    print("MC files saved:")
    print("  ", sum_json)
    print("  ", mc_png)


# ---------------- Main ----------------
if __name__ == "__main__":
    args = parse_args()
    run_robust_from_nominal(args)
