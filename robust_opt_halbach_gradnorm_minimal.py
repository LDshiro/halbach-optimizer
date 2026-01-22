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

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.io import savemat

from halbach.constants import FACTOR, m0, mu0, phi0
from halbach.geom import (
    build_r_bases_from_vars,
    build_roi_points,
    build_symmetry_indices,
    pack_x,
    unpack_x,
)
from halbach.io import load_nominal_npz
from halbach.logging_utils import configure_logging
from halbach.objective import objective_with_grads_fixed
from halbach.physics import compute_B_and_B0, objective_only
from halbach.robust import Float1DArray, fun_grad_gradnorm_fixed
from halbach.solvers.lbfgsb import bounds_from_arrays, solve_lbfgsb
from halbach.solvers.types import LBFGSBOptions
from halbach.types import Geometry


class SummaryStats(TypedDict):
    name: str
    mean: float
    std: float
    median: float
    p90: float
    p95: float
    p99: float


logger = logging.getLogger(__name__)


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
        description="Robust (gradient-norm) optimization from nominal .npz, with MC & saving"
    )
    ap.add_argument(
        "--in_npz",
        type=str,
        default="/mnt/data/diam_opt_saved_results.npz",
        help="path to nominal .npz",
    )
    ap.add_argument("--out_dir", type=str, default="/mnt/data", help="output directory")

    # ROI for objective
    ap.add_argument("--roi_r", type=float, default=0.140, help="ROI sphere radius [m]")
    ap.add_argument("--roi_step", type=float, default=0.020, help="ROI grid step [m]")

    # Robust hyper-parameters (y-space)
    ap.add_argument("--sigma_alpha_deg", type=float, default=0.5, help="1-sigma angle tol [deg]")
    ap.add_argument("--sigma_r_mm", type=float, default=0.20, help="1-sigma radius tol [mm]")
    ap.add_argument("--rho_gn", type=float, default=1e-4, help="gradient-norm weight (y-space)")
    ap.add_argument(
        "--eps_hvp", type=float, default=1e-6, help="HVP step base for y-space central difference"
    )

    # Optimizer
    ap.add_argument("--maxiter", type=int, default=900, help="L-BFGS-B iterations")
    ap.add_argument("--ftol", type=float, default=1e-12, help="L-BFGS-B ftol")
    ap.add_argument("--gtol", type=float, default=1e-12, help="L-BFGS-B gtol")
    ap.add_argument(
        "--min_radius_drop_mm",
        type=float,
        default=20.0,
        help="lower bound: r_vars >= r_init - this [mm]",
    )

    # Monte Carlo
    ap.add_argument("--mc_samples", type=int, default=600, help="Monte Carlo samples")
    ap.add_argument("--seed", type=int, default=20250926, help="random seed")
    ap.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return ap.parse_args()


# ---------------- Optimization, MC & Saving ----------------
def run_robust_from_nominal(args: argparse.Namespace) -> None:
    """
    名目 .npz を読み込み → y-space 勾配ノルム正則化でロバスト最適化 → 保存 → MC 比較。
    - 返却値は不要（ファイル保存が主な成果物）。途中経過は標準出力へ。
    """
    # 1) 名目データの読取と幾何セットアップ
    nom = load_nominal_npz(args.in_npz)
    r0 = float(np.median(nom["r_bases"]))
    geom = Geometry(
        theta=nom["theta"],
        sin2=nom["sin2"],
        cth=nom["cth"],
        sth=nom["sth"],
        z_layers=nom["z_layers"],
        ring_offsets=nom["ring_offsets"],
        N=nom["N"],
        R=nom["R"],
        K=nom["K"],
        dz=nom["dz"],
        Lz=nom["Lz"],
    )
    lower_var, upper_var, _ = build_symmetry_indices(geom.K)

    # 2) ROI の離散点（目的関数評価用）
    pts = build_roi_points(args.roi_r, args.roi_step)

    # 3) 初期点の構成（x-space）：
    #    - alphas は名目のまま
    #    - r_vars は「下半分」層の r_bases を切り出す
    al0 = nom["alphas"].copy()
    rv0 = np.array([nom["r_bases"][k] for k in lower_var], dtype=np.float64)
    x0 = cast(Float1DArray, pack_x(al0, rv0))

    # 4) Box 制約（半径の下限: 初期値 - min_radius_drop_mm）
    delta_m = args.min_radius_drop_mm * 1e-3
    P = geom.R * geom.K
    lb = np.full(x0.size, -np.inf, dtype=np.float64)
    ub = np.full(x0.size, np.inf, dtype=np.float64)
    lb[P:] = rv0 - delta_m
    bounds = bounds_from_arrays(lb, ub)

    # 5) ロバスト正則化の公差・重み（y-space）
    sigma_alpha = np.deg2rad(args.sigma_alpha_deg)  # [rad]
    sigma_r = args.sigma_r_mm * 1e-3  # [m]

    # 6) L-BFGS-B ????????
    def fun_grad_solver(x: Float1DArray) -> tuple[float, Float1DArray, dict[str, Any]]:
        Jgn, gx, B0, Jn, gn2 = fun_grad_gradnorm_fixed(
            x,
            geom,
            pts,
            sigma_alpha,
            sigma_r,
            args.rho_gn,
            args.eps_hvp,
            r0,
            lower_var,
            upper_var,
        )
        extras: dict[str, Any] = {"J": float(Jn), "B0": float(B0), "gn2": float(gn2)}
        return float(Jgn), gx, extras

    def iter_cb(
        k: int,
        xk: Float1DArray,
        fk: float,
        gk: Float1DArray,
        extras: dict[str, Any],
    ) -> None:
        if k % 10 == 0:
            logger.info(
                "iter=%d Jgn=%.6e J=%.6e gn2=%.3e |B0|=%.3f mT gnorm=%.3e",
                k,
                fk,
                float(extras.get("J", np.nan)),
                float(extras.get("gn2", np.nan)),
                float(extras.get("B0", np.nan)) * 1e3,
                float(np.linalg.norm(gk)),
            )

    opt = LBFGSBOptions(
        maxiter=args.maxiter,
        gtol=args.gtol,
        ftol=args.ftol,
        disp=True,
    )
    res = solve_lbfgsb(fun_grad_solver, x0, bounds, opt, iter_callback=iter_cb)

    J_hist = np.array(res.trace.f, dtype=float)
    Jn_hist = np.array(
        [float(d.get("J", np.nan)) for d in res.trace.extras],
        dtype=float,
    )
    B0_hist = np.array(
        [float(d.get("B0", np.nan)) for d in res.trace.extras],
        dtype=float,
    )
    gn2_hist = np.array(
        [float(d.get("gn2", np.nan)) for d in res.trace.extras],
        dtype=float,
    )
    # 8) 出力整形
    x_opt = res.x
    al_opt, rv_opt = unpack_x(x_opt, geom.R, geom.K)
    rb_opt = build_r_bases_from_vars(rv_opt, geom.K, r0, lower_var, upper_var)

    Jgn_f, _, B0_f, Jn_f, gn2_f = fun_grad_gradnorm_fixed(
        x_opt,
        geom,
        pts,
        sigma_alpha,
        sigma_r,
        args.rho_gn,
        args.eps_hvp,
        r0,
        lower_var,
        upper_var,
    )
    logger.info(
        f"[done] success={res.success}, iters={res.nit}, Jgn={Jgn_f:.6e}, "
        f"J={Jn_f:.6e}, ||Σ^1/2∇J||^2={gn2_f:.3e}, |B0|={B0_f*1e3:.3f} mT"
    )

    # 9) 保存（robust 専用 & viewer 互換）
    os.makedirs(args.out_dir, exist_ok=True)
    robust_npz = os.path.join(args.out_dir, "diam_opt_saved_results_robust.npz")
    robust_meta = os.path.join(args.out_dir, "diam_opt_saved_meta_robust.json")

    np.savez_compressed(
        robust_npz,
        alphas_opt=al_opt,
        r_bases_opt=rb_opt,
        theta=geom.theta,
        sin2=geom.sin2,
        z_layers=geom.z_layers,
        ring_offsets=geom.ring_offsets,
        J_hist=np.array(J_hist),
        Jn_hist=np.array(Jn_hist),
        B0_hist=np.array(B0_hist),
        gn2_hist=np.array(gn2_hist),
    )

    meta = dict(
        N=int(geom.N),
        K=int(geom.K),
        R=int(geom.R),
        r0=float(r0),
        ring_offsets=geom.ring_offsets.tolist(),
        z_layers=geom.z_layers.tolist(),
        Lz=float(geom.Lz),
        dz=float(geom.dz),
        phi0=float(phi0),
        m0=float(m0),
        roi_r=float(args.roi_r),
        roi_step=float(args.roi_step),
        robust=dict(
            type="grad-norm(y-space)",
            rho_gn=float(args.rho_gn),
            eps_hvp=float(args.eps_hvp),
            sigma_alpha_deg=float(args.sigma_alpha_deg),
            sigma_r_mm=float(args.sigma_r_mm),
        ),
        optimizer=dict(
            method="L-BFGS-B",
            maxiter=int(args.maxiter),
            ftol=float(args.ftol),
            gtol=float(args.gtol),
            min_radius_drop_mm=float(args.min_radius_drop_mm),
        ),
        scipy_result=dict(
            success=bool(res.success),
            status=0 if res.success else 1,
            message=str(res.message),
            nit=int(res.nit),
            nfev=int(res.nfev),
            njev=-1 if res.njev is None else int(res.njev),
        ),
    )
    with open(robust_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # viewer 互換ファイル（.npz/.json 名を上書きして 3D/2D ビューア群がそのまま読めるように）
    std_npz = os.path.join(args.out_dir, "diam_opt_saved_results.npz")
    std_meta = os.path.join(args.out_dir, "diam_opt_saved_meta.json")
    np.savez_compressed(
        std_npz,
        alphas_opt=al_opt,
        r_bases_opt=rb_opt,
        theta=geom.theta,
        sin2=geom.sin2,
        z_layers=geom.z_layers,
        ring_offsets=geom.ring_offsets,
        J_hist=np.array(J_hist),
        Jn_hist=np.array(Jn_hist),
        B0_hist=np.array(B0_hist),
        gn2_hist=np.array(gn2_hist),
    )
    with open(std_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 10) 履歴図（最適化の進行状況）
    iters = np.arange(1, len(J_hist) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    ax[0].plot(iters, J_hist, lw=1.2)
    ax[0].set_xlabel("iter")
    ax[0].set_ylabel("J_gn [T^2]")
    ax[0].set_title("Robust Obj (GN)")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(iters, Jn_hist, lw=1.2)
    ax[1].set_xlabel("iter")
    ax[1].set_ylabel("J [T^2]")
    ax[1].set_title("Nominal J")
    ax[1].grid(True, alpha=0.3)
    ax[2].plot(iters, np.array(B0_hist) * 1e3, lw=1.2)
    ax[2].set_xlabel("iter")
    ax[2].set_ylabel("|B0| [mT]")
    ax[2].set_title("Center field")
    ax[2].grid(True, alpha=0.3)
    hist_png = os.path.join(args.out_dir, "histories_gradnorm.png")
    fig.savefig(hist_png, dpi=180)
    plt.close(fig)

    # 11) MATLAB 互換 .mat（3D 可視化ツールがそのまま読めるように）
    mat_path = os.path.join(args.out_dir, "diam_opt_saved_results.mat")
    savemat(
        mat_path,
        {
            "alphas_opt": al_opt,
            "r_bases_opt": rb_opt.reshape(-1, 1),
            "theta": geom.theta.reshape(-1, 1),
            "sin2th": geom.sin2.reshape(-1, 1),
            "z_layers": geom.z_layers.reshape(-1, 1),
            "ring_offsets": geom.ring_offsets.reshape(-1, 1),
            "J_hist": np.array(J_hist).reshape(-1, 1),
            "Jn_hist": np.array(Jn_hist).reshape(-1, 1),
            "B0_hist": np.array(B0_hist).reshape(-1, 1),
            "gn2_hist": np.array(gn2_hist).reshape(-1, 1),
        },
        do_compression=True,
    )

    logger.info("Saved robust results:")
    logger.info("  %s", robust_npz)
    logger.info("  %s", robust_meta)
    logger.info("  %s", std_npz)
    logger.info("  %s", std_meta)
    logger.info("  %s", hist_png)
    logger.info("  %s", mat_path)

    # 12) Monte Carlo（名目 vs ロバスト） … 同一サンプルの共通乱数で公平比較
    logger.info("Running Monte Carlo evaluation ...")
    pts_mc = pts  # 同一 ROI サンプルをそのまま流用

    def mc_eval(
        alphas_base: NDArray[np.float64],
        r_bases_base: NDArray[np.float64],
        S: int,
        seed: int,
        sigma_alpha: float,
        sigma_r: float,
    ) -> NDArray[np.float64]:
        """
        y-space へ独立ガウス摂動（角度=rad, 半径=m）を加えて J をサンプリング。
        """
        rng = np.random.default_rng(seed)
        out = np.zeros(S, dtype=np.float64)
        for s in range(S):
            dA = rng.standard_normal(size=(geom.R, geom.K)) * sigma_alpha
            dR = rng.standard_normal(size=(geom.K,)) * sigma_r
            out[s] = objective_only(
                alphas_base + dA,
                r_bases_base + dR,
                geom.theta,
                geom.sin2,
                geom.cth,
                geom.sth,
                geom.z_layers,
                geom.ring_offsets,
                pts_mc,
                FACTOR,
                phi0,
                m0,
            )
        return out

    sigma_alpha = np.deg2rad(args.sigma_alpha_deg)
    sigma_r = args.sigma_r_mm * 1e-3

    # 名目とロバストの MC サンプリング（シードをずらして同一乱数を両者に適用）
    J_nom_mc = mc_eval(
        nom["alphas"], nom["r_bases"], args.mc_samples, args.seed, sigma_alpha, sigma_r
    )
    J_rob_mc = mc_eval(al_opt, rb_opt, args.mc_samples, args.seed + 1, sigma_alpha, sigma_r)

    def summarize(arr: NDArray[np.float64], name: str) -> SummaryStats:
        q = np.quantile(arr, [0.50, 0.90, 0.95, 0.99])
        return dict(
            name=name,
            mean=float(arr.mean()),
            std=float(arr.std()),
            median=float(q[0]),
            p90=float(q[1]),
            p95=float(q[2]),
            p99=float(q[3]),
        )

    sum_nom = summarize(J_nom_mc, "nominal")
    sum_rob = summarize(J_rob_mc, "robust-GN")
    improv = dict(
        mean_ratio=float(sum_rob["mean"] / sum_nom["mean"]),
        p90_ratio=float(sum_rob["p90"] / sum_nom["p90"]),
        p95_ratio=float(sum_rob["p95"] / sum_nom["p95"]),
        p99_ratio=float(sum_rob["p99"] / sum_nom["p99"]),
    )
    summary = dict(
        settings=dict(
            in_npz=args.in_npz,
            roi_r=float(args.roi_r),
            roi_step=float(args.roi_step),
            sigma_alpha_deg=float(args.sigma_alpha_deg),
            sigma_r_mm=float(args.sigma_r_mm),
            rho_gn=float(args.rho_gn),
            eps_hvp=float(args.eps_hvp),
            maxiter=int(args.maxiter),
            ftol=float(args.ftol),
            gtol=float(args.gtol),
            min_radius_drop_mm=float(args.min_radius_drop_mm),
            mc_samples=int(args.mc_samples),
            seed=int(args.seed),
        ),
        nominal=sum_nom,
        robust=sum_rob,
        improvement=improv,
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
    plt.xlabel("Objective J under perturbations [T^2]")
    plt.ylabel("Density")
    plt.title(
        f"Monte Carlo robustness (σ_α={args.sigma_alpha_deg:.2f}°, σ_r={args.sigma_r_mm:.2f} mm)"
    )
    plt.legend()
    plt.yscale("linear")
    plt.xscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(mc_png, dpi=180)
    plt.close()

    logger.info("MC files saved:")
    logger.info("  %s", sum_json)
    logger.info("  %s", mc_png)


# ---- Public API for tests (no runtime behavior change) ----
__all__ = [
    "compute_B_and_B0",
    "objective_only",
    "objective_with_grads_fixed",
    "build_roi_points",
    "mu0",
    "FACTOR",
    "phi0",
    "m0",
]


# ---------------- Main ----------------
if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    run_robust_from_nominal(args)
