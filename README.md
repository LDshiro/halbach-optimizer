# halbach-optimizer

Halbach 配列の**磁場均一化最適化**（L-BFGS-B）と 2D/3D 可視化を行うツール群です。  
主に `generate_run` / `optimize_run` / GUI（Streamlit）を使って、
幾何・角度モデル・自己無撞着モデルを切り替えながら最適化できます。

---

## 1. セットアップ

### 1.1 仮想環境
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-dev.txt -r requirements-gui.txt
```

### 1.2 JAX（JAX必須のモデル）
`delta-rep-x0` / `fourier-x0` と自己無撞着モデル（self-consistent）は JAX が必須です。
```powershell
python -m pip install jax jaxlib
```

---

## 2. 使い方

### 2.1 初期 run の生成
```powershell
python -m halbach.cli.generate_run --out runs/demo `
  --N 48 --R 3 --K 24 --Lz 0.64 --diameter-mm 400 --ring-offset-step-mm 12
```

生成物:
- `runs/demo/results.npz`
- `runs/demo/meta.json`

### 2.2 最適化（L-BFGS-B）
```powershell
python -m halbach.cli.optimize_run --in runs/demo --out runs/demo_opt `
  --maxiter 900 --gtol 1e-12 `
  --roi-r 0.14 --roi-mode surface-fibonacci --roi-samples 300 `
  --field-scale 1e6 `
  --angle-model legacy-alpha --grad-backend analytic `
  --fix-center-radius-layers 2 `
  --r-bound-mode relative --r-lower-delta-mm 30 --r-upper-delta-mm 30
```

出力:
- `runs/demo_opt/results.npz`
- `runs/demo_opt/meta.json`
- `runs/demo_opt/trace.json`
- `runs/demo_opt/opt.log`

### 2.3 GUI
```powershell
python -m streamlit run app\streamlit_app.py
```

GUI でできること:
- run の可視化（2D/3D）
- `generate_run` / `optimize_run` の実行
- 角度モデル / 自己無撞着 / ROI 等のパラメータ設定

---

## 3. 計算モデル

### 3.1 角度モデル（angle_model）

**legacy-alpha**  
リングごとに `alpha[r,k]` を持つ簡易モデル。
\[
\phi(\theta) = 2\theta + \phi_0 + \alpha_{r,k}\sin(2\theta)
\]

**delta-rep-x0**  
x=0 面の対称条件を満たすように、\(\delta\phi\) を表現するモデル。

**fourier-x0**  
\(\delta\phi\) を Fourier 展開で表現するモデル。
\[
\delta\phi(\theta)=\sum_{h=0}^{H-1} a_h\cos((2h+1)\theta)
+ \sum_{h=0}^{H-1} b_h\sin(2(h+1)\theta)
\]

### 3.2 磁化モデル（mag_model）

**fixed**  
磁化モーメントの大きさは固定（`m0`）。

**self-consistent-easy-axis**  
近傍相互作用による自己無撞着（p-solver）で `p_i` を更新。
- H 計算は `H_FACTOR = FACTOR / mu0 = 1/(4π)` を使用
- **field_scale には依存しない**（不変性チェックあり）

### 3.3 近傍カーネル（near_kernel）
自己無撞着の近傍相互作用に使用。

- **dipole**: 点双極子の近傍相互作用
- **multi-dipole**: サブ双極子分割（`subdip_n`）で近似
- **cellavg**: セル平均 demag tensor による近傍相互作用
- **gl-double-mixed**: Gauss?Legendre 二重平均（混合）
  - 低次数（n=2）を全エッジに、
  - **face-to-face** エッジにのみ高次数（n=3）を適用
  - `gl_order` を `2 / 3 / mixed` で切り替え可能

---

## 4. 計算フロー（最適化）

1. **run を読み込み**（幾何・初期角度など）
2. **ROI 点群を生成**（volume-grid / surface-fibonacci 等）
3. **目的関数 J と勾配を評価**
   - 固定モデルなら `m0` 固定
   - self-consistent の場合は **p-solver** を解く
4. **L-BFGS-B 反復**（制約付き最適化）
5. **結果保存**（`results.npz` / `trace.json`）

self-consistent の内部:
- NearGraph で近傍を構築
- p-solver（固定点反復 / 解析的反復）で `p` を更新
- **p は field_scale に依存しない**

---

## 5. 物理モデルと目的関数

### 5.1 双極子磁場
\[
B(r) = \frac{\mu_0}{4\pi}\left(\frac{3(m\cdot\hat{r})\hat{r} - m}{\|r\|^3}\right)
\]
定数は `FACTOR = mu0/(4π)` として扱います。

### 5.2 目的関数
ROI 上の磁場差分の平均二乗。
\[
J = \frac{1}{M}\sum_{p\in ROI} \|B(p) - B_0\|^2
\]

### 5.3 field_scale
評価用のスケーリング。
\[
FACTOR_{eff} = FACTOR \times field\_scale
\]
自己無撞着側（p-solver）には **field_scale を混ぜない**。

---

## 6. ROI サンプリング

`halbach.geom.build_roi_points(...)` により ROI 点を作ります。

- `volume-grid`
- `volume-subsample`
- `surface-fibonacci`
- `surface-random`

`optimize_run` の既定は `surface-fibonacci` です。

---

## 7. 出力ファイル

```
runs/<run_name>/
  results.npz
  meta.json
  trace.json
  opt.log
```

### 7.1 results.npz（主キー）
- `alphas_opt` (R,K)
- `r_bases_opt` (K,)
- `theta`, `sin2th`, `cth`, `sth`, `z_layers`, `ring_offsets`
- `J_hist`, `Jn_hist`, `B0_hist`

角度モデルに応じて追加:
- `delta_rep_opt` (K, n_rep)
- `fourier_coeffs_opt` (K, 2H)

自己無撞着が有効な場合:
- `sc_p_flat` / `sc_cfg_fingerprint` が保存されることがあります

---

## 8. 可視化

### 8.1 2D Error Map (ppm)
\[
ppm = \frac{|B|-|B_0|}{|B_0|}\times 10^6
\]

### 8.2 3D 可視化
Plotly による 3D 表示。
- `fast` / `pretty`
- `cubes` / `cubes_arrows`

---

## 9. 主要ファイル

```
app/
  streamlit_app.py          # GUI
halbach/
  constants.py              # mu0, FACTOR, phi0, m0
  physics.py                # 磁場計算
  objective.py              # 目的関数 + 勾配
  geom.py                   # ROI/幾何ヘルパ
  run_io.py / run_types.py  # run IO
  viz2d.py                  # 2D ppm
  viz3d.py                  # 3D 可視化
  cli/
    generate_run.py
    optimize_run.py
  gui/
    opt_job.py              # GUI → CLI 呼び出し
  solvers/
    lbfgsb.py               # L-BFGS-B ラッパ
```

---

## 10. コマンド
```powershell
ruff check .
mypy .
python -m pytest -q
```

pre-commit:
```powershell
pre-commit install
pre-commit run -a
```
