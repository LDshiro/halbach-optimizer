# halbach-optimizer

Halbach 磁石アレイの最適化・可視化ツールです。  
L-BFGS-B による最適化と、Streamlit + Plotly による 2D/3D 表示を提供します。

主な機能:
- `generate_run` で初期パラメータ（run）を生成
- `optimize_run` で最適化（角度モデルの切替を含む）
- GUI で run の比較・可視化・最適化実行

> 本リポジトリでは「ロバスト化（GN正則化・Monte Carlo 評価）」の機能は除去されています。  
> 最適化計算の基本機能は維持しつつ、関連パラメータや UI を削除しています。

---

## 1. セットアップ

### 1.1 仮想環境
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-dev.txt -r requirements-gui.txt
```

### 1.2 JAX（角度モデル delta/fourier 用）
`delta-rep-x0` / `fourier-x0`、または legacy の JAX 勾配を使う場合は JAX が必要です。
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

### 2.2 最適化（CLI）
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

### 2.3 GUI 起動
```powershell
python -m streamlit run app\streamlit_app.py
```

GUI から:
- run の選択・比較（2D/3D）
- `generate_run` → `optimize_run` の実行
- 角度モデルや境界条件の設定

---

## 3. 角度モデル

### 3.1 legacy-alpha
リングごとに `alpha[r,k]` を持つ従来モデル。
\[
\phi(\theta) = 2\theta + \phi_0 + \alpha_{r,k}\sin(2\theta)
\]

### 3.2 delta-rep-x0（鏡映反対称）
x=0 面（x→-x）の鏡映対称を満たす δφ 表現です。  
鏡映ペアでは δφ が符号反転し、固定点は 0 固定になります。

### 3.3 fourier-x0（鏡映反対称 Fourier）
鏡映反対称を満たす Fourier 基底で δφ を表現します。
\[
\delta\phi(\theta)=\sum_{h=0}^{H-1} a_h\cos((2h+1)\theta)
 + \sum_{h=0}^{H-1} b_h\sin(2(h+1)\theta)
\]

### 3.4 JAX と勾配
- `delta-rep-x0` / `fourier-x0` は JAX 勾配を使用
- `legacy-alpha` は `analytic` / `jax` を選択可能

---

## 4. 物理モデルと目的関数

### 4.1 幾何
`N` 個の磁石角度:
\[
\theta_i = \frac{2\pi i}{N}
\]

半径と位置:
\[
\rho_{r,k} = r\_bases[k] + ring\_offsets[r]
\]
\[
p_{i,r,k} = (\rho_{r,k}\cos\theta_i,\ \rho_{r,k}\sin\theta_i,\ z\_layers[k])
\]

### 4.2 磁化ベクトル
磁化は xy 平面内:
\[
m(\phi) = m_0(\cos\phi,\ \sin\phi,\ 0)
\]

### 4.3 点双極子磁場
磁石中心から観測点へのベクトルを \(r\) とすると:
\[
B(r) = \frac{\mu_0}{4\pi}\left(\frac{3(m\cdot\hat{r})\hat{r} - m}{\|r\|^3}\right)
\]
実装では `FACTOR = mu0/(4π)` を用います。

### 4.4 目的関数
原点の磁場を \(B_0 = B(0)\) とすると:
\[
J = \frac{1}{M}\sum_{p\in ROI} \|B(p) - B_0\|^2
\]

### 4.5 field_scale
数値安定化のため `field_scale` を導入します:
\[
FACTOR_{eff} = FACTOR \times field\_scale
\]
ログ表示の `|B0|` は `B0 / field_scale` に戻して表示します。

---

## 5. ROI サンプリング

`halbach.geom.build_roi_points(...)` により ROI 点を生成します。

- `volume-grid` : 3D グリッド
- `volume-subsample` : グリッドからランダムサブサンプル
- `surface-fibonacci` : 球面フィボナッチ
- `surface-random` : 球面ランダム

`optimize_run` のデフォルトは `surface-fibonacci` です。

---

## 6. 最適化パラメータの要点

### 6.1 半径 bounds
`--r-bound-mode` により `r_bases` の bounds を設定します。

- `relative`:
  - `r >= r0 - r_lower_delta_mm`
  - `r <= r0 + r_upper_delta_mm`（`--r-no-upper` で上限無効）
- `absolute`:
  - `r_min_mm <= r <= r_max_mm`
- `none`: bounds 無効

### 6.2 中心 z レイヤの半径固定
`--fix-center-radius-layers {0,2,4}`  
中心 z レイヤの **半径のみ** を固定し、角度は常に最適化対象です。

---

## 7. 出力ファイル構造

```
runs/<run_name>/
  results.npz
  meta.json
  trace.json
  opt.log
```

### 7.1 results.npz
主要キー:
- `alphas_opt` (R,K)
- `r_bases_opt` (K,)
- `theta`, `sin2th`, `cth`, `sth`, `z_layers`, `ring_offsets`
- `J_hist`, `Jn_hist`, `B0_hist`

角度モデルに応じて追加:
- `delta_rep_opt` (K, n_rep)
- `fourier_coeffs_opt` (K, 2H)

### 7.2 meta.json
例:
- `angle_model`
- `grad_backend`
- `fourier_H`
- `regularization`（lambda0/lambda_theta/lambda_z）
- `r_bound_mode` / `r_lower_delta_mm` / `r_upper_delta_mm` など
- `fix_center_radius_layers`

---

## 8. 可視化

### 8.1 2D Error Map (ppm)
ppm は以下で計算します:
\[
ppm = \frac{|B|-|B_0|}{|B_0|}\times 10^6
\]

### 8.2 3D 表示
Plotly で以下のモードを利用できます:
- `fast` / `pretty`
- `cubes`（立方体）
- `cubes_arrows`（立方体 + 磁化方向）

GUI では 3D 比較（Initial vs Optimized）も利用可能です。

---

## 9. 主要ファイル

```
app/
  streamlit_app.py          # GUI
halbach/
  constants.py              # mu0, FACTOR, phi0, m0
  physics.py                # 磁場計算
  objective.py              # 目的関数 + 勾配
  geom.py                   # ROI/パラメータ操作
  run_io.py / run_types.py  # run ロード
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

## 10. 開発コマンド
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
