# halbach-optimizer

Halbach 配列（永久磁石を磁気双極子で近似）を対象に、**円筒状にスタックしたリング型 Halbach 配列**の「磁場の一様性」を改善するための最適化・ロバスト化・評価（Monte Carlo）を行うコード群です。

このリポジトリは **研究用/検証用** の実装であり、磁石は「有限体積」ではなく **点双極子** で近似しています（端面・形状効果や減磁などは含みません）。

---

## 目次

- [1. 何ができるか](#1-何ができるか)
- [2. 計算モデル（物理・幾何）](#2-計算モデル物理幾何)
- [3. 最適化問題の定式化](#3-最適化問題の定式化)
- [4. ロバスト化（勾配ノルム正則化）](#4-ロバスト化勾配ノルム正則化)
- [5. データ形式（results.npz / meta.json / trace.json / .mat）](#5-データ形式resultsnpz--metajson--tracejson--mat)
- [6. コード構成（どこに何が実装されているか）](#6-コード構成どこに何が実装されているか)
- [7. 実行方法（最小例）](#7-実行方法最小例)
- [8. 2D 可視化（ppm 誤差マップ）](#8-2d-可視化ppm-誤差マップ)
- [9. ベンチマークと性能上の注意](#9-ベンチマークと性能上の注意)
- [10. 開発（black/ruff/mypy/pytest, pre-commit）](#10-開発blackruffmypypytest-pre-commit)
- [11. 既知の制約・未実装/今後の予定](#11-既知の制約未実装今後の予定)

---

## 1. 何ができるか

現状（本 zip に含まれるコード）でできる主なことは以下です。

### 最適化 / ロバスト化（スクリプト）
- 名目（nominal）設計 `.npz` を読み込み
- **勾配ノルム正則化（y-space）**を付与した目的関数で **L-BFGS-B** 最適化
- 半径変数に **箱型制約（下限）**を付与（例：`r >= r_init - 20 mm`）
- 最適化履歴（目的関数・中心磁場など）を保存し、簡単な履歴プロット（png）も出力
- 最適化後の設計について **Monte Carlo 評価（製造誤差を模した摂動）**を行い、名目 vs ロバストの分布を比較して保存

→ エンドツーエンド実行：`robust_opt_halbach_gradnorm_minimal.py`

### 入出力（run ローダ / 正規化）
- `results.npz` / `meta.json` / `trace.json` の「run ディレクトリ」を読み込んで、内部の `RunBundle` にまとめる
- 既存の `.npz`（キーが揺れている場合も）を **標準化**して `results.npz` 形式に揃える

→ `halbach/run_io.py`, `halbach/tools/normalize_run.py`

### 2D 可視化用のデータ生成
- ROI 球内で **指定した平面（xy/xz/yz）**の点群における磁場を計算し、
  - `|B|` の ppm 誤差マップ（`( |B(p)| - |B0| ) / |B0| * 1e6`）を生成
  - `y=0` の断面ライン（x方向）を抽出

→ `halbach/viz2d.py`

> **注意（GUIについて）**  
> この zip の内容には **Streamlit 等の GUI 実装は含まれていません**（`app/` ディレクトリ等が存在しません）。  
> ただし、`run_io`/`viz2d`/`run_types` は将来の GUI から利用しやすいように「run ディレクトリ」形式での入出力を整備しています。

---

## 2. 計算モデル（物理・幾何）

### 2.1 座標・単位
- 座標系：右手系 `(x, y, z)`
- リング：`xy` 平面上に配置し、`z` 方向に層をスタック
- 内部計算の長さ単位：**m**
- 角度：**rad**
- 磁場：**T**（テスラ）

### 2.2 ハルバッハリングの幾何（N, R, K）
パラメータ：
- `N` : 1リングあたりの磁石個数（等角配置）
- `R` : 半径方向のリング数（同じ z 位置に同心円状の複数リング）
- `K` : z 方向の層数（リングを z 方向に積む）

リング位置：
- 磁石の方位角：`theta_i = 2π i / N`（`i=0..N-1`）
- z 層位置：`z_layers[k]`（`k=0..K-1`）
- 半径方向リングのオフセット：`ring_offsets[r]`（`r=0..R-1`）
- 各層の基準半径：`r_bases[k]`
- したがって、磁石（点双極子）の配置座標は

\[
\rho_{r,k} = r\_bases[k] + ring\_offsets[r]
\]

\[
p_{i,r,k} = \left(\rho_{r,k}\cos\theta_i,\ \rho_{r,k}\sin\theta_i,\ z\_layers[k]\right)
\]

### 2.3 磁石モデル：点磁気双極子
各磁石は磁気双極子（磁気モーメント）で近似します。

- `m0` : 双極子モーメントの大きさ（現状は `1.0` の無次元値）
- モーメントは **xy 平面内**にのみ向く（`m_z = 0`）

\[
m(\phi) = m_0(\cos\phi,\ \sin\phi,\ 0)
\]

磁束密度（真空中・点双極子）：

\[
B(r) = \frac{\mu_0}{4\pi}\left(\frac{3(m\cdot\hat{r})\hat{r} - m}{\|r\|^3}\right)
\]

コード上は
- `mu0 = 4π×10^{-7}`
- `FACTOR = mu0/(4π)`
として `factor * (...) / r^3` を計算します。

### 2.4 ハルバッハ配列の向き（phi のパラメタ化）
この実装では、各磁石の向き `phi` を **完全自由（N 個）**にはせず、以下の低次元パラメタで表します：

\[
\phi(\theta) = 2\theta + \phi_0 + \alpha_{r,k}\sin(2\theta)
\]

- `phi0 = -π/2`：理想ハルバッハの位相（中心磁場の向きを回転させるための定数）
- `alphas[r,k]`：リング `r`、層 `k` の「角度補正」パラメータ
- `sin2(theta)` を掛けることで、理想 `2θ` からの形状を少し歪める自由度を与えています

> **重要**  
> 本リポジトリの「角度最適化」は **磁石ごとに独立角度を最適化する方式ではありません**。  
> `alphas[r,k]` という少数パラメタでリング全体の角度分布を制御します（変数数を抑える設計）。

---

## 3. 最適化問題の定式化

### 3.1 ROI（関心領域）
目的関数は、球状 ROI 内のサンプル点 `pts` における磁場の一様性で評価します。

`halbach/geom.py: build_roi_points(roi_r, roi_step)` は以下を行います：
- `[-roi_r, +roi_r]` の立方格子を作る
- `x^2 + y^2 + z^2 <= roi_r^2` を満たす点だけ残す

出力：
- `pts`: shape `(M, 3)`（`M` はサンプル点数）

### 3.2 ノミナル目的関数（ベクトル誤差）
磁場の向き情報を保持するため、ノミナル目的関数は **磁束密度ベクトル**の差で定義します。

- 参照点：原点（中心）`p=0` の磁場 `B0`
- 評価点：ROI 内の点 `p` の磁場 `B(p)`

\[
J = \frac{1}{M}\sum_{p \in ROI} \| B(p) - B_0 \|^2
\]

- `B0` は **毎回の計算で更新**されます（設計変数に依存）
- 実装：
  - `halbach/physics.py: compute_B_and_B0()` が ROI 全点の `B` と、原点の `B0` を同時に計算
  - `halbach/physics.py: objective_only()` が `J` を計算（Numba）

### 3.3 設計変数（x-space）と物理パラメタ（y-space）
このコードでは、半径変数に z 対称性を入れて変数削減します。

- **y-space（物理パラメタ）**  
  \[
  y = [\alpha_{r,k}\ \text{(全要素)};\ r\_bases[k]\ \text{(全層)}]
  \]
  - `alphas`: shape `(R,K)`
  - `r_bases`: shape `(K,)`

- **x-space（最適化変数）**  
  \[
  x = [\alpha_{r,k}\ \text{(全要素)};\ r\_vars[j]\ \text{(下半分のみ)}]
  \]
  - `alphas`: shape `(R,K)`（ここは全層を持つ）
  - `r_vars`: shape `(K/2-2,)`（中央 4 層固定の前提）

z 対称のマッピング（`halbach/geom.py: build_symmetry_indices`）：
- 中央 4 層（`K//2-2 .. K//2+1`）は **固定**（`r0`）
- 下側の変数 `r_vars[j]` を上側にもコピー

\[
r\_bases[k] = \begin{cases}
r\_vars[j] & (k=k\_{low}(j)\ \text{or}\ k\_{up}(j)) \\
r_0 & (k \in \text{fixed center})
\end{cases}
\]

> **注意**  
> 「中央固定」は **半径のみ**であり、角度補正 `alphas` は現行実装では固定していません（全層を持ちます）。

---

## 4. ロバスト化（勾配ノルム正則化）

### 4.1 ねらい（感度の抑制）
製造誤差を `y-space` に独立ガウス摂動として与えるとします：

- 角度摂動：`δα ~ N(0, σ_α^2)`
- 半径摂動：`δr ~ N(0, σ_r^2)`

このとき、線形近似での目的関数の変化は `∇_y J` に比例するため、
\[
\mathrm{Var}[\delta J] \approx \nabla_y J^\top \Sigma_y \nabla_y J
\]
となります。よって `|| Σ^{1/2} ∇J ||^2` を小さくすると、感度（ばらつき）を抑えられる期待があります。

### 4.2 ロバスト目的関数
本実装のロバスト目的関数は

\[
J_{gn}(x) = J(x) + \frac{1}{2}\rho_{gn} \left\| \Sigma_y^{1/2}\nabla_y J \right\|^2
\]

- `Σ_y` は対角：`σ_α^2`（alphas 全要素）と `σ_r^2`（r_bases 全要素）
- 実装：
  - `halbach/robust.py: fun_grad_gradnorm_fixed()`

### 4.3 ロバスト勾配（HVP）
ロバスト項の勾配は

\[
\nabla_x J_{gn} = \nabla_x J + \rho_{gn}\ (dy/dx)^\top \left[ H_y(J)\ (\Sigma_y \nabla_y J) \right]
\]

ここで `H_y(J) v`（Hessian-vector product）は解析 Hessian を作らず、
**y-space で中心差分**により近似します：

\[
H_y(J)v \approx \frac{\nabla_y J(y+h v) - \nabla_y J(y-h v)}{2h}
\]

- 実装：
  - `halbach/robust.py: hvp_y()`（中心差分）
  - `halbach/objective.py: objective_with_grads_fixed()`（解析 `∇_y J`）

---

## 5. データ形式（results.npz / meta.json / trace.json / .mat）

### 5.1 `results.npz`（必須）
最低限、以下のキーを含むことを想定しています（`run_io` は互換的に読みます）。

| key | shape | 単位 | 説明 |
|---|---:|---|---|
| `alphas_opt` または `alphas` | (R,K) | rad | 角度補正パラメタ |
| `r_bases_opt` または `r_bases` | (K,) | m | 層ごとの基準半径 |
| `theta` | (N,) | rad | 磁石配置角（等角） |
| `sin2th` または `sin2` | (N,) | - | `sin(2θ)`（なければ内部生成） |
| `cth` / `sth` | (N,) | - | `cosθ`, `sinθ`（なければ内部生成） |
| `z_layers` | (K,) | m | 層の z 位置 |
| `ring_offsets` | (R,) | m | 同心リングの半径オフセット |

また、スクリプト `robust_opt_halbach_gradnorm_minimal.py` は履歴も同梱します：
- `J_hist`, `Jn_hist`, `B0_hist`, `gn2_hist` など（任意）

### 5.2 `meta.json`（任意）
`run_io` は `meta.json` があれば読み込みます（無くても動きます）。  
ロバスト最適化スクリプトは、以下のような情報を保存します：
- `N,K,R,r0,Lz,dz`
- `roi_r, roi_step`
- robust 設定（`rho_gn`, `sigma_*`, `eps_hvp`）
- optimizer 設定（`maxiter, ftol, gtol, min_radius_drop_mm`）
- SciPy 結果のサマリ

### 5.3 `trace.json`（任意）
`solve_lbfgsb()` が返す trace を保存しておくための形式です（GUI 等での可視化用）。
- `iters`: iteration index
- `f`: objective value
- `gnorm`: gradient norm
- `extras`: 任意の追加情報（`J`, `B0`, `gn2` など）

### 5.4 `.mat`（MATLAB 互換）
`robust_opt_halbach_gradnorm_minimal.py` は `diam_opt_saved_results.mat` を出力します。  
MATLAB 側のビューア/可視化コードで読みやすいように、列ベクトルに整形して保存します。

---

## 6. コード構成（どこに何が実装されているか）

```
halbach-optimizer-main/
  robust_opt_halbach_gradnorm_minimal.py   # エンドツーエンド（最適化+MC+保存）
  halbach/
    constants.py        # mu0, FACTOR, phi0, m0
    geom.py             # ROI生成, z対称インデックス, x<->(alphas,r_vars)
    physics.py          # Numba: compute_B_and_B0(), objective_only()
    objective.py        # Numba: 解析勾配 grad_alpha_and_radius_fixed()
    robust.py           # y-space GN正則化, HVP, robust目的+勾配
    solvers/
      lbfgsb.py         # SciPy L-BFGS-B ラッパ（cache/trace/callback）
      types.py          # solver型（options/result/trace）
    run_io.py           # results/meta/trace 読み込み → RunBundle
    run_types.py        # RunBundle / RunResults dataclasses
    viz2d.py            # 2D ppm誤差マップ（平面）と断面抽出
    tools/
      normalize_run.py  # run正規化ツール（results.npz/meta/trace）
  bench/
    bench_kernels.py    # objective/grad/HVP の簡易ベンチ
  tests/
    test_smoke_objective.py
    test_gradcheck_yspace.py
    test_run_io.py
    test_viz2d.py
```

### 6.1 「磁場計算」の本体
- `halbach/physics.py: compute_B_and_B0()`
  - ROI 全点の B と原点 B0 を同時に計算（Numba）
  - 計算量：`O(K*R*N*M)`（M=ROI点数）

### 6.2 「ノミナル目的関数」
- `halbach/physics.py: objective_only()`
  - `J = mean ||B(p)-B0||^2` を返す（Numba）

### 6.3 「解析勾配（y-space）」
- `halbach/objective.py: objective_with_grads_fixed()`
  - `J` と `∂J/∂alphas`, `∂J/∂r_bases` を返す
- `halbach/objective.py: grad_alpha_and_radius_fixed()`
  - 解析勾配の重いループ（Numba）

### 6.4 「ロバスト目的＋勾配（x-space）」
- `halbach/robust.py: fun_grad_gradnorm_fixed()`
  - `x -> (alphas, r_vars) -> r_bases` の写像
  - `Jgn` と `∇_x Jgn` を返す
- `halbach/robust.py: hvp_y()`
  - `H_y(J) v` を y-space 中心差分で近似

### 6.5 「最適化ソルバ」
- `halbach/solvers/lbfgsb.py: solve_lbfgsb()`
  - SciPy `minimize(method="L-BFGS-B")` を呼ぶ
  - `fun` と `jac` を別々に呼ばれても整合するよう **キャッシュ**を実装
  - 反復履歴（trace）を記録

### 6.6 「入出力 / run ローダ」
- `halbach/run_io.py: load_run()`
  - run ディレクトリ or `.npz` 単体から読み込み
  - `RunBundle(results, meta, geometry, trace)` を構築
- `halbach/tools/normalize_run.py`
  - ファイル名/キーが揺れている run を `results.npz/meta.json/trace.json` に揃える

### 6.7 「2D 可視化（データ生成）」
- `halbach/viz2d.py: compute_error_map_ppm_plane()`
  - 指定平面における `|B|` の ppm 誤差マップ
- `extract_cross_section_y0()` 等：断面抽出と表示スケールのユーティリティ

---

## 7. 実行方法（最小例）

## 7.1 セットアップ（Windows / PowerShell, Python 3.11）
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-dev.txt
```

## 7.2 ロバスト最適化 + MC（エンドツーエンド）
名目設計 `.npz`（例：過去に保存した `diam_opt_saved_results.npz`）を入力として実行します。

```powershell
python .\robust_opt_halbach_gradnorm_minimal.py `
  --in_npz .\path\to\nominal_results.npz `
  --out_dir .\out\robust_run1 `
  --roi_r 0.140 `
  --roi_step 0.020 `
  --sigma_alpha_deg 0.5 `
  --sigma_r_mm 0.20 `
  --rho_gn 1e-4 `
  --eps_hvp 1e-6 `
  --maxiter 900 `
  --min_radius_drop_mm 20 `
  --mc_samples 600 `
  --seed 20250926
```

出力（例）：
- `diam_opt_saved_results_robust.npz`
- `diam_opt_saved_meta_robust.json`
- （互換用に上書き）`diam_opt_saved_results.npz`, `diam_opt_saved_meta.json`
- `diam_opt_saved_results.mat`
- `histories_gradnorm.png`
- `robust_summary_gradnorm.json`
- `robust_mc_compare_gradnorm.png`

> **注意（上書き）**  
> 上のスクリプトは「ビューア互換」を優先して、`out_dir` に `diam_opt_saved_results.npz` を **上書き保存**します。  
> 複数 run を並行で保持したい場合は `out_dir` を都度変えるか、ファイル名を変更するよう改造してください。

---

## 8. 2D 可視化（ppm 誤差マップ）

`halbach/viz2d.py` は「プロット画像」ではなく、プロットに必要なデータ（グリッドと ppm 値）を生成します。  
例：xy 平面（z=0）で 1 mm 刻みの ppm マップを作る：

```python
from halbach.run_io import load_run
from halbach.viz2d import compute_error_map_ppm_plane, extract_cross_section_y0

run = load_run("path/to/run_dir_or_results.npz")
m = compute_error_map_ppm_plane(run, plane="xy", coord0=0.0, roi_r=0.14, step=0.001)

line = extract_cross_section_y0(m)
print("B0 [mT] =", m.B0_T * 1e3)
print("ppm line shape =", line.ppm.shape)
```

matplotlib で描く場合は、`m.ppm` を `imshow` に渡して `vmin/vmax` を揃え、等高線は `contour` を使うのが典型です。

---

## 9. ベンチマークと性能上の注意

### 9.1 Numba の JIT
最初の呼び出しは **Numba のコンパイル時間**が乗ります。  
同じプロセス内の2回目以降は高速になります（`cache=True`）。

### 9.2 ROI 点数と計算量
`compute_B_and_B0` の計算量は `K*R*N*M` に比例します。  
ROI 点数 `M` は `roi_step` を小さくすると急増します。

- 目的関数評価用：`roi_step=0.02`（20 mm）など粗めにして軽量化
- 可視化用：`step=0.001`（1 mm）など細かくすると重い（ただし平面のみ）

### 9.3 ベンチマーク
カーネルの実行時間を測る：

```powershell
python -m bench.bench_kernels --preset prod --repeats 10
```

`prod` プリセットは概ね以下を想定しています：
- `N=48, K=24, R=3`
- `roi_r=0.14, roi_step=0.02`
- `ring_offsets=[0, 0.012, 0.024]`（m）

---

## 10. 開発（black/ruff/mypy/pytest, pre-commit）

### 10.1 ローカル品質チェック
```powershell
black .
isort .
ruff check .
mypy .
pytest
```

### 10.2 pre-commit
```powershell
pre-commit install
pre-commit run -a
```

> **トラブルシュート：InvalidConfigError（YAML）**  
> `.pre-commit-config.yaml` に BOM（不可視文字）が混入していると、Windowsで YAML 解析エラーになることがあります。  
> その場合はファイル先頭の BOM を削除（UTF-8 with BOM をやめる）してください。

---

## 11. 既知の制約・未実装/今後の予定

### 11.1 物理モデルの制約
- 磁石を点双極子近似（形状・寸法・材質・減磁等は未考慮）
- 双極子モーメントは xy 平面内のみ（`m_z=0`）
- 近接点での特異性回避として `+1e-30` を加えている（厳密ではない）

### 11.2 最適化変数の制約
- `alphas[r,k]` による低次元パラメタ化であり、**磁石ごとの独立角度最適化ではない**
- 半径は z 対称＋中央 4 層固定（現状仕様）

### 11.3 GUI / 3D 可視化
- この zip には GUI 実装は含まれていません（将来的に `Streamlit + Plotly` などを想定）。
- そのため、README の「GUIでできること」はここでは記載できません。
  - ただし `run_io` / `viz2d` / `trace.json` は GUI 実装を前提に設計してあります。

### 11.4 ライセンス
`pyproject.toml` は `LICENSE` ファイルを参照しますが、この zip には `LICENSE` が含まれていません。  
MIT ライセンス運用を想定している場合は、リポジトリ直下に `LICENSE` を追加してください。

---

