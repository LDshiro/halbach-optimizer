# 最適化後に保存されるデータ仕様

この文書は、最適化や関連コマンドの実行後に保存されるデータの構造を、コード実装に合わせて整理したものです。

## 1. 対象

本書の対象は次の 3 系統です。

1. 標準 Halbach 最適化 run
   - `python -m halbach.cli.optimize_run`
2. DC 系 run
   - `python -m halbach.cli.dc_optimize_run`
   - `python -m halbach.cli.dc_ccp_optimize_run`
   - `python -m halbach.cli.dc_ccp_sc_optimize_run`
3. standalone の legacy/minimal script
   - `python robust_opt_halbach_gradnorm_minimal.py`

この文書は「最適化後に保存される本体データ」を対象にしています。`sc_debug` のデバッグ bundle、可視化用の一時ファイル、評価専用スクリプトの派生出力は補足扱いです。

## 2. 共通ルール

### 2.1 単位

- 長さは原則 `m`
- 角度は原則 `rad`
- `magnet_dimensions_mm` のようにキー名に `mm` が付くものは `mm`
- `theta_id` / `layer_id` / `ring_id` は 0-based index

### 2.2 代表記号

- `R_max`: 保存配列上の半径方向スロット数
- `K`: z 層数
- `N`: 各 ring の角度サンプル数
- `M_dense = R_max * K * N`
- `M_active = sum(ring_active_mask) * N`

### 2.3 Flatten 順序

`phi_rkn.reshape(-1)` や `r0_rkn.reshape(-1, 3)` を使う箇所では、C-order の `(r, k, n)` 順で flatten されます。

- `r`: ring index
- `k`: layer index
- `n`: theta index

`magnet_*` 系の active-only export は、この dense 配列から `ring_active_mask` で inactive ring/layer を落とした順序です。各 row は `magnet_ring_id`, `magnet_layer_id`, `magnet_theta_id` と 1 対 1 で対応します。

## 3. 標準 Halbach run (`halbach.cli.optimize_run`)

### 3.1 ディレクトリ構成

```text
runs/<run_name>/
  results.npz
  meta.json
  trace.json
  opt.log
```

### 3.2 `results.npz`

`results.npz` は NumPy archive です。数値配列は原則 `float64` / `bool` / 整数配列で保存されます。条件付きで文字列や object を含むキーもあります。

#### 3.2.1 常に保存されるキー

| key | shape | unit | 説明 |
| --- | --- | --- | --- |
| `alphas_opt` | `(R_max, K)` | rad | 最適化後の `alpha` 配列。`legacy-alpha` では実値、他 angle model では互換用の dense 配列です。 |
| `r_bases_opt` | `(K,)` | m | 各 z 層の基準半径。 |
| `theta` | `(N,)` | rad | 各磁石セルの方位角サンプル。 |
| `sin2th` | `(N,)` | 1 | `sin(2 * theta)`。 |
| `cth` | `(N,)` | 1 | `cos(theta)`。 |
| `sth` | `(N,)` | 1 | `sin(theta)`。 |
| `z_layers` | `(K,)` | m | 各 z 層中心位置。 |
| `ring_offsets` | `(R_max,)` | m | ring ごとの半径オフセット。 |
| `radial_count_per_layer` | `(K,)` | count | 各 z 層で active な ring 数。 |
| `ring_active_mask` | `(R_max, K)` | bool | dense 配列中で active な ring/layer。`True` のみ物理磁石として有効。 |
| `J_hist` | `(T_iter,)` | objective unit | solver が見ていた目的関数履歴。現行 `optimize_run` では `Jn_hist` と同じ値です。 |
| `Jn_hist` | `(T_iter,)` | objective unit | trace extras 内の `J` を抜き出した履歴。 |
| `B0_hist` | `(T_iter,)` | T | 各反復での中心磁場強度 `|B0|`。`field_scale` を戻した実単位です。 |

`objective unit` は `field_scale` の影響を受けます。`B0_hist` は物理単位の Tesla ですが、`J_hist` / `Jn_hist` は最適化時に使ったスケーリング付き objective 値です。

#### 3.2.2 angle model に応じて追加されるキー

| key | shape | unit | 条件 | 説明 |
| --- | --- | --- | --- | --- |
| `beta_tilt_x_opt` | `(R_max, K)` | rad | `--enable-beta-tilt-x` | 各 ring/layer の x 軸まわり tilt。 |
| `delta_rep_opt` | `(K, n_rep)` | rad | `--angle-model delta-rep-x0` | x=0 mirror 制約下の代表角度差分。 |
| `fourier_coeffs_opt` | `(K, 2H)` | rad | `--angle-model fourier-x0` | Fourier 基底の係数。 |

補足:

- `delta-rep-x0` / `fourier-x0` でも `alphas_opt` は互換用に保存されますが、実際の角度自由度は `delta_rep_opt` / `fourier_coeffs_opt` を見てください。

#### 3.2.3 self-consistent 磁化モデルで追加されるキー

| key | shape | unit | 条件 | 説明 |
| --- | --- | --- | --- | --- |
| `sc_p_flat` | `(M_dense,)` | internal `p` unit | `mag_model=self-consistent-easy-axis` かつ保存成功時 | easy-axis の自己無撞着 `p`。dense flatten 順序 `(r, k, n)`。inactive 磁石は 0 になり得ます。 |
| `sc_cfg_fingerprint` | scalar | - | 上と同じ | `sc_p_flat` がどの self-consistent 設定で計算されたかを識別する fingerprint。 |
| `extras_sc_stats` | object | - | 上と同じ | 最終 self-consistent 評価から得た補助統計。内部実装依存で schema 固定ではありません。 |

#### 3.2.4 Fusion360 連携用の磁石 export キー

以下のキーは、現行の `optimize_run` で常に保存されます。

| key | shape | unit | 説明 |
| --- | --- | --- | --- |
| `magnet_centers_m` | `(M_active, 3)` | m | active 磁石の中心座標。 |
| `magnet_phi_rad` | `(M_active,)` | rad | 各磁石の方位角 `phi`。 |
| `magnet_beta_rad` | `(M_active,)` | rad | 各磁石の `beta_tilt_x`。無効時は 0。 |
| `magnet_u` | `(M_active, 3)` | 1 | 磁化方向の単位ベクトル `[ux, uy, uz]`。 |
| `magnet_ring_id` | `(M_active,)` | index | ring index。 |
| `magnet_layer_id` | `(M_active,)` | index | layer index。 |
| `magnet_theta_id` | `(M_active,)` | index | theta sample index。 |

self-consistent 有効時のみ、共通寸法として次も保存されます。

| key | shape | unit | 説明 |
| --- | --- | --- | --- |
| `magnet_dimensions_m` | `(3,)` | m | 各磁石の共通寸法 `[sx, sy, sz]`。現行実装では `volume_mm3` から復元した等価立方体 `[a, a, a]` です。 |
| `magnet_dimensions_mm` | `(3,)` | mm | 上と同じ内容の mm 版。 |

重要:

- `magnet_dimensions_*` は「各磁石ごとの配列」ではなく、「run 全体で共通な寸法」です。
- 現在の self-consistent `multi-dipole` / `cellavg` / `gl-double-mixed` は直方体の 3 辺を別々には持っておらず、`volume_mm3` から求めた等価立方体寸法を保存します。
- fixed model では元の直方体寸法パラメータがないため、現時点では `magnet_dimensions_*` は保存されません。

### 3.3 `meta.json`

`meta.json` は JSON object で、主に条件設定と run の文脈を保存します。現行 `optimize_run` のトップレベルには、概ね次のキーがあります。

- `input_run`: 入力 run パス
- `out_dir`: 出力先 run パス
- `start_time`, `end_time`: ISO-8601 timestamp
- `git_hash`: 実行時 git hash。取得失敗時は `null`
- `roi`: ROI 条件
- `scaling`: `field_scale`
- `optimizer`: L-BFGS-B と半径 bound の設定
- `angle_model`, `grad_backend`, `angle_init`, `fourier_H`
- `regularization`: `lambda0`, `lambda_theta`, `lambda_z`
- `angle_extra.beta_tilt_x`: beta tilt の定義と有効化状態
- `radial_profile`: `mode`, `base_R`, `end_R`, `end_layers_per_side`, `R_max`
- `magnetization`: `model_requested`, `model_effective`, `self_consistent`
- `fusion360_export`: 外部 CAD 連携向けメタデータ
- `fix_center_radius_layers`, `fix_radius_layer_mode`, `fixed_k_radius`
- `scipy_result`: `success`, `status`, `message`, `nit`, `nfev`, `njev`

`fusion360_export` の中身は現行コードで次の意味を持ちます。

- `coordinates_unit`: 現在は `"m"`
- `angles_unit`: 現在は `"rad"`
- `active_magnets_only`: `true`
- `keys`: `results.npz` 内の対応キー名
- `magnet_dimensions_source`: 寸法の由来。self-consistent の等価立方体なら `"self-consistent-volume-equivalent-cube"`
- `magnet_dimensions_m`, `magnet_dimensions_mm`: `results.npz` の寸法キーと同内容

`dry-run` 実行時は `results.npz` と `trace.json` は出力されず、`meta.json` のみが保存されます。このとき `meta.json` には `dry_run: true` が入ります。

### 3.4 `trace.json`

`trace.json` は solver 履歴の JSON 版です。

```json
{
  "iters": [0, 1, 2],
  "f": [123.0, 120.0, 119.5],
  "gnorm": [10.0, 4.0, 1.2],
  "extras": [
    {"J": 123.0, "B0": 0.0041},
    {"J": 120.0, "B0": 0.0040},
    {"J": 119.5, "B0": 0.0039}
  ]
}
```

ルール:

- `iters`, `f`, `gnorm`, `extras` は同じ長さです。
- `extras[i]` は反復 `iters[i]` に対応します。
- `J` と `B0` は常に含まれます。
- self-consistent 時は `sc_extras` 由来の追加キーが入ることがあります。
- JSON 化できない object は文字列化されることがあります。`trace.json` を厳密 schema として固定利用するより、`results.npz` / `meta.json` を主とし、`trace.json` は履歴閲覧用途と考える方が安全です。

### 3.5 `opt.log`

`opt.log` は人間向けログです。解析対象にしてもよいですが、機械可読 schema は保証しません。

## 4. `load_run` から見た互換条件

`halbach.run_io.load_run` が標準 run として扱うときの必須 / 省略可キーは次の通りです。

### 4.1 必須

- `alphas_opt` または `alphas`
- `r_bases_opt` または `r_bases`
- `theta`
- `z_layers`

### 4.2 省略可

- `sin2th` または `sin2`
  - 省略時は `sin(2 * theta)` を再計算
- `cth`
  - 省略時は `cos(theta)` を再計算
- `sth`
  - 省略時は `sin(theta)` を再計算
- `ring_offsets`
  - 省略時は全 0
  - 長さ 1 で `R_max > 1` の場合は全 ring に broadcast

### 4.3 extras の扱い

上記以外の `results.npz` キーは `RunResults.extras` にそのまま入ります。外部ツール向けの追加キーを増やしても、既存 loader 互換は保ちやすい設計です。

## 5. DC 系 run

DC 系 run は `meta.json["framework"] == "dc"` を持ちます。標準 Halbach run と異なり、`results.npz` に `theta`, `z_layers`, `ring_offsets` を直接持たない場合があります。

`load_run` は `meta.json["geom"]` から幾何を再構成し、DC run を `RunBundle` に読み込みます。

### 5.1 共通メタデータ

DC 系 `meta.json` には概ね次が入ります。

- `framework: "dc"`
- `dc_model`: 実行モデル名
- `geom`: `R`, `K`, `N`, `radius_m`, `length_m`
- `roi`
- `objective`
- `factor`

CCP 系ではさらに `ccp` セクションや `init_from_run` などが入ります。

### 5.2 `dc_optimize_run`

`results.npz` の主キー:

- `x_opt`
- `phi_flat`
- `p_flat`
- `m_flat`
- `r0_flat`
- `pts`

### 5.3 `dc_ccp_optimize_run`

`results.npz` の主キー:

- `z_opt`
- `x_opt`
- `phi_nom_flat`
- `phi_opt`
- `norm_opt`
- `m_flat`
- `r0_flat`
- `pts`
- `By`
- `By_diff`
- `center_idx`
- 条件付き: `z_init`, `phi_init_flat`

### 5.4 `dc_ccp_sc_optimize_run`

`results.npz` の主キー:

- `z_opt`
- `p_opt`
- `x_opt`
- `p_init`
- `phi_nom_flat`
- `phi_opt`
- `z_norm`
- `m_flat`
- `r0_flat`
- `pts`
- `By`
- `By_diff`
- `center_idx`
- `p_sc_post`
- `x_sc_post`
- `f_sc_post`
- 条件付き: `z_init`, `phi_init_flat`

注意:

- DC 系 `results.npz` は標準 Halbach run と schema が異なります。
- 可視化や後段処理では、まず `meta.json["framework"]` を見て分岐するのが安全です。

## 6. standalone / legacy script (`robust_opt_halbach_gradnorm_minimal.py`)

このスクリプトは標準 run directory とは別形式のファイル群を出力します。

### 6.1 主な出力ファイル

- `diam_opt_saved_results_robust.npz`
- `diam_opt_saved_meta_robust.json`
- `diam_opt_saved_results.npz`
- `diam_opt_saved_meta.json`
- `diam_opt_saved_results.mat`
- `histories_gradnorm.png`
- `robust_summary_gradnorm.json`
- `robust_mc_compare_gradnorm.png`

### 6.2 `.npz` / `.mat` の主キー

- `alphas_opt`
- `r_bases_opt`
- `theta`
- `sin2` または `sin2th`
- `z_layers`
- `ring_offsets`
- `J_hist`
- `Jn_hist`
- `B0_hist`
- `gn2_hist`

### 6.3 `meta` の主な内容

- `N`, `K`, `R`
- `r0`
- `ring_offsets`, `z_layers`
- `Lz`, `dz`
- `phi0`, `m0`
- `roi_r`, `roi_step`
- `robust`
- `optimizer`
- `scipy_result`

補足:

- この script の保存物は、現行 `runs/<name>/results.npz` 形式より古い viewer 互換名を優先しています。
- 現時点では Fusion360 用の `magnet_centers_m` / `magnet_dimensions_mm` などの追加キーはこの script には未実装です。

## 7. 外部ツール連携の推奨読み方

Fusion360 アドインや CAD/CAM 側で使う場合は、まず次の順で読むのが安全です。

1. `meta.json`
   - `framework`
   - `fusion360_export`
   - `radial_profile`
   - `magnetization`
2. `results.npz`
   - 標準 Halbach run なら `magnet_centers_m`, `magnet_phi_rad`, `magnet_beta_rad`, `magnet_dimensions_mm`
   - DC 系なら `r0_flat`, `phi_opt` などモデル固有キー
3. 必要なら `trace.json`
   - 反復履歴や診断用

標準 Halbach run で磁石穴あけに必要な最小キーは通常次です。

- 位置: `magnet_centers_m`
- 角度: `magnet_phi_rad`
- 傾き: `magnet_beta_rad`
- 寸法: `magnet_dimensions_mm` または `magnet_dimensions_m`

ただし寸法は現在 self-consistent でのみ保存され、しかも `volume_mm3` 由来の等価立方体です。実際の直方体 3 辺が必要なら、別途そのパラメータを run schema に追加する必要があります。
