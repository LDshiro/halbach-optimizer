# Plan C Implementation Plan

## 0. 目的

この文書は、`plan_c_specification_final.md` の「案C: クラスタ箱分け + 組立時再測定 + オンラインMPC割当」を、`halbach-optimizer` に実装するための詳細計画である。

主目的は、実組立前にシミュレーションとして以下を推定できるようにすることである。

- 個々の磁石の磁化強度ばらつき
- 個々の磁石の磁化方向ばらつき
- 測定ノイズ
- クラスタ箱分け
- 外れ値隔離
- 4姿勢候補
- 逐次割当

これらを考慮したとき、ROIでの磁場不均一性を案Cでどの程度抑制できるかを、ランダム配置と比較して評価する。

実組立モードは、シミュレーションで検証した割当エンジンと状態機械を再利用し、測定値入力元だけを `synthetic` から `manual` / `csv` / `serial` に差し替える方針で実装する。

## 1. 実装方針

### 1.1 一括実装ではなく段階実装にする

案Cは、物理評価、クラスタ在庫、逐次最適化、作業セッション、UI、ログ再開、将来の測定器I/Fを含む大きな後段システムである。一括実装すると、物理モデルの誤り、割当アルゴリズムの誤り、UI状態管理の誤りを切り分けにくい。

そのため、以下の順序で段階実装する。

1. データ構造、スロット抽出、4姿勢、測定値変換
2. 仮想磁石生成、クラスタリング、外れ値隔離
3. 固定モデルでの磁場評価、ランダム配置baseline
4. 線形感度計算
5. 線形感度MPCによる `simulation_auto_run`
6. 出力ファイル、CLI、統計集計
7. セッション状態機械、`simulation_step_by_step`、実組立モード基盤
8. Streamlit UI
9. `sequential_self_consistent` 候補再評価
10. serial測定器I/Fと運用硬化

### 1.2 既存最適化コードから独立させる

案Cは幾何・角度・半径プロファイルを再最適化しない。既存の `generate_run` / `optimize_run` / 目的関数 / 保存フォーマットを壊さず、最適化済みrunを入力として読む後段モジュールにする。

追加先は原則として以下に限定する。

```text
halbach/assembly/
halbach/cli/plan_c_*.py
app/pages/plan_c_*.py
tests/test_plan_c_*.py
docs または新規md
```

既存ファイルに触る場合は、import公開、Streamlitページ登録、README追記などの最小差分にする。

### 1.3 CLIを先に作り、UIは後に載せる

最初にCLIとpytestで再現可能なシミュレーション核を作る。

理由:

- seed固定で結果を検証できる
- UIなしで統計評価を回せる
- 実組立UIの前に、割当ロジックの正しさをテストできる
- 作業セッションの状態遷移をUIから切り離して検証できる

Streamlit UIは、CLIで固めたサービス層を呼ぶ薄い表示層として実装する。

### 1.4 最初の物理モデルはfixed + linear_sensitivity

最初の実用MVPは以下とする。

```text
simulation_auto_run
decision_engine = linear_sensitivity
mag_model_eval = fixed
baseline = random placement
```

`sequential_self_consistent` は後段に回す。既存リポジトリには per-magnet `p0_flat` 対応の自己無撞着ソルバがあるため、基盤が固まってから候補再評価として追加する。

## 2. 推奨ディレクトリ構成

### 2.1 assembly package

```text
halbach/assembly/
  __init__.py
  types.py
  slots.py
  work_units.py
  orientations.py
  measurement.py
  variation.py
  clustering.py
  inventory.py
  field_eval.py
  sensitivity.py
  online_assignment.py
  simulation.py
  session.py
  self_consistent_assignment.py
  io.py
```

### 2.2 CLI

```text
halbach/cli/plan_c_compute_sensitivity.py
halbach/cli/plan_c_prepare_inventory.py
halbach/cli/plan_c_simulate.py
halbach/cli/plan_c_session.py
```

### 2.3 Streamlit

最初は既存 `app/streamlit_app.py` に大きく混ぜず、Streamlit multipageとして分離する。

```text
app/pages/plan_c_simulation.py
app/pages/plan_c_assembly.py
```

既存アプリとの統合は後段で検討する。

## 3. 主要データ構造

初期実装で必要な型は `halbach/assembly/types.py` に集約する。

候補:

```python
SlotId = tuple[int, int, int]  # ring_id, layer_id, theta_id

@dataclass(frozen=True)
class AssemblySlot:
    slot_flat_id: int
    ring_id: int
    layer_id: int
    theta_id: int
    center_m: FloatArray  # shape (3,)
    nominal_u: FloatArray  # shape (3,)
    nominal_phi_rad: float
    active: bool
    work_unit_id: str
    mirror_pair_id: str | None
    physical_slot_number: int

@dataclass(frozen=True)
class MagnetError:
    epsilon_parallel: float
    delta_perp_1: float
    delta_perp_2: float

@dataclass(frozen=True)
class MeasuredMagnet:
    error: MagnetError
    moment_magnitude: float
    direction: FloatArray  # shape (3,)
    quality: float | None
    cluster_id: str | None

@dataclass(frozen=True)
class OrientationCandidate:
    id: str
    angle_deg: float
    instruction: str

@dataclass(frozen=True)
class Placement:
    slot_flat_id: int
    orientation_id: str
    magnet_error: MagnetError
    cluster_requested: str
    insert_order: int
    decision_engine: str
```

NumPy配列のshapeと単位はdocstringに明記する。

## 4. データフロー

### 4.1 シミュレーション

```text
optimized run
  -> slot extraction
  -> ROI points
  -> nominal field
  -> virtual true magnets
  -> pre-measurement noise
  -> clustering
  -> outlier isolation
  -> work unit planning
  -> online assignment
  -> final placement
  -> final field evaluation with true magnet errors
  -> random baseline comparison
  -> summary outputs
```

### 4.2 実組立

```text
optimized run
  -> precomputed sensitivity
  -> prepared inventory
  -> session start
  -> choose cluster
  -> measurement provider
  -> validate measured magnet
  -> solve placement
  -> operator confirmation
  -> session log append
  -> state save
  -> final placement csv
```

シミュレーションと実組立では、`choose cluster` 以降の状態機械を共有する。違いは測定値の供給元だけにする。

## 5. 実装ステップ詳細

## Step 0: ベースライン確認と開発前整備

### 目的

案C実装前の状態を明確にし、後続PRで既存機能を壊していないことを確認しやすくする。

### 実装内容

コード変更は原則なし。必要に応じて開発メモのみ追加する。

確認項目:

- Python環境
- `ruff`
- `mypy`
- 既存pytestの収集可否
- `fastapi` / `pydantic` など不足依存の有無

### 検証

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pytest -q tests\test_smoke_objective.py tests\test_run_io.py tests\test_magnet_export.py
```

### 完了条件

- 既存の品質ゲート状態を記録できている
- 案C実装による回帰か、環境起因かを判別できる

## Step 1: スロット抽出、作業単位、4姿勢、測定値変換

### 目的

最適化済みrunから、案Cが扱う「挿入スロット」と「磁石誤差」を安定して表現できるようにする。

### 実装ファイル

```text
halbach/assembly/__init__.py
halbach/assembly/types.py
halbach/assembly/slots.py
halbach/assembly/work_units.py
halbach/assembly/orientations.py
halbach/assembly/measurement.py
tests/test_plan_c_slots.py
tests/test_plan_c_orientations.py
tests/test_plan_c_measurement.py
```

### 実装内容

`slots.py`:

- `RunBundle` から active slot を列挙する
- `radial_profile_from_run(run).ring_active_mask` を使う
- flatten順序は既存仕様の `(r, k, n)` C-orderに合わせる
- 各slotについて以下を保持する
  - `slot_flat_id`
  - `ring_id`
  - `layer_id`
  - `theta_id`
  - `center_m`
  - `nominal_u`
  - `nominal_phi_rad`
  - `physical_slot_number`

`work_units.py`:

- `all_slots`
- `single_physical_ring`
- `ring_group`
- `auto`
- 初期実装では `custom` は読込仕様だけ予約してもよい
- 鏡像ペア候補を `layer_id` の `k` と `K-1-k` から作る

`orientations.py`:

- `O0/O90/O180/O270`
- `rotate_error_for_orientation(error, orientation)`
- 2D横ずれベクトルを名目磁化軸まわりに回す

`measurement.py`:

- `moment_magnitude` と `direction` から `MagnetError` を作る
- 小角近似:
  - `epsilon_parallel = moment_magnitude / nominal_magnitude - 1`
  - `delta_perp_1 ~= u_x / u_z`
  - `delta_perp_2 ~= u_y / u_z`
- `quality` の任意入力を保持

### 検証

単体テスト:

- active slot数が `sum(ring_active_mask) * N` と一致する
- inactive ring/layerが除外される
- slot flatten順が `((r*K + k)*N + n)` と一致する
- `all_slots` は全slotを1回ずつ含む
- `single_physical_ring` は `(ring_id, layer_id)` ごとに分かれる
- `auto` は仕様書の閾値に従う
- `O90` 回転で `[dx, dy] -> [-dy, dx]` になる
- `O180` 回転で符号反転する
- `direction=[0,0,1]` で横ずれゼロになる

コマンド:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_plan_c_slots.py tests\test_plan_c_orientations.py tests\test_plan_c_measurement.py
.\.venv\Scripts\python.exe -m ruff check halbach\assembly tests\test_plan_c_*.py
.\.venv\Scripts\python.exe -m mypy
```

### 完了条件

- 最適化済みrunから案C用slot tableを作れる
- 4姿勢の回転がテストで固定されている
- 測定器座標から磁石誤差ベクトルを作れる

## Step 2: 仮想磁石生成、クラスタリング、在庫、外れ値隔離

### 目的

シミュレーション用の仮想磁石群を生成し、事前測定値からクラスタ箱と在庫を作れるようにする。

### 実装ファイル

```text
halbach/assembly/variation.py
halbach/assembly/clustering.py
halbach/assembly/inventory.py
halbach/assembly/io.py
tests/test_plan_c_variation.py
tests/test_plan_c_clustering.py
tests/test_plan_c_inventory.py
```

### 実装内容

`variation.py`:

- iid normal
- Student-t
- two-lot mixture
- linear drift
- random walk drift
- direction normal
- seed固定
- 予備磁石数を含めて `required_count + reserve_count` を生成

初期実装では全分布を一度に完璧に作らず、以下を最小MVPにする。

- `iid_normal`
- `two_lot_mixture`
- `direction_normal`
- `linear_drift`

Student-t、random walk、CSV bootstrapはStep 6以降でもよい。

`clustering.py`:

- 強度 quantile bin
- 方向 transverse norm quantile bin
- `S{strength_bin:02d}_A{angle_bin:02d}` 形式
- thresholds modeは後段でもよい

`inventory.py`:

- clusterごとのcount, mean, covariance
- quarantine count
- inventory decrement / increment
- negative count防止

外れ値隔離:

- `none`
- `isolate_up_to_fraction`
- 優先順位:
  1. measurement unstable
  2. direction norm outlier
  3. strength outlier

初期実装では、qualityが閾値未満のものをunstable、方向normと強度絶対値のscoreで隔離対象を決める。

### 検証

単体テスト:

- seed固定で仮想磁石列が再現する
- quantile cluster数が設定値を超えない
- cluster id形式が安定している
- inventory合計が入力磁石数と一致する
- quarantine数が `floor(max_fraction * total)` を超えない
- inventory decrementで負数にならない
- covariance shapeが `(3, 3)` になる

コマンド:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_plan_c_variation.py tests\test_plan_c_clustering.py tests\test_plan_c_inventory.py
.\.venv\Scripts\python.exe -m mypy
```

### 完了条件

- シミュレーション用磁石群を生成できる
- 事前測定値からcluster inventoryを作れる
- 最大10%程度の外れ値隔離を安全に扱える

## Step 3: 固定モデル磁場評価とランダム配置baseline

### 目的

任意の最終配置について、ROI磁場均一度を評価できるようにする。案Cの改善度を測るため、ランダム配置baselineを先に実装する。

### 実装ファイル

```text
halbach/assembly/field_eval.py
halbach/assembly/simulation.py
tests/test_plan_c_field_eval.py
tests/test_plan_c_random_baseline.py
```

### 実装内容

`field_eval.py`:

- slot table + placement + magnet true errorから `r0_flat`, `m_flat` を作る
- `compute_B_and_B0_from_m_flat` を使ってROI磁場を評価
- 指標:
  - `rms_homogeneity_ppm`
  - `max_homogeneity_ppm`
  - `p95_homogeneity_ppm`
  - `p99_homogeneity_ppm`
  - `B0_norm`
  - `J_vector = mean(||B(p)-B0||^2)`
- 最初はfixed modelのみ

注意:

- 仕様書の主指標は `|B(p)-B0| / |B0|` のppm
- 既存2D表示は `|B|-|B0|` のppmを使う箇所があるため、案C評価ではベクトル差分指標を明示する

`simulation.py`:

- random placement
- orientationもrandomまたは固定O0を選べるようにする
- random baselineで同じ磁石集合と同じ外れ値条件を使う

### 検証

単体テスト:

- ばらつきゼロ、全O0ならnominalに近い指標になる
- 同じseedのrandom placementは同じ結果
- slot重複があるplacementを拒否する
- placement不足を拒否する
- `B0_norm` が小さすぎる場合は明確なエラー

統合テスト:

- `N=8,K=4,R=1` の小規模runでrandom baselineが完走する

コマンド:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_plan_c_field_eval.py tests\test_plan_c_random_baseline.py
```

### 完了条件

- 案Cなしのランダム配置で、ばらつきによるROI不均一性を測れる
- 後続の案C改善率の分母を安定して生成できる

## Step 4: 線形感度計算と感度ファイル

### 目的

slot・orientationごとの磁石誤差がROI均一度residualに与える線形寄与を事前計算する。

### 実装ファイル

```text
halbach/assembly/sensitivity.py
halbach/cli/plan_c_compute_sensitivity.py
tests/test_plan_c_sensitivity.py
tests/test_plan_c_sensitivity_io.py
```

### 実装内容

感度の基本量:

```text
y = [B(p_1)-B0, ..., B(p_M)-B0]
C_s = d y / d [epsilon_parallel, delta_perp_1, delta_perp_2]
```

初期実装方針:

- fixed modelのみ
- ROI点は `build_roi_points` または surface fibonacci
- 各slotについて3基底誤差の寄与を計算
- 姿勢は以下どちらかで扱う
  - 誤差ベクトルをorientationで回転して `C_s @ x_o`
  - `C_{s,o}` を保存
- 実装を単純にするため、初期は `C_{s,o}` 保存を推奨

低次元化:

- MVPでは圧縮なしでもよい
- 次の段階でSVD/PCAによる `dimension=30` を追加
- `projection_basis` は `plan_c_sensitivity.npz` に保存

感度ファイル:

```text
plan_c_sensitivity.npz
  slot_flat_id
  ring_id
  layer_id
  theta_id
  centers_m
  nominal_u
  orientation_id
  C
  roi_points
  normalization_B0
  metadata_json
```

CLI:

```powershell
.\.venv\Scripts\python.exe -m halbach.cli.plan_c_compute_sensitivity --run runs/demo_opt --out runs/demo_opt/plan_c_sensitivity.npz --roi-r 0.14 --roi-mode surface-fibonacci --roi-samples 300
```

### 検証

単体テスト:

- `C` shapeが `(n_slots, n_orientations, residual_dim, 3)` または定義したshapeと一致する
- `x=0` の寄与はゼロ
- 1slotに小さい誤差を入れた直接磁場差分と `C @ x` が一致する
- `O180` は横ずれ成分の符号反転として反映される
- 感度npzのread/writeで値が保持される

数値許容:

- fixed modelなら相対誤差 `1e-5` 程度を目標
- 有限差分を使う場合は `finite_difference_step` に依存するため、テストは絶対/相対許容をやや緩める

コマンド:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_plan_c_sensitivity.py tests\test_plan_c_sensitivity_io.py
```

### 完了条件

- 感度ファイルを生成・再読込できる
- 線形近似が小さい誤差で直接評価と一致する
- 後続のonline assignmentが高速に候補評価できる

## Step 5: 線形感度オンライン割当と `simulation_auto_run`

### 目的

案Cの中核である「クラスタ箱指定、再測定、slot・姿勢決定」を、自動シミュレーションとして最後まで実行できるようにする。

### 実装ファイル

```text
halbach/assembly/online_assignment.py
halbach/assembly/simulation.py
halbach/cli/plan_c_simulate.py
tests/test_plan_c_online_assignment.py
tests/test_plan_c_simulation_auto.py
```

### 実装内容

`online_assignment.py`:

- residual vectorを保持
- 空きslot集合を保持
- 現在磁石 `x_now` とcluster計画から未来仮想磁石を生成
- 初期実装の候補選択:
  1. 次clusterを選ぶ
  2. そのclusterから仮想または実測磁石を取る
  3. 全空きslot x 4姿勢を線形評価
  4. 最良候補を選ぶ
- 未来仮想磁石を使うMPCは段階的に追加する

MVPの割当アルゴリズム:

```text
score(s, o) = ||residual + C[s,o] @ x_now||^2
choose argmin score
```

次の段階で追加:

- cluster mean future magnets
- quantile future magnets
- local search swaps
- restarts

`simulation.py`:

- trial loop
- random baseline
- plan C linear run
- outlierあり/なし比較
- summary metrics集計

CLI:

```powershell
.\.venv\Scripts\python.exe -m halbach.cli.plan_c_simulate --run runs/demo_opt --out runs/demo_opt/plan_c_sim --trials 10 --seed 1234 --strength-sigma 0.01 --direction-sigma 0.001 --engine linear_sensitivity
```

### 検証

単体テスト:

- 同じseedで最終配置が再現する
- slotが重複使用されない
- inventoryが負にならない
- orientationが4候補から選ばれる
- residual更新が `residual += contribution` と一致する
- cluster使用数が計画と一致または許容範囲内

統合テスト:

- 小規模runで `simulation_auto_run` が完走
- 人工的に相殺しやすい誤差集合ではrandomよりplan Cが改善する
- 改善しない可能性があるランダムケースでは、失敗扱いではなくsummaryに比率を出す

コマンド:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_plan_c_online_assignment.py tests\test_plan_c_simulation_auto.py
```

### 完了条件

- fixed + linear_sensitivityで案Cシミュレーションが最後まで回る
- ランダム配置との比較指標が出る
- seed固定再現性がある

## Step 6: 出力ファイル、統計集計、設定ファイル

### 目的

シミュレーション結果を後から比較・解析できる形式で保存する。

### 実装ファイル

```text
halbach/assembly/io.py
halbach/assembly/config.py
tests/test_plan_c_io.py
tests/test_plan_c_config.py
```

### 実装内容

出力:

```text
simulation_summary.json
simulation_trials.csv
final_placement_trial_XXX.csv
cluster_usage_trial_XXX.csv
work_unit_summary_trial_XXX.csv
field_metrics_trial_XXX.json
streamlit_session_log_trial_XXX.jsonl
```

設定:

- 初期はJSONを標準にしてもよい
- 仕様書は `plan_c_config.yaml` だが、依存追加を避けるならJSONから始める
- YAML対応が必要なら `pyyaml` 追加の是非をOwner確認する

保存時の注意:

- CSV列は仕様書13.5に合わせる
- JSONは `schema_version` を持つ
- 単位をmetadataに明記する
- trialごとのseedを保存する

### 検証

単体テスト:

- summary JSONが読める
- CSV列が仕様どおり
- placement CSVからslot重複なしを再検証できる
- `schema_version` が保存される
- config defaultが仕様書の既定値に近い

コマンド:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_plan_c_io.py tests\test_plan_c_config.py
```

### 完了条件

- CLI出力だけで案C改善率を評価できる
- 後続UIが同じ出力・状態を読める

## Step 7: セッション状態機械、step-by-step simulation、実組立モード基盤

### 目的

実組立と同じ1個ずつの手順を、UIなしでも状態機械として実行できるようにする。

### 実装ファイル

```text
halbach/assembly/session.py
halbach/assembly/measurement.py
halbach/cli/plan_c_session.py
tests/test_plan_c_session.py
tests/test_plan_c_measurement_providers.py
```

### 実装内容

状態:

```text
PREPARE_SESSION
SELECT_WORK_UNIT
BUILD_WORK_UNIT
EVALUATE_MIRROR_PAIR_SWAP
INSTALL_OR_CONFIRM_PAIR
COMPLETE
PAUSED
ERROR
```

作業単位内サブ状態:

```text
CHOOSE_CLUSTER
WAIT_FOR_MAGNET_MEASUREMENT
VALIDATE_MEASUREMENT
SOLVE_PLACEMENT
WAIT_FOR_INSERT_CONFIRMATION
UPDATE_STATE
```

measurement providers:

- `synthetic`
- `manual`
- `csv`
- `serial` はinterfaceとfake実装のみ

session log:

- JSONL append
- 各step後にstate snapshot保存
- placement確定時にevent保存

Undo:

- 直近1stepのみMVPで対応
- residual
- slot occupancy
- inventory
- placement log

### 検証

単体テスト:

- 状態遷移が期待順に進む
- `simulation_auto_run` と同じseed・同じ測定順でstep-by-stepが同じfinal placementになる
- 直近Undoで状態が戻る
- resume後にslot重複が起きない
- CSV providerが順番に測定値を返す

コマンド:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_plan_c_session.py tests\test_plan_c_measurement_providers.py
```

### 完了条件

- UIなしで実組立相当の手順を再現できる
- session logから中断・再開できる
- 実測定器入力を差し込む場所が決まっている

## Step 8: Streamlit UI

### 目的

作業者がシミュレーションと実組立を同じ流れで操作できるUIを追加する。

### 実装ファイル

```text
app/pages/plan_c_simulation.py
app/pages/plan_c_assembly.py
halbach/assembly/ui_payload.py
tests/test_plan_c_ui_payload.py
```

### 実装内容

UIは薄くし、状態更新は `session.py` に委譲する。

表示内容:

- 現在のモード
- 現在の作業単位
- 鏡像ペアID
- 現在作業中ring
- 残りslot数
- 次に取るcluster
- cluster在庫残数
- 再測定値
- 推奨ring
- 推奨slot
- 推奨orientation
- orientation instruction
- 現在の予測均一度
- 隔離数
- ログ保存状態

リング表示:

- MVPではPlotlyまたは表形式
- 大型リングは円周slot map
- 小規模やgroupはtable併用

操作:

- Start session
- Next synthetic measurement
- Accept measurement
- Reject / quarantine
- Confirm insertion
- Undo last
- Pause / resume

### 検証

UIそのものは手動確認が必要だが、ロジックはpayloadでテストする。

単体テスト:

- UI payloadに必要項目がそろう
- 推奨slotがhighlight対象に入る
- orientation説明文が返る
- occupied / empty / recommended 状態が区別される

手動確認:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

または:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\pages\plan_c_simulation.py
```

### 完了条件

- `simulation_step_by_step` をUIで進められる
- `simulation_auto_run` のsummaryをUIから起動・閲覧できる
- `assembly_real` のmanual入力で推奨slot・姿勢が表示される

## Step 9: sequential_self_consistent 候補再評価

### 目的

線形感度近似では捉えにくい自己相互作用を、候補削減付きの自己無撞着評価で確認できるようにする。

### 実装ファイル

```text
halbach/assembly/self_consistent_assignment.py
tests/test_plan_c_self_consistent_assignment.py
```

### 実装内容

方針:

1. 線形感度で候補slot・orientationをtop-k抽出
2. 各候補について仮配置を作る
3. 既存JAX自己無撞着ソルバでROI指標を評価
4. 最良候補を採用

既存再利用候補:

- `compute_B_and_B0_from_m_flat`
- `build_m_flat_from_phi_and_p`
- `solve_p_easy_axis_near_with_p0_flat`
- `solve_p_easy_axis_near_multi_dipole_with_p0_flat`
- `solve_p_easy_axis_near_gl_double_mixed`

制限:

- 初期は小規模run向け
- `cellavg` は既存 `perturbation_eval.py` 同様に後回しでもよい
- beta tilt込みの3D方向は既存関数との整合を確認してから拡張

### 検証

単体テスト:

- `chi=0` ではfixed評価と一致または近い
- 小規模runで完走
- top-k制限を超えて評価しない
- seed固定で再現
- p0_flatのshape不一致を拒否

性能ログ:

- 1step平均評価時間
- self-consistent評価回数
- linear候補とSC最良候補の差

コマンド:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_plan_c_self_consistent_assignment.py
```

### 完了条件

- 比較対象「案C逐次自己無撞着モード」が小規模で実行できる
- 線形感度モードとの比較がsummaryに出る

## Step 10: serial測定器I/Fと実運用硬化

### 目的

実測定器が用意されたときに、最小差分で接続できるようにする。

### 実装ファイル

```text
halbach/assembly/measurement.py
tests/test_plan_c_serial_provider.py
```

### 実装内容

serial provider:

- JSON line形式
- port, baudrate, timeout
- parse error処理
- 欠損field検出
- direction正規化
- quality閾値

注意:

- `pyserial` 依存追加が必要な場合はOwner確認
- 依存追加を避けるならinterfaceのみ先に作り、実装はoptional importにする

運用硬化:

- session log破損検出
- duplicate slot検出
- inventory mismatch検出
- manual overrideログ
- quarantine理由ログ

### 検証

単体テスト:

- fake serial streamでJSON lineを読める
- malformed JSONをrejectできる
- timeoutを扱える
- missing directionをrejectできる
- quality低下を `Q_MEASUREMENT_UNSTABLE` にできる

### 完了条件

- 実測定器接続前でもI/F仕様が固定されている
- serial接続の失敗がセッション破壊につながらない

## 6. 比較対象の実装順

仕様書の比較対象は以下である。

1. ランダム配置
2. 案C: 線形感度
3. 案C: 線形感度 + 外れ値隔離
4. 案C: 逐次自己無撞着
5. 案C: 逐次自己無撞着 + 外れ値隔離

実装順は以下にする。

```text
Phase A:
  random placement
  plan C linear_sensitivity

Phase B:
  random placement + outlier isolation
  plan C linear_sensitivity + outlier isolation

Phase C:
  plan C sequential_self_consistent

Phase D:
  plan C sequential_self_consistent + outlier isolation
```

Phase Aが完成した時点で、主要な設計判断とデータ構造はかなり検証できる。

## 7. 受け入れ条件の段階定義

### MVP受け入れ条件

MVPは「実組立UIなしで、シミュレーション評価ができる」状態とする。

必須:

- 最適化済みrunからslotを抽出できる
- `O0/O90/O180/O270` を扱える
- synthetic magnet variationを生成できる
- quantile clusteringができる
- 外れ値隔離なし/ありを選べる
- fixed modelでfinal field metricsを評価できる
- random baselineとplan C linearを比較できる
- `simulation_summary.json` と `simulation_trials.csv` を出せる
- seed固定で再現する
- pytestで主要ロジックを検証できる

### 実組立基盤受け入れ条件

- session state machineがある
- manual providerで測定値を入力できる
- slot・姿勢の推奨を1stepずつ出せる
- JSONL session logを保存できる
- 中断・再開できる
- 直近Undoができる

### 高精度評価受け入れ条件

- sequential self-consistent候補再評価が小規模runで動く
- top-k候補削減が効く
- linear modeとの比較がsummaryに出る
- 1step平均計算時間が記録される

## 8. 推奨テスト一覧

### 単体テスト

```text
tests/test_plan_c_slots.py
tests/test_plan_c_work_units.py
tests/test_plan_c_orientations.py
tests/test_plan_c_measurement.py
tests/test_plan_c_variation.py
tests/test_plan_c_clustering.py
tests/test_plan_c_inventory.py
tests/test_plan_c_field_eval.py
tests/test_plan_c_sensitivity.py
tests/test_plan_c_online_assignment.py
tests/test_plan_c_simulation_auto.py
tests/test_plan_c_io.py
tests/test_plan_c_session.py
tests/test_plan_c_measurement_providers.py
tests/test_plan_c_ui_payload.py
tests/test_plan_c_self_consistent_assignment.py
```

### 統合テスト

小規模run:

```text
N=8, K=4, R=1
```

確認:

- random baseline完走
- plan C linear完走
- slot重複なし
- inventory負数なし
- seed固定再現
- 外れ値率上限遵守
- orientation候補が評価される
- session autoとstep-by-stepが同じ最終配置を再現できる設定がある

### 回帰テスト

案Cは後段機能なので、既存機能に影響しないことを確認する。

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_smoke_objective.py tests\test_run_io.py tests\test_magnet_export.py tests\test_perturbation_eval_smoke.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
```

環境が整っている場合:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 9. 実装上の注意点

### 9.1 目的関数・座標系を変更しない

案Cでは、既存の幾何最適化結果を入力として使う。以下はOwner指示なしに変更しない。

- 目的関数定義
- 座標系
- 単位
- `results.npz` / `meta.json` の既存キー
- `x-space` / `y-space` の定義
- Numbaカーネル境界

### 9.2 評価指標を明確に分ける

既存可視化には `|B|-|B0|` 型のppmがある。一方、案C仕様では主指標を以下にする。

```text
sqrt(mean(||B(p)-B0||^2)) / ||B0|| * 1e6
max(||B(p)-B0||) / ||B0|| * 1e6
```

混同を避けるため、案Cの出力では名前を明確にする。

推奨名:

```text
rms_vector_homogeneity_ppm
max_vector_homogeneity_ppm
p95_vector_homogeneity_ppm
p99_vector_homogeneity_ppm
```

### 9.3 個体IDなし運用を守る

実組立モードでは個体IDを前提にしない。ただし品質記録として、挿入順、測定値、cluster、slot、orientationはログに残す。

### 9.4 依存追加は最小限にする

YAMLやserialのために追加依存が必要な場合は、最初から必須依存にしない。

候補:

- YAML: optional `pyyaml`
- serial: optional `pyserial`

初期はJSON configとfake/manual/csv providerで進める。

### 9.5 大規模計算への備え

数千磁石 x ROI点 x orientation候補は重い。初期実装から以下を意識する。

- slot感度はnpzに保存して再利用
- online residualは低次元化可能な形にする
- candidate scoreはNumPy vectorizeする
- self-consistentはtop-k候補だけに制限
- Streamlitで重い計算を直接繰り返さない

## 10. 推奨PR分割

### PR 1: assembly基礎型

内容:

- `types.py`
- `slots.py`
- `work_units.py`
- `orientations.py`
- `measurement.py`
- 単体テスト

受け入れ:

- slot抽出、4姿勢、測定値変換が動く

### PR 2: variation / clustering / inventory

内容:

- `variation.py`
- `clustering.py`
- `inventory.py`
- 外れ値隔離
- 単体テスト

受け入れ:

- synthetic磁石をcluster箱に分けられる

### PR 3: field_eval / random baseline

内容:

- `field_eval.py`
- random placement
- fixed model metrics

受け入れ:

- random配置のROI不均一性を評価できる

### PR 4: sensitivity

内容:

- `sensitivity.py`
- `plan_c_compute_sensitivity` CLI
- sensitivity npz

受け入れ:

- 小さい誤差で直接評価と線形感度が一致する

### PR 5: linear simulation auto-run

内容:

- `online_assignment.py`
- `simulation.py`
- `plan_c_simulate` CLI
- summary出力

受け入れ:

- random vs plan C linearを比較できる

### PR 6: session state machine

内容:

- `session.py`
- `plan_c_session` CLI
- JSONL log
- resume / undo MVP

受け入れ:

- step-by-step simulationができる

### PR 7: Streamlit simulation UI

内容:

- `app/pages/plan_c_simulation.py`
- UI payload

受け入れ:

- 仮想測定と仮想挿入をUIで進められる

### PR 8: Streamlit real assembly UI

内容:

- `app/pages/plan_c_assembly.py`
- manual/csv measurement
- 作業者向け姿勢説明

受け入れ:

- 実測値手入力でslot・姿勢指示が出る

### PR 9: sequential self-consistent

内容:

- `self_consistent_assignment.py`
- top-k再評価
- summary比較

受け入れ:

- 小規模runでSC候補再評価が完走する

### PR 10: serial provider and hardening

内容:

- serial optional provider
- log validation
- operation safeguards

受け入れ:

- fake serialで測定値を読める
- 異常入力でセッションが壊れない

## 11. 最初に着手する具体タスク

最初の実装タスクは、PR 1相当を推奨する。

編集対象:

```text
halbach/assembly/__init__.py
halbach/assembly/types.py
halbach/assembly/slots.py
halbach/assembly/work_units.py
halbach/assembly/orientations.py
halbach/assembly/measurement.py
tests/test_plan_c_slots.py
tests/test_plan_c_work_units.py
tests/test_plan_c_orientations.py
tests/test_plan_c_measurement.py
```

この段階では、磁場評価、感度計算、MPC、UIには入らない。

理由:

- 仕様の土台になるslot orderとorientation定義を先に固定できる
- 後続の全機能がこのデータ構造を使う
- 既存物理式に触れないためリスクが低い
- テストが軽く、回帰検出しやすい

## 12. 未決事項

実装前または該当Step前にOwner判断が必要な事項:

- `plan_c_config.yaml` のために `pyyaml` を依存追加するか、初期はJSONにするか
- serial providerのために `pyserial` を依存追加するか
- 実作業で使うslot番号の採番規則
- mirror pair swapを最初から有効にするか、シミュレーションのみ先行にするか
- 作業単位 `ring_group` の自動グループサイズ
- 方向誤差の2横成分に対する重み
- UIでの姿勢説明文の実物治具に合わせた表現
- self-consistent modeで `cellavg` を初期対応に含めるか

## 13. まとめ

案Cは、まずCLIベースの `simulation_auto_run` と fixed + `linear_sensitivity` を完成させるのが最短で価値が出る。これにより、磁石ばらつきに対してROI磁場不均一性をどの程度抑制できるかを、ランダム配置と比較して定量評価できる。

実組立モードは、その後に同じsession state machineへ `manual` / `csv` / `serial` providerを差し込む。Streamlit UIは、状態機械と割当エンジンを呼ぶ薄い表示層として作る。

この順序なら、既存の幾何最適化・目的関数・保存フォーマットを保ったまま、案Cを独立した後段機能として安全に追加できる。
