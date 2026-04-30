# Plan C Ring-by-Ring Strategy Specification

## 0. 目的

この文書は、案Cに大規模リング向けの「リング1枚ずつ組み立てる戦略」を追加するための仕様・実装計画である。

現在の Plan C simulation は、実質的に全 active slot を同時候補として扱う小規模リング向け戦略である。これは、磁石総数が少なく、全リングを同時に組み立てる運用では妥当である。一方、大規模リングでは、作業スペース、治具、取り違え防止、実作業の安定性の観点から、物理リングを1枚ずつ組み立てる戦略が必要になる。

本仕様の目的は、以下を満たす ring-by-ring 戦略を段階的に実装することである。

- 物理リングごとの平均磁石強度をできるだけ揃える。
- 円筒スタックの両端から中心へ向かって組み立てる。
- ROIに近い中心付近リングには、強度が揃い、方向誤差が小さい磁石を優先する。
- 両端付近リングでは、ばらつきの大きい磁石を優先的に消費する。
- 外側リングで発生した磁場誤差を、内側リングの磁石ばらつきで補償する。
- 強度クラスタと角度誤差クラスタを使い、作業者には「どのクラスタ箱から1個取るか」を指示する。
- 取った磁石は組み立て直前に再測定し、実測値に基づいて現在リング内の slot/orientation を決める。
- シミュレーションでは、各リングにどの程度のエラーを持つ磁石が割り当てられたかを可視化できる。
- GUIでは、リングが1枚ずつ組み上がる様子を観察できる。

## 1. 現状整理

### 1.1 現在実装済みの小規模向け戦略

現在の Plan C simulation は以下の動作をする。

1. 全 active slot を候補集合にする。
2. 磁石を順番に処理する。
3. 各磁石の `measured_error` を使う。
4. 全 remaining slot と 4 姿勢を評価する。
5. 線形感度、または線形上位候補の自己無撞着再評価で slot/orientation を決める。
6. 配置後に residual を更新する。

これは `all_slots` 戦略であり、小規模リングや全リング同時組み立てに相当する。

### 1.2 既存の土台

以下は既に実装済みで、ring-by-ring 戦略でも再利用する。

- `AssemblySlot`
- `WorkUnit`
- `build_work_units(..., "single_physical_ring")`
- `assign_quantile_clusters`
- `ClusterInventory`
- `SensitivityTable`
- `linear_sensitivity` assignment
- `sequential_self_consistent` assignment
- `self_consistent_from_run` の JAX 評価
- simulation output csv/json
- session state machine の基礎

### 1.3 まだ不足しているもの

ring-by-ring 戦略に必要だが未実装のものは以下である。

- 外側から中心へ進む work unit ordering。
- mirror layer pair を意識した work unit。
- 現在 work unit 内だけに候補 slot を制限する assignment。
- ring ごとの強度平均 target。
- ring ごとの強度クラスタ quota。
- 中心優先の角度誤差クラスタ配分。
- cluster を選んでから磁石を再測定する simulation。
- cluster mean/cov を使った cluster-level MPC。
- future inventory risk を含む Level 3 MPC。
- ring-by-ring playback 用の timeline 出力。
- GUIで ring summary / stack progress / cluster inventory を表示する機能。

## 2. 用語

### 2.1 Physical Ring

1枚の物理リングを、現行データ構造では以下で表す。

```text
physical_ring_key = (layer_id, ring_id)
```

- `layer_id`: z方向スタック位置
- `ring_id`: 半径方向リング番号
- `theta_id`: 同一物理リング内の円周方向 slot 番号

### 2.2 Mirror Ring Pair

z方向の鏡像ペアを以下で定義する。

```text
mirror_pair(layer_id=k) = (k, K - 1 - k)
```

`K` が奇数の場合、中央層 `k == K//2` は単独 ring として扱う。

半径方向に複数リング `R > 1` がある場合は、同じ `ring_id` ごとに mirror pair を作る。

```text
((layer_id=k, ring_id=r), (layer_id=K-1-k, ring_id=r))
```

### 2.3 Assembly Strategy

少なくとも以下の2戦略をサポートする。

```text
assembly_strategy = all_slots_global
assembly_strategy = ring_by_ring_outer_to_inner
```

`all_slots_global` は現行方式で、小規模リング向けである。

`ring_by_ring_outer_to_inner` は大規模リング向けで、物理リングを1枚ずつ、両端から中心へ向かって組み立てる。

### 2.4 Ring Importance

ROIは円筒中心にあるため、中心付近の ring ほど重要度を高くする。

例:

```text
z_abs_norm(k) = abs(z_layer[k]) / max(abs(z_layer))
ring_importance(k) = 1 - z_abs_norm(k)
```

中心 ring ほど `ring_importance` が大きい。

この値は、角度誤差の小さい磁石を中心に残すための重みとして使う。

## 3. 基本方針

### 3.1 2段階の意思決定

ring-by-ring 戦略では、意思決定を2段階に分ける。

#### Stage A: どのクラスタ箱から磁石を取るか

これは ring 平均強度、角度誤差配分、将来在庫を考えて決める。

作業者への指示:

```text
S03_A04 から磁石を1個取ってください
```

#### Stage B: 取った磁石をどの slot/orientation に入れるか

取った磁石を再測定し、その `measured_error` を使って、現在組み立て中の ring 内の未配置 slot と姿勢を選ぶ。

候補は現在 active な work unit に限定する。

```text
candidate_slots = remaining slots in current physical ring
candidate_orientations = O0/O90/O180/O270
```

基本 score:

```text
score = || residual + C[slot, orientation] @ measured_error ||^2
```

`sequential_self_consistent` の場合は、まず線形 score で上位候補を選び、その候補だけ JAX self-consistent 評価で再ランクする。

### 3.2 ring 平均強度の制約

各 physical ring で使う磁石の平均強度誤差は、全体 target に近づける。

```text
ring_mean_epsilon = mean(epsilon_parallel of magnets in ring)
target_mean_epsilon = mean(epsilon_parallel of all usable magnets)
```

基本 penalty:

```text
strength_balance_cost =
  lambda_ring_mean * (ring_mean_epsilon - target_mean_epsilon)^2
```

mirror pair の左右差も抑える。

```text
mirror_strength_cost =
  lambda_mirror_mean * (mean_epsilon_left - mean_epsilon_right)^2
```

### 3.3 端から中心への品質配分

角度誤差 cluster は中心ほど良いものを使う。

例:

```text
angle_cluster A00: 方向誤差が最小
angle_cluster A04: 方向誤差が最大
```

端 ring:

```text
A03/A04 を優先的に消費
```

中心 ring:

```text
A00/A01 を優先的に使用
```

ただし、強度平均の制約を崩してまで角度品質を優先しない。中心付近では強度も方向も良い磁石を残すため、将来在庫 reserve を明示的に管理する。

### 3.4 外側 ring の誤差を内側 ring で補償する

外側 ring を組んだ後、ROI residual を更新する。

より内側の ring では、以下の両方を考える。

1. ring 平均強度 target を守る。
2. 現在 residual を小さくする。

これは次のような multi-objective score として扱う。

```text
total_cost =
  field_cost
+ strength_balance_cost
+ angle_quality_cost
+ mirror_pair_balance_cost
+ future_inventory_risk
```

Level 1/2 では一部項だけを使い、Level 3 で全体を使う。

## 4. 推奨アルゴリズム

### 4.1 Level 1: quota planning

最初の実装では、各 ring に cluster quota を事前配分する。

入力:

- cluster inventory
- cluster stats: count, mean, cov
- physical rings
- ring importance
- target mean epsilon

出力:

```text
RingQuotaPlan:
  ring_key
  target_count
  target_mean_epsilon
  allowed_clusters
  quota_by_cluster
```

強度方向は、全 ring の平均が揃うように低強度 cluster と高強度 cluster を組み合わせる。

例:

```text
S00 + S09
S01 + S08
S02 + S07
S03 + S06
S04 + S05
```

端 ring では強度の極端な組み合わせも許容し、中心 ring では `S04/S05` 近傍を増やす。ただし最終的な ring mean は target に合わせる。

角度方向は、端から中心へ向かって以下のように割り当てる。

```text
outermost: A04, A03
mid:       A02
center:    A01, A00
```

### 4.2 Level 2: ring-constrained online assignment

Level 2 では、既存の `linear_sensitivity` を current ring 制約付きで使う。

手順:

1. current physical ring を決める。
2. その ring の quota から、次に使う cluster を選ぶ。
3. simulation では、その cluster から1個の磁石をサンプリングする。
4. 実組立では、作業者がその cluster 箱から1個取る。
5. 取った磁石を再測定する。
6. current ring の remaining slot だけを候補にする。
7. `measured_error` で slot/orientation を選ぶ。
8. residual と ring summary を更新する。
9. current ring が埋まったら次の ring に進む。

この段階では cluster 選択は quota 消費を優先し、field residual は slot/orientation 選択で使う。

### 4.3 Level 3: cluster-level MPC

Level 3 では、cluster 選択にも residual と将来在庫を使う。

cluster `c` の測定誤差分布を以下で近似する。

```text
x_c ~ N(mean_c, cov_c)
x = [epsilon_parallel, delta_perp_1, delta_perp_2]
```

slot/orientation `j` の線形寄与を `C_j` とすると、cluster c から取った磁石を候補 j に置いたときの期待 field cost は、

```text
E[||residual + C_j x_c||^2]
  = ||residual + C_j mean_c||^2
  + trace(C_j cov_c C_j^T)
```

cluster 選択では、current ring 内の最良候補で近似する。

```text
field_cost(c) = min_j E[||residual + C_j x_c||^2]
```

最終的な cluster score:

```text
cluster_score(c) =
  field_cost(c)
+ lambda_quota * quota_deviation_after_pick(c)
+ lambda_ring_mean * ring_mean_deviation_after_pick(c)
+ lambda_angle * angle_quality_penalty(c, ring_importance)
+ lambda_mirror * mirror_pair_balance_after_pick(c)
+ lambda_future * future_inventory_risk_after_pick(c)
```

`future_inventory_risk` は、中心 ring 用に必要な良品 cluster が不足しそうな場合に罰する。

例:

```text
required_center_good_count[A00/A01, S04/S05] - remaining_good_count
```

不足が出る場合のみ二乗 penalty にする。

### 4.4 実測後のズレへの対応

作業時には、指定 cluster から取った磁石を再測定する。再測定値が元 cluster の範囲から外れる場合がある。

対応方針:

1. 小さいズレ: 指定 cluster から消費したものとして扱い、実測 `measured_error` で配置する。
2. 大きいズレ: `measurement_reclassified` としてログし、実測 cluster に振り替える。
3. 品質不良: `Q_MEASUREMENT_UNSTABLE` として reject し、同じ cluster 指示を再発行する。
4. 方向/強度外れ: quarantine し、ring quota を再計算または補充 cluster を選ぶ。

初期実装では 1 と 3 を実装し、2 と 4 は後段でよい。

## 5. 組み立て順序

### 5.1 基本順序

`K` 層の stack に対し、両端から中心へ進む。

例: `K = 8`

```text
k=0
k=7
k=1
k=6
k=2
k=5
k=3
k=4
```

mirror pair ごとに計画する場合:

```text
pair 0: k=0 and k=7
pair 1: k=1 and k=6
pair 2: k=2 and k=5
pair 3: k=3 and k=4
```

実作業は1リングずつ行うが、quota と mirror balance は pair 単位で見る。

### 5.2 R > 1 の場合

半径方向に複数 ring がある場合、初期実装では以下の順序にする。

```text
for each layer in outer_to_inner_layer_order:
  for ring_id in ascending order:
    assemble physical ring (layer_id, ring_id)
```

必要になれば、半径方向の重要度も導入する。

## 6. 実装ステップ見積もり

Level 3 と GUI まで含めると、10ステップ程度に分けるのが安全である。

### Step R1: assembly strategy と outer-to-inner work unit

目的:

ring-by-ring 戦略の作業単位を表現する。

実装:

- `WorkUnitMode` に以下を追加する。
  - `mirror_ring_pair`
  - `ring_by_ring_outer_to_inner`
- `build_outer_to_inner_work_units(slots, mode)` を追加する。
- physical ring key `(layer_id, ring_id)` を扱う helper を追加する。
- `K` が偶数/奇数の場合の layer order を定義する。
- `assign_work_unit_ids` を新 work unit に対応させる。

検証:

- `K=8` で order が `0,7,1,6,2,5,3,4` になる。
- `K=7` で中央 `k=3` が最後に単独になる。
- `R>1` で `(layer_id, ring_id)` が漏れなく含まれる。
- slot 重複・漏れがない。

### Step R2: ring summary と timeline data model

目的:

各 ring にどのような磁石が入ったかを記録・集計できるようにする。

実装:

- `RingKey`
- `RingSummary`
- `RingPairSummary`
- `AssemblyTimelineEvent`
- `ring_summary_from_placements(...)`
- `timeline_from_simulation_result(...)`

RingSummary の項目:

- `layer_id`
- `ring_id`
- `count`
- `mean_epsilon`
- `std_epsilon`
- `min_epsilon`
- `max_epsilon`
- `mean_angle_error`
- `std_angle_error`
- `cluster_counts`
- `mean_true_error`
- `mean_measured_error`
- `B0_norm_after_ring`
- `rms_homogeneity_ppm_after_ring`
- `J_vector_after_ring`

検証:

- 手作り placement から ring 平均が正しく計算される。
- cluster count が一致する。
- mirror pair summary の左右差が正しく出る。

### Step R3: quota planner Level 1

目的:

各 ring に割り当てる cluster quota を事前に作る。

実装:

- `RingQuotaPlan`
- `RingQuotaPlannerConfig`
- `plan_ring_cluster_quotas(...)`
- strength target 計算
- ring importance 計算
- angle quality schedule
- mirror pair balance constraint

初期 score:

```text
quota_cost =
  lambda_ring_mean * ring_mean_deviation^2
+ lambda_angle * angle_quality_penalty
+ lambda_inventory * inventory_shortage_penalty
```

検証:

- 全 ring の quota count が slot count に一致する。
- 全 inventory count を超えて quota を割り当てない。
- 各 ring の expected mean epsilon が target に近い。
- outer ring に大きい angle bin が多く、center ring に小さい angle bin が多い。

### Step R4: ring-constrained assignment Level 2

目的:

既存の線形感度割当を current work unit に制限する。

実装:

- `run_ring_constrained_linear_assignment(...)`
- `remaining_slot_flat_ids` を current ring に限定する。
- current ring が埋まったら次 work unit へ進む。
- `all_slots_global` と同じ結果経路を壊さない。
- random baseline も同じ ring order で配置できるようにする。

検証:

- placement order が work unit order に従う。
- 各 ring が完了するまで次 ring の slot が使われない。
- `all_slots_global` は既存挙動と一致する。
- random baseline と linear baseline が同じ制約下で比較される。

### Step R5: quota-driven cluster pickup simulation

目的:

「指定 cluster から1個取る」をシミュレーションに入れる。

実装:

- `ClusterPickupPolicy = quota_ordered`
- cluster ごとの magnet pool を作る。
- current ring quota に従って次 cluster を選ぶ。
- simulation ではその cluster から magnet を1個取り出す。
- 取り出した magnet の `measured_error` で slot/orientation を決める。
- inventory count と quota remaining を更新する。

検証:

- 使用 cluster count が quota と一致する。
- cluster 不足時に明確なエラーまたは fallback が発生する。
- ring mean epsilon が quota の expected mean に近づく。

### Step R6: cluster-level MPC Level 3

目的:

cluster 選択そのものを residual 補償に使う。

実装:

- `ClusterMPCConfig`
- `score_cluster_for_current_ring(...)`
- cluster mean/cov による期待 field cost
- quota deviation penalty
- ring mean penalty
- angle quality penalty
- future inventory reserve penalty
- mirror pair balance penalty

cluster expected score:

```text
field_cost(c) =
  min_{slot, orientation in current ring}
    ||residual + C @ mean_c||^2 + trace(C cov_c C^T)
```

検証:

- synthetic case で residual と逆向きに効く cluster が選ばれる。
- center reserve を有効にすると、良品 cluster が外側で使われすぎない。
- `lambda_future=0` では greedy に近い挙動になる。

### Step R7: mirror pair balancing and compensation

目的:

両端から中心へ進むとき、mirror pair の左右差を抑えつつ、内側 ring で外側 ring の誤差を補償する。

実装:

- mirror pair 状態を管理する。
- pair 内の左右 ring mean 差を penalty に入れる。
- pair 完了時に pair summary を出す。
- outer rings の residual を inner rings へ引き継ぐ。
- optional: pair 内で left/right の順序を入れ替える strategy。

検証:

- pair 左右の expected mean epsilon 差が penalty で小さくなる。
- penalty をゼロにすると差が大きくなりうることを synthetic test で確認する。
- outer residual を打ち消す方向に inner ring cluster/slot が選ばれる。

### Step R8: self-consistent integration for ring strategy

目的:

ring-by-ring 戦略でも `sequential_self_consistent` と `self_consistent_from_run` 評価を使えるようにする。

実装:

- ring-constrained top-k candidate を JAX self-consistent で再評価する。
- final evaluation は既存 `self_consistent_from_run` JAX backend を使う。
- ring completion ごとの評価は高コストなので optional にする。
- cache または評価頻度設定を追加する。

検証:

- `linear_sensitivity` と `sequential_self_consistent` の配置順制約が同一。
- final evaluation metadata に JAX backend が残る。
- ring-by-ring でも既存 Plan C self-consistent tests が通る。

### Step R9: outputs for visualization

目的:

GUIなしでも、ring-by-ring の進行を解析できるファイルを出力する。

実装:

- `ring_summary_trial_000.csv`
- `ring_pair_summary_trial_000.csv`
- `assembly_timeline_trial_000.jsonl`
- `cluster_quota_plan_trial_000.csv`
- `cluster_pickup_log_trial_000.csv`

timeline event 例:

```json
{
  "step": 123,
  "event": "insert_confirmed",
  "work_unit_id": "W_K000_R000",
  "layer_id": 0,
  "ring_id": 0,
  "theta_id": 57,
  "cluster_requested": "S02_A04",
  "epsilon_parallel": -0.008,
  "angle_error": 0.0031,
  "orientation_id": "O180",
  "residual_norm": 1.2e-8,
  "ring_mean_epsilon_so_far": -0.0003
}
```

検証:

- CSV/JSONL が placement 数と一致する。
- ring summary の count 合計が placement 数に一致する。
- timeline を読み戻して最終配置を復元できる。

### Step R10: Streamlit GUI and playback

目的:

リングが1枚ずつ組み上がる様子と、各 ring に割り当てられた磁石誤差を観察できる GUI を作る。

実装:

- `app/pages/plan_c_ring_simulation.py`
- run selector
- strategy selector
- cluster config panel
- simulation launch panel
- playback viewer
- ring summary viewer
- inventory heatmap
- final comparison panel

検証:

- small test output を読み込んでGUIが表示できる。
- playback slider で step/ring を切り替えられる。
- active ring の slot 色分けが更新される。
- ring summary heatmap が CSV と一致する。

## 7. GUI構成案

### 7.1 Page layout

```text
Plan C Ring Simulation

Sidebar:
  - Run selection
  - Assembly strategy
  - Work unit mode
  - Strength cluster count
  - Angle cluster count
  - Quality / quarantine settings
  - MPC weights
  - Evaluation model
  - Trial count / seed

Main:
  Tab 1: Simulation Summary
  Tab 2: Assembly Playback
  Tab 3: Ring Error Map
  Tab 4: Cluster Inventory
  Tab 5: Trial Files / Logs
```

### 7.2 Simulation Summary

表示:

- random vs ring-by-ring Plan C の RMS ppm
- random vs ring-by-ring Plan C の `J_vector`
- self-consistent evaluation の有無
- final B0 norm
- improvement ratio
- ring count
- used magnet count
- quarantined count

### 7.3 Assembly Playback

目的:

リングが1枚ずつ組み上がる様子を見る。

UI:

- playback slider
  - step 単位
  - ring 単位
- play/pause button
- current ring indicator
- current cluster instruction
- current residual norm

表示:

1. Side stack view
   - z方向の layer を縦に並べる。
   - completed ring は色付き。
   - active ring は枠線で強調。
   - ring color は `mean_epsilon` または `mean_angle_error`。

2. Active ring polar view
   - 円周 slot を polar plot で表示。
   - slot color は配置済み磁石の `epsilon_parallel`。
   - marker size は angle error。
   - marker outline は orientation。
   - hover で cluster, magnet id, measured/true error を表示。

3. Residual trend
   - insert step に対する residual norm。
   - ring completion points を縦線で表示。

### 7.4 Ring Error Map

目的:

どの ring にどの程度のエラー磁石が入ったかを見る。

表示:

- layer_id x ring_id の heatmap
  - mean epsilon
  - std epsilon
  - mean angle error
  - angle error std
  - cluster entropy
  - RMS ppm after ring
- mirror pair imbalance plot
  - mean epsilon left-right difference
  - angle error left-right difference

### 7.5 Cluster Inventory

目的:

強度 x 角度誤差の 2D クラスタ在庫が、外側から中心へどう消費されたかを見る。

表示:

- 10 x 5 heatmap
  - initial count
  - remaining count
  - used count
  - reserved for center
- selected ring の quota
- actual usage vs planned quota
- shortage warning

### 7.6 Trial Files / Logs

表示:

- output directory
- summary json
- placement csv
- ring summary csv
- timeline jsonl
- cluster quota plan csv

## 8. CLI案

既存 `plan_c_simulate.py` に以下を追加する。

```powershell
--assembly-strategy all_slots_global|ring_by_ring_outer_to_inner
--work-unit-mode single_physical_ring|mirror_ring_pair|all_slots
--strength-clusters 10
--angle-clusters 5
--cluster-pickup-policy quota_ordered|cluster_mpc
--ring-quality-profile outer_to_inner
--lambda-ring-mean 1.0
--lambda-angle-quality 1.0
--lambda-mirror-balance 1.0
--lambda-future-inventory 1.0
--ring-eval-frequency final|per_ring|per_pair
```

デフォルト:

```text
assembly_strategy = all_slots_global
```

既存挙動を壊さない。

大規模向け推奨:

```powershell
.\.venv\Scripts\python.exe -m halbach.cli.plan_c_simulate `
  --run runs\... `
  --out runs\...\plan_c_ring `
  --assembly-strategy ring_by_ring_outer_to_inner `
  --work-unit-mode single_physical_ring `
  --strength-clusters 10 `
  --angle-clusters 5 `
  --cluster-pickup-policy cluster_mpc `
  --evaluation-model self_consistent_from_run
```

## 9. 評価指標

### 9.1 全体評価

- final RMS homogeneity ppm
- final `J_vector`
- final B0 norm
- random baseline ratio
- all-slots Plan C ratio
- ring-by-ring Plan C ratio

### 9.2 ring別評価

- ring mean epsilon
- ring std epsilon
- ring mean angle error
- ring std angle error
- ring cluster usage
- ring residual norm after completion
- ring completion RMS ppm

### 9.3 mirror pair評価

- left/right mean epsilon difference
- left/right angle error difference
- pair completion residual norm
- pair completion RMS ppm

### 9.4 inventory評価

- cluster initial count
- cluster used count
- cluster remaining count
- center reserve violation
- quota deviation
- rejected count
- quarantine count

## 10. 受け入れ条件

### 10.1 物理・数式不変条件

- 既存の幾何最適化、目的関数、保存形式を変更しない。
- Plan C は最適化済み run を読む後段機能であり、最適化変数を変更しない。
- `self_consistent_from_run` 評価では、最適化時 metadata に基づく JAX backend を使う。
- 既存 `all_slots_global` の結果は、追加実装で変わらない。

### 10.2 ring-by-ring 条件

- `ring_by_ring_outer_to_inner` では、current ring 外の slot を使わない。
- ring order が deterministic である。
- 各 ring の placement count が slot count と一致する。
- ring summary の count 合計が placement count と一致する。
- quota-driven mode では、使用 cluster count が quota と一致または明示的 fallback として記録される。

### 10.3 GUI条件

- simulation output だけから playback が再現できる。
- ring単位で mean epsilon / angle error / cluster usage が見える。
- active ring の slot配置が polar view で確認できる。
- cluster inventory の初期、使用済み、残数が見える。

## 11. 推奨実装順

最初に実装すべき最小セットは以下である。

```text
R1 -> R2 -> R3 -> R4 -> R5
```

この時点で、以下が可能になる。

- リング1枚ずつの配置制約
- 外側から中心への組み立て順
- ring summary 出力
- quota による cluster 消費
- 小規模 all-slots 方式との比較

その後、Level 3 の性能改善として以下を実装する。

```text
R6 -> R7 -> R8
```

最後に可視化・実運用支援として以下を実装する。

```text
R9 -> R10
```

## 12. 未決事項

以下は実装前に、シミュレーション結果を見ながら調整する。

- `ring_importance` の具体的な関数形。
- 端 ring にどの程度大きな angle error cluster を許容するか。
- 中心 ring 用 reserve の割合。
- `lambda_*` の初期値。
- mirror pair を完全に pair work unit とするか、quotaだけ pair で実作業は single ring にするか。
- `R > 1` の半径方向 ring order。
- 再測定で cluster が変わった磁石の扱い。
- ring completion ごとの self-consistent 評価頻度。

初期実装では、作業性と比較再現性を優先し、複雑な局所探索や長い horizon MPC は後回しにする。
