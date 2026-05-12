# Plan C Specification Final: Clustered Online Assignment for Halbach Magnet Assembly

## 0. 文書の位置づけ

本仕様書は、`halbach-optimizer` に追加する「案C: クラスタ箱分け + 組立時再測定 + オンラインMPC割当」の最終仕様案である。

案Cは、数千個規模の立方体磁石をすべて個別ID管理して広い作業スペースに展開することを避けながら、実測された磁気モーメント強度・磁化方向のばらつきがROI磁場均一度を劣化させることを抑えるための組立方式である。

本仕様書は、後続のCodex実装指示書を作るための上位仕様として、以下を定義する。

- 案Cを採用する背景
- 案Cの作業手順
- クラスタリング仕様
- 磁石の測定・マーキング・姿勢候補仕様
- 線形感度モードと逐次自己無撞着モード
- シミュレーションモード
- 実組立モード
- Streamlit UI要件
- データ構造
- 評価指標
- 受け入れ基準

---

## 1. 背景

### 1.1 現行プロジェクトの前提

`halbach-optimizer` は、円筒状のハルバッハ配列に多数の立方体磁石を配置し、ROI内に均一な磁場を発生させるための幾何・着磁角度最適化コードである。

案Cは、現行の幾何最適化結果を前提として、その後段に追加される「実測磁石をどのスロットへ、どの姿勢で入れるか」を扱う機能である。

案Cは、ハルバッハ配列の幾何寸法、半径プロファイル、角度補正パラメータを再最適化する機能ではない。

### 1.2 製造上の問題

現実の立方体磁石には、以下のばらつきがある。

- 磁気モーメント強度のばらつき
- 磁化方向のずれ
- ロット差
- 取り出し順に伴う中心値ドリフト
- 外れ値
- 測定ノイズ

理想的には、必要な全磁石、例えば2000個程度を測定し、個別IDを振り、全磁石を一括で組合せ最適化し、指定スロットへ個別に挿入するのが最適に近い。

しかし、この方法には以下の問題がある。

- 数千個の磁石を個別ID管理する必要がある。
- 測定後の磁石を、後で取り出せるように広い作業スペースへ展開する必要がある。
- 強磁性体を大量に並べるため、吸着・取り違え・破損のリスクが高い。
- 作業者が個体番号を取り違えるリスクが高い。
- 組立作業が複雑になり、人間の負担が大きい。

案Cは、個体ID管理を不要にしながら、測定情報をクラスタ在庫として利用して磁場均一度を高めるための手法である。

---

## 2. 案Cの基本方針

### 2.1 案Cの要約

案Cでは、磁石を個別IDで管理せず、事前測定結果に基づいてクラスタ箱へ分ける。

組立時には、ソフトウェアが作業者へ次を逐次指示する。

1. どのクラスタ箱から磁石を1個取るか。
2. その磁石を再測定した結果が受理可能か。
3. 鏡像ペアのうち、どちらのリングのどの番号付きスロットへ入れるか。
4. 4姿勢候補のうち、どの姿勢で入れるか。

作業者は、基本的に以下を繰り返す。

```text
指定クラスタ箱から1個取る
→ 測定器にセットして再測定する
→ UIに表示されたリング・スロット・姿勢へ挿入する
→ 完了ボタンを押す
```

### 2.2 目的

案C実装の目的は以下である。

1. 実測磁石の強度・方向ばらつきを考慮し、ROI磁場均一度の劣化を抑える。
2. 全磁石の個別ID管理を不要にする。
3. 全磁石を広い作業スペースに展開することを不要にする。
4. 大型配列ではリング1枚ずつ組み立てられるようにする。
5. 小規模配列では、複数リングまたは全リングをまとめた作業単位で組み立てられるようにする。
6. 鏡像関係にあるリングペアをUI上で2枚同時に表示し、作業位置を明確にする。
7. 鏡像リングペアの上下入れ替え判断を支援する。
8. 強度クラスタ数と方向誤差クラスタ数を可変にする。
9. 外れ値を最大10%程度まで隔離・除外できるようにする。
10. 実運用前に、案Cで得られる磁場均一度をシミュレーション評価できるようにする。
11. 実組立時には、Streamlit UIで作業者へ逐次指示を出せるようにする。

### 2.3 最優先指標

最優先する性能指標はROI磁場均一度である。

中心磁場強度の絶対値は、組立時の割当最適化では主目的にしない。中心磁場強度はログ・参考指標として記録してよいが、割当判断では、均一な強度スケール変化よりもROI内の空間的不均一を優先して抑える。

### 2.4 非目標

案Cは以下を目的としない。

- ハルバッハ配列の幾何寸法を再最適化すること。
- `alpha`、Fourier係数、半径プロファイルを再設計すること。
- 案A、案Bを比較実装すること。
- 全磁石に個体IDを付けること。
- 全磁石の厳密な一括組合せ最適化を行うこと。
- 中心磁場強度を目標値へ精密に合わせること。

---

## 3. 実装対象と比較対象

### 3.1 比較対象

シミュレーションモードでは、以下のみを比較対象にする。

1. ランダム配置
2. 案C: クラスタ付き逐次MPC、線形感度モード
3. 案C: クラスタ付き逐次MPC、線形感度モード + 外れ値隔離
4. 案C: クラスタ付き逐次MPC、逐次自己無撞着モード
5. 案C: クラスタ付き逐次MPC、逐次自己無撞着モード + 外れ値隔離

案Aと案Bは比較対象から外す。

### 3.2 ランダム配置

ランダム配置は、案Cによる改善度を評価するためのベースラインである。

ランダム配置では、使用対象磁石をスロットへランダムに割り当てる。外れ値隔離を有効にした比較では、同じ隔離条件をランダム配置にも適用できるようにする。

### 3.3 案Cのオンライン判断エンジン

案Cには、オンライン割当の評価エンジンとして以下の2モードを実装する。

#### `linear_sensitivity`

事前計算したスロット・姿勢ごとの感度を用い、組立中の残留誤差を線形更新する。

利点:

- 高速である。
- Streamlit UIでの逐次応答に適する。
- 多数の仮想候補・局所探索を評価しやすい。

欠点:

- 大きなばらつきや強い自己相互作用がある場合、非線形効果を近似しきれない可能性がある。

#### `sequential_self_consistent`

各挿入ステップまたは候補評価で、自己無撞着 easy-axis モデルを逐次的に再計算し、配置判断に使う。

利点:

- 近接相互作用や磁化強度変化の影響をより直接的に扱える。
- 線形感度近似の妥当性検証にも使える。

欠点:

- 計算コストが高い。
- UI応答時間が長くなる可能性がある。
- 候補全探索には向かないため、候補削減・キャッシュ・ウォームスタートが必要である。

初期実装では、実作業での標準モードは `linear_sensitivity`、検証・高精度シミュレーション用に `sequential_self_consistent` を使うことを推奨する。

---

## 4. 用語定義

### 4.1 スロット

スロットとは、設計済みハルバッハ配列における1個の磁石の組込位置である。

スロットは以下で識別する。

```text
slot_id = (ring_id, layer_id, theta_id)
```

または、平坦化された整数IDで識別する。

- `ring_id`: 半径方向リング番号
- `layer_id`: 軸方向レイヤ番号
- `theta_id`: 円周方向番号

UIでは、作業者が確認しやすいように、すべての物理リングの枠に作業用スロット番号を付ける。

### 4.2 物理リング

本仕様書で「物理リング」と呼ぶ場合、xy平面上に存在する1枚のリング状組立単位を意味する。

大型配列では、1物理リングあたり磁石数が約100個になることを想定する。この場合は1リングずつ組み立てる。

小規模配列では、全体磁石数が100個程度の場合がある。この場合、1リング単位では磁石数が少なすぎるため、複数リングまたは全リングを1つの作業単位として扱えるようにする。

### 4.3 作業単位

作業単位は、オンラインMPC割当を実行する対象スロット集合である。

設定値:

```text
work_unit_mode = single_physical_ring
work_unit_mode = mirror_pair
work_unit_mode = ring_group
work_unit_mode = all_slots
work_unit_mode = custom
```

#### `single_physical_ring`

大型配列向け。1物理リングを1作業単位とする。

#### `mirror_pair`

鏡像関係にある2枚のリングを連続した作業単位として扱う。UIでは2枚を同時に表示し、現在作業中のリングを強調する。

#### `ring_group`

複数リングをまとめて1作業単位とする。小規模配列または1リングあたり磁石数が少ない場合に使う。

#### `all_slots`

全スロットを1作業単位として扱う。全体磁石数が100個程度の小規模配列で使用する。

#### `custom`

ユーザーがCSVまたはJSONで作業単位を明示的に指定する。

初期設定では、以下の自動判定を推奨する。

```text
if magnets_per_physical_ring >= 60:
    work_unit_mode = single_physical_ring
elif total_magnets <= 150:
    work_unit_mode = all_slots
else:
    work_unit_mode = ring_group
```

この閾値は設定ファイルで変更可能にする。

### 4.4 鏡像リングペア

軸方向に対して鏡像関係にある2つのリングを、鏡像リングペアと呼ぶ。

例:

```text
+z_k ring
-z_k ring
```

鏡像位置関係にあるペアは、完成後に上下位置を入れ替え可能とする。ただし、直径や機械制約により入れ替え可能なペアだけを登録する。

### 4.5 磁石測定値

測定器から得る値は以下を想定する。

```text
moment_magnitude
moment_direction_vector
measurement_quality(optional)
timestamp(optional)
```

個体IDは不要である。

ただし、組立ログには「どのスロットへ、どの測定値の磁石を入れたか」を記録する。これは完成品の品質記録であり、磁石個体IDではない。

### 4.6 磁石誤差ベクトル

磁石の誤差は、名目着磁方向を基準とする局所座標で表す。

```text
x = [epsilon_parallel, delta_perp_1, delta_perp_2]
```

- `epsilon_parallel`: 磁気モーメント強度の相対誤差
- `delta_perp_1`: 名目着磁方向に直交する第1横成分
- `delta_perp_2`: 名目着磁方向に直交する第2横成分

方向ずれが小さい場合、角度ラジアンと横成分比はほぼ等しいため、小角近似で扱ってよい。

### 4.7 測定ノイズ

初期シミュレーションでは、各測定成分の標準偏差を0.1%程度と仮定する。

既定値:

```yaml
measurement_noise:
  strength_sigma: 0.001
  transverse_component_1_sigma: 0.001
  transverse_component_2_sigma: 0.001
```

方向成分の0.001は、横成分比または小角近似で約0.001 radに相当する。

---

## 5. 磁石マーキングと4姿勢仕様

### 5.1 測定時の座標系

個々の磁石は、測定器にセットする際、名目磁化方向が測定器座標系の上方向、すなわち `+Z_meas` 方向を向くように置く。

測定器座標系は以下を持つ。

```text
+Z_meas: 名目磁化方向、上方向
+X_meas: 測定器上の左右方向など、治具で定義する基準方向
+Y_meas: 測定器上の前後方向など、治具で定義する基準方向
```

実際の磁石の磁化方向は、完全には `+Z_meas` を向かず、わずかな横ずれを持つ。

測定値は、以下のように分解する。

```text
moment_magnitude = |m|
unit_moment_meas = [u_x, u_y, u_z]
```

小角近似では、

```text
delta_x ≈ u_x / u_z
delta_y ≈ u_y / u_z
```

として横ずれ成分を扱う。

### 5.2 マーキング手順

測定時の姿勢を基準姿勢とし、測定器にセットした状態で作業者から見て手前となる基準面に、磁化方向を示す矢印マーキングを行う。

このマーキングは、以下の目的で使う。

- 測定時の磁石座標系を組立時にも再現する。
- 磁石をスロットへ入れる際、4姿勢のうちどれを使うかを作業者が識別する。
- 磁化方向誤差の横成分を、スロット座標系内で0度、90度、180度、270度に回転させて扱えるようにする。

マーキング面と矢印の具体的な作業表現は、治具・測定器の実物に合わせて設定ファイルで定義する。

### 5.3 4姿勢の物理的意味

理想的に磁化した磁石であれば、スロットの設計着磁方向に合わせて挿入すればよい。

しかし実際の磁石には、名目磁化方向からの小さな角度誤差がある。この角度誤差を持つ磁石をそのまま全て同じ姿勢で挿入すると、ROI磁場均一度を劣化させる低次モードが残る可能性がある。

そこで案Cでは、磁石の名目磁化軸、すなわち測定時の `+Z_meas` 軸まわりに、以下の4姿勢を許す。

```text
O0
O90
O180
O270
```

これにより、磁石の主磁化成分はスロットの理想着磁方向に合わせつつ、横ずれ誤差成分だけをスロット座標系内で4方向に回転させることができる。

4姿勢は以下の回転角で定義する。

```text
O0   : ψ = 0°
O90  : ψ = 90°
O180 : ψ = 180°
O270 : ψ = 270°
```

横ずれベクトルを

```text
d = [delta_x, delta_y]
```

とすると、姿勢 `o` における横ずれは、

```text
d_o = Rz(ψ_o) d
```

である。

ここで `Rz(ψ_o)` は、測定時の名目磁化軸まわりの2次元回転である。

### 5.4 感度モデル内での姿勢の扱い

スロット `s` の名目着磁方向を `u_s` とする。スロットごとに、`u_s` に直交する2つの横方向基底を定義する。

```text
e_parallel = u_s
e_perp_1
e_perp_2
```

磁石 `j` を姿勢 `o` でスロット `s` に入れる場合、磁石誤差ベクトルは以下になる。

```text
x_{j,o} = [epsilon_parallel_j, delta_perp_1_{j,o}, delta_perp_2_{j,o}]
```

線形感度モードでは、寄与を以下で評価する。

```text
residual += C_s @ x_{j,o}
```

または、実装上都合がよければ、姿勢込みの感度行列を事前計算して、

```text
residual += C_{s,o} @ x_j
```

としてもよい。

### 5.5 UIでの姿勢指示

UIでは、内部ID `O0/O90/O180/O270` だけを表示してはならない。作業者が物理的に理解できる説明を併記する。

表示例:

```text
姿勢: O90
マーキング矢印をリング枠の青い基準マークから時計回りに90°回転した向きで挿入
```

実際の文言は、測定器・治具・リング枠のマーキング方式に合わせて設定ファイルで定義する。

設定例:

```yaml
orientations:
  mode: discrete4
  rotation_axis: measured_nominal_magnetization_axis
  candidates:
    - id: O0
      angle_deg: 0
      instruction: "マーキング矢印をスロット基準マークに合わせる"
    - id: O90
      angle_deg: 90
      instruction: "マーキング矢印をスロット基準マークから時計回りに90度"
    - id: O180
      angle_deg: 180
      instruction: "マーキング矢印をスロット基準マークと反対向き"
    - id: O270
      angle_deg: 270
      instruction: "マーキング矢印をスロット基準マークから反時計回りに90度"
```

### 5.6 将来拡張

磁化方向誤差が想定より大きく、4姿勢では補償力が不足する場合は、以下へ拡張可能にする。

```text
orientation_mode = discreteN
orientation_mode = continuous
```

ただし、作業ミスを避けるため、初期実装と標準UIでは `discrete4` を用いる。

---

## 6. クラスタリング仕様

### 6.1 クラスタリングの目的

クラスタリングの目的は、個体IDなしで未来磁石の分布を狭め、オンラインMPC割当の予測精度を上げることである。

案Cでは、事前測定によって磁石をクラスタ箱へ分配する。組立時には、ソフトが次に使うクラスタ箱を指定するため、次に来る磁石の分布が全体分布ではなくクラスタ内分布になる。

### 6.2 強度クラスタ

強度誤差 `epsilon_parallel` を5〜10種程度に分ける。

推奨初期設定:

```yaml
clusters:
  strength:
    mode: quantile
    count: 10
```

小規模配列または磁石数が少ない場合は、5〜6クラスタへ減らす。

固定閾値方式も許可する。

```yaml
clusters:
  strength:
    mode: thresholds
    thresholds: [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
```

初期実装では、クラスタ在庫が偏りにくい分位点方式を標準とする。

### 6.3 方向誤差クラスタ

磁化方向のずれを2〜5種程度に分ける。

推奨初期設定:

```yaml
clusters:
  angle:
    mode: transverse_norm_quantile
    count: 3
    transverse_2_weight: 1.0
```

方向誤差ノルムは以下で定義する。

```text
angle_score = sqrt(delta_x^2 + transverse_2_weight * delta_y^2)
```

測定器座標系の2つの横方向で誤差影響に差がある場合は、重みを設定できるようにする。

方向誤差が実測上ほとんど問題にならない場合は、方向クラスタ数を2にしてよい。

方向誤差が大きい場合は、以下のように横方向2成分を別々に分類する拡張を許可する。

```yaml
clusters:
  angle:
    mode: separate_transverse_components
    transverse_1_count: 3
    transverse_2_count: 3
```

### 6.4 クラスタID

通常クラスタIDは以下とする。

```text
S{strength_bin}_A{angle_bin}
```

例:

```text
S03_A01
S07_A00
```

特殊クラスタ:

```text
Q_DIRECTION_OUTLIER
Q_STRENGTH_OUTLIER
Q_MEASUREMENT_UNSTABLE
REJECTED
RESERVE
```

### 6.5 外れ値隔離

最大10%程度まで磁石を隔離・除外できる。

外れ値処理は、単純な強度上位・下位の切り捨てではなく、以下の優先順位で行う。

1. 測定が不安定な磁石
2. 方向誤差が大きい磁石
3. 極端な強度外れ
4. クラスタ在庫計画上、使いにくい磁石

隔離磁石は通常は使わない。ただし、シミュレーション上の選択肢として、補償に有利な場合に隔離箱から復帰使用するオプションを用意してもよい。

初期実装では、以下の2モードで十分とする。

```text
outlier_policy = none
outlier_policy = isolate_up_to_fraction
```

`isolate_up_to_fraction` の既定最大値:

```yaml
reject:
  max_fraction: 0.10
```

### 6.6 組立時再測定

クラスタ箱から取り出した磁石は、挿入前に必ず再測定することを標準手順とする。

目的:

- 箱分け時の測定誤差を補正する。
- 箱境界付近の誤分類を許容する。
- 箱への混入を検出する。
- 挿入時の実測値をログに残す。
- オンライン割当の精度を上げる。

再測定値がクラスタの想定範囲から大きく外れた場合、Streamlit UIで以下の選択肢を表示する。

- 測定をやり直す。
- 現在値で使用する。
- 別クラスタとして扱い直す。
- 隔離箱へ移す。

---

## 7. 物理モデルとオンライン判断エンジン

### 7.1 共通の残留誤差表現

案Cでは、すでに挿入済みの磁石がROI均一度へ与える累積誤差を `residual` として管理する。

線形感度モードでは、以下のように更新する。

```text
residual += C_s @ x_{j,o}
```

または、姿勢込み感度を使う場合、

```text
residual += C_{s,o} @ x_j
```

とする。

逐次自己無撞着モードでは、`residual` はキャッシュされたフル磁場評価またはROI均一度評価結果として扱う。

### 7.2 ROI均一度誤差

ROI点群における磁場差分を基本量とする。

```text
y = [B(p_1) - B0, B(p_2) - B0, ..., B(p_M) - B0]
```

中心磁場強度の一様変化を主目的から外すため、必要に応じて射影 `P` を用いる。

```text
y_homogeneity = P @ y
```

目的関数:

```text
J = ||y_homogeneity||^2
```

### 7.3 線形感度モード

#### 7.3.1 事前感度計算

スロット `s` について、強度・方向誤差に対する感度行列を計算する。

```text
C_s = [d y / d epsilon_parallel,
       d y / d delta_perp_1,
       d y / d delta_perp_2]
```

差分または自動微分・解析微分で求める。

初期実装では、差分法でよい。

姿勢は、誤差ベクトル側を回転して扱うか、姿勢込み感度 `C_{s,o}` として事前保存する。

#### 7.3.2 低次元化

ROI点が多い場合、`y_homogeneity` は高次元になる。オンライン割当では、SVD/PCA等で低次元化する。

推奨:

```yaml
sensitivity:
  dimension: 10..50
```

初期値:

```yaml
sensitivity:
  dimension: 30
```

#### 7.3.3 利用場面

線形感度モードは、以下で使用する。

- 標準シミュレーション
- 実組立時の標準オンライン割当
- 多数候補の局所探索
- Streamlit UIでの逐次応答

### 7.4 逐次自己無撞着モード

#### 7.4.1 基本方針

逐次自己無撞着モードでは、各挿入ステップで候補配置を評価する際に、自己無撞着 easy-axis モデルを使ってROI均一度を再評価する。

目的は、線形感度近似では捉えにくい以下の効果を評価することである。

- 近傍磁石との相互作用による磁化強度変化
- 配置済み磁石の実測ばらつきによる局所場変化
- 方向ずれが自己無撞着解へ与える影響

#### 7.4.2 候補評価の制限

全空きスロット・全姿勢・全仮想未来配置に対してフル自己無撞着計算を行うと、計算量が大きすぎる可能性が高い。

そのため、逐次自己無撞着モードでは以下の2段階評価を標準とする。

1. 線形感度モデルで候補を粗く絞る。
2. 上位 `K` 候補だけを自己無撞着計算で再評価する。

設定例:

```yaml
self_consistent_mpc:
  candidate_pruning: linear_top_k
  top_k_slots: 8
  top_k_orientations_per_slot: 2
  max_sc_evaluations_per_step: 16
```

#### 7.4.3 逐次更新

各挿入確定後、以下を更新する。

- スロット占有状態
- 実測磁石誤差
- 姿勢
- クラスタ在庫
- 現在の自己無撞着解 `p_i`
- ROI磁場評価
- 次ステップ用キャッシュ

自己無撞着ソルバは、前ステップの解を初期値としてウォームスタートする。

---

## 8. オンラインMPC割当アルゴリズム

### 8.1 基本方針

案Cのオンライン割当は、現在測定した磁石と、未来に使う予定のクラスタ分布から作った仮想磁石群を使って、残り空きスロットに対する仮想バッチ割当を解く。

その仮想最適解のうち、現在の実測磁石に割り当てられたスロット・姿勢だけを実際に採用する。

### 8.2 未来仮想磁石の生成

残りクラスタ使用数を `h[b]` とする。

各クラスタ `b` について、クラスタ分布 `F_b` から代表磁石を作る。

初期実装では、以下のいずれかを選べるようにする。

```text
future_virtual_mode = cluster_mean
future_virtual_mode = quantile
future_virtual_mode = bootstrap
```

推奨初期値:

```text
future_virtual_mode = quantile
```

多次元誤差ベクトルの場合、初期実装ではクラスタ平均の複製でもよい。ただし、シミュレーションでは `bootstrap` も実装しておくと、分布の裾を評価しやすい。

### 8.3 仮想バッチ割当問題

現在磁石 `x_now` と未来仮想磁石 `q_l` を合わせて、残り空きスロット `R` に割り当てる。

目的関数:

```text
minimize J(residual + contribution(current) + contribution(future))
```

線形感度モードでは、

```text
contribution = C_s @ x_{j,o}
```

または、

```text
contribution = C_{s,o} @ x_j
```

とする。

逐次自己無撞着モードでは、候補配置ごとに自己無撞着解を求めて `J` を評価する。

制約:

- 各空きスロットは1回だけ使う。
- 各仮想磁石は1回だけ使う。
- 許可された挿入姿勢だけを使う。
- 現在磁石は必ず1つのスロットへ割り当てる。

### 8.4 局所探索

リングまたは作業単位が100個程度であれば、局所探索で十分とする。

手順:

1. 初期配置を作る。
2. 2点交換で改善する。
3. 必要に応じて姿勢変更も候補に入れる。
4. 複数初期値から最良解を選ぶ。

線形感度モードでは、2点交換の差分を高速計算する。

方向誤差を含む場合:

```text
Δresidual = C_a @ x_{b,o_b} + C_b @ x_{a,o_a}
          - C_a @ x_{a,o_a} - C_b @ x_{b,o_b}
```

逐次自己無撞着モードでは、線形感度の局所探索で得た候補群を自己無撞着再評価する。

### 8.5 次に使うクラスタ箱の選択

次に使うクラスタ `b` は、以下を考慮して選ぶ。

- 作業単位内での残り使用予定数
- グローバル在庫計画からの逸脱
- 現在の残留誤差を打ち消しやすいか
- 極端なクラスタを後半まで残しすぎていないか
- 中庸クラスタを最後の調整用に残せるか

初期実装のヒューリスティック:

1. 残り使用予定数が0のクラスタは候補外。
2. 前半では極端な強度・方向クラスタを優先する。
3. 中盤では現在残留を減らす期待値が高いクラスタを優先する。
4. 後半では中庸クラスタを残しすぎないように調整する。

---

## 9. Streamlit UI仕様

### 9.1 UIの基本方針

Streamlit UIは、シミュレーションモードと実組立モードで同じ作業フローを持つ。

違いは、測定値の取得元だけである。

```text
simulation_step_by_step: 仮想磁石から測定値を生成
simulation_auto_run    : 仮想磁石を自動測定・自動挿入
assembly_real          : 実測定器または手入力から測定値を取得
```

### 9.2 鏡像リングペア表示

UIでは、鏡像ペアとなるリングを2枚分表示する。

表示要件:

- 左右または上下に、ペアの2リングを同時表示する。
- 各リングの全スロット番号を表示する。
- 現在作業中のリングを強調する。
- 挿入候補スロットを強調する。
- 挿入済みスロット、空きスロット、現在推奨スロットを色や記号で区別する。
- 現在の磁石に対する推奨姿勢を表示する。
- ペアのもう一方のリングの進捗も同時に表示する。

大型リングの場合:

- xy平面上の円周スロットマップを表示する。
- theta番号または作業用スロット番号を表示する。
- 推奨スロットを大きく強調する。

小規模配列または複数リング作業単位の場合:

- ring/layer/thetaを表形式で表示する。
- 必要に応じて2Dマップと表を併用する。

### 9.3 1ステップで表示する内容

各挿入ステップで、UIは以下を表示する。

```text
現在のモード
現在の作業単位
鏡像ペアID
現在作業中のリングID
残りスロット数
次に取るクラスタ箱
クラスタ在庫残数
再測定値
測定値の受理・再測定・隔離ボタン
推奨リング
推奨スロット番号
推奨姿勢 O0/O90/O180/O270
姿勢の作業者向け説明
完了ボタン
やり直しボタン
セッション一時停止ボタン
現在の予測均一度指標
外れ値隔離数
ログ保存状態
```

### 9.4 作業者への指示文の例

UIは、以下のような作業指示を生成する。

```text
1. クラスタ S06_A01 から磁石を1個取り出してください。
2. 測定器に、磁化方向が上向きになるようにセットしてください。
3. 測定結果を確認しました。
4. 鏡像ペア P03 のリング B、スロット 42 に挿入してください。
5. 姿勢は O90 です。
   マーキング矢印をスロット基準マークから時計回りに90度回転した向きにしてください。
6. 挿入後、完了ボタンを押してください。
```

### 9.5 UI状態

実組立モードは、以下の状態を持つ。

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

作業単位内では以下のサブ状態を持つ。

```text
CHOOSE_CLUSTER
WAIT_FOR_MAGNET_MEASUREMENT
VALIDATE_MEASUREMENT
SOLVE_PLACEMENT
WAIT_FOR_INSERT_CONFIRMATION
UPDATE_STATE
```

### 9.6 Undo / やり直し

実組立中の誤操作に備え、直近ステップのUndoをサポートする。

Undoに必要な情報:

- 直前に挿入したスロット
- 直前の測定値
- 直前のクラスタ使用数
- 直前の残留誤差
- 直前の自己無撞着キャッシュ

作業者が物理的に磁石を抜き取れない場合、Undoは使わない。

### 9.7 セッション再開

セッション状態は各ステップ後に保存する。

必要要件:

- 停止後に同じ状態から再開できる。
- 既に挿入済みのスロットを重複使用しない。
- クラスタ在庫を復元できる。
- ログ破損時に警告する。
- シミュレーションと実組立で同じ状態機械を使える。

---

## 10. 測定器インターフェース

### 10.1 基本方針

測定器は将来的にシリアル通信で接続する予定である。ただし測定器は今後作成されるため、初期実装では測定値入力を抽象化する。

測定値プロバイダ:

```text
measurement_provider = synthetic
measurement_provider = manual
measurement_provider = csv
measurement_provider = serial
```

#### `synthetic`

シミュレーション用。仮想磁石から測定値を生成する。

#### `manual`

測定値をStreamlitフォームに手入力する。

#### `csv`

CSVから順番に測定値を読み込む。

#### `serial`

シリアルポートから測定値を読む。

### 10.2 シリアル通信の仮仕様

設定例:

```yaml
measurement:
  provider: serial
  serial:
    port: COM3
    baudrate: 115200
    timeout_s: 2.0
    line_format: json
```

初期実装ではJSON line形式を仮仕様とする。

例:

```json
{"moment_magnitude": 1.0021, "direction": [0.002, -0.001, 0.999997], "quality": 0.98}
```

ソフト側で、測定時の基準方向 `+Z_meas` に対する強度誤差と横ずれ成分へ変換する。

---

## 11. 作業手順

### 11.1 全体手順

1. 現行 `halbach-optimizer` でハルバッハ配列の幾何・角度を最適化する。
2. 最適化済みrunからスロット情報を読み出す。
3. 各スロットの感度を計算する。
4. 4姿勢候補を考慮できるように、姿勢変換または姿勢込み感度を準備する。
5. 使用予定磁石を事前測定する。
6. 測定時に、基準面へ磁化方向マーキングを行う。
7. 強度・方向誤差でクラスタ箱に分ける。
8. 外れ値を最大10%程度まで隔離する。
9. 作業単位を決める。
10. クラスタ在庫をもとに、作業単位ごとのクラスタ使用計画を作る。
11. Streamlit UIで組立セッションを開始する。
12. UIが鏡像リングペアを2枚表示し、現在作業中のリングを示す。
13. ソフトが次に取るクラスタ箱を指示する。
14. 作業者が指定箱から1個取る。
15. その磁石を再測定する。
16. ソフトがスロットと姿勢を決める。
17. 作業者が指定スロットへ指定姿勢で挿入する。
18. ソフトがログ・残留誤差・在庫を更新する。
19. 作業単位が完了するまで繰り返す。
20. 鏡像ペアの上下入れ替えが可能な場合は、2通りを評価して良い方を採用する。
21. 全作業単位完了後、最終磁場均一度を評価する。

### 11.2 1ステップの詳細

```text
入力:
  - 現在の作業単位
  - 鏡像ペア情報
  - 空きスロット集合 R
  - 現在の残留誤差または現在の自己無撞着状態
  - 残りクラスタ使用予定数 h
  - グローバル在庫

手順:
  1. ソフトが次に使うクラスタ b を選ぶ
  2. Streamlit UIに「クラスタ b から1個取る」と表示する
  3. 作業者が1個取り、測定器に置く
  4. 測定値 x_now を取得する
  5. 測定値を検証する
  6. 必要なら再測定・隔離・継続使用を選ぶ
  7. オンラインMPC割当を解く
  8. リング、スロット s、姿勢 o を決める
  9. Streamlit UIにペア2リング図、推奨スロット、姿勢指示を表示する
  10. 作業者が挿入して完了ボタンを押す
  11. ソフトが状態を更新する
  12. ログを保存する
```

### 11.3 鏡像リングペアの上下入れ替え

鏡像ペアの2枚のリングを `A`, `B` とする。

完成後、以下の2通りを評価する。

```text
option_1: A -> +z, B -> -z
option_2: A -> -z, B -> +z
```

評価関数はROI均一度目的関数である。良い方を採用する。

---

## 12. シミュレーションモード仕様

### 12.1 目的

シミュレーションモードは、実組立前に案Cでどの程度のROI磁場均一度が期待できるかを評価するための機能である。

### 12.2 UI方針

シミュレーションモードは、実組立モードと同じStreamlit UIを使う。

違いは、測定器から実測値を読む代わりに、あらかじめ定義したばらつきモデルまたは仮想磁石リストから「仮想測定値」を生成する点である。

UIには以下の実行モードを用意する。

```text
simulation_step_by_step
simulation_auto_run
assembly_real
```

#### `simulation_step_by_step`

実組立と同じように、1個ずつ仮想測定・仮想挿入を進める。

用途:

- UI動作確認
- 作業手順の検証
- アルゴリズムの挙動確認

#### `simulation_auto_run`

仮想組立を自動で最後まで実行する。

用途:

- 多数試行の統計評価
- パラメータ探索
- 外れ値隔離率やクラスタ数の比較

#### `assembly_real`

実測定器から値を取得し、実際の組立指示を出す。

### 12.3 シミュレーション入力

入力項目:

```text
- 最適化済みrunディレクトリ
- ROI設定
- 磁化モデル設定 fixed / self-consistent-easy-axis
- オンライン判断モード linear_sensitivity / sequential_self_consistent
- 磁石数
- 予備磁石数
- 強度ばらつきモデル
- 方向ずればらつきモデル
- ロットドリフトモデル
- 測定ノイズモデル
- クラスタ設定
- 外れ値隔離設定
- 作業単位設定
- 鏡像ペア設定
- 姿勢候補設定
- 試行回数
- 乱数seed
```

### 12.4 磁石ばらつきモデル

最低限、以下をサポートする。

#### iid正規分布

```text
epsilon ~ Normal(mu, sigma)
```

#### 裾の重い分布

```text
epsilon ~ StudentT(df, scale)
```

#### 2ロット混合

```text
epsilon ~ w * Normal(mu1, sigma1) + (1-w) * Normal(mu2, sigma2)
```

#### 線形ドリフト

```text
mu_t = mu_0 + drift_rate * t
```

#### ランダムウォークドリフト

```text
mu_{t+1} = mu_t + noise
```

#### 方向誤差

```text
delta_x ~ Normal(0, sigma_delta_x)
delta_y ~ Normal(0, sigma_delta_y)
```

#### 実測CSVリサンプリング

実測済み磁石データCSVがある場合、そのデータからブートストラップサンプリングできるようにする。

### 12.5 シミュレーション手順

1. 仮想磁石群の真値を生成する。
2. 事前測定ノイズを加える。
3. 事前測定値でクラスタ箱に分ける。
4. 外れ値隔離を適用する。
5. 作業単位ごとのクラスタ使用計画を作る。
6. Streamlit UI上で、実組立と同じ手順を仮想的に進める。
7. ソフトがクラスタ箱を指定する。
8. 指定箱から仮想磁石を1個取り出す。
9. 再測定ノイズを加えて仮想測定値を生成する。
10. オンラインMPC割当でリング・スロット・姿勢を決める。
11. 実際の真値で配置ログを更新する。
12. 作業単位が完了するまで繰り返す。
13. 鏡像ペア上下入れ替えが可能なら評価する。
14. 全体完成後、固定モデルまたは自己無撞着モデルで磁場を評価する。
15. ランダム配置と比較する。

### 12.6 出力

候補出力:

```text
simulation_summary.json
simulation_trials.csv
final_placement_trial_XXX.csv
cluster_usage_trial_XXX.csv
work_unit_summary_trial_XXX.csv
field_metrics_trial_XXX.json
streamlit_session_log_trial_XXX.jsonl
```

集計項目:

- RMS均一度
- 最大偏差
- p95偏差
- p99偏差
- 中心磁場強度参考値
- ランダム配置比の改善率
- 除外磁石数
- 隔離磁石数
- クラスタ別使用数
- 作業単位別平均強度誤差
- 作業単位別方向誤差統計
- 鏡像ペア別残留誤差
- 自己無撞着モードの平均計算時間
- UIステップ数
- 異常値発生回数

### 12.7 評価指標

磁場均一度は中心磁場強度で規格化したppmとして報告する。

```text
rms_homogeneity_ppm = sqrt(mean(|B(p)-B0|^2)) / |B0| * 1e6
max_homogeneity_ppm = max(|B(p)-B0|) / |B0| * 1e6
```

ただし、最適化目的関数では中心磁場強度の絶対値合わせを主目的にしない。

---

## 13. データファイル仕様

### 13.1 設定ファイル

候補名:

```text
plan_c_config.yaml
```

例:

```yaml
run_dir: runs/demo_opt

work_units:
  mode: auto
  large_ring_threshold: 60
  small_total_threshold: 150
  allow_mirror_pair_swap: true

roi:
  mode: surface-fibonacci
  samples: 300
  radius_m: 0.14

online_assignment:
  decision_engine: linear_sensitivity
  alternative_engine: sequential_self_consistent
  future_virtual_mode: quantile
  local_search_swaps: 20000
  restarts: 8
  seed: 1234

sensitivity:
  dimension: 30
  remove_center_strength_mode: true
  finite_difference_step: 0.001

self_consistent_mpc:
  enabled: true
  candidate_pruning: linear_top_k
  top_k_slots: 8
  top_k_orientations_per_slot: 2
  max_sc_evaluations_per_step: 16
  warm_start: true

clusters:
  strength:
    mode: quantile
    count: 10
  angle:
    mode: transverse_norm_quantile
    count: 3
    transverse_2_weight: 1.0

reject:
  policy: isolate_up_to_fraction
  max_fraction: 0.10
  prefer_direction_outliers: true

measurement_noise:
  strength_sigma: 0.001
  transverse_component_1_sigma: 0.001
  transverse_component_2_sigma: 0.001

measurement:
  provider: synthetic
  serial:
    port: COM3
    baudrate: 115200
    timeout_s: 2.0
    line_format: json

marking:
  measurement_axis: +Z_meas
  reference_face: operator_front_face_in_measurement_fixture
  arrow_meaning: measured_magnetization_direction

orientations:
  mode: discrete4
  rotation_axis: measured_nominal_magnetization_axis
  candidates:
    - id: O0
      angle_deg: 0
      instruction: "マーキング矢印をスロット基準マークに合わせる"
    - id: O90
      angle_deg: 90
      instruction: "マーキング矢印をスロット基準マークから時計回りに90度"
    - id: O180
      angle_deg: 180
      instruction: "マーキング矢印をスロット基準マークと反対向き"
    - id: O270
      angle_deg: 270
      instruction: "マーキング矢印をスロット基準マークから反時計回りに90度"

streamlit:
  show_mirror_pair_two_ring_view: true
  enable_step_by_step_simulation: true
  enable_auto_run_simulation: true
  enable_real_assembly: true
```

### 13.2 感度ファイル

候補名:

```text
plan_c_sensitivity.npz
```

保存内容:

```text
slot_id
ring_id
layer_id
theta_id
work_unit_id
mirror_pair_id
centers_m
nominal_u
orientation_id
C
projection_basis
roi_points
normalization_B0
metadata_json
```

### 13.3 クラスタ在庫ファイル

候補名:

```text
cluster_inventory.json
```

例:

```json
{
  "schema_version": 1,
  "clusters": {
    "S04_A01": {
      "count": 153,
      "mean": [0.001, 0.0002, -0.0001],
      "cov": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      "status": "normal"
    }
  },
  "quarantine": {
    "Q_DIRECTION_OUTLIER": {"count": 12},
    "Q_STRENGTH_OUTLIER": {"count": 8}
  }
}
```

### 13.4 セッションログ

候補名:

```text
assembly_session.jsonl
```

1行1イベントで保存する。

主なイベント:

```text
session_started
work_unit_selected
cluster_requested
measurement_received
measurement_rejected
placement_solved
insert_confirmed
undo_requested
work_unit_completed
mirror_pair_swap_evaluated
pair_installed
session_completed
```

### 13.5 最終配置ファイル

候補名:

```text
placement_final.csv
```

列:

```text
work_unit_id
mirror_pair_id
installed_layer_id
installed_side
ring_id
layer_id
theta_id
slot_flat_id
physical_slot_number
cluster_requested
epsilon_parallel
delta_perp_1
delta_perp_2
orientation_id
orientation_angle_deg
insert_order
decision_engine
```

---

## 14. 評価指標

### 14.1 磁場均一度指標

主指標:

```text
rms_homogeneity_ppm
max_homogeneity_ppm
p95_homogeneity_ppm
p99_homogeneity_ppm
```

補助指標:

```text
B0_norm
mean_B_deviation
low_mode_residual_norm
self_consistent_final_J
linear_model_final_J
```

### 14.2 組立品質指標

```text
used_magnet_count
quarantined_count
rejected_count
reject_fraction
cluster_plan_deviation
work_unit_mean_strength_error
work_unit_direction_error_stats
mirror_pair_residual_norm
operator_retry_count
undo_count
measurement_outlier_count
```

### 14.3 比較指標

案Cの性能は、ランダム配置に対して比較する。

```text
random_baseline_ratio = J_plan_c / J_random
improvement_factor = J_random / J_plan_c
```

案A・案Bとの比較指標は不要である。

---

## 15. 実装分割案

案Cは、既存の幾何最適化コードから独立した後段モジュールとして実装する。

候補ディレクトリ:

```text
halbach/assembly/
```

候補モジュール:

```text
halbach/assembly/sensitivity.py
halbach/assembly/clustering.py
halbach/assembly/inventory.py
halbach/assembly/work_units.py
halbach/assembly/marking.py
halbach/assembly/orientations.py
halbach/assembly/online_assignment.py
halbach/assembly/self_consistent_assignment.py
halbach/assembly/simulation.py
halbach/assembly/session.py
halbach/assembly/measurement.py
halbach/assembly/io.py
```

候補Streamlitページ:

```text
app/pages/plan_c_simulation.py
app/pages/plan_c_assembly.py
```

または既存Streamlitアプリにタブとして追加する。

候補CLI:

```text
python -m halbach.cli.plan_c_compute_sensitivity
python -m halbach.cli.plan_c_prepare_inventory
python -m halbach.cli.plan_c_simulate
python -m halbach.cli.plan_c_session
```

CLIはバッチ実行・テスト用、Streamlitは実組立・ステップ実行用とする。

---

## 16. テスト方針

### 16.1 単体テスト

- クラスタ境界生成
- 測定値からクラスタIDへの変換
- 外れ値判定
- 作業単位自動生成
- 鏡像ペア検出
- 磁石マーキング座標から誤差ベクトルへの変換
- 4姿勢回転 `O0/O90/O180/O270` の変換
- 感度ファイルの読み書き
- 仮想未来磁石生成
- 線形感度モードの2点交換差分
- 姿勢候補選択
- スロット重複防止
- セッション状態保存・復元
- synthetic/manual/csv/serial測定プロバイダの抽象インターフェース

### 16.2 統合テスト

小規模配列で以下を確認する。

```text
N=8, K=4, R=1
```

確認項目:

- ランダム配置より案C線形感度モードが改善するケースがある。
- 案C逐次自己無撞着モードが最後まで実行できる。
- 全スロットが1回だけ使用される。
- クラスタ在庫が負にならない。
- 外れ値隔離率が上限を超えない。
- 鏡像ペア2リング表示に必要なデータが生成される。
- 4姿勢候補が評価される。
- seed固定でシミュレーション結果が再現する。
- Streamlitの`simulation_step_by_step`と`simulation_auto_run`が同じ最終配置を再現できる設定がある。

### 16.3 回帰テスト

既存の `generate_run`、`optimize_run`、目的関数評価、自己無撞着モデルに影響を与えないことを確認する。

---

## 17. 受け入れ基準

初期実装の受け入れ基準は以下とする。

1. 最適化済みrunからスロット情報を読み出せる。
2. 作業単位を `single_physical_ring`、`ring_group`、`all_slots` で扱える。
3. 鏡像ペアを登録し、Streamlit UIで2リング同時表示できる。
4. 鏡像ペアの上下入れ替え評価ができる。
5. 各物理リングのスロット番号をUIに表示できる。
6. 測定時マーキングを前提とした4姿勢 `O0/O90/O180/O270` を扱える。
7. 推奨姿勢を作業者向け説明文付きで表示できる。
8. 強度クラスタ数を5〜10の範囲で設定できる。
9. 方向誤差クラスタ数を2〜5の範囲で設定できる。
10. 外れ値隔離率を最大10%程度まで設定できる。
11. 測定ノイズ0.1%程度を既定値にしたシミュレーションができる。
12. 線形感度モードで案Cの逐次MPC割当ができる。
13. 逐次自己無撞着モードで候補削減付きの案C割当ができる。
14. シミュレーションモードで、ランダム配置と案C各モードを比較できる。
15. Streamlit UIで、仮想測定・仮想挿入を実組立と同じ手順で進められる。
16. Streamlit UIで、実測定器接続予定のシリアルI/Fを抽象化して扱える。
17. 個体IDなしで運用できる。
18. 各スロットの最終測定値ログを保存できる。
19. セッションを中断・再開できる。
20. 乱数seedを固定すればシミュレーション結果が再現する。
21. pytest等で主要ロジックの単体試験が可能である。

---

## 18. 仕様の要約

案Cは、以下の制約を満たすための組立方式である。

```text
個体ID管理なし
全磁石の平面展開なし
クラスタ箱管理
組立時再測定
1個ずつ挿入
リングまたは作業単位ごとに組立
ROI磁場均一度を最優先
```

中核となる処理は以下である。

```text
事前測定でクラスタ箱へ分ける
測定時に磁石へ姿勢マーキングを行う
Streamlit UIが次に取る箱を指示する
取り出した1個を再測定する
線形感度または逐次自己無撞着評価でリング・スロット・姿勢を決める
鏡像ペア2リングをUI上に表示する
指定スロットへ指定姿勢で挿入する
挿入後に状態を更新する
鏡像リングペアは上下入れ替えを評価する
```

実装するモードは以下である。

```text
simulation_step_by_step
simulation_auto_run
assembly_real
```

比較対象は以下に限定する。

```text
ランダム配置
案C 線形感度モード
案C 線形感度モード + 外れ値隔離
案C 逐次自己無撞着モード
案C 逐次自己無撞着モード + 外れ値隔離
```

最終判断指標はROI磁場均一度であり、中心磁場強度の絶対値は組立時の主目的にはしない。
