# AGENTS.md
Codex（VS Code上のAIアシスタント）にリファクタリングを依頼する際の前提・禁止事項・不変条件をまとめます。
このファイルは「AIが迷わず正しく動く」ための仕様書です。

## 1. Roles（役割）
- **Human（Owner）**
  - 目的・仕様の最終決定者。レビューとマージ判断を行う。
- **Codex（Implementer）**
  - 指示された範囲でコード変更を行う。必ず小さな差分で提案する。
- **Codex（Reviewer モード）**
  - 変更方針・設計レビュー・危険箇所の指摘・テスト設計を行う。

## 2. Non-negotiable rules（絶対ルール）
1. **小さな差分（Small PR）**で進める  
   - 1コミット = 1意図（例：モジュール分割、型付け、ログ置換、など）
2. **数式・定義の勝手変更禁止**  
   - 目的関数、座標系、単位、符号、対称性は「Ownerの指示」がない限り変更しない
3. **ファイルフォーマット互換性維持**  
   - 保存npz/json/matのキー名・形状は互換を壊さない（壊すなら必ず移行手順を用意）
4. **テストを増やして安全に**  
   - 変更後は pytest が通る状態を維持（最低：スモーク・勾配チェック）
5. **パフォーマンス配慮**  
   - 重要ループに余計なPythonオブジェクト生成を入れない
   - Numba関数の境界を崩さない（Numba対象をPythonクラス化しない等）

## 3. Invariants（不変条件：現行仕様）
### 3.1 単位・座標
- 長さ：m（内部計算）、ユーザー入力に mm があれば m へ変換
- 角度：rad（内部計算）、ユーザー入力に deg があれば rad へ変換
- 座標系：xy平面上に磁石リング、z方向に層スタック

### 3.2 設計変数
- x-space: `x = [alphas(:); r_vars]`
  - `alphas`: shape (R, K) … 全リング×全層の角度補正パラメータ
  - `r_vars`: shape (K/2-2,) … z対称性で下側のみ（中央4層固定の前提）
- y-space（物理パラメータ）: `y = [alphas(:); r_bases]` で shape (R*K + K,)

### 3.3 目的関数（ノミナル）
- ROI（球内サンプル点）での **磁束密度ベクトル誤差の2乗平均**
- `J = mean(||B(p) - B0||^2)`  
  - B0 は中心点の磁束密度ベクトル（毎回更新）

### 3.4 ロバスト（本スクリプトのGN版）
- 勾配ノルム正則化は **y-space** で定義する（MC摂動空間と整合）
- `J_gn(x) = J(x) + 0.5*rho*||Sigma_y^{1/2} grad_y J||^2`
- `H_y(J) v` は y-space の中心差分で近似（解析 grad_y を使う）

### 3.5 保存フォーマット（互換要件）
- NPZ（最低限）
  - `alphas_opt`, `r_bases_opt`, `theta`, `sin2` or `sin2th`, `z_layers`, `ring_offsets`
- JSON（メタ情報）
  - `N,K,R,r0,Lz,dz,roi_r,roi_step` など
- MAT（MATLAB viewer互換）
  - `alphas_opt`, `r_bases_opt`, `theta`, `sin2th`, `z_layers`, `ring_offsets`

## 4. Coding standards
- Python: 3.11
- Formatter: black
- Linter: ruff
- Type check: mypy **strict**（ただし numba 境界は現実的に緩和が必要な場合あり）
- Docstring: 「shape」「単位」「入出力」を明記
- 依存追加は最小限（配布予定なし、CIなし）

## 5. How to ask Codex（指示テンプレ）
Codexには毎回、以下を明示して依頼する：
1. **目的**（なぜやるか）
2. **変更範囲**（編集して良いファイルと関数）
3. **不変条件**（Invariantsのどれを守るか）
4. **受け入れ条件**（pytest通過、互換ファイル出力、性能劣化なし等）
5. **出力形式**（差分パッチ or 変更ファイル一覧 + 変更理由）

例：
- “`physics.py` に numbaカーネルを移し、API互換で import を修正して。pytest通過が条件。変更は1コミット分に収めること。”
