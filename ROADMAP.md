
---

## ROADMAP.md

```markdown
# ROADMAP.md
本プロジェクトのリファクタリング計画（Codexに渡す作業計画書）です。
目的：`robust_opt_halbach_gradnorm_minimal.py` を保守性の高い構造へ分割し、厳格な型チェックとテストで安全に改良できるようにする。

## 前提
- Windows + PowerShell
- Python 3.11 + venv
- GitHubで新規リポ作成（第1章で実施）
- CIは導入しない（ローカルでpytest/format/typecheckを回す）
- 型チェックは強め（mypy strict）
- License: MIT
- 配布予定なし（pip package化はしない）
- ROI上限の目安：円筒内側直径の約80%程度（ただしROIは引数で可変）

---

## Milestones（章立て）
### 0) Agent docs（完了条件）
- [ ] AGENTS.md / WORKFLOW.md / ROADMAP.md を追加
- [ ] “不変条件（目的関数/保存互換）” を明文化

### 1) Repo bootstrap（ローカル→GitHub）
- [ ] venv（3.11）手順をREADMEに書く
- [ ] MIT LICENSE を追加
- [ ] .gitignore を整備
- [ ] ベースコードを `src/` or ルートに配置し、初回タグ `v0-baseline`

### 2) Tooling（formatter/lint/typecheck）
- [ ] black / ruff / isort / mypy(strict) / pytest を導入（pyproject.toml）
- [ ] ローカル実行手順をWORKFLOWへ追記
- [ ] （任意）pre-commit導入（CIなしでも品質維持）

### 3) Tests（最小安全網）
- [ ] スモークテスト（小ROIで objective が動く）
- [ ] gradcheck（解析勾配 vs FD：1e-6〜1e-9級で一致）
- [ ] 保存データ互換テスト（キーとshape）

### 4) Module split（役割分離）
- [ ] `halbach/geom.py`（ROI生成、対称インデックス）
- [ ] `halbach/physics.py`（Numbaカーネル）
- [ ] `halbach/objective.py`（目的関数＋解析勾配）
- [ ] `halbach/robust.py`（GN正則化/HVP）
- [ ] `halbach/io.py`（npz/json/mat保存）
- [ ] CLI（入口）で従来と同等に動く

### 5) Types & Logging
- [ ] dataclass（Geometry, Settings, Results）
- [ ] type hints（mypy strict）
- [ ] print を logging に移行（verboseで制御）

### 6) Solver abstraction（拡張の足場）
- [ ] Optimizerインターフェース導入
- [ ] L-BFGS-B実装をクラス化
- [ ] 今後 Gauss-Newton/Diagonal-CCP を追加可能に

### 7) Performance harness（速度検証）
- [ ] 簡易ベンチマークスクリプト
- [ ] 重要ループの不要アロケーション削減
- [ ] numba parallel の検討（任意）

### 8) Robust term abstraction
- [ ] RobustTermインターフェース
- [ ] GN実装をStrategy化（将来CVaR/SAAの追加余地）

### 9) Documentation（CIなし想定）
- [ ] README更新（セットアップ/実行/トラブルシュート）
- [ ] よくあるエラー集（Windows/PowerShell/venv/numba）

---

## Acceptance criteria（最終受け入れ）
- [ ] ベース目的関数（ノミナル）と保存フォーマット互換が維持されている
- [ ] pytest がローカルで通る
- [ ] mypy strict が（numba境界を除き）通る
- [ ] 最適化・MC・保存が従来と同様に実行できる
