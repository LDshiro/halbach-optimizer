# Baseline runs

回帰比較用のベースライン run を再現可能に生成するための手順です。

## 実行方法
```powershell
python scripts/make_baseline_runs.py --force
```

## ケースとコマンド
### baseline_tiny_legacy_analytic

Generate:
```powershell
python -m halbach.cli.generate_run --out runs\baseline_tiny_legacy_analytic --N 12 --R 1 --K 6 --Lz 0.20 --diameter-mm 200 --ring-offset-step-mm 0
```
Optimize:
```powershell
python -u -m halbach.cli.optimize_run --in runs\baseline_tiny_legacy_analytic --out runs\baseline_tiny_legacy_analytic_opt --maxiter 3 --gtol 1.0e-12 --roi-r 0.05 --roi-mode surface-fibonacci --roi-samples 50 --roi-seed 20250924 --field-scale 1e+06 --angle-model legacy-alpha --fix-center-radius-layers 2 --r-bound-mode relative --r-lower-delta-mm 5.0 --r-upper-delta-mm 5.0 --grad-backend analytic
```

### baseline_tiny_legacy_jax

Generate:
```powershell
python -m halbach.cli.generate_run --out runs\baseline_tiny_legacy_jax --N 12 --R 1 --K 6 --Lz 0.20 --diameter-mm 200 --ring-offset-step-mm 0
```
Optimize:
```powershell
python -u -m halbach.cli.optimize_run --in runs\baseline_tiny_legacy_jax --out runs\baseline_tiny_legacy_jax_opt --maxiter 3 --gtol 1.0e-12 --roi-r 0.05 --roi-mode surface-fibonacci --roi-samples 50 --roi-seed 20250924 --field-scale 1e+06 --angle-model legacy-alpha --fix-center-radius-layers 2 --r-bound-mode relative --r-lower-delta-mm 5.0 --r-upper-delta-mm 5.0 --grad-backend jax
```

### baseline_tiny_delta_rep

Generate:
```powershell
python -m halbach.cli.generate_run --out runs\baseline_tiny_delta_rep --N 12 --R 1 --K 6 --Lz 0.20 --diameter-mm 200 --ring-offset-step-mm 0
```
Optimize:
```powershell
python -u -m halbach.cli.optimize_run --in runs\baseline_tiny_delta_rep --out runs\baseline_tiny_delta_rep_opt --maxiter 3 --gtol 1.0e-12 --roi-r 0.05 --roi-mode surface-fibonacci --roi-samples 50 --roi-seed 20250924 --field-scale 1e+06 --angle-model delta-rep-x0 --fix-center-radius-layers 2 --r-bound-mode relative --r-lower-delta-mm 5.0 --r-upper-delta-mm 5.0
```

## 取得メトリクス
- J（trace.extras[-1].J）
- |B0| [T]（trace.extras[-1].B0）
- iters（trace.iters[-1]）
- roi_seed（固定）

## サマリ

| case | angle_model | grad_backend | status | J | |B0| [T] | iters | roi_seed | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_tiny_legacy_analytic | legacy-alpha | analytic | ok | 4.515922e+03 | 5.695224e-03 | 3 | 20250924 | - |
| baseline_tiny_legacy_jax | legacy-alpha | jax | ok | 4.515922e+03 | 5.695224e-03 | 3 | 20250924 | - |
| baseline_tiny_delta_rep | delta-rep-x0 | - | ok | 4.510121e+03 | 5.695244e-03 | 3 | 20250924 | - |

## baseline_runs.json
`docs/baseline_runs.json` はスクリプトが生成する機械比較用の出力です。
主な構造:
```json
{
  "generated_at": "ISO-8601 timestamp",
  "cases": [
    {
      "name": "baseline_tiny_legacy_analytic",
      "angle_model": "legacy-alpha",
      "grad_backend": "analytic",
      "status": "ok|skipped|error",
      "metrics": {
        "J": 0.0,
        "B0": 0.0,
        "iters": 0
      },
      "roi_seed": 20250924,
      "input_dir": "runs/...",
      "opt_dir": "runs/..._opt",
      "reason": "optional"
    }
  ]
}
```