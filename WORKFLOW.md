# WORKFLOW.md
Windows + PowerShell + VS Code + Codex を前提にした開発ワークフローです（CIなし・ローカルで完結）。

## 1. Local setup（Windows / PowerShell）
### 1.1 Python venv（3.14）
```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-dev.txt -r requirements-gui.txt
```

### 1.2 Formatting / Lint / Typecheck
`mypy` は `pyproject.toml` の設定に従い、既定で `halbach/` のみをチェックします。

```powershell
black .
isort .
python -m ruff check .
python -m mypy .
python -m pytest -q
```

### 1.3 GUI
```powershell
python -m streamlit run app\streamlit_app.py
```
