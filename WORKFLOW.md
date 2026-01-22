# WORKFLOW.md
Windows + PowerShell + VS Code + Codex を前提にした開発ワークフローです（CIなし・ローカルで完結）。

## 1. Local setup（Windows / PowerShell）
### 1.1 Python venv（3.11）
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

### Formatting / Lint / Typecheck
```powershell
black .
isort .
ruff check .
mypy .
pytest