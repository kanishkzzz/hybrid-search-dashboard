$ErrorActionPreference = "Stop"

Write-Host "Starting Hybrid Search System"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m backend.app.ingest --input data/raw --out data/processed

$env:QUERY_LOG_DB = "data/metrics/queries.db"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "uvicorn backend.app.api.main:app --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "streamlit run frontend/dashboard.py"
