#!/bin/bash

echo "Starting Hybrid Search System"

# create virtual environment if not exists
python -m venv .venv

# activate venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run ingestion
python -m backend.app.ingest --input data/raw --out data/processed

# start API
uvicorn backend.app.api.main:app --reload &

# start dashboard
streamlit run frontend/dashboard.py