#!/bin/bash

# FastAPI 백엔드 실행 (백그라운드)
uvicorn studio_project.backend.main:app --host 0.0.0.0 --port 8000 &

# Streamlit 프론트엔드 실행
streamlit run studio_project/frontend/app.py --server.port 8501 --server.address 0.0.0.0
