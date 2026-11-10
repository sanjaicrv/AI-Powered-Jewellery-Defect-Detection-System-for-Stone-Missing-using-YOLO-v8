@echo off
title AI Jewellery Anomaly Detection - YOLOv8
cd /d "%~dp0"
echo =======================================================
echo   Starting AI Jewellery Anomaly Detection Web App...
echo =======================================================
.\venv\Scripts\python.exe web_app.py
pause
