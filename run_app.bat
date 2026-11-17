@echo off
TITLE Wearable-Signal Stress-Risk Classifier
echo ========================================================
echo   Launching Wearable Stress Risk Classifier Dashboard
echo ========================================================
cd /d "%~dp0Wearable_Stress_Classification"
python -m streamlit run app.py
pause
