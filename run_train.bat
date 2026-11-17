@echo off
TITLE Train Wearable Stress Model
echo ========================================================
echo   Training Random Forest Stress Classification Model
echo ========================================================
cd /d "%~dp0Wearable_Stress_Classification"
python train_model.py
pause
