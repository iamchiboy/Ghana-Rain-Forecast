@echo off
REM Ghana Rain Forecast - Windows Batch Runner
REM This script provides easy access to all application commands

setlocal enabledelayexpand

echo.
echo =====================================
echo  GHANA RAIN FORECAST - Control Panel
echo =====================================
echo.
echo Choose an option:
echo.
echo 1. Collect Weather Data
echo 2. Preprocess Data
echo 3. Train Model
echo 4. Make Prediction
echo 5. Launch Dashboard
echo 6. View Logs
echo 7. Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo.
    echo Starting data collection (press Ctrl+C to stop)...
    python -m src.collect_data
) else if "%choice%"=="2" (
    echo.
    echo Preprocessing data...
    python -m src.preprocess
) else if "%choice%"=="3" (
    echo.
    echo Training model...
    python -m src.train_model
) else if "%choice%"=="4" (
    echo.
    echo Making prediction...
    python -m src.predict
) else if "%choice%"=="5" (
    echo.
    echo Launching dashboard at http://localhost:8501
    streamlit run app/dashboard.py
) else if "%choice%"=="6" (
    echo.
    echo Application logs:
    echo.
    type logs\app.log
) else if "%choice%"=="7" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice. Please try again.
)

echo.
echo Press any key to return to menu...
pause
goto :start
