@echo off
echo ===================================================
echo AI Surveillance System - Setup and Run (Backend)
echo ===================================================

cd /d "%~dp0"

echo Checking for Python...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH! Please install Python 3.10+.
    pause
    exit /b 1
)

echo Setting up virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Starting FastAPI server...
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
