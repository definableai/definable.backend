@echo off
REM Windows batch script to start Celery worker

echo Starting Celery Worker for KB Service...
echo ========================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Set environment variables if .env file exists
if exist ".env" (
    echo Loading environment from .env file...
    for /f "usebackq tokens=*" %%i in (".env") do (
        set %%i
    )
)

REM Start the worker with default settings
echo Starting worker with 2 concurrent processes...
python run_celery_worker.py --concurrency 2 --loglevel info

pause