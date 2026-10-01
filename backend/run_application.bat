@echo off
REM Activates the local virtual environment and runs application.py via waitress
REM Also starts the Celery background worker and beat scheduler.

set VENV_PATH=..\..\.venv
if not exist "%VENV_PATH%\Scripts\python.exe" (
    echo Error: Virtual environment not found at %VENV_PATH%
    echo Please make sure the virtual environment exists in the 'helloworld' root directory.
    exit /b 1
)

echo Using Python from virtual environment: %VENV_PATH%

echo Starting Celery worker...
start "Celery Worker" cmd /c "%VENV_PATH%\Scripts\celery.exe -A celery_app.celery worker --pool=solo --loglevel=info"

echo Starting Celery beat...
start "Celery Beat" cmd /c "%VENV_PATH%\Scripts\celery.exe -A celery_app.celery beat --loglevel=info"

echo Starting Flask API with Eventlet/SocketIO on port 5000...
"%VENV_PATH%\Scripts\python.exe" application.py
