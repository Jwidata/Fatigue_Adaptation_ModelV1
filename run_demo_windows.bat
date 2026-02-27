@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\activate" (
  echo [ERROR] Virtual environment not found at .venv\Scripts\activate
  echo Create it with:
  echo   python -m venv .venv
  echo   .venv\Scripts\activate
  echo   pip install -e .
  pause
  exit /b 1
)

echo [INFO] Starting FastAPI server in a new window...
start "Fatigue XR Server" cmd /k "call .venv\Scripts\activate && python -m fatigue_xr.cli serve --host 0.0.0.0 --port 8000"

timeout /t 1 /nobreak >nul
echo [INFO] Opening http://127.0.0.1:8000/
start "" "http://127.0.0.1:8000/"

endlocal
