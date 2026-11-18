@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set USE_NEURAL_AI=1

if exist ".venv-ai\Scripts\python.exe" (
    set "PYEXE=.venv-ai\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYEXE=.venv\Scripts\python.exe"
) else (
    echo [ERROR] Khong tim thay Python venv (.venv-ai hoac .venv)!
    pause
    exit /b 1
)

echo ===============================================================
echo                   KHOI DONG GAME CO VUA AI
echo ===============================================================
echo [*] Dang khoi chay may chu va mo trinh duyet tai http://127.0.0.1:8000...
echo.

"%PYEXE%" run_game.py %*
if errorlevel 1 pause
