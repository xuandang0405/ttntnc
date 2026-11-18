@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set USE_NEURAL_AI=1

if not exist ".venv-ai\Scripts\python.exe" (
    echo [ERROR] Khong tim thay moi truong .venv-ai!
    echo Vui long chay CAI_DAT_AI.ps1 truoc.
    pause
    exit /b 1
)

echo ===============================================================
echo          KHOI DONG CO VUA AI - DEEP LEARNING CHAMPION
echo ===============================================================
echo [*] Dang khoi chay may chu va mo trinh duyet...
echo.

".venv-ai\Scripts\python.exe" run_game.py %*
if errorlevel 1 pause
