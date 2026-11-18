@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

if not exist ".venv-ai\Scripts\python.exe" (
    echo [ERROR] Khong tim thay moi truong .venv-ai!
    echo Vui long chay CAI_DAT_AI.ps1 truoc.
    pause
    exit /b 1
)

".venv-ai\Scripts\python.exe" -X utf8 train_pipeline.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] Qua trinh chay gap su co.
)
echo.
pause
