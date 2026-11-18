@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

if not exist ".venv-ai\Scripts\python.exe" (
    echo [ERROR] Khong tim thay moi truong .venv-ai!
    pause
    exit /b 1
)

".venv-ai\Scripts\python.exe" -X utf8 train_pipeline.py --leaderboard

echo.
pause
