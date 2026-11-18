@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
where python >nul 2>&1
if errorlevel 1 (
    echo Can cai Python va them Python vao PATH truoc.
    goto failed
)
where npm >nul 2>&1
if errorlevel 1 (
    echo Can cai Node.js LTS va npm truoc.
    goto failed
)
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
pushd frontend
call npm ci
if errorlevel 1 goto frontend_failed
call npm run build
if errorlevel 1 goto frontend_failed
popd
echo.
echo Da cai xong. Mo CHAY_GAME.bat de choi.
pause
exit /b 0
:frontend_failed
popd
:failed
echo.
echo Cai dat chua hoan tat. Xem thong bao loi o tren.
pause
exit /b 1
