param(
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cpu',
    [string]$PythonExe = ''
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv-ai/Scripts/python.exe')) {
        if (-not $PythonExe) {
            $uvRoot = Join-Path $env:APPDATA 'uv/python'
            if (Test-Path -LiteralPath $uvRoot) {
                $candidate = Get-ChildItem -LiteralPath $uvRoot -Directory -Filter 'cpython-3.12.*-windows-x86_64-none' |
                    Sort-Object Name -Descending | Select-Object -First 1
                if ($candidate) { $PythonExe = Join-Path $candidate.FullName 'python.exe' }
            }
        }
        if ($PythonExe) { & $PythonExe -m venv '.venv-ai' }
        else { & py -3.12 -m venv '.venv-ai' }
        if ($LASTEXITCODE -ne 0) { throw 'Can Python 3.12; truyen -PythonExe neu may dung duong dan khac.' }
    }
    $runtime = Join-Path $PSScriptRoot '.venv-ai/Scripts/python.exe'
    $build = if ($Device -eq 'cuda') { 'cu128' } else { 'cpu' }
    & $runtime -m pip install --upgrade "torch==2.10.0+$build" --index-url "https://download.pytorch.org/whl/$build"
    if ($LASTEXITCODE -ne 0) { throw 'Khong cai duoc PyTorch.' }
    & $runtime -m pip install -r 'requirements-ai.txt'
    if ($LASTEXITCODE -ne 0) { throw 'Khong cai duoc cac thu vien AI.' }
    & $runtime -c 'import torch; print("torch:",torch.__version__); print("CUDA:",torch.cuda.is_available())'
    if ($Device -eq 'cuda') {
        & $runtime -c 'import torch; assert torch.cuda.is_available(), "CUDA chua san sang; kiem tra driver NVIDIA."'
        if ($LASTEXITCODE -ne 0) { throw 'PyTorch da cai nhung GPU chua san sang.' }
    }
} finally { Pop-Location }
