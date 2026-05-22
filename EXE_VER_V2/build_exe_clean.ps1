$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPath = Join-Path $repoRoot ".venv-exe"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment at $venvPath" }
}

& $pythonExe -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade packaging tools" }

& $pythonExe -m pip uninstall -y pathlib
if ($LASTEXITCODE -ne 0) { throw "Failed to uninstall pathlib from build environment" }

& $pythonExe -m pip install -r "src_exe\requirements-exe.txt"
if ($LASTEXITCODE -ne 0) { throw "Failed to install build requirements" }

& $pythonExe -m PyInstaller --noconfirm "src_exe\build_exe.spec"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

$exePath = Join-Path $repoRoot "dist\ONC_PTWL_Tool_Standalone\ONC_PTWL_Tool_Standalone.exe"
if (-not (Test-Path $exePath)) { throw "Build finished without expected output: $exePath" }

Write-Host "Build complete: $exePath"
