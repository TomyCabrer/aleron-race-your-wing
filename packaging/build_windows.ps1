# packaging/build_windows.ps1 -- build the Windows folder (dist\<Name>\<Name>.exe) for Steam.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#
# Needs Python 3.11+ (64-bit, python.org installer) on PATH. PyInstaller cannot
# cross-build: run this on a Windows PC (or let .github/workflows/steam-build.yml do it).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
if (-not $env:SKIP_VENV) {
    & $py -m venv build\venv
    $py = "build\venv\Scripts\python.exe"
    & $py -m pip install --upgrade pip
    & $py -m pip install -r packaging\requirements-build.txt
}
if (-not (Test-Path packaging\icons\game.ico)) { & $py packaging\make_icons.py }
& $py -m PyInstaller --noconfirm --clean --distpath dist --workpath build\pyinstaller packaging\game.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
$name = & $py -c "from drive import branding; print(branding.ascii_name())"
Write-Host "built dist\$name\$name.exe"
Write-Host "smoke test: `$env:CARSIM_DATA_DIR='C:\temp\fresh'; dist\$name\$name.exe"
