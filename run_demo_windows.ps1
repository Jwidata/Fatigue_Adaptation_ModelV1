$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$activatePath = ".venv\Scripts\Activate.ps1"
if (!(Test-Path $activatePath)) {
  Write-Host "[ERROR] Virtual environment not found at .venv\Scripts\Activate.ps1"
  Write-Host "Create it with:"
  Write-Host "  python -m venv .venv"
  Write-Host "  .venv\Scripts\Activate.ps1"
  Write-Host "  pip install -e ."
  Write-Host "If activation is blocked, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass"
  exit 1
}

Write-Host "[INFO] Starting FastAPI server in a new window..."
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "& $activatePath; python -m fatigue_xr.cli serve --host 0.0.0.0 --port 8000"

Start-Sleep -Seconds 1
Write-Host "[INFO] Opening http://127.0.0.1:8000/"
Start-Process "http://127.0.0.1:8000/"
