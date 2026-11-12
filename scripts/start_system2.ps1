# System 2 (low GPU): live loop + dashboard. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\start_system2.ps1
# Add -Standin to also run the headless System 1 stand-in on this machine.
param([switch]$Standin, [double]$Speed = 1.0)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "Checking the Redis link..."
& $py scripts\check_link.py
if ($LASTEXITCODE -ne 0) { Write-Host "Redis not reachable. Check .env and DEVELOPMENT_PLAN.md section 7.2"; exit 1 }

if ($Standin) {
  Start-Process -FilePath $py -ArgumentList "-u", "twin\system1_standin.py", "--speed", $Speed -WindowStyle Normal
}
Start-Process -FilePath $py -ArgumentList "-u", "orchestrator\live.py" -WindowStyle Normal
Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000" -WindowStyle Normal
Start-Sleep -Seconds 4
Start-Process "http://localhost:8000"
Write-Host "Dashboard: http://localhost:8000   (other laptops on the network: http://<this-ip>:8000)"
