# Headless System 1 stand-in (no Unity). Use for testing, or as the demo fallback.
#   powershell -ExecutionPolicy Bypass -File scripts\start_standin.ps1 -Speed 1 [-Gui]
param([double]$Speed = 1.0, [switch]$Gui)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$pyArgs = @("-u", "twin\system1_standin.py", "--speed", $Speed)
if ($Gui) { $pyArgs += "--gui" }
& $py @pyArgs
