param(
    [string]$AsOf = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptDir "ALycos.venv\Scripts\python.exe"
$configPath = Join-Path $scriptDir "config\scurve_config_fast.yaml"

if (-not (Test-Path $pythonExe)) {
    throw "Could not find venv python at $pythonExe. Run from your AGORA repo root and ensure ALycos.venv exists."
}

if (-not $AsOf) {
    $AsOf = Get-Date -Format "yyyy-MM-dd"
}

& $pythonExe -m scurve.run --asof $AsOf --config $configPath
