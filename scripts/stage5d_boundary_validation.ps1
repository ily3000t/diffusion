param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$')]
    [string]$RunId,
    [string]$Config='configs/experiments/stage5d_boundary_validation.yaml'
)
$ErrorActionPreference='Stop'
$projectRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
$pythonPath=Join-Path $projectRoot '.venv-model\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
$outputPath=Join-Path $projectRoot ('artifacts\runs\'+$RunId)
if (Test-Path -LiteralPath $outputPath) { throw 'Choose a fresh RunId; prior outputs are retained.' }
$env:PYTHONPATH=Join-Path $projectRoot 'src'
$env:SUMO_HOME='E:\Program Files\sumo-1.22.0'
$env:PYTHONUNBUFFERED='1'
& $pythonPath -m sumodiff.diffusion.boundary_validation --config $Config --output $outputPath --formal
if ($LASTEXITCODE -ne 0) { throw ('Diagnostic failed with exit code '+$LASTEXITCODE+'; outputs are retained.') }
