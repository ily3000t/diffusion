param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Preflight','Run','Train','WarmupCheck','Sample','Report','AuditBaseline')]
    [string]$Stage,
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$')]
    [string]$RunId,
    [string]$Config='configs/experiments/stage5d_training.yaml',
    [string]$Resume,
    [string]$TrainingRun,
    [string]$SamplingRun
)
$ErrorActionPreference='Stop'
if ($Resume -and $Stage -ne 'Train') { throw '-Resume is only valid with Train.' }
if ($TrainingRun -and $Stage -notin @('Sample','Report')) { throw '-TrainingRun is only valid with Sample/Report.' }
if ($SamplingRun -and $Stage -ne 'Report') { throw '-SamplingRun is only valid with Report.' }
$projectRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
$pythonPath=Join-Path $projectRoot '.venv-model\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
$env:PYTHONPATH=Join-Path $projectRoot 'src'
$env:SUMO_HOME='E:\Program Files\sumo-1.22.0'
$env:PYTHONUNBUFFERED='1'
$outputPath=Join-Path $projectRoot ('artifacts\runs\'+$RunId)
if (Test-Path -LiteralPath $outputPath) { throw 'Choose a fresh RunId; prior outputs are preserved.' }
$arguments=@('-m','sumodiff.diffusion.stage5d',$Stage.ToLowerInvariant().Replace('warmupcheck','warmup-check').Replace('auditbaseline','audit-baseline'),
    '--config',$Config,'--output',$outputPath,'--formal')
if ($Stage -in @('Run','Train')) { $arguments+='--allow-long-run' }
if ($Resume) { $arguments+=@('--resume',(Resolve-Path -LiteralPath $Resume).Path) }
if ($Stage -in @('Sample','Report')) {
    if (-not $TrainingRun) { throw 'Sample/Report require -TrainingRun.' }
    $arguments+=@('--training-run',(Resolve-Path -LiteralPath $TrainingRun).Path)
}
if ($Stage -eq 'Report') {
    if (-not $SamplingRun) { throw 'Report requires -SamplingRun.' }
    $arguments+=@('--sampling-run',(Resolve-Path -LiteralPath $SamplingRun).Path)
}
& $pythonPath @arguments
if ($LASTEXITCODE -ne 0) { throw ('Stage5D failed with exit code '+$LASTEXITCODE+'; outputs/checkpoints are retained.') }
