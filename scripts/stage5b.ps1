param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Preflight','Run','TrainArm','SampleArm','Report')]
    [string]$Stage,
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$')]
    [string]$RunId,
    [ValidateSet('A','B','C','D')]
    [string]$Arm,
    [string]$Resume,
    [string]$TrainingRun,
    [string[]]$ArmRuns
)
$ErrorActionPreference = 'Stop'
if ($Resume -and $Stage -ne 'TrainArm') { throw '-Resume is only valid with TrainArm.' }
if ($TrainingRun -and $Stage -ne 'SampleArm') { throw '-TrainingRun is only valid with SampleArm.' }
if ($ArmRuns -and $Stage -ne 'Report') { throw '-ArmRuns is only valid with Report.' }
if ($Arm -and $Stage -notin @('TrainArm','SampleArm')) { throw '-Arm is only valid with TrainArm or SampleArm.' }
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv-model\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:SUMO_HOME = 'E:\Program Files\sumo-1.22.0'
$env:PYTHONUNBUFFERED = '1'
$outputPath = Join-Path $projectRoot ('artifacts\runs\' + $RunId)
if (Test-Path -LiteralPath $outputPath) { throw 'Run output already exists; choose a fresh RunId.' }
$arguments = @('-m','sumodiff.diffusion.stage5b')
switch ($Stage) {
    'Preflight' { $arguments += 'preflight' }
    'Run' { $arguments += @('run','--allow-long-run') }
    'TrainArm' {
        if (-not $Arm) { throw 'TrainArm requires -Arm.' }
        $arguments += @('train-arm','--arm',$Arm,'--allow-long-run')
        if ($Resume) {
            $arguments += @('--resume',(Resolve-Path -LiteralPath $Resume).Path)
        }
    }
    'SampleArm' {
        if (-not $Arm -or -not $TrainingRun) { throw 'SampleArm requires -Arm and -TrainingRun.' }
        $arguments += @('sample-arm','--arm',$Arm,'--training-run',(Resolve-Path -LiteralPath $TrainingRun).Path)
    }
    'Report' {
        if (-not $ArmRuns -or $ArmRuns.Count -ne 4) { throw 'Report requires four -ArmRuns in A B C D order.' }
        $arguments += '--arm-runs'
        foreach ($runPath in $ArmRuns) {
            $arguments += (Resolve-Path -LiteralPath $runPath).Path
        }
        $arguments += 'report'
    }
}
$arguments += @('--config','configs/experiments/stage5b_ablation.yaml','--output',$outputPath,'--formal')
& $pythonPath @arguments
if ($LASTEXITCODE -ne 0) { throw ('Stage5B command failed with exit code ' + $LASTEXITCODE + '; original outputs are retained.') }
