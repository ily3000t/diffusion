[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('Pilot','Collect','Prepare','Audit','Calibrate','Preflight','Train','Resume','Validate','Seal','Test')][string]$Action,
    [string]$Name='stage5a-fresh',
    [string]$RunSuffix='',
    [string]$Checkpoint='',
    [string]$PilotReport='',
    [string]$ReusePilot='',
    [string]$ScaleAudit='',
    [string]$TrainingConfig='',
    [string]$ValidationRun='',
    [string]$TestSeal='',
    [switch]$ReuseCompleted,
    [string]$SumoHome='E:\Program Files\sumo-1.22.0'
)
$ErrorActionPreference='Stop'
if ($Name -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$' -or ($RunSuffix -and $RunSuffix -notmatch '^[a-zA-Z0-9_-]{1,32}$')) {
    throw 'Name/RunSuffix must be simple identifiers.'
}
if ($Checkpoint -and $Action -notin @('Resume','Validate','Seal','Test')) {
    throw 'Checkpoint is only valid for Resume/Validate/Seal/Test; Train always starts from random initialization.'
}
if ($RunSuffix -and $Action -eq 'Prepare') { throw 'Prepare creates the named dataset once; choose a fresh -Name rather than a suffix.' }
$taskRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython=Join-Path $taskRoot '.venv-model\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { throw 'Project Python is missing; see docs/stage5a.md.' }
$env:PYTHONPATH=Join-Path $taskRoot 'src'
$env:SUMO_HOME=$SumoHome
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$taskRunName=$Name
if ($RunSuffix) { $taskRunName="$Name-$RunSuffix" }
$taskPlan='configs/experiments/stage5a.yaml'
$taskRecipe='configs/train/stage5a_epochs.yaml'
$taskData="artifacts/processed/$Name/dataset"
$taskAudit="artifacts/runs/$Name-scales/scale_audit.json"
if ($ScaleAudit) { $taskAudit=$ScaleAudit }
if (-not $TrainingConfig) { $TrainingConfig="artifacts/runs/$Name-calibration/training_config.yaml" }
if (-not $PilotReport) { $PilotReport="artifacts/runs/$Name-pilot/preparation/quality_report.json" }
if (-not $ValidationRun) { $ValidationRun="artifacts/runs/$Name-validate" }
if (-not $TestSeal) { $TestSeal="artifacts/runs/$Name-test-seal/test_seal.json" }
$taskRepository=(Get-Location).Path
function Invoke-Stage5APython {
    param([string[]]$Arguments)
    & $taskPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit $LASTEXITCODE; inspect retained status/report before proceeding." }
}
switch ($Action) {
    'Pilot' {
        $taskArgs=@('-m','sumodiff.experiments.stage5a','pilot','--config',$taskPlan,
            '--output',"artifacts/runs/$taskRunName-pilot",'--repository',$taskRepository,'--formal')
        if ($ReusePilot) { $taskArgs+=@('--reuse-pilot',$ReusePilot) }
        Invoke-Stage5APython $taskArgs
    }
    'Collect' {
        $taskArgs=@('-m','sumodiff.experiments.stage5a','collect','--config',$taskPlan,
            '--pilot-report',$PilotReport,'--campaign',"artifacts/raw/$Name",
            '--output',"artifacts/runs/$taskRunName-collect",'--repository',$taskRepository,'--allow-bulk','--formal')
        if ($ReuseCompleted) { $taskArgs+='--reuse-completed' }
        Invoke-Stage5APython $taskArgs
    }
    'Prepare' {
        Invoke-Stage5APython @('-m','sumodiff.experiments.stage5a','prepare','--config',$taskPlan,
            '--sources',"artifacts/raw/$Name/sources.json",'--output',"artifacts/processed/$Name",'--repository',$taskRepository,'--formal')
        Invoke-Stage5APython @('-m','sumodiff.diffusion','audit-scales','--dataset',$taskData,
            '--config',$taskRecipe,'--output',"artifacts/runs/$Name-scales",'--repository',$taskRepository,'--formal')
    }
    'Audit' {
        Invoke-Stage5APython @('-m','sumodiff.diffusion','audit-scales','--dataset',$taskData,
            '--config',$taskRecipe,'--output',"artifacts/runs/$taskRunName-scales",'--repository',$taskRepository,'--formal')
    }
    'Calibrate' {
        Invoke-Stage5APython @('-m','sumodiff.diffusion','calibrate','--dataset',$taskData,
            '--model-config','configs/model/initial.yaml','--config',$taskRecipe,
            '--output',"artifacts/runs/$taskRunName-calibration",'--repository',$taskRepository,'--formal')
    }
    { $_ -in 'Train','Resume','Preflight' } {
        if (-not (Test-Path -LiteralPath $TrainingConfig -PathType Leaf)) { throw 'Run Calibrate first; -TrainingConfig must be its frozen output.' }
        $taskArgs=@('-m','sumodiff.diffusion','train','--dataset',$taskData,
            '--model-config','configs/model/initial.yaml','--config',$TrainingConfig,
            '--scale-audit',$taskAudit,'--output',"artifacts/runs/$taskRunName-$($Action.ToLower())",
            '--repository',$taskRepository,'--allow-long-run','--formal')
        if ($Action -eq 'Resume') {
            if (-not $Checkpoint -or -not (Test-Path -LiteralPath $Checkpoint -PathType Leaf) -or -not $RunSuffix) {
                throw 'Resume requires an existing -Checkpoint and a fresh -RunSuffix.'
            }
            $taskArgs+=@('--resume',$Checkpoint)
        }
        if ($Action -eq 'Preflight') { $taskArgs+=@('--prefix-updates','8') }
        Invoke-Stage5APython $taskArgs
    }
    { $_ -in 'Validate','Seal','Test' } {
        if (-not $Checkpoint -or -not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw 'An explicit selected -Checkpoint is required.' }
        if ($Action -eq 'Seal') {
            Invoke-Stage5APython @('-m','sumodiff.diffusion','seal-test','--dataset',$taskData,
                '--checkpoint',$Checkpoint,'--config','configs/experiments/stage5a_test.yaml',
                '--validation-run',$ValidationRun,'--output',"artifacts/runs/$taskRunName-test-seal",'--repository',$taskRepository,'--formal')
        } else {
            $taskSampleConfig='configs/experiments/stage5a_validation.yaml'
            if ($Action -eq 'Test') { $taskSampleConfig='configs/experiments/stage5a_test.yaml' }
            $taskArgs=@('-m','sumodiff.diffusion','sample','--dataset',$taskData,
                '--checkpoint',$Checkpoint,'--config',$taskSampleConfig,
                '--output',"artifacts/runs/$taskRunName-$($Action.ToLower())",'--repository',$taskRepository,'--formal')
            if ($Action -eq 'Test') { $taskArgs+=@('--test-seal',$TestSeal) }
            Invoke-Stage5APython $taskArgs
        }
    }
}
