[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('Pilot','Collect','Prepare','Audit','Train','Resume')][string]$Action,
    [string]$Name='stage5a-v1',
    [string]$RunSuffix='',
    [string]$Checkpoint='',
    [string]$PilotReport='',
    [string]$ScaleAudit='',
    [switch]$ReuseCompleted,
    [string]$SumoHome='E:\Program Files\sumo-1.22.0'
)
$ErrorActionPreference='Stop'
if ($Name -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$' -or ($RunSuffix -and $RunSuffix -notmatch '^[a-zA-Z0-9_-]{1,32}$')) {
    throw 'Name/RunSuffix must be simple identifiers, not paths.'
}
$taskRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython=Join-Path $taskRoot '.venv-model\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { throw 'Project .venv-model Python is missing; see docs/stage5a.md.' }
$env:PYTHONPATH=Join-Path $taskRoot 'src'
$env:SUMO_HOME=$SumoHome
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$taskRunName=$Name
if ($RunSuffix) { $taskRunName="$Name-$RunSuffix" }
$taskPlan='configs/experiments/stage5a.yaml'
$taskData="artifacts/processed/$Name/dataset"
$taskAudit="artifacts/runs/$Name-scales/scale_audit.json"
if ($ScaleAudit) { $taskAudit=$ScaleAudit }
$taskRepository=(Get-Location).Path
if (-not $PilotReport) { $PilotReport="artifacts/runs/$Name-pilot/preparation/quality_report.json" }

function Invoke-Stage5APython {
    param([string[]]$Arguments)
    & $taskPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage5A command failed with exit $LASTEXITCODE; inspect retained status.json/report before dependent steps." }
}
switch ($Action) {
    'Pilot' {
        Invoke-Stage5APython @('-m','sumodiff.experiments.stage5a','pilot','--config',$taskPlan,
            '--output',"artifacts/runs/$taskRunName-pilot",'--repository',$taskRepository,'--formal')
    }
    'Collect' {
        $taskArgs=@('-m','sumodiff.experiments.stage5a','collect','--config',$taskPlan,
            '--pilot-report',$PilotReport,
            '--campaign',"artifacts/raw/$Name",'--output',"artifacts/runs/$taskRunName-collect",
            '--repository',$taskRepository,'--allow-bulk','--formal')
        if ($ReuseCompleted) { $taskArgs+='--reuse-completed' }
        Invoke-Stage5APython $taskArgs
    }
    'Prepare' {
        Invoke-Stage5APython @('-m','sumodiff.experiments.stage5a','prepare','--config',$taskPlan,
            '--sources',"artifacts/raw/$Name/sources.json",'--output',"artifacts/processed/$Name",
            '--repository',$taskRepository,'--formal')
        Invoke-Stage5APython @('-m','sumodiff.diffusion','audit-scales','--dataset',$taskData,
            '--config','configs/train/stage5a_epochs.yaml','--output',"artifacts/runs/$Name-scales",
            '--repository',$taskRepository,'--formal')
    }
    'Audit' {
        Invoke-Stage5APython @('-m','sumodiff.diffusion','audit-scales','--dataset',$taskData,
            '--config','configs/train/stage5a_epochs.yaml','--output',"artifacts/runs/$taskRunName-scales",
            '--repository',$taskRepository,'--formal')
    }
    'Train' {
        Invoke-Stage5APython @('-m','sumodiff.diffusion','train','--dataset',$taskData,
            '--model-config','configs/model/initial.yaml','--config','configs/train/stage5a_epochs.yaml',
            '--scale-audit',$taskAudit,'--output',"artifacts/runs/$taskRunName-train",
            '--repository',$taskRepository,'--allow-long-run','--formal')
    }
    'Resume' {
        if (-not $Checkpoint -or -not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
            throw 'Resume requires -Checkpoint pointing to an existing committed-step checkpoint.'
        }
        if (-not $RunSuffix) { throw 'Resume requires a fresh -RunSuffix; original run is never overwritten.' }
        Invoke-Stage5APython @('-m','sumodiff.diffusion','train','--dataset',$taskData,
            '--model-config','configs/model/initial.yaml','--config','configs/train/stage5a_epochs.yaml',
            '--scale-audit',$taskAudit,'--resume',$Checkpoint,'--output',"artifacts/runs/$taskRunName-train",
            '--repository',$taskRepository,'--allow-long-run','--formal')
    }
}
