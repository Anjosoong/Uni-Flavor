param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("odor", "taste")]
    [string]$Task = "odor",

    [Parameter(Mandatory = $false)]
    [ValidateSet("all", "mpnn", "chemberta")]
    [string]$Model = "all",

    [Parameter(Mandatory = $false)]
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"
$BenchmarkRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $BenchmarkRoot

$Config = "configs/$Task.json"

function Run-Step {
    param([string]$Name, [string[]]$Cmd)
    Write-Host "`n========== $Name ==========" -ForegroundColor Cyan
    $exe = $Cmd[0]
    $stepArgs = @()
    if ($Cmd.Length -gt 1) {
        $stepArgs = $Cmd[1..($Cmd.Length - 1)]
    }
    Write-Host ">>> $exe $($stepArgs -join ' ')" -ForegroundColor DarkGray
    & $exe @stepArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipSetup) {
    Run-Step "Setup deps" @("python", "setup_deps.py", "--config", $Config)
}

if ($Model -eq "all" -or $Model -eq "mpnn") {
    Run-Step "MPNN" @("python", "mpnn/train_mpnn.py", "--config", $Config)
}

if ($Model -eq "all" -or $Model -eq "chemberta") {
    Run-Step "ChemBERTa" @("python", "chemberta/train_chemberta.py", "--config", $Config)
}

Run-Step "Aggregate" @("python", "aggregate_results.py", "--task", $Task)

Write-Host "`nDone. See output/$Task/benchmark_summary.csv" -ForegroundColor Green
