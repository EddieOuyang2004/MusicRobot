[CmdletBinding()]
param(
    [ValidateSet("all", "preflight", "offline", "literature", "full", "ablation", "longrun", "features", "consolidate", "supplement", "reanalyse")]
    [string]$Stage = "all",
    [switch]$Resume,
    [switch]$DryRun,
    [string]$OutputDir = "",
    [double]$ControlRateHz = 120.0,
    [switch]$Pilot,
    [string]$SupplementRoot = "",
    [string]$ResultOutputDir = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "run_humanoid_matcher_experiments.py"
$featureBuilder = Join-Path $PSScriptRoot "build_aistpp_fact_feature_bundle.py"
$consolidator = Join-Path $PSScriptRoot "consolidate_thesis_experiments.py"
$preflightValidator = Join-Path $PSScriptRoot "validate_thesis_preflight.py"
$rawRoot = Join-Path $PSScriptRoot "output\thesis_final"
$resultRoot = Join-Path $repoRoot "docs\thesis\experiment_results"
$catalog = Join-Path $repoRoot "realtime\humanoid_robot\data\music_catalog\catalog.json"
$negativeRoot = Join-Path $repoRoot "realtime\humanoid_robot\data\test_audio\matcher_negative_set"
$changeRoot = Join-Path $repoRoot "realtime\humanoid_robot\data\test_audio\matcher_change_streams"

function Invoke-PythonStep {
    param([string]$Name, [string[]]$CommandArguments)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    Write-Host ($python + " " + ($CommandArguments -join " "))
    & $python @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Test-SelectedStage {
    param([string]$Name)
    return $Stage -eq "all" -or $Stage -eq $Name
}

function Invoke-RunnerSuite {
    param([string]$Name, [string]$Suite)
    $commandArguments = @(
        $runner,
        "--catalog", $catalog,
        "--output-dir", (Join-Path $rawRoot $Name),
        "--execute-suite", $Suite,
        "--no-audio-evaluation"
    )
    if ($Resume) { $commandArguments += "--resume" }
    if ($DryRun) { $commandArguments += "--dry-run" }
    Invoke-PythonStep -Name $Name -CommandArguments $commandArguments
}

Set-Location $repoRoot
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python was not found at $python"
}
if ($Stage -eq "supplement" -or $Stage -eq "reanalyse") {
    $entry = if ($Stage -eq "supplement") { "run_thesis_supplement.py" } else { "reanalyse_thesis_experiments.py" }
    $commandArguments = @((Join-Path $PSScriptRoot $entry))
    if ($Resume -and $Stage -eq "supplement") { $commandArguments += "--resume" }
    if ($DryRun) { $commandArguments += "--dry-run" }
    if ($Stage -eq "supplement") {
        $commandArguments += @("--control-rate-hz", $ControlRateHz.ToString([Globalization.CultureInfo]::InvariantCulture))
        if ($OutputDir) { $commandArguments += @("--output-dir", $OutputDir) }
        if ($Pilot) { $commandArguments += "--pilot" }
    } else {
        if ($SupplementRoot) { $commandArguments += @("--supplement-root", $SupplementRoot) }
        if ($ResultOutputDir) { $commandArguments += @("--output-dir", $ResultOutputDir) }
    }
    Invoke-PythonStep -Name $Stage -CommandArguments $commandArguments
    Write-Host "Requested stage '$Stage' completed." -ForegroundColor Green
    exit 0
}
New-Item -ItemType Directory -Force -Path $rawRoot | Out-Null
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

if (Test-SelectedStage "preflight") {
    foreach ($required in @($runner, $featureBuilder, $consolidator, $preflightValidator, $catalog)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required file is missing: $required"
        }
    }
    $catalogJson = Get-Content -LiteralPath $catalog -Raw | ConvertFrom-Json
    $trackCount = @($catalogJson.tracks.PSObject.Properties).Count
    $motionCount = @($catalogJson.motions.PSObject.Properties).Count
    $gmrRoot = Join-Path $repoRoot "realtime\humanoid_robot\data\aistpp_gmr"
    $gmrFileCount = @(Get-ChildItem $gmrRoot -File).Count
    $gmrMotionCount = @(Get-ChildItem $gmrRoot -Filter "*.pkl" -File).Count
    if ($trackCount -ne 60 -or $motionCount -ne 411 -or $gmrFileCount -ne 476 -or $gmrMotionCount -ne 411) {
        throw "Dataset count mismatch: tracks=$trackCount motions=$motionCount GMR-files=$gmrFileCount GMR-pkl=$gmrMotionCount"
    }
    $driveRoot = [System.IO.Path]::GetPathRoot($repoRoot)
    $freeBytes = ([System.IO.DriveInfo]::new($driveRoot)).AvailableFreeSpace
    if ($freeBytes -lt 10GB) {
        throw ("At least 10 GB free space is required; available: {0:N2} GB" -f ($freeBytes / 1GB))
    }
    if (@(Get-ChildItem $negativeRoot -Filter "*.wav" -File -ErrorAction SilentlyContinue).Count -ne 20) {
        Invoke-PythonStep -Name "build negative set" -CommandArguments @(
            (Join-Path $PSScriptRoot "build_matcher_negative_set.py"), "--output-dir", $negativeRoot
        )
    }
    if (-not (Test-Path -LiteralPath (Join-Path $changeRoot "manifest.json") -PathType Leaf)) {
        Invoke-PythonStep -Name "build change streams" -CommandArguments @(
            (Join-Path $PSScriptRoot "build_matcher_change_streams.py"), "--output-dir", $changeRoot
        )
    }
    Invoke-PythonStep -Name "validate preflight environment" -CommandArguments @(
        $preflightValidator,
        "--catalog", $catalog,
        "--output", (Join-Path $rawRoot "preflight\preflight_validation.json")
    )
    Invoke-RunnerSuite -Name "preflight" -Suite "smoke"
}

if (Test-SelectedStage "offline") {
    $commandArguments = @(
        $runner,
        "--catalog", $catalog,
        "--output-dir", (Join-Path $rawRoot "offline"),
        "--negative-audio-root", $negativeRoot
    )
    if ($DryRun) { $commandArguments += "--no-audio-evaluation" }
    Invoke-PythonStep -Name "offline" -CommandArguments $commandArguments
}

if (Test-SelectedStage "literature") { Invoke-RunnerSuite -Name "literature" -Suite "literature" }
if (Test-SelectedStage "full") { Invoke-RunnerSuite -Name "full" -Suite "full" }
if (Test-SelectedStage "ablation") { Invoke-RunnerSuite -Name "ablation" -Suite "ablation" }
if (Test-SelectedStage "longrun") { Invoke-RunnerSuite -Name "longrun" -Suite "longrun" }

if (Test-SelectedStage "features") {
    if ($DryRun) {
        Write-Host "`n=== features (dry run) ===" -ForegroundColor Cyan
        Write-Host "Feature extraction skipped because it requires completed literature traces."
    } else {
        Invoke-PythonStep -Name "features" -CommandArguments @(
            $featureBuilder,
            "--catalog", $catalog,
            "--run-status", (Join-Path $rawRoot "literature\run_status.json"),
            "--output", (Join-Path $rawRoot "features\aistpp_fact_features.npz")
        )
    }
}

if (Test-SelectedStage "consolidate") {
    if ($DryRun) {
        Write-Host "`n=== consolidate (dry run) ===" -ForegroundColor Cyan
        Write-Host "Consolidation skipped because dry-run records must never create READY_FOR_THESIS."
    } else {
        Invoke-PythonStep -Name "consolidate" -CommandArguments @(
            $consolidator, "--raw-root", $rawRoot, "--output-dir", $resultRoot
        )
    }
}

Write-Host "`nRequested stage '$Stage' completed." -ForegroundColor Green
if (-not $DryRun -and ($Stage -eq "all" -or $Stage -eq "consolidate")) {
    Write-Host "Ready marker: $(Join-Path $resultRoot 'READY_FOR_THESIS')"
}
