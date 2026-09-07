param(
    [ValidateSet("Preflight", "Systematic", "Analyze")]
    [string]$Mode = "Preflight",
    [string]$Python = "python",
    [ValidateSet("cuda", "cpu", "auto")]
    [string]$Device = "cuda",
    [int]$QueryBatchSize = 0,
    [int]$ChunkSize = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

$branch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne "m1/recent-mmkgc-baselines") {
    throw "MKG-Y Y-E2 must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$contract = "docs/protocols/MKG_Y_Y_E2_X4_OOF_CONTRACT.json"
$outputRoot = "outputs/complementarity_identifiability/mkg_y_y_e2_information"
$yE1Manifest = "outputs/complementarity_identifiability/mkg_y_y_e1_landscape/audit_manifest.json"
$yE1Stats = "outputs/complementarity_identifiability/mkg_y_y_e1_landscape/pair_statistics.csv"
$learners = @("linear_huber", "hist_gbdt", "mlp_low", "mlp_high")
$pairs = @(
    [pscustomobject]@{ Id = "mkg_y_mhyper_native"; Directory = "mhyper_native_seed123" },
    [pscustomobject]@{ Id = "mkg_y_mhyper_adamf"; Directory = "mhyper_adamf_seed123" },
    [pscustomobject]@{ Id = "mkg_y_native_adamf"; Directory = "native_adamf_seed123" }
)

function Invoke-PythonChecked {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $Python $($Arguments -join ' ')"
    }
}

function Assert-FrozenInputs {
    Invoke-PythonChecked @(
        "scripts/preflight_mkg_y_y_e2_x4.py",
        "--contract", $contract,
        "--output", "$outputRoot/preflight.json",
        "--overwrite"
    )
    $assessment = Get-Content -LiteralPath $yE1Manifest -Raw | ConvertFrom-Json
    if ($assessment.assessment.outcome -ne "Y_E1_AVAILABLE_COMPLEMENTARITY_PRESENT") {
        throw "MKG-Y Y-E1 prerequisite assessment is not present."
    }
    foreach ($pair in $pairs) {
        $utility = "$outputRoot/utility_tables/$($pair.Id)_dev_utility_table.csv.gz"
        $utilityManifest = "$outputRoot/utility_tables/$($pair.Id)_dev_source_manifest.json"
        $summary = "outputs/mkg_y/anchored_dynamic/$($pair.Directory)/full_ranking/dev_summary.json"
        $runManifest = "outputs/complementarity_identifiability/mkg_y_y2_full_ranking/$($pair.Id)_dev_source_manifest.json"
        foreach ($path in @($utility, $utilityManifest, $summary, $runManifest)) {
            if (-not (Test-Path -LiteralPath $path)) { throw "Missing frozen Y-E2 input: $path" }
        }
    }
}

Assert-FrozenInputs

if ($Mode -eq "Preflight") {
    Invoke-PythonChecked @(
        "-c",
        "import numpy,pandas,sklearn,torch,yaml; print('runtime dependencies OK; cuda=' + str(torch.cuda.is_available()))"
    )
    foreach ($pair in $pairs) {
        $arguments = @(
            "scripts/build_aacpi_action_response_features.py",
            "--utility-table", "$outputRoot/utility_tables/$($pair.Id)_dev_utility_table.csv.gz",
            "--full-ranking-summary", "outputs/mkg_y/anchored_dynamic/$($pair.Directory)/full_ranking/dev_summary.json",
            "--frozen-run-manifest", "outputs/complementarity_identifiability/mkg_y_y2_full_ranking/$($pair.Id)_dev_source_manifest.json",
            "--output-dir", "$outputRoot/preflight/phase3a/$($pair.Id)",
            "--dry-run", "--overwrite"
        )
        Invoke-PythonChecked $arguments
    }
    Write-Host "[OK] MKG-Y Y-E2 preflight passed. No checkpoint inference, learner training, or TEST access occurred."
    exit 0
}

if ($Mode -eq "Systematic") {
    Invoke-PythonChecked @(
        "-c",
        "import numpy,pandas,sklearn,torch,yaml; assert torch.cuda.is_available() or '$Device' != 'cuda'; print('systematic runtime OK; cuda=' + str(torch.cuda.is_available()))"
    )
    foreach ($pair in $pairs) {
        $phase3aDir = "$outputRoot/context/$($pair.Id)/phase3a"
        $phase3aTable = "$phase3aDir/dev_action_response_features.csv.gz"
        $phase3aManifest = "$phase3aDir/candidate_score_source_manifest.json"
        if (-not (Test-Path -LiteralPath $phase3aTable) -or -not (Test-Path -LiteralPath $phase3aManifest) -or $Overwrite) {
            $arguments = @(
                "scripts/build_aacpi_action_response_features.py",
                "--utility-table", "$outputRoot/utility_tables/$($pair.Id)_dev_utility_table.csv.gz",
                "--full-ranking-summary", "outputs/mkg_y/anchored_dynamic/$($pair.Directory)/full_ranking/dev_summary.json",
                "--frozen-run-manifest", "outputs/complementarity_identifiability/mkg_y_y2_full_ranking/$($pair.Id)_dev_source_manifest.json",
                "--output-dir", $phase3aDir,
                "--device", $Device,
                "--overwrite"
            )
            if ($QueryBatchSize -gt 0) { $arguments += @("--query-batch-size", "$QueryBatchSize") }
            if ($ChunkSize -gt 0) { $arguments += @("--chunk-size", "$ChunkSize") }
            Invoke-PythonChecked $arguments
        } else {
            Write-Host "[SKIP] frozen candidate-score reconstruction exists: $($pair.Id)"
        }

        $contextDir = "$outputRoot/context/$($pair.Id)"
        $contextTable = "$contextDir/context_features.csv.gz"
        $contextManifest = "$contextDir/context_feature_manifest.json"
        if (-not (Test-Path -LiteralPath $contextTable) -or -not (Test-Path -LiteralPath $contextManifest) -or $Overwrite) {
            Invoke-PythonChecked @(
                "scripts/build_aacpi_phase4a_context_features.py",
                "--phase3a-feature-table", $phase3aTable,
                "--phase3a-source-manifest", $phase3aManifest,
                "--output-table", $contextTable,
                "--output-manifest", $contextManifest,
                "--overwrite"
            )
        } else {
            Write-Host "[SKIP] frozen TRAIN-only X4 context exists: $($pair.Id)"
        }
    }

    $assetArguments = @(
        "scripts/build_mkg_y_y_e2_information_assets.py",
        "--contract", $contract,
        "--phase4a-root", "$outputRoot/context",
        "--output-dir", "$outputRoot/assets"
    )
    if ($Overwrite) { $assetArguments += "--overwrite" }
    $allAssetsReady = $true
    foreach ($pair in $pairs) {
        $allAssetsReady = $allAssetsReady -and (Test-Path -LiteralPath "$outputRoot/assets/$($pair.Id)_query_information.npz")
        $allAssetsReady = $allAssetsReady -and (Test-Path -LiteralPath "$outputRoot/assets/$($pair.Id)_query_information_manifest.json")
    }
    if (-not $allAssetsReady -or $Overwrite) {
        Invoke-PythonChecked $assetArguments
    } else {
        Write-Host "[SKIP] all frozen MKG-Y X4 query assets exist"
    }

    foreach ($pair in $pairs) {
        foreach ($learner in $learners) {
            $runDir = "$outputRoot/runs/$($pair.Id)/x4/$learner"
            if ((Test-Path -LiteralPath "$runDir/metrics.json") -and (Test-Path -LiteralPath "$runDir/oof_action_predictions.npz") -and -not $Overwrite) {
                Write-Host "[SKIP] $($pair.Id) X4 $learner"
                continue
            }
            $arguments = @(
                "scripts/run_exp2_information_nested_oof.py",
                "--pair-id", $pair.Id,
                "--representation", "X4",
                "--learner", $learner,
                "--asset-dir", "$outputRoot/assets",
                "--utility-manifest-dir", "$outputRoot/utility_tables",
                "--exp1-stats", $yE1Stats,
                "--exp1-manifest", $yE1Manifest,
                "--contract", $contract,
                "--output-root", "$outputRoot/runs",
                "--device", $Device,
                "--overwrite"
            )
            Invoke-PythonChecked $arguments
        }
    }
    Write-Host "[OK] MKG-Y Y-E2 frozen X4 systematic runs completed. Run -Mode Analyze next."
    exit 0
}

if ($Mode -eq "Analyze") {
    $arguments = @(
        "scripts/analyze_mkg_y_y_e2_x4_oof.py",
        "--contract", $contract,
        "--run-root", "$outputRoot/runs",
        "--asset-dir", "$outputRoot/assets",
        "--utility-manifest-dir", "$outputRoot/utility_tables",
        "--exp1-stats", $yE1Stats,
        "--output-dir", $outputRoot,
        "--report", "docs/reports/mkg_y_information_identifiability_audit_2026-09-08.md"
    )
    if ($Overwrite) { $arguments += "--overwrite" }
    Invoke-PythonChecked $arguments
    Write-Host "[OK] MKG-Y Y-E2 X4 descriptive replication audit generated. X6 and method development remain closed."
}
