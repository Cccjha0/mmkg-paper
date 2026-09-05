param(
    [ValidateSet("Preflight", "Systematic", "Analyze")]
    [string]$Mode = "Preflight",
    [string]$Python = "python",
    [ValidateSet("cuda", "cpu", "auto")]
    [string]$Device = "cuda",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

$branch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne "m1/recent-mmkgc-baselines") {
    throw "Experiment 2 must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$exp1 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp1_landscape/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp1.gate.decision -ne "GO") {
    throw "Experiment 1 Available Complementarity Gate is not GO."
}

$contract = "docs/protocols/EXP2_INFORMATION_FEATURE_CONTRACT.json"
$outputRoot = "outputs/complementarity_identifiability/exp2_information"
$pairs = @(
    "mkgw_mhyper_native",
    "mkgw_mhyper_adamf",
    "mkgw_native_adamf",
    "db15k_mhyper_native",
    "db15k_mhyper_adamf",
    "db15k_native_adamf"
)
$tabularLearners = @("linear_huber", "hist_gbdt", "mlp_low", "mlp_high")
$representations = @("X1", "X2", "X3", "X4", "X5")
$overwriteArg = @()
if ($Overwrite) { $overwriteArg = @("--overwrite") }

function Invoke-PythonChecked {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $Python $($Arguments -join ' ')"
    }
}

if ($Mode -eq "Preflight") {
    Invoke-PythonChecked @(
        "scripts/build_exp2_information_assets.py",
        "--contract", $contract,
        "--output-dir", "$outputRoot/preflight/assets",
        "--dry-run", "--overwrite"
    )
    foreach ($pair in $pairs) {
        Invoke-PythonChecked @(
            "scripts/build_exp2_union_top100.py",
            "--pair-id", $pair,
            "--contract", $contract,
            "--output-dir", "$outputRoot/preflight/candidate_assets",
            "--dry-run", "--overwrite"
        )
    }
    Write-Host "[OK] Experiment 2 preflight completed. No model was trained and no candidate scores were reconstructed."
    exit 0
}

if ($Mode -eq "Systematic") {
    Invoke-PythonChecked @("-c", "import numpy,pandas,sklearn,torch,yaml; assert torch.cuda.is_available() or '$Device' != 'cuda'; print('runtime dependencies OK')")

    $assetsReady = $true
    foreach ($pair in $pairs) {
        $assetsReady = $assetsReady -and (Test-Path -LiteralPath "$outputRoot/assets/${pair}_query_information.npz")
        $assetsReady = $assetsReady -and (Test-Path -LiteralPath "$outputRoot/assets/${pair}_query_information_manifest.json")
    }
    if (-not $assetsReady -or $Overwrite) {
        $arguments = @(
            "scripts/build_exp2_information_assets.py",
            "--contract", $contract,
            "--output-dir", "$outputRoot/assets",
            "--overwrite"
        )
        Invoke-PythonChecked $arguments
    }

    foreach ($pair in $pairs) {
        $candidateManifest = "$outputRoot/candidate_assets/${pair}_union_top100_manifest.json"
        $candidateAsset = "$outputRoot/candidate_assets/${pair}_union_top100.npz"
        if (-not (Test-Path -LiteralPath $candidateManifest) -or -not (Test-Path -LiteralPath $candidateAsset) -or $Overwrite) {
            $arguments = @(
                "scripts/build_exp2_union_top100.py",
                "--pair-id", $pair,
                "--contract", $contract,
                "--output-dir", "$outputRoot/candidate_assets",
                "--device", $Device,
                "--overwrite"
            )
            Invoke-PythonChecked $arguments
        }

        foreach ($representation in $representations) {
            foreach ($learner in $tabularLearners) {
                $metrics = "$outputRoot/runs/$pair/$($representation.ToLower())/$learner/metrics.json"
                $predictions = "$outputRoot/runs/$pair/$($representation.ToLower())/$learner/oof_action_predictions.npz"
                if ((Test-Path -LiteralPath $metrics) -and (Test-Path -LiteralPath $predictions) -and -not $Overwrite) {
                    Write-Host "[SKIP] $pair $representation $learner"
                    continue
                }
                $arguments = @(
                    "scripts/run_exp2_information_nested_oof.py",
                    "--pair-id", $pair,
                    "--representation", $representation,
                    "--learner", $learner,
                    "--contract", $contract,
                    "--device", $Device,
                    "--overwrite"
                )
                Invoke-PythonChecked $arguments
            }
        }

        $x6Metrics = "$outputRoot/runs/$pair/x6/set_encoder/metrics.json"
        $x6Predictions = "$outputRoot/runs/$pair/x6/set_encoder/oof_action_predictions.npz"
        if (-not (Test-Path -LiteralPath $x6Metrics) -or -not (Test-Path -LiteralPath $x6Predictions) -or $Overwrite) {
            $arguments = @(
                "scripts/run_exp2_information_nested_oof.py",
                "--pair-id", $pair,
                "--representation", "X6",
                "--learner", "set_encoder",
                "--contract", $contract,
                "--device", $Device,
                "--overwrite"
            )
            Invoke-PythonChecked $arguments
        }
    }
    Write-Host "[OK] Systematic Experiment 2 runs completed. Run this script again with -Mode Analyze."
    exit 0
}

if ($Mode -eq "Analyze") {
    $arguments = @(
        "scripts/analyze_exp2_information_audit.py",
        "--contract", $contract,
        "--output-dir", $outputRoot,
        "--report", "docs/reports/information_identifiability_audit_2026-09-05.md"
    ) + $overwriteArg
    Invoke-PythonChecked $arguments
    Write-Host "[OK] Experiment 2 audit and frozen preliminary gate were generated. No method development was started."
}
