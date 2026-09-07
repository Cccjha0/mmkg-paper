param(
    [ValidateSet("Preflight", "Export", "Verify")]
    [string]$Mode = "Preflight",
    [ValidateSet("All", "MHyperNative", "MHyperAdaMF", "NativeAdaMF")]
    [string]$Pair = "All",
    [string]$Python = "python",
    [ValidateSet("cuda", "cpu", "auto")]
    [string]$Device = "cuda",
    [int]$ProgressEvery = 25,
    [switch]$NoResume,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$branch = (& git branch --show-current).Trim()
if ($branch -ne "m1/recent-mmkgc-baselines") {
    throw "MKG-Y Y2 must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$contractPath = "docs/protocols/MKG_Y_Y2_DEV_FULL_RANKING_CONTRACT.json"
$auditOutput = "outputs/complementarity_identifiability/mkg_y_y2_full_ranking"

function Invoke-PythonChecked {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $Python $($Arguments -join ' ')"
    }
}

function Invoke-Preflight {
    $arguments = @(
        "scripts/audit_mkg_y_y2_dev_export.py",
        "--contract", $contractPath,
        "--output-dir", $auditOutput,
        "--mode", "preflight"
    )
    if ($Overwrite) { $arguments += "--overwrite" }
    Invoke-PythonChecked $arguments
}

if ($Mode -eq "Preflight") {
    Invoke-Preflight
    exit 0
}

if ($Mode -eq "Verify") {
    $arguments = @(
        "scripts/audit_mkg_y_y2_dev_export.py",
        "--contract", $contractPath,
        "--output-dir", $auditOutput,
        "--mode", "verify"
    )
    if ($Overwrite) { $arguments += "--overwrite" }
    Invoke-PythonChecked $arguments
    exit 0
}

Invoke-Preflight
& $Python -c "import torch; assert torch.cuda.is_available() if '$Device' == 'cuda' else True; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "PyTorch/device preflight failed." }

$jobs = @(
    [pscustomobject]@{
        Key = "MHyperNative"
        PairName = "mkg_y_mhyper_native"
        ExpertA = "M-Hyper"
        ExpertB = "NativE"
        OutputDir = "outputs/mkg_y/anchored_dynamic/mhyper_native_seed123/full_ranking"
        Runs = @(
            "ml/artifacts/outputs/mkg_y_mhyper/20260903_215952_seed1::ml/artifacts/outputs/mkg_y_native/20260903_221947_seed1",
            "ml/artifacts/outputs/mkg_y_mhyper/20260903_220630_seed2::ml/artifacts/outputs/mkg_y_native/20260903_230141_seed2",
            "ml/artifacts/outputs/mkg_y_mhyper/20260903_221310_seed3::ml/artifacts/outputs/mkg_y_native/20260903_234233_seed3"
        )
    },
    [pscustomobject]@{
        Key = "MHyperAdaMF"
        PairName = "mkg_y_mhyper_adamf"
        ExpertA = "M-Hyper"
        ExpertB = "AdaMF-MAT"
        OutputDir = "outputs/mkg_y/anchored_dynamic/mhyper_adamf_seed123/full_ranking"
        Runs = @(
            "ml/artifacts/outputs/mkg_y_mhyper/20260903_215952_seed1::ml/artifacts/outputs/mkg_y_adamf_mat/20260904_002716_seed1",
            "ml/artifacts/outputs/mkg_y_mhyper/20260903_220630_seed2::ml/artifacts/outputs/mkg_y_adamf_mat/20260904_020546_seed2",
            "ml/artifacts/outputs/mkg_y_mhyper/20260903_221310_seed3::ml/artifacts/outputs/mkg_y_adamf_mat/20260904_035250_seed3"
        )
    },
    [pscustomobject]@{
        Key = "NativeAdaMF"
        PairName = "mkg_y_native_adamf"
        ExpertA = "NativE"
        ExpertB = "AdaMF-MAT"
        OutputDir = "outputs/mkg_y/anchored_dynamic/native_adamf_seed123/full_ranking"
        Runs = @(
            "ml/artifacts/outputs/mkg_y_native/20260903_221947_seed1::ml/artifacts/outputs/mkg_y_adamf_mat/20260904_002716_seed1",
            "ml/artifacts/outputs/mkg_y_native/20260903_230141_seed2::ml/artifacts/outputs/mkg_y_adamf_mat/20260904_020546_seed2",
            "ml/artifacts/outputs/mkg_y_native/20260903_234233_seed3::ml/artifacts/outputs/mkg_y_adamf_mat/20260904_035250_seed3"
        )
    }
)

$selectedJobs = if ($Pair -eq "All") { $jobs } else { @($jobs | Where-Object { $_.Key -eq $Pair }) }
foreach ($job in $selectedJobs) {
    Write-Host "[START] $($job.PairName) -> $($job.OutputDir)"
    $arguments = @(
        "scripts/eval_heterogeneous_complementarity.py",
        "--pair-name", $job.PairName,
        "--expert-a-name", $job.ExpertA,
        "--expert-b-name", $job.ExpertB,
        "--split", "dev",
        "--output-dir", $job.OutputDir,
        "--device", $Device,
        "--alphas", "0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00",
        "--rrf-k", "60",
        "--relation-min-support", "60",
        "--progress-every", $ProgressEvery.ToString(),
        "--dev-only-no-test-access",
        "--no-reference-check"
    )
    foreach ($run in $job.Runs) { $arguments += @("--run-pair", $run) }
    if ($NoResume) { $arguments += "--no-resume" }
    Invoke-PythonChecked $arguments
}

if ($Pair -eq "All") {
    $verifyArguments = @(
        "scripts/audit_mkg_y_y2_dev_export.py",
        "--contract", $contractPath,
        "--output-dir", $auditOutput,
        "--mode", "verify",
        "--overwrite"
    )
    Invoke-PythonChecked $verifyArguments
} else {
    Write-Host "[INFO] Partial pair export completed. Run -Mode Verify after all three pairs exist."
}
