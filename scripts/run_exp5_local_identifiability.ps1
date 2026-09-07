param(
    [ValidateSet("Preflight", "Systematic", "Analyze")]
    [string]$Mode = "Preflight",
    [string]$Python = "python",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("", "mkgw_mhyper_native", "mkgw_mhyper_adamf", "mkgw_native_adamf", "db15k_mhyper_native", "db15k_mhyper_adamf", "db15k_native_adamf")]
    [string]$PairId = "",
    [int]$CenterBatchSize = 512,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

$branch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne "m1/recent-mmkgc-baselines") {
    throw "Experiment 5 must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$exp1 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp1_landscape/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp1.gate.decision -ne "GO") { throw "Experiment 1 Available Complementarity Gate is not GO." }
$exp2 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp2_information/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp2.split -ne "dev" -or $exp2.operational_audit.test_access -ne 0) { throw "Experiment 2 is not a clean DEV-only input." }
$exp4 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp4_cross_seed_transfer/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp4.split -ne "dev" -or $exp4.operational_audit.test_access -ne 0) { throw "Experiment 4 is not a clean frozen DEV-only input." }

& $Python -c "import numpy,pandas; print('table dependencies OK')"
if ($LASTEXITCODE -ne 0) { throw "Experiment 5 requires numpy and pandas." }
if ($Mode -eq "Systematic") {
    & $Python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "Experiment 5 systematic mode requires PyTorch." }
    if ($Device -eq "cuda") {
        & $Python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
        if ($LASTEXITCODE -ne 0) { throw "CUDA mode requested but torch.cuda.is_available() is false." }
    }
}

$pythonMode = $Mode.ToLowerInvariant()
$arguments = @(
    "scripts/audit_local_identifiability.py",
    "--mode", $pythonMode,
    "--contract", "docs/protocols/EXP5_LOCAL_IDENTIFIABILITY_CONTRACT.json",
    "--output-dir", "outputs/complementarity_identifiability/exp5_local_identifiability",
    "--report", "docs/reports/local_identifiability_action_ambiguity_audit_2026-09-07.md"
)
if ($Mode -eq "Systematic") {
    $arguments += @("--device", $Device, "--center-batch-size", $CenterBatchSize)
}
if ($PairId -ne "") { $arguments += @("--pair-id", $PairId) }
if ($Overwrite) { $arguments += "--overwrite" }

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Experiment 5 command failed: $Python $($arguments -join ' ')"
}

switch ($Mode) {
    "Preflight" { Write-Host "[OK] Experiment 5 preflight completed; X4/folds verified, no kNN computation." }
    "Systematic" { Write-Host "[OK] Experiment 5 systematic artifacts completed; no selector was trained." }
    "Analyze" { Write-Host "[OK] Experiment 5 report/classification completed; no subsequent experiment was started." }
}
