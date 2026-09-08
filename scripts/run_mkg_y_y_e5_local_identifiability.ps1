param(
    [ValidateSet("Preflight", "Systematic", "Analyze")]
    [string]$Mode = "Preflight",
    [string]$Python = "python",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("", "mkg_y_mhyper_native", "mkg_y_mhyper_adamf", "mkg_y_native_adamf")]
    [string]$PairId = "",
    [int]$CenterBatchSize = 4096,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

$branch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne "m1/recent-mmkgc-baselines") {
    throw "MKG-Y Y-E5 must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$exp1 = Get-Content -LiteralPath "outputs/complementarity_identifiability/mkg_y_y_e1_landscape/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp1.split -ne "dev" -or $exp1.assessment.outcome -ne "Y_E1_AVAILABLE_COMPLEMENTARITY_PRESENT") {
    throw "MKG-Y Y-E1 is not a frozen completed DEV replication."
}
$exp2 = Get-Content -LiteralPath "outputs/complementarity_identifiability/mkg_y_y_e2_information/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp2.split -ne "dev" -or $exp2.assessment.outcome -ne "Y_E2_X4_REPLICATION_REPORTED") {
    throw "MKG-Y Y-E2 is not a frozen completed DEV replication."
}
$exp4 = Get-Content -LiteralPath "outputs/complementarity_identifiability/mkg_y_y_e4_cross_seed_transfer/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp4.split -ne "dev" -or $exp4.assessment.outcome -ne "Y_E4_CROSS_SEED_REPLICATION_REPORTED") {
    throw "MKG-Y Y-E4 is not a frozen completed DEV replication."
}

& $Python -c "import numpy,pandas; print('table dependencies OK')"
if ($LASTEXITCODE -ne 0) { throw "MKG-Y Y-E5 requires numpy and pandas." }
if ($Mode -eq "Systematic") {
    & $Python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "MKG-Y Y-E5 systematic mode requires torch." }
}

$arguments = @(
    "scripts/audit_local_identifiability.py",
    "--profile", "mkg_y",
    "--mode", $Mode.ToLowerInvariant(),
    "--contract", "docs/protocols/MKG_Y_Y_E5_LOCAL_IDENTIFIABILITY_CONTRACT.json",
    "--exp1-root", "outputs/complementarity_identifiability/mkg_y_y_e1_landscape",
    "--exp2-root", "outputs/complementarity_identifiability/mkg_y_y_e2_information",
    "--exp4-root", "outputs/complementarity_identifiability/mkg_y_y_e4_cross_seed_transfer",
    "--utility-manifest-dir", "outputs/complementarity_identifiability/mkg_y_y_e2_information/utility_tables",
    "--output-dir", "outputs/complementarity_identifiability/mkg_y_y_e5_local_identifiability",
    "--report", "docs/reports/mkg_y_local_identifiability_audit_2026-09-08.md",
    "--device", $Device,
    "--center-batch-size", $CenterBatchSize
)
if ($PairId -ne "") { $arguments += @("--pair-id", $PairId) }
if ($Overwrite) { $arguments += "--overwrite" }

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "MKG-Y Y-E5 command failed: $Python $($arguments -join ' ')"
}

switch ($Mode) {
    "Preflight" { Write-Host "[OK] MKG-Y Y-E5 preflight completed; X4/folds verified, no kNN computation." }
    "Systematic" { Write-Host "[OK] MKG-Y Y-E5 systematic artifacts completed; no selector was trained." }
    "Analyze" { Write-Host "[OK] MKG-Y Y-E5 descriptive audit completed; TEST remains locked." }
}
