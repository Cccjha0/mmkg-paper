param(
    [ValidateSet("Preflight", "Run")]
    [string]$Mode = "Preflight",
    [string]$Python = "python",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

$branch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne "m1/recent-mmkgc-baselines") {
    throw "Experiment 4 must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$exp1 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp1_landscape/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp1.gate.decision -ne "GO") {
    throw "Experiment 1 Available Complementarity Gate is not GO."
}
$exp2 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp2_information/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp2.split -ne "dev" -or $exp2.operational_audit.test_access -ne 0) {
    throw "Experiment 2 audit is not a clean DEV-only input."
}
$exp3 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp3_resolution/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp3.decision -ne "ROUTE_C_LIMITS" -or $exp3.operational_audit.test_access -ne 0) {
    throw "Experiment 3 frozen ROUTE_C_LIMITS audit is missing or inconsistent."
}

& $Python -c "import numpy,pandas; print('runtime dependencies OK')"
if ($LASTEXITCODE -ne 0) { throw "Experiment 4 requires numpy and pandas." }

$arguments = @(
    "scripts/audit_cross_seed_transfer.py",
    "--contract", "docs/protocols/EXP4_CROSS_SEED_TRANSFER_CONTRACT.json",
    "--output-dir", "outputs/complementarity_identifiability/exp4_cross_seed_transfer",
    "--report", "docs/reports/cross_seed_transferable_complementarity_audit_2026-09-07.md"
)
if ($Mode -eq "Preflight") {
    $arguments += "--dry-run"
} elseif ($Overwrite) {
    $arguments += "--overwrite"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Experiment 4 command failed: $Python $($arguments -join ' ')"
}

if ($Mode -eq "Preflight") {
    Write-Host "[OK] Experiment 4 preflight completed; DEV tables only, no checkpoint execution."
} else {
    Write-Host "[OK] Experiment 4 audit completed; Experiment 5 was not started."
}
