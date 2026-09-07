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
    throw "Part II must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$exp2 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp2_information/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp2.operational_audit.test_access -ne 0 -or $exp2.operational_audit.final_policy_development -ne 0) {
    throw "Experiment 2 is not a clean frozen DEV-only input."
}
$exp6 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp6_stable_headroom/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp6.split -ne "dev" -or $exp6.operational_audit.test_access -ne 0) {
    throw "Experiment 6 is not a clean frozen DEV-only input."
}

& $Python -c "import numpy,pandas; print('runtime dependencies OK')"
if ($LASTEXITCODE -ne 0) { throw "Part II requires numpy and pandas." }

$arguments = @(
    "scripts/audit_high_value_selectivity.py",
    "--contract", "docs/protocols/FROZEN_HIGH_VALUE_SELECTIVITY_AUDIT.json",
    "--output-dir", "outputs/complementarity_identifiability/part2_selectivity_audit",
    "--report", "docs/reports/high_value_opportunity_selectivity_audit_2026-09-07.md"
)
if ($Mode -eq "Preflight") {
    $arguments += "--dry-run"
} elseif ($Overwrite) {
    $arguments += "--overwrite"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Part II command failed: $Python $($arguments -join ' ')"
}

if ($Mode -eq "Preflight") {
    Write-Host "[OK] Part II preflight passed; contract and DEV-only source hashes verified."
} else {
    Write-Host "[OK] Part II frozen selectivity audit completed."
    Write-Host "[UNCHANGED] Closure route remains FINAL_SELECTIVE_RARE_OPPORTUNITY."
    Write-Host "[LOCKED] No TEST command was run or unlocked."
}
