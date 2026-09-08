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
    throw "MKG-Y Y-E6 must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$exp1 = Get-Content -LiteralPath "outputs/complementarity_identifiability/mkg_y_y_e1_landscape/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp1.split -ne "dev" -or $exp1.assessment.outcome -ne "Y_E1_AVAILABLE_COMPLEMENTARITY_PRESENT") {
    throw "MKG-Y Y-E1 is not a frozen completed DEV replication."
}
$exp4 = Get-Content -LiteralPath "outputs/complementarity_identifiability/mkg_y_y_e4_cross_seed_transfer/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp4.split -ne "dev" -or $exp4.assessment.outcome -ne "Y_E4_CROSS_SEED_REPLICATION_REPORTED") {
    throw "MKG-Y Y-E4 is not a frozen completed DEV replication."
}
$exp5 = Get-Content -LiteralPath "outputs/complementarity_identifiability/mkg_y_y_e5_local_identifiability/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp5.split -ne "dev" -or $exp5.final_classification -ne "Y_E5_LOCAL_IDENTIFIABILITY_REPLICATION_REPORTED") {
    throw "MKG-Y Y-E5 is not a frozen completed DEV replication."
}

& $Python -c "import numpy,pandas; print('runtime dependencies OK')"
if ($LASTEXITCODE -ne 0) { throw "MKG-Y Y-E6 requires numpy and pandas." }

$arguments = @(
    "scripts/audit_mkg_y_y_e6_stable_headroom.py",
    "--contract", "docs/protocols/MKG_Y_Y_E6_STABLE_HEADROOM_CONTRACT.json",
    "--output-dir", "outputs/complementarity_identifiability/mkg_y_y_e6_stable_headroom",
    "--report", "docs/reports/mkg_y_stable_headroom_gain_concentration_audit_2026-09-08.md"
)
if ($Mode -eq "Preflight") {
    $arguments += "--dry-run"
} elseif ($Overwrite) {
    $arguments += "--overwrite"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "MKG-Y Y-E6 command failed: $Python $($arguments -join ' ')"
}

if ($Mode -eq "Preflight") {
    Write-Host "[OK] MKG-Y Y-E6 preflight completed; DEV assets only, no bootstrap or TEST access."
} else {
    Write-Host "[OK] MKG-Y Y-E6 descriptive closure replication completed."
    Write-Host "[LOCKED] MKG-Y TEST remains prohibited."
}
