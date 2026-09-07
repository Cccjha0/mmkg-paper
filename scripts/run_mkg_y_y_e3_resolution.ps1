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
    throw "MKG-Y Y-E3 must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

& $Python -c "import numpy,pandas; print('runtime dependencies OK')"
if ($LASTEXITCODE -ne 0) {
    throw "MKG-Y Y-E3 requires numpy and pandas."
}

$arguments = @(
    "scripts/audit_mkg_y_y_e3_resolution.py",
    "--contract", "docs/protocols/MKG_Y_Y_E3_ADAPTIVE_RESOLUTION_CONTRACT.json",
    "--output-dir", "outputs/complementarity_identifiability/mkg_y_y_e3_resolution",
    "--report", "docs/reports/mkg_y_adaptive_resolution_audit_2026-09-08.md"
)
if ($Mode -eq "Preflight") {
    $arguments += "--dry-run"
} elseif ($Overwrite) {
    $arguments += "--overwrite"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "MKG-Y Y-E3 command failed: $Python $($arguments -join ' ')"
}

if ($Mode -eq "Preflight") {
    Write-Host "[OK] MKG-Y Y-E3 preflight completed; no policy or learner was trained."
} else {
    Write-Host "[OK] MKG-Y Y-E3 adaptive-resolution replication completed. MKG-Y TEST remains locked."
}
