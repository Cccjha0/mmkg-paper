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
    throw "MKG-Y acceptance must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

& $Python -c "import numpy,pandas,yaml; print('runtime dependencies OK')"
if ($LASTEXITCODE -ne 0) { throw "MKG-Y acceptance requires numpy, pandas, and PyYAML." }

$arguments = @(
    "scripts/audit_mkg_y_y0_y1_acceptance.py",
    "--contract", "docs/protocols/MKG_Y_Y0_Y1_ACCEPTANCE_CONTRACT.json",
    "--output-dir", "outputs/complementarity_identifiability/mkg_y_y0_y1_acceptance",
    "--report", "docs/reports/mkg_y_y0_y1_acceptance_audit_2026-09-07.md"
)
if ($Mode -eq "Preflight") {
    $arguments += "--dry-run"
} elseif ($Overwrite) {
    $arguments += "--overwrite"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "MKG-Y Y0/Y1 acceptance command failed: $Python $($arguments -join ' ')"
}

if ($Mode -eq "Preflight") {
    Write-Host "[OK] MKG-Y Y0/Y1 acceptance preflight passed."
} else {
    Write-Host "[OK] MKG-Y Y0/Y1 acceptance audit completed."
    Write-Host "[LOCKED] This does not unlock or access TEST."
}
