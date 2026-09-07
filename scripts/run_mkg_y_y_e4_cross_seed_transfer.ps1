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
    throw "MKG-Y Y-E4 must run on m1/recent-mmkgc-baselines; current branch: $branch"
}
& $Python -c "import numpy,pandas; print('runtime dependencies OK')"
if ($LASTEXITCODE -ne 0) { throw "MKG-Y Y-E4 requires numpy and pandas." }
$arguments = @(
    "scripts/audit_mkg_y_y_e4_cross_seed_transfer.py",
    "--contract", "docs/protocols/MKG_Y_Y_E4_CROSS_SEED_TRANSFER_CONTRACT.json",
    "--output-dir", "outputs/complementarity_identifiability/mkg_y_y_e4_cross_seed_transfer",
    "--report", "docs/reports/mkg_y_cross_seed_transfer_audit_2026-09-08.md"
)
if ($Mode -eq "Preflight") { $arguments += "--dry-run" } elseif ($Overwrite) { $arguments += "--overwrite" }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "MKG-Y Y-E4 command failed: $Python $($arguments -join ' ')" }
if ($Mode -eq "Preflight") {
    Write-Host "[OK] MKG-Y Y-E4 preflight completed; no checkpoint was executed."
} else {
    Write-Host "[OK] MKG-Y Y-E4 transfer/LOSO audit completed. MKG-Y TEST remains locked."
}
