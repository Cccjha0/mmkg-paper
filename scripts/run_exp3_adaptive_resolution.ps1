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
    throw "Experiment 3 must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$exp1 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp1_landscape/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp1.gate.decision -ne "GO") {
    throw "Experiment 1 Available Complementarity Gate is not GO."
}
$exp2 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp2_information/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp2.split -ne "dev" -or $exp2.operational_audit.test_access -ne 0) {
    throw "Experiment 2 audit is not a clean DEV-only input."
}

& $Python -c "import numpy,pandas; print('runtime dependencies OK')"
if ($LASTEXITCODE -ne 0) { throw "Experiment 3 requires numpy and pandas." }

$arguments = @(
    "scripts/audit_adaptive_resolution.py",
    "--contract", "docs/protocols/EXP3_ADAPTIVE_RESOLUTION_CONTRACT.json",
    "--output-dir", "outputs/complementarity_identifiability/exp3_resolution",
    "--report", "docs/reports/adaptive_resolution_audit_2026-09-07.md"
)
if ($Mode -eq "Preflight") {
    $arguments += "--dry-run"
} else {
    if ($Overwrite) { $arguments += "--overwrite" }
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Experiment 3 command failed: $Python $($arguments -join ' ')"
}

if ($Mode -eq "Preflight") {
    Write-Host "[OK] Experiment 3 preflight completed; no policy or model was trained."
} else {
    Write-Host "[OK] Experiment 3 granularity audit completed; no subsequent method development was started."
}
