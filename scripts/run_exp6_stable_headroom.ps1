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
    throw "Experiment 6 must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$exp1 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp1_landscape/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp1.gate.decision -ne "GO" -or $exp1.operational_audit.test_access -ne 0) {
    throw "Experiment 1 is not a clean DEV-only GO input."
}
$exp4 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp4_cross_seed_transfer/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp4.split -ne "dev" -or $exp4.operational_audit.test_access -ne 0) {
    throw "Experiment 4 is not a clean frozen DEV-only input."
}
$exp5 = Get-Content -LiteralPath "outputs/complementarity_identifiability/exp5_local_identifiability/audit_manifest.json" -Raw | ConvertFrom-Json
if ($exp5.split -ne "dev" -or $exp5.operational_audit.test_access -ne 0) {
    throw "Experiment 5 is not a clean frozen DEV-only input."
}

& $Python -c "import numpy,pandas; print('runtime dependencies OK')"
if ($LASTEXITCODE -ne 0) { throw "Experiment 6 requires numpy and pandas." }

$arguments = @(
    "scripts/audit_stable_headroom.py",
    "--contract", "docs/protocols/EXP6_STABLE_HEADROOM_CONTRACT.json",
    "--output-dir", "outputs/complementarity_identifiability/exp6_stable_headroom",
    "--report", "docs/reports/stable_headroom_gain_concentration_audit_2026-09-07.md",
    "--decision-memo", "docs/protocols/COMPLEMENTARITY_CLOSURE_DECISION_MEMO.md"
)
if ($Mode -eq "Preflight") {
    $arguments += "--dry-run"
} elseif ($Overwrite) {
    $arguments += "--overwrite"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Experiment 6 command failed: $Python $($arguments -join ' ')"
}

if ($Mode -eq "Preflight") {
    Write-Host "[OK] Experiment 6 preflight completed; DEV assets only, no bootstrap or TEST access."
} else {
    Write-Host "[OK] Experiment 6 DEV closure completed and decision memo generated."
    Write-Host "[LOCKED] Do not run TEST until the decision memo and frozen TEST scripts are committed."
}
