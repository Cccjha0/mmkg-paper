param(
    [ValidateSet("Preflight", "Analyze")]
    [string]$Mode = "Analyze",
    [string]$Python = "python",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$Arguments = @(
    "scripts/synthesize_cross_dataset_closure.py",
    "--contract", "docs/protocols/CROSS_DATASET_COMPLEMENTARITY_CLOSURE_SYNTHESIS_CONTRACT.json",
    "--output-dir", "outputs/complementarity_identifiability/cross_dataset_closure_synthesis",
    "--report", "docs/reports/cross_dataset_complementarity_closure_synthesis_2026-09-08.md"
)

if ($Mode -eq "Preflight") {
    $Arguments += "--dry-run"
}
if ($Overwrite) {
    $Arguments += "--overwrite"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Python command failed: $Python $($Arguments -join ' ')"
}
