[CmdletBinding()]
param(
    [ValidateSet('Plan', 'Dev', 'Test', 'Summarize', 'Bundle', 'All')]
    [string]$Stage = 'Plan',
    [string]$Python = 'python',
    [string]$OutputRoot = 'outputs/paper_a_safe_correction/dynasemble_controls_v1'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $stages = if ($Stage -eq 'All') { @('dev', 'test', 'summarize', 'bundle') } else { @($Stage.ToLowerInvariant()) }
    foreach ($currentStage in $stages) {
        Write-Host "DynaSemble controls: $currentStage"
        & $Python 'scripts/paper_a_dynasemble_controls.py' --stage $currentStage --device cuda --output-root $OutputRoot
        if ($LASTEXITCODE -ne 0) { throw "Stage $currentStage failed (exit $LASTEXITCODE). Results are retained; do not remove failed seeds." }
    }
} finally {
    Pop-Location
}
