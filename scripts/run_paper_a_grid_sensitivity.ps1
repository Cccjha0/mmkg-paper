param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $ProjectRoot
try {
    foreach ($Phase in @('prepare', 'score-dev', 'score-test')) {
        & $Python 'scripts/paper_a_grid_sensitivity.py' $Phase
        if ($LASTEXITCODE -ne 0) { throw "Grid sensitivity phase failed: $Phase" }
    }
    & $Python 'scripts/paper_a_grid_sensitivity.py' 'report' '--pack'
    if ($LASTEXITCODE -ne 0) { throw 'Grid sensitivity report failed' }
} finally {
    Pop-Location
}
