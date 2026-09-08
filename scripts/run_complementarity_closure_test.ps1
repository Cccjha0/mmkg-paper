param(
    [ValidateSet("Preflight", "Evaluate", "Analyze")]
    [string]$Mode = "Preflight",
    [string]$Python = "python",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

function Invoke-Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $Python $($Arguments -join ' ')"
    }
}

$branch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne "m1/recent-mmkgc-baselines") {
    throw "Closure TEST must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$memoPath = "docs/protocols/COMPLEMENTARITY_CLOSURE_DECISION_MEMO.md"
$contractPath = "docs/protocols/COMPLEMENTARITY_CLOSURE_TEST_CONTRACT.json"
$frozenPaths = @(
    $memoPath,
    $contractPath,
    "scripts/run_complementarity_closure_test.ps1",
    "scripts/preflight_complementarity_closure_test.py",
    "scripts/apply_frozen_x4_closure_test.py",
    "scripts/audit_complementarity_closure_test.py",
    "scripts/eval_heterogeneous_complementarity.py"
)
foreach ($frozenPath in $frozenPaths) {
    & git cat-file -e "HEAD:$frozenPath"
    if ($LASTEXITCODE -ne 0) {
        throw "TEST remains locked: frozen protocol/code is absent from committed HEAD: $frozenPath"
    }
    & git diff --quiet HEAD -- $frozenPath
    if ($LASTEXITCODE -ne 0) {
        throw "TEST remains locked: frozen protocol/code differs from committed HEAD: $frozenPath"
    }
}

$memo = Get-Content -LiteralPath $memoPath -Raw
if (
    $memo -notmatch "FROZEN_THREE_DATASET_DEV_CLOSURE_BEFORE_TEST" -or
    $memo -notmatch "TEST_STATUS_READY_FOR_ONE_TIME_CONFIRMATORY_RUN" -or
    $memo -notmatch "FINAL_SELECTIVE_RARE_OPPORTUNITY"
) {
    throw "Decision memo does not contain the frozen three-dataset TEST boundary."
}

$contract = Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json
$preflightPath = "outputs/complementarity_identifiability/closure_test/preflight.json"
Invoke-Python -Arguments @(
    "scripts/preflight_complementarity_closure_test.py",
    "--contract", $contractPath,
    "--output", $preflightPath
)
if ($Mode -eq "Preflight") {
    Write-Host "[OK] Three-dataset closure TEST preflight passed. No TEST split was loaded."
    exit 0
}

$rawRoot = "outputs/complementarity_identifiability/closure_test/raw"
$x4Root = "outputs/complementarity_identifiability/closure_test/x4"

if ($Mode -eq "Analyze") {
    Invoke-Python -Arguments @(
        "scripts/audit_complementarity_closure_test.py",
        "--contract", $contractPath,
        "--raw-root", $rawRoot,
        "--x4-root", $x4Root,
        "--output-dir", "outputs/complementarity_identifiability/closure_test/audit",
        "--report", "docs/reports/complementarity_closure_test_audit_2026-09-08.md"
    )
    Write-Host "[OK] One-time three-dataset TEST analysis completed; frozen DEV gates and narrative remain unchanged."
    exit 0
}

& $Python -c "import torch,sys; sys.exit(0 if ('$Device' != 'cuda' or torch.cuda.is_available()) else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA was requested but is unavailable." }

foreach ($pair in $contract.pairs) {
    $pairId = [string]$pair.pair_id
    $selectionPath = [string]$pair.dev_selection.path
    $outDir = Join-Path $rawRoot $pairId
    $testSummary = Join-Path $outDir "test_summary.json"
    $runPairs = @()
    $runsA = @($contract.experts.PSObject.Properties[[string]$pair.expert_a_key].Value)
    $runsB = @($contract.experts.PSObject.Properties[[string]$pair.expert_b_key].Value)
    if ($runsA.Count -ne 3 -or $runsB.Count -ne 3) {
        throw "Frozen checkpoint inventory is incomplete for $pairId"
    }
    for ($index = 0; $index -lt 3; $index++) {
        if ([int]$runsA[$index].seed -ne [int]$runsB[$index].seed) {
            throw "Frozen checkpoint seeds are misaligned for $pairId"
        }
        $runPairs += "$([string]$runsA[$index].run_dir)::$([string]$runsB[$index].run_dir)"
    }

    if (Test-Path -LiteralPath $testSummary) {
        $existing = Get-Content -LiteralPath $testSummary -Raw | ConvertFrom-Json
        if ($existing.split -ne "test" -or $existing.export_alpha_grid -ne $true -or $existing.export_x4_features -ne $true) {
            throw "A nonconforming prior TEST export exists for $pairId; do not overwrite it."
        }
        Write-Host "[LOCKED] ${pairId}: conforming TEST export already exists and will not be rerun."
    } else {
        $arguments = @(
            "scripts/eval_heterogeneous_complementarity.py",
            "--pair-name", $pairId,
            "--expert-a-name", [string]$pair.expert_a_name,
            "--expert-b-name", [string]$pair.expert_b_name,
            "--split", "test",
            "--selection-json", $selectionPath,
            "--output-dir", $outDir,
            "--device", $Device,
            "--export-alpha-grid",
            "--export-x4-features"
        )
        foreach ($runPair in $runPairs) { $arguments += @("--run-pair", $runPair) }
        Invoke-Python -Arguments $arguments
    }

    $x4Summary = Join-Path (Join-Path $x4Root $pairId) "test_x4_summary.json"
    if (Test-Path -LiteralPath $x4Summary) {
        Write-Host "[LOCKED] ${pairId}: frozen X4 TEST application already exists and will not be rerun."
    } else {
        Invoke-Python -Arguments @(
            "scripts/apply_frozen_x4_closure_test.py",
            "--contract", $contractPath,
            "--pair-id", $pairId,
            "--raw-root", $rawRoot,
            "--output-root", $x4Root,
            "--device", $Device
        )
    }
}

Write-Host "[OK] One-time frozen TEST full-ranking and X4 application completed for all 9 pairs. Run Analyze exactly once."
