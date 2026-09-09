[CmdletBinding()]
param(
    [ValidateSet('dev', 'test', 'analyze', 'all')]
    [string]$Stage = 'all',
    [string]$Python = 'python',
    [ValidateSet('cuda', 'cpu', 'auto')]
    [string]$Device = 'cuda',
    [switch]$NoResume
)

$ErrorActionPreference = 'Stop'
$outputRoot = 'outputs/paper_a_safe_correction/boundary_pairs'
$protocol = 'docs/protocols/paper_a_boundary_pair_protocol.md'

$mkgwNative = @(
    'ml/artifacts/outputs/mkg_w_native/20260902_021506_seed1',
    'ml/artifacts/outputs/mkg_w_native/20260902_033610_seed2',
    'ml/artifacts/outputs/mkg_w_native/20260902_044121_seed3'
)
$mkgwAdamf = @(
    'ml/artifacts/outputs/mkg_w_adamf_mat/20260902_060855_seed1',
    'ml/artifacts/outputs/mkg_w_adamf_mat/20260902_083530_seed2',
    'ml/artifacts/outputs/mkg_w_adamf_mat/20260902_142555_seed3'
)
$db15kNative = @(
    'ml/artifacts/outputs/db15k_native/20260831_223530_seed1',
    'ml/artifacts/outputs/db15k_native/20260902_182809_seed2',
    'ml/artifacts/outputs/db15k_native/20260903_012002_seed3'
)
$db15kAdamf = @(
    'ml/artifacts/outputs/db15k_adamf_mat/20260901_030904_seed1',
    'ml/artifacts/outputs/db15k_adamf_mat/20260903_044823_seed2',
    'ml/artifacts/outputs/db15k_adamf_mat/20260903_090422_seed3'
)

$pairs = @(
    [pscustomobject]@{
        Dataset = 'mkg_w'; Pair = 'native_adamf'; PairName = 'mkgw_native_adamf_dynasemble'
        A = $mkgwNative; B = $mkgwAdamf
        Baseline = 'outputs/mkg_w/anchored_dynamic/native_adamf_seed123/full_ranking'
        ExistingTestRows = 'outputs/complementarity_identifiability/closure_test/raw/mkgw_native_adamf/test_query_rows.csv'
        P3 = 'outputs/mkg_w/anchored_dynamic/native_adamf_seed123/p3_ablation/dev_p3_summary.json'
    },
    [pscustomobject]@{
        Dataset = 'db15k'; Pair = 'native_adamf'; PairName = 'db15k_native_adamf_dynasemble'
        A = $db15kNative; B = $db15kAdamf
        Baseline = 'outputs/db15k/anchored_dynamic/native_adamf_seed123/full_ranking'
        ExistingTestRows = 'outputs/complementarity_identifiability/closure_test/raw/db15k_native_adamf/test_query_rows.csv'
        P3 = 'outputs/db15k/anchored_dynamic/native_adamf_seed123/p3_ablation/dev_p3_summary.json'
    }
)

function Invoke-CheckedPython {
    param([string[]]$CommandArgs, [string]$FailureMessage)
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) { throw $FailureMessage }
}

function Assert-Input {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required input missing: $Path" }
}

function Resolve-TestRows {
    param([object]$Spec)
    if (Test-Path -LiteralPath $Spec.ExistingTestRows) {
        Write-Host "[REUSE] Existing exact TEST rows $($Spec.ExistingTestRows)" -ForegroundColor DarkCyan
        return $Spec.ExistingTestRows
    }

    $pairRoot = Join-Path $outputRoot "$($Spec.Dataset)/$($Spec.Pair)"
    $testFullRanking = Join-Path $pairRoot 'test_full_ranking'
    $testRows = Join-Path $testFullRanking 'test_query_rows.csv'
    if (Test-Path -LiteralPath $testRows) {
        Write-Host "[REUSE] Boundary-pair exact TEST rows $testRows" -ForegroundColor DarkCyan
        return $testRows
    }

    Write-Host (
        "[EXPORT] Existing TEST rows are absent; exporting exact full-ranking rows " +
        "from frozen checkpoints for $($Spec.Dataset)/$($Spec.Pair)"
    ) -ForegroundColor Cyan
    $commandArgs = @(
        'scripts/eval_heterogeneous_complementarity.py',
        '--pair-name', $Spec.PairName.Replace('_dynasemble', ''),
        '--expert-a-name', 'NativE',
        '--expert-b-name', 'AdaMF-MAT',
        '--split', 'test',
        '--selection-json', (Join-Path $Spec.Baseline 'selection.json'),
        '--output-dir', $testFullRanking,
        '--device', $Device,
        '--export-alpha-grid'
    )
    for ($index = 0; $index -lt 3; $index++) {
        $commandArgs += @('--run-pair', ($Spec.A[$index] + '::' + $Spec.B[$index]))
    }
    if ($NoResume) { $commandArgs += '--no-resume' }
    Invoke-CheckedPython -CommandArgs $commandArgs -FailureMessage (
        "Exact TEST row export failed: $($Spec.Dataset)/$($Spec.Pair)"
    ) | Out-Host
    Assert-Input $testRows
    return $testRows
}

function Invoke-AnchoredDev {
    param([object]$Spec)
    $pairRoot = Join-Path $outputRoot "$($Spec.Dataset)/$($Spec.Pair)"
    $devLock = Join-Path $pairRoot 'anchored/dev_lock'
    $lockJson = Join-Path $devLock 'anchored_dev_lock.json'
    if (Test-Path -LiteralPath $lockJson) {
        Write-Host "[REUSE] Anchored DEV lock $lockJson" -ForegroundColor DarkCyan
        return
    }
    Invoke-CheckedPython -CommandArgs @(
        'scripts/lock_apply_anchored_dynamic.py', 'lock',
        '--dev-query-rows', (Join-Path $Spec.Baseline 'dev_query_rows.csv'),
        '--selection-json', (Join-Path $Spec.Baseline 'selection.json'),
        '--crossfit-summary', $Spec.P3,
        '--output-dir', $devLock
    ) -FailureMessage "Anchored DEV lock failed: $($Spec.Dataset)/$($Spec.Pair)"
}

function Invoke-AnchoredTest {
    param([object]$Spec, [string]$TestRows)
    $pairRoot = Join-Path $outputRoot "$($Spec.Dataset)/$($Spec.Pair)"
    $devLock = Join-Path $pairRoot 'anchored/dev_lock/anchored_dev_lock.json'
    Assert-Input $devLock
    Invoke-CheckedPython -CommandArgs @(
        'scripts/lock_apply_anchored_dynamic.py', 'apply',
        '--test-query-rows', $TestRows,
        '--lock-json', $devLock,
        '--output-dir', (Join-Path $pairRoot 'anchored/test_anchored')
    ) -FailureMessage "Anchored immutable TEST apply failed: $($Spec.Dataset)/$($Spec.Pair)"
}

function Invoke-DynaSemble {
    param([object]$Spec, [string]$CurrentStage, [string]$TestRows = '')
    $pairRoot = Join-Path $outputRoot "$($Spec.Dataset)/$($Spec.Pair)"
    $dynaRoot = Join-Path $pairRoot 'dynasemble'
    if ($CurrentStage -eq 'dev') {
        $reference = Join-Path $Spec.Baseline 'dev_query_rows.csv'
        $comparison = Join-Path $pairRoot 'anchored/dev_lock/dev_locked_query_rows.csv'
    } else {
        if (-not $TestRows) { throw 'TEST rows path is required for DynaSemble TEST.' }
        $reference = $TestRows
        $comparison = Join-Path $pairRoot 'anchored/test_anchored/test_locked_query_rows.csv'
    }
    Assert-Input $reference
    Assert-Input $comparison
    $commandArgs = @(
        'scripts/eval_paper_a_dynasemble.py',
        '--stage', $CurrentStage,
        '--pair-name', $Spec.PairName,
        '--expert-a-name', 'NativE',
        '--expert-b-name', 'AdaMF-MAT',
        '--baseline-selection-json', (Join-Path $Spec.Baseline 'selection.json'),
        '--reference-query-rows', $reference,
        '--comparison-query-rows', $comparison,
        '--output-dir', $dynaRoot,
        '--lock-filename', 'lock.json',
        '--protocol-path', $protocol,
        '--device', $Device
    )
    for ($index = 0; $index -lt 3; $index++) {
        $commandArgs += @('--run-pair', ($Spec.A[$index] + '::' + $Spec.B[$index]))
    }
    if ($NoResume) { $commandArgs += '--no-resume' }
    Invoke-CheckedPython -CommandArgs $commandArgs -FailureMessage (
        "DynaSemble failed: $($Spec.Dataset)/$($Spec.Pair) stage=$CurrentStage"
    )
}

if (($Python -match '[\\/]') -and -not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

if ($Stage -in @('dev', 'test', 'all')) {
    Invoke-CheckedPython -CommandArgs @(
        '-c',
        "import sys, torch; requested=sys.argv[1]; assert torch.cuda.is_available() or requested != 'cuda', 'CUDA unavailable'; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())",
        $Device
    ) -FailureMessage 'Runtime check failed.'
}

if ($Stage -in @('dev', 'all')) {
    Invoke-CheckedPython -CommandArgs @(
        'scripts/audit_reliable_primary_regime.py'
    ) -FailureMessage 'Reliable-primary DEV audit failed.'
    foreach ($pair in $pairs) {
        Invoke-AnchoredDev -Spec $pair
        Invoke-DynaSemble -Spec $pair -CurrentStage 'dev'
    }
}

if ($Stage -in @('test', 'all')) {
    Assert-Input 'outputs/paper_a_safe_correction/reliable_primary/reliable_primary_decisions.csv'
    foreach ($pair in $pairs) {
        $testRows = Resolve-TestRows -Spec $pair
        Invoke-AnchoredTest -Spec $pair -TestRows $testRows
        Invoke-DynaSemble -Spec $pair -CurrentStage 'test' -TestRows $testRows
    }
}

if ($Stage -in @('analyze', 'all')) {
    Invoke-CheckedPython -CommandArgs @(
        'scripts/analyze_paper_a_boundary_pairs.py'
    ) -FailureMessage 'Boundary-pair analysis failed.'
}

Write-Host "[DONE] Paper A boundary-pair workflow stage=$Stage" -ForegroundColor Green
