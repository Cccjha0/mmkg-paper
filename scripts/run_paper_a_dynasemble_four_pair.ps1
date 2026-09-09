[CmdletBinding()]
param(
    [ValidateSet('dev', 'test', 'all')]
    [string]$Stage = 'all',
    [string]$Python = 'python',
    [ValidateSet('cuda', 'cpu', 'auto')]
    [string]$Device = 'cuda',
    [switch]$NoResume
)

$ErrorActionPreference = 'Stop'

$protocol = 'docs/protocols/paper_a_dynasemble_four_pair_protocol.md'
$outputRoot = 'outputs/paper_a_safe_correction/dynasemble'

$mkgwMhyper = @(
    'ml/artifacts/outputs/mkg_w_mhyper/20260902_014323_seed1',
    'ml/artifacts/outputs/mkg_w_mhyper/20260902_015400_seed2',
    'ml/artifacts/outputs/mkg_w_mhyper/20260902_020435_seed3'
)
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
$db15kMhyper = @(
    'ml/artifacts/outputs/db15k_mhyper/20260831_221627_seed1',
    'ml/artifacts/outputs/db15k_mhyper/20260902_175015_seed2',
    'ml/artifacts/outputs/db15k_mhyper/20260902_180912_seed3'
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
        Dataset = 'mkg_w'; Pair = 'mhyper_native'; PairName = 'mkgw_mhyper_native_dynasemble'
        ExpertB = 'NativE'; A = $mkgwMhyper; B = $mkgwNative
        Baseline = 'outputs/mkg_w/anchored_dynamic/mhyper_native_seed123/full_ranking'
        Anchored = 'outputs/mkg_w/anchored_dynamic/mhyper_native_seed123'
    },
    [pscustomobject]@{
        Dataset = 'mkg_w'; Pair = 'mhyper_adamf'; PairName = 'mkgw_mhyper_adamf_dynasemble'
        ExpertB = 'AdaMF-MAT'; A = $mkgwMhyper; B = $mkgwAdamf
        Baseline = 'outputs/mkg_w/anchored_dynamic/mhyper_adamf_seed123/full_ranking'
        Anchored = 'outputs/mkg_w/anchored_dynamic/mhyper_adamf_seed123'
    },
    [pscustomobject]@{
        Dataset = 'db15k'; Pair = 'mhyper_native'; PairName = 'db15k_mhyper_native_dynasemble'
        ExpertB = 'NativE'; A = $db15kMhyper; B = $db15kNative
        Baseline = 'outputs/db15k/anchored_dynamic/mhyper_native_seed123/full_ranking'
        Anchored = 'outputs/db15k/anchored_dynamic/mhyper_native_seed123'
    },
    [pscustomobject]@{
        Dataset = 'db15k'; Pair = 'mhyper_adamf'; PairName = 'db15k_mhyper_adamf_dynasemble'
        ExpertB = 'AdaMF-MAT'; A = $db15kMhyper; B = $db15kAdamf
        Baseline = 'outputs/db15k/anchored_dynamic/mhyper_adamf_seed123/full_ranking'
        Anchored = 'outputs/db15k/anchored_dynamic/mhyper_adamf_seed123'
    }
)

function Invoke-DynaSembleStage {
    param([object]$Spec, [string]$CurrentStage)

    $outputDir = Join-Path $outputRoot "$($Spec.Dataset)/$($Spec.Pair)"
    if ($CurrentStage -eq 'dev') {
        $reference = Join-Path $Spec.Baseline 'dev_query_rows.csv'
        $comparison = Join-Path $Spec.Anchored 'dev_lock/dev_locked_query_rows.csv'
    } else {
        $reference = Join-Path $Spec.Anchored 'test_full_ranking/test_query_rows.csv'
        $comparison = Join-Path $Spec.Anchored 'test_anchored/test_locked_query_rows.csv'
    }
    $arguments = @(
        'scripts/eval_paper_a_dynasemble.py',
        '--stage', $CurrentStage,
        '--pair-name', $Spec.PairName,
        '--expert-a-name', 'M-Hyper',
        '--expert-b-name', $Spec.ExpertB,
        '--baseline-selection-json', (Join-Path $Spec.Baseline 'selection.json'),
        '--reference-query-rows', $reference,
        '--comparison-query-rows', $comparison,
        '--output-dir', $outputDir,
        '--lock-filename', 'lock.json',
        '--protocol-path', $protocol,
        '--device', $Device
    )
    for ($index = 0; $index -lt 3; $index++) {
        $arguments += @('--run-pair', ($Spec.A[$index] + '::' + $Spec.B[$index]))
    }
    if ($NoResume) {
        $arguments += '--no-resume'
    }
    Write-Host "[START] $($Spec.Dataset)/$($Spec.Pair) stage=$CurrentStage" -ForegroundColor Cyan
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "DynaSemble failed: $($Spec.Dataset)/$($Spec.Pair) stage=$CurrentStage"
    }
}

if (($Python -match '[\\/]') -and -not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}
& $Python -c "import sys, torch; requested = sys.argv[1]; assert torch.cuda.is_available() or requested != 'cuda', 'CUDA unavailable'; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())" $Device
if ($LASTEXITCODE -ne 0) { throw 'Runtime check failed.' }

$stages = if ($Stage -eq 'all') { @('dev', 'test') } else { @($Stage) }
foreach ($currentStage in $stages) {
    foreach ($pair in $pairs) {
        Invoke-DynaSembleStage -Spec $pair -CurrentStage $currentStage
    }
}

if ($Stage -ne 'dev') {
    & $Python scripts/build_paper_a_dynasemble_four_pair_summary.py --root $outputRoot
    if ($LASTEXITCODE -ne 0) { throw 'Four-pair summary build failed.' }
}

Write-Host '[DONE] Paper A four-pair DynaSemble workflow' -ForegroundColor Green
