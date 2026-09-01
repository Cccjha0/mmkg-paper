[CmdletBinding()]
param(
    [ValidateSet('dev', 'test')]
    [string]$Stage = 'dev',
    [ValidateSet('all', 'mhyper_adamf', 'mhyper_native')]
    [string]$Pair = 'all',
    [switch]$NoResume
)

$ErrorActionPreference = 'Stop'

Write-Host '[CHECK] Python 3.10+'
& python -c "import sys; print('Python:', sys.version.split()[0]); print('Executable:', sys.executable); assert sys.version_info >= (3, 10)"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.10+ check failed.' }

Write-Host '[CHECK] CUDA availability'
& python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA check failed.' }

$mhyper = @(
    'ml/artifacts/outputs/openbg_img_mhyper/20260828_182757_seed1',
    'ml/artifacts/outputs/openbg_img_mhyper/20260830_020356_seed2',
    'ml/artifacts/outputs/openbg_img_mhyper/20260830_060535_seed3'
)
$adamf = @(
    'ml/artifacts/outputs/openbg_img_adamf_mat/20260828_145519_seed1',
    'ml/artifacts/outputs/openbg_img_adamf_mat/20260829_155711_seed2',
    'ml/artifacts/outputs/openbg_img_adamf_mat/20260829_204154_seed3'
)
$native = @(
    'ml/artifacts/outputs/openbg_img_native/20260828_110323_seed1',
    'ml/artifacts/outputs/openbg_img_native/20260829_093518_seed2',
    'ml/artifacts/outputs/openbg_img_native/20260829_124029_seed3'
)

$jobs = @(
    [pscustomobject]@{
        Key = 'mhyper_adamf'
        Name = 'openbg_mhyper_adamf'
        ExpertA = 'M-Hyper'
        ExpertB = 'AdaMF-MAT'
        RunsA = $mhyper
        RunsB = $adamf
    },
    [pscustomobject]@{
        Key = 'mhyper_native'
        Name = 'openbg_mhyper_native'
        ExpertA = 'M-Hyper'
        ExpertB = 'NativE'
        RunsA = $mhyper
        RunsB = $native
    }
)

if ($Pair -ne 'all') {
    $jobs = @($jobs | Where-Object Key -eq $Pair)
}

foreach ($job in $jobs) {
    $outDir = Join-Path 'outputs/openbg_img/heterogeneous_complementarity' $job.Key
    $arguments = @(
        'scripts/eval_heterogeneous_complementarity.py',
        '--pair-name', $job.Name,
        '--expert-a-name', $job.ExpertA,
        '--expert-b-name', $job.ExpertB,
        '--split', $Stage,
        '--output-dir', $outDir,
        '--device', 'cuda',
        '--rrf-k', '60',
        '--relation-min-support', '60'
    )
    for ($index = 0; $index -lt 3; $index++) {
        $arguments += @('--run-pair', ($job.RunsA[$index] + '::' + $job.RunsB[$index]))
    }
    if ($Stage -eq 'test') {
        $selection = Join-Path $outDir 'selection.json'
        if (-not (Test-Path -LiteralPath $selection)) {
            throw "Locked DEV selection not found: $selection"
        }
        $arguments += @('--selection-json', $selection)
    }
    if ($NoResume) {
        $arguments += '--no-resume'
    }

    Write-Host ''
    Write-Host "[START] pair=$($job.Key) stage=$Stage" -ForegroundColor Cyan
    & python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Complementarity evaluation failed: pair=$($job.Key) stage=$Stage"
    }
    Write-Host "[DONE] pair=$($job.Key) stage=$Stage" -ForegroundColor Green
}

$archive = "outputs/openbg_img/openbg_heterogeneous_complementarity_$Stage.zip"
Compress-Archive -Path 'outputs/openbg_img/heterogeneous_complementarity/*' -DestinationPath $archive -Force
Write-Host "[OK] Archive written to $archive" -ForegroundColor Green
