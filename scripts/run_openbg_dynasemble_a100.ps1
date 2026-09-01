[CmdletBinding()]
param(
    [ValidateSet('dev', 'test')]
    [string]$Stage = 'dev',
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
$native = @(
    'ml/artifacts/outputs/openbg_img_native/20260828_110323_seed1',
    'ml/artifacts/outputs/openbg_img_native/20260829_093518_seed2',
    'ml/artifacts/outputs/openbg_img_native/20260829_124029_seed3'
)

$outputDir = 'outputs/openbg_img/dynasemble/mhyper_native'
$baselineDir = 'outputs/openbg_img/heterogeneous_complementarity/mhyper_native'
$arguments = @(
    'scripts/eval_openbg_dynasemble.py',
    '--stage', $Stage,
    '--baseline-selection-json', (Join-Path $baselineDir 'selection.json'),
    '--reference-query-rows', (Join-Path $baselineDir ($Stage + '_query_rows.csv')),
    '--output-dir', $outputDir,
    '--device', 'cuda'
)
for ($index = 0; $index -lt 3; $index++) {
    $arguments += @('--run-pair', ($mhyper[$index] + '::' + $native[$index]))
}
if ($NoResume) {
    $arguments += '--no-resume'
}

Write-Host "[START] OpenBG-IMG DynaSemble stage=$Stage" -ForegroundColor Cyan
& python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "DynaSemble failed: stage=$Stage"
}
Write-Host "[DONE] OpenBG-IMG DynaSemble stage=$Stage" -ForegroundColor Green

$archive = "outputs/openbg_img/openbg_dynasemble_$Stage.zip"
$paths = @($outputDir)
Compress-Archive -Path $paths -DestinationPath $archive -Force
Write-Host "[ARCHIVE] $archive" -ForegroundColor Green
