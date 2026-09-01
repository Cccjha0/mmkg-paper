[CmdletBinding()]
param(
    [int]$ChunkSize = 8192,
    [int]$QueryBatchSize = 32,
    [string[]]$ForceExperts = @(),
    [switch]$NoArchive
)

$ErrorActionPreference = 'Stop'

$outDir = 'outputs/openbg_img/recent_baselines/dev'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host '[CHECK] Python 3.10+'
& python -c "import sys; print('Python:', sys.version.split()[0]); print('Executable:', sys.executable); assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'"
if ($LASTEXITCODE -ne 0) {
    throw 'Python version check failed. Activate the Python 3.10 environment and retry.'
}

Write-Host '[CHECK] CUDA availability'
& python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))"
if ($LASTEXITCODE -ne 0) {
    throw 'CUDA check failed.'
}

$jobs = @(
    [pscustomobject]@{ Expert = 'mhyper'; Seed = 1; RunDir = 'ml/artifacts/outputs/openbg_img_mhyper/20260828_182757_seed1'; Stem = 'mhyper_query_eval_seed1' },
    [pscustomobject]@{ Expert = 'mhyper'; Seed = 2; RunDir = 'ml/artifacts/outputs/openbg_img_mhyper/20260830_020356_seed2'; Stem = 'mhyper_query_eval_seed2' },
    [pscustomobject]@{ Expert = 'mhyper'; Seed = 3; RunDir = 'ml/artifacts/outputs/openbg_img_mhyper/20260830_060535_seed3'; Stem = 'mhyper_query_eval_seed3' },
    [pscustomobject]@{ Expert = 'native'; Seed = 1; RunDir = 'ml/artifacts/outputs/openbg_img_native/20260828_110323_seed1'; Stem = 'native_query_eval_seed1' },
    [pscustomobject]@{ Expert = 'native'; Seed = 2; RunDir = 'ml/artifacts/outputs/openbg_img_native/20260829_093518_seed2'; Stem = 'native_query_eval_seed2' },
    [pscustomobject]@{ Expert = 'native'; Seed = 3; RunDir = 'ml/artifacts/outputs/openbg_img_native/20260829_124029_seed3'; Stem = 'native_query_eval_seed3' },
    [pscustomobject]@{ Expert = 'adamf_mat'; Seed = 1; RunDir = 'ml/artifacts/outputs/openbg_img_adamf_mat/20260828_145519_seed1'; Stem = 'adamf_mat_query_eval_seed1' },
    [pscustomobject]@{ Expert = 'adamf_mat'; Seed = 2; RunDir = 'ml/artifacts/outputs/openbg_img_adamf_mat/20260829_155711_seed2'; Stem = 'adamf_mat_query_eval_seed2' },
    [pscustomobject]@{ Expert = 'adamf_mat'; Seed = 3; RunDir = 'ml/artifacts/outputs/openbg_img_adamf_mat/20260829_204154_seed3'; Stem = 'adamf_mat_query_eval_seed3' },
    [pscustomobject]@{ Expert = 'apkgc'; Seed = 1; RunDir = 'ml/artifacts/outputs/openbg_img_apkgc/20260827_181839_seed1'; Stem = 'apkgc_query_eval_seed1' },
    [pscustomobject]@{ Expert = 'apkgc'; Seed = 2; RunDir = 'ml/artifacts/outputs/openbg_img_apkgc/20260828_223440_seed2'; Stem = 'apkgc_query_eval_seed2' },
    [pscustomobject]@{ Expert = 'apkgc'; Seed = 3; RunDir = 'ml/artifacts/outputs/openbg_img_apkgc/20260829_034230_seed3'; Stem = 'apkgc_query_eval_seed3' }
)

$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($job in $jobs) {
    $csvPath = Join-Path -Path $outDir -ChildPath ($job.Stem + '.csv')
    $summaryPath = Join-Path -Path $outDir -ChildPath ($job.Stem + '_summary.json')
    $forceExport = $ForceExperts -contains $job.Expert

    if (Test-Path -LiteralPath $csvPath) {
        if ($forceExport) {
            Write-Host "[REPLACE] $($job.Expert) seed=$($job.Seed)" -ForegroundColor Magenta
            Remove-Item -LiteralPath $csvPath
            if (Test-Path -LiteralPath $summaryPath) {
                Remove-Item -LiteralPath $summaryPath
            }
        }
        else {
            $existingRows = @(Import-Csv -LiteralPath $csvPath).Count
            if ($existingRows -eq 10000) {
                Write-Host "[SKIP] $($job.Expert) seed=$($job.Seed): already has 10000 rows" -ForegroundColor Yellow
                continue
            }
            throw "Unexpected row count in existing file: $csvPath ($existingRows rows)"
        }
    }

    $checkpointPath = Join-Path -Path $job.RunDir -ChildPath 'best.ckpt'
    if (-not (Test-Path -LiteralPath $checkpointPath)) {
        throw "Checkpoint not found: $checkpointPath"
    }

    Write-Host ''
    Write-Host "[START] $($job.Expert) seed=$($job.Seed)" -ForegroundColor Cyan
    $jobTimer = [System.Diagnostics.Stopwatch]::StartNew()

    & python scripts/export_query_eval.py `
        --expert $job.Expert `
        --run-dir $job.RunDir `
        --seed $job.Seed `
        --split dev `
        --device cuda `
        --chunk-size $ChunkSize `
        --query-batch-size $QueryBatchSize `
        --out $csvPath `
        --summary-json $summaryPath

    if ($LASTEXITCODE -ne 0) {
        throw "Export failed: $($job.Expert) seed=$($job.Seed)"
    }

    $rowCount = @(Import-Csv -LiteralPath $csvPath).Count
    if ($rowCount -ne 10000) {
        throw "Unexpected exported row count: $csvPath ($rowCount rows)"
    }

    $jobTimer.Stop()
    Write-Host "[DONE] $($job.Expert) seed=$($job.Seed), rows=$rowCount, elapsed=$($jobTimer.Elapsed)" -ForegroundColor Green
}

$totalTimer.Stop()
$csvFiles = @(Get-ChildItem -LiteralPath $outDir -Filter '*_query_eval_seed*.csv' -File)
if ($csvFiles.Count -ne 12) {
    throw "Expected 12 DEV query-eval CSV files, found $($csvFiles.Count)."
}

Write-Host "[OK] All 12 exports completed in $($totalTimer.Elapsed)." -ForegroundColor Green

if (-not $NoArchive) {
    $zipPath = 'outputs/openbg_img/openbg_recent_baseline_dev_query_eval.zip'
    Compress-Archive -Path (Join-Path -Path $outDir -ChildPath '*') -DestinationPath $zipPath -Force
    Write-Host "[OK] Archive written to $zipPath" -ForegroundColor Green
}
