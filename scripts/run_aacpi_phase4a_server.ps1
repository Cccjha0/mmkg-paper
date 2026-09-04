param(
    [ValidateSet("all", "features", "latents", "oof", "analyze")]
    [string]$Stage = "all",
    [ValidateSet("cuda", "cpu", "auto")]
    [string]$Device = "cuda",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$pairs = @(
    "mkgw_mhyper_native",
    "mkgw_mhyper_adamf",
    "mkgw_native_adamf",
    "db15k_mhyper_native",
    "db15k_mhyper_adamf",
    "db15k_native_adamf"
)
$root = "outputs/aacpi/phase4a"
$overwriteArg = @()
if ($Overwrite) { $overwriteArg = @("--overwrite") }

foreach ($pair in $pairs) {
    $asset = "outputs/aacpi/action_response_assets/$pair"
    $featureTable = "$asset/dev_action_response_features.csv.gz"
    $sourceManifest = "$asset/candidate_score_source_manifest.json"
    $contextTable = "$root/raw/${pair}_context_features.csv.gz"
    $latentFile = "$root/latents/${pair}_query_latents.npz"
    if ($Stage -in @("all", "features")) {
        $contextManifest = "$root/$pair/context_feature_manifest.json"
        if (-not $Overwrite -and (Test-Path -LiteralPath $contextTable) -and (Test-Path -LiteralPath $contextManifest)) {
            Write-Host "[SKIP] Context features already complete: $pair"
        } else {
            python scripts/build_aacpi_phase4a_context_features.py `
                --phase3a-feature-table $featureTable `
                --phase3a-source-manifest $sourceManifest `
                --output-table $contextTable `
                --output-manifest $contextManifest @overwriteArg
            if ($LASTEXITCODE -ne 0) { throw "Context feature build failed: $pair" }
        }
    }
    if ($Stage -in @("all", "latents")) {
        $latentManifest = "$root/$pair/latent_extraction_manifest.json"
        if (-not $Overwrite -and (Test-Path -LiteralPath $latentFile) -and (Test-Path -LiteralPath $latentManifest)) {
            Write-Host "[SKIP] Frozen latents already complete: $pair"
        } else {
            python scripts/extract_aacpi_frozen_query_latents.py `
                --phase3a-feature-table $featureTable `
                --phase3a-source-manifest $sourceManifest `
                --output-latents $latentFile `
                --output-manifest $latentManifest `
                --device $Device @overwriteArg
            if ($LASTEXITCODE -ne 0) { throw "Latent extraction failed: $pair" }
        }
    }
    if ($Stage -in @("all", "oof")) {
        $runManifest = "$root/$pair/phase4a_run_manifest.json"
        if (-not $Overwrite -and (Test-Path -LiteralPath $runManifest)) {
            Write-Host "[SKIP] Nested OOF already complete: $pair"
        } else {
            python scripts/run_aacpi_phase4a_context_oof.py `
                --context-table $contextTable `
                --latent-file $latentFile `
                --phase3a-r3-oof "outputs/aacpi/phase3a/$pair/r3/dev_oof_predictions.csv.gz" `
                --output-dir $root `
                --device $Device @overwriteArg
            if ($LASTEXITCODE -ne 0) { throw "Nested OOF failed: $pair" }
        }
    }
}

if ($Stage -in @("all", "analyze")) {
    python scripts/analyze_aacpi_phase4a_context_identifiability.py `
        --phase4a-root $root `
        --phase3a-root outputs/aacpi/phase3a `
        --report docs/reports/aacpi_phase4a_contextual_identifiability_audit_2026-09-05.md @overwriteArg
    if ($LASTEXITCODE -ne 0) { throw "Phase 4A analysis failed" }
}
