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

$branch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne "m1/recent-mmkgc-baselines") {
    throw "Closure TEST must run on m1/recent-mmkgc-baselines; current branch: $branch"
}

$memoPath = "docs/protocols/COMPLEMENTARITY_CLOSURE_DECISION_MEMO.md"
$frozenPaths = @(
    $memoPath,
    "docs/protocols/EXP6_STABLE_HEADROOM_CONTRACT.json",
    "scripts/run_complementarity_closure_test.ps1",
    "scripts/audit_complementarity_closure_test.py",
    "outputs/complementarity_identifiability/exp6_stable_headroom/final_joint_interpretation.json",
    "outputs/complementarity_identifiability/exp6_stable_headroom/audit_manifest.json"
)
foreach ($frozenPath in $frozenPaths) {
    & git cat-file -e "HEAD:$frozenPath"
    if ($LASTEXITCODE -ne 0) { throw "TEST remains locked: frozen closure artifact is absent from committed HEAD: $frozenPath" }
    & git diff --quiet HEAD -- $frozenPath
    if ($LASTEXITCODE -ne 0) { throw "TEST remains locked: frozen closure artifact differs from committed HEAD: $frozenPath" }
}
$memo = Get-Content -LiteralPath $memoPath -Raw
if ($memo -notmatch "FROZEN_AFTER_DEV_CLOSURE_BEFORE_TEST" -or $memo -notmatch "TEST_STATUS_LOCKED_PENDING_MEMO_COMMIT") {
    throw "Decision memo does not contain the frozen TEST boundary."
}

$pairIds = @(
    "mkgw_mhyper_native", "mkgw_mhyper_adamf", "mkgw_native_adamf",
    "db15k_mhyper_native", "db15k_mhyper_adamf", "db15k_native_adamf"
)
$rawRoot = "outputs/complementarity_identifiability/closure_test/raw"

if ($Mode -eq "Analyze") {
    & $Python "scripts/audit_complementarity_closure_test.py" `
        "--raw-root" $rawRoot `
        "--dev-manifest-dir" "outputs/aacpi/utility_tables" `
        "--output-dir" "outputs/complementarity_identifiability/closure_test/audit" `
        "--report" "docs/reports/complementarity_closure_test_audit.md"
    if ($LASTEXITCODE -ne 0) { throw "Frozen closure TEST analysis failed." }
    Write-Host "[OK] One-time frozen TEST analysis completed; no DEV gate or narrative was changed."
    exit 0
}

if ($Mode -eq "Evaluate") {
    & $Python -c "import torch,sys; sys.exit(0 if ('$Device' != 'cuda' or torch.cuda.is_available()) else 1)"
    if ($LASTEXITCODE -ne 0) { throw "CUDA was requested but is unavailable." }
}

foreach ($pairId in $pairIds) {
    $manifestPath = "outputs/aacpi/utility_tables/${pairId}_dev_source_manifest.json"
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.split -ne "dev" -or $manifest.pair_id -ne $pairId) { throw "Invalid DEV source manifest: $manifestPath" }
    $selectionPath = [string]$manifest.source_selection.path
    $selectionHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $selectionPath).Hash.ToLowerInvariant()
    if ($selectionHash -ne [string]$manifest.source_selection.sha256) { throw "Frozen DEV selection hash mismatch: $selectionPath" }
    $selection = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
    $devSummaryPath = [string]$manifest.source_full_ranking_summary.path
    $summaryHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $devSummaryPath).Hash.ToLowerInvariant()
    if ($summaryHash -ne [string]$manifest.source_full_ranking_summary.sha256) { throw "Frozen DEV summary hash mismatch: $devSummaryPath" }
    $devSummary = Get-Content -LiteralPath $devSummaryPath -Raw | ConvertFrom-Json
    $runPairs = @()
    foreach ($seed in @(1, 2, 3)) {
        $a = @($devSummary.endpoint_reproduction | Where-Object { $_.expert -eq "A" -and $_.seed -eq $seed })
        $b = @($devSummary.endpoint_reproduction | Where-Object { $_.expert -eq "B" -and $_.seed -eq $seed })
        if ($a.Count -ne 1 -or $b.Count -ne 1) { throw "Missing frozen seed-$seed run pair for $pairId" }
        foreach ($runPath in @([string]$a[0].run_dir, [string]$b[0].run_dir)) {
            if (-not (Test-Path -LiteralPath (Join-Path $runPath "best.ckpt"))) { throw "Frozen checkpoint missing: $runPath" }
        }
        $runPairs += "$($a[0].run_dir)::$($b[0].run_dir)"
    }
    $outDir = Join-Path $rawRoot $pairId
    if ($Mode -eq "Preflight") {
        Write-Host "[OK] ${pairId}: frozen alpha=$($selection.global_alpha); 3 checkpoint pairs verified."
        continue
    }
    if (Test-Path -LiteralPath (Join-Path $outDir "test_summary.json")) {
        Write-Host "[LOCKED] ${pairId}: completed TEST export already exists and will not be rerun."
        continue
    }
    $arguments = @(
        "scripts/eval_heterogeneous_complementarity.py",
        "--pair-name", $pairId,
        "--expert-a-name", [string]$selection.expert_a_name,
        "--expert-b-name", [string]$selection.expert_b_name,
        "--split", "test",
        "--selection-json", $selectionPath,
        "--output-dir", $outDir,
        "--device", $Device,
        "--export-alpha-grid"
    )
    foreach ($runPair in $runPairs) { $arguments += @("--run-pair", $runPair) }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Frozen TEST evaluation failed for $pairId" }
}

if ($Mode -eq "Preflight") {
    Write-Host "[OK] Closure TEST preflight completed; DEV locks and checkpoints verified. No TEST split was loaded."
} else {
    Write-Host "[OK] One-time frozen TEST full-ranking evaluation completed. Run Analyze exactly once."
}
