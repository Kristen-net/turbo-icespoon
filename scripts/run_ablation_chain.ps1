# Ablation chain runner v2 - corrected per document
# Queue: ablation_no_uncertainty, ablation_freeze_backbone, ablation_cascade (baseline)
$REPO = "C:\Users\2457025871\.trae-cn\work\turbo-icespoon"
$LOG_FILE = "$REPO\outputs\ablation_chain.log"
$CONDA_ENV = "dehaze_fusion"
$CHECK_INTERVAL = 60

function Write-Log {
    param([string]$msg)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Output $line
    Add-Content -Path $LOG_FILE -Value $line -Encoding UTF8
}

function Test-TrainingRunning {
    $proc = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet64 -gt 100MB } | Select-Object -First 1
    return ($null -ne $proc)
}

function Wait-TrainingDone {
    Write-Log "Waiting for current training to finish..."
    $idleCount = 0
    while ($idleCount -lt 3) {
        $running = Test-TrainingRunning
        if (-not $running) {
            $idleCount++
            Write-Log "  No training process detected (idle $idleCount/3)"
            if ($idleCount -lt 3) { Start-Sleep -Seconds $CHECK_INTERVAL }
        } else {
            $idleCount = 0
            Start-Sleep -Seconds $CHECK_INTERVAL
        }
    }
    Write-Log "Current training process has ended"
}

function Run-Experiment {
    param([string]$configName, [string]$outputName)
    Write-Log "=== Starting experiment: $configName ==="

    $cfgPath = "$REPO\configs\train\$configName.yaml"
    if (-not (Test-Path $cfgPath)) {
        Write-Log "  [ERROR] Config not found: $cfgPath"
        return $false
    }

    $runScript = "$REPO\scripts\run_ablation.py"
    Write-Log "  CMD: conda run -n $CONDA_ENV python $runScript --config $configName --output $outputName"

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "conda"
    $psi.Arguments = "run -n $CONDA_ENV python `"$runScript`" --config $configName --output $outputName"
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $proc.Start() | Out-Null

    $completed = $proc.WaitForExit(14400000)
    if (-not $completed) {
        $proc.Kill()
        Write-Log "  [ERROR] Experiment timed out (4h)"
        return $false
    }

    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $exitCode = $proc.ExitCode

    Write-Log "  Exit code: $exitCode"

    $lines = $stdout -split "`n" | Where-Object { $_.Trim() -ne "" } | Select-Object -Last 20
    foreach ($line in $lines) { Write-Log "  [out] $line" }

    if ($stderr -ne "") {
        $errLines = $stderr -split "`n" | Where-Object { $_.Trim() -ne "" } | Select-Object -Last 10
        foreach ($line in $errLines) { Write-Log "  [err] $line" }
    }

    $metricsPath = "$REPO\outputs\$outputName\metrics.json"
    if (Test-Path $metricsPath) {
        $m = Get-Content $metricsPath -Raw | ConvertFrom-Json
        Write-Log "  Best PSNR: $($m.best_psnr)"
        Write-Log "  Epochs: $($m.history.Count)"
        return $true
    } else {
        Write-Log "  [WARN] metrics.json not found"
        return ($exitCode -eq 0)
    }
}

# Main
Write-Log "============================================================"
Write-Log "Ablation chain runner v2 (corrected per document)"
Write-Log "Queue: ablation_no_uncertainty, ablation_freeze_backbone, ablation_cascade"
Write-Log "============================================================"

Wait-TrainingDone

# Exp 2: no_uncertainty (ablation)
$s1 = Run-Experiment "ablation_no_uncertainty" "ablation_no_uncertainty"
if ($s1) { Write-Log "Experiment ablation_no_uncertainty DONE" }
else { Write-Log "Experiment ablation_no_uncertainty FAILED" }

Start-Sleep -Seconds 30

# Exp 3: freeze_backbone (ablation, per document)
$s2 = Run-Experiment "ablation_freeze_backbone" "ablation_freeze_backbone"
if ($s2) { Write-Log "Experiment ablation_freeze_backbone DONE" }
else { Write-Log "Experiment ablation_freeze_backbone FAILED" }

Start-Sleep -Seconds 30

# Cascade baseline (separate comparison row in document table 6.4)
$s3 = Run-Experiment "ablation_cascade" "ablation_cascade"
if ($s3) { Write-Log "Experiment ablation_cascade DONE" }
else { Write-Log "Experiment ablation_cascade FAILED" }

Write-Log "============================================================"
Write-Log "All experiments finished"
Write-Log "============================================================"

Write-Log ""
Write-Log "=== Summary ==="
$allOutputs = @("joint_v2_w5", "ablation_no_boxfeat", "ablation_no_uncertainty", "ablation_freeze_backbone", "ablation_cascade")
foreach ($name in $allOutputs) {
    $metricsPath = "$REPO\outputs\$name\metrics.json"
    if (Test-Path $metricsPath) {
        $m = Get-Content $metricsPath -Raw | ConvertFrom-Json
        $psnr = [math]::Round($m.best_psnr, 2)
        $epochs = $m.history.Count
        Write-Log "  $name : Best PSNR = $psnr dB, Epochs = $epochs"
    } else {
        Write-Log "  $name : [not found]"
    }
}
