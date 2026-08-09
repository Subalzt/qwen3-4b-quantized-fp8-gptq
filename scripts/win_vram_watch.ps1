# Peak-VRAM sampler for the Windows-side L1 loop (LM Studio).
#
# LM Studio's stats bar reports tok/s and TTFT but NOT peak VRAM, and
# torch.cuda.max_memory_allocated() cannot see llama.cpp at all. So we sample
# the driver from a separate process, which is the only method that gives a
# number comparable to the WSL harness later.
#
# Usage:
#   1. Start this.
#   2. Let it collect ~5s of idle baseline BEFORE loading the model.
#   3. Load the model in LM Studio, run your prompt.
#   4. Ctrl+C. Read PEAK and DELTA.
#
#   .\win_vram_watch.ps1 -Label bf16_q4test -Hz 5

param(
    [string]$Label = "run",
    [double]$Hz = 5,
    [string]$OutDir = "$PSScriptRoot\..\results\raw"
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$csv = Join-Path $OutDir "vram_$($Label)_$(Get-Date -Format yyyyMMdd_HHmmss).csv"
$periodMs = [int](1000 / $Hz)

"timestamp_ms,used_mb,total_mb" | Out-File -FilePath $csv -Encoding utf8

Write-Host "Sampling GPU memory at $Hz Hz -> $csv" -ForegroundColor Cyan
Write-Host "Collect ~5s of idle baseline, THEN load the model. Ctrl+C when done.`n" -ForegroundColor Yellow

$peak = 0.0
$baseline = $null
$samples = 0
$t0 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()

try {
    while ($true) {
        $raw = (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits) -split ','
        $used = [double]$raw[0].Trim()
        $total = [double]$raw[1].Trim()
        $now = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $samples++

        # First 5 seconds define the idle baseline (desktop compositor etc).
        if (($now - $t0) -lt 5000) {
            if ($null -eq $baseline -or $used -lt $baseline) { $baseline = $used }
        }
        if ($used -gt $peak) { $peak = $used }

        "$now,$used,$total" | Add-Content -Path $csv -Encoding utf8

        $delta = if ($null -ne $baseline) { $peak - $baseline } else { 0 }
        $phase = if (($now - $t0) -lt 5000) { "BASELINE" } else { "RECORDING" }
        Write-Host -NoNewline ("`r[{0,-9}] now {1,6:N0} MiB | peak {2,6:N0} | baseline {3,6:N0} | DELTA {4,6:N0} MiB | n={5}   " -f $phase, $used, $peak, $baseline, $delta, $samples)

        Start-Sleep -Milliseconds $periodMs
    }
}
finally {
    Write-Host "`n"
    Write-Host "=== $Label ===" -ForegroundColor Green
    Write-Host ("  peak      : {0:N0} MiB" -f $peak)
    Write-Host ("  baseline  : {0:N0} MiB" -f $baseline)
    Write-Host ("  DELTA     : {0:N0} MiB   <- this is the model's VRAM cost" -f ($peak - $baseline))
    Write-Host ("  total     : {0:N0} MiB" -f $total)
    Write-Host ("  samples   : {0} @ {1} Hz" -f $samples, $Hz)
    Write-Host "  csv       : $csv"
}
