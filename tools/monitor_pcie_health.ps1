#requires -Version 5.1
<#
    Samples the GPU's PCIe replay counter (nvidia-smi's "Replays Since Reset")
    and Microsoft-Windows-WHEA-Logger "corrected hardware error" events over a
    window, at a fixed interval. Reproduces the table format from the P7 D14
    diagnosis in docs/tuning/RESULTS.md (elapsed / replays / GPU util%), so a
    rerun after any hardware change (reseat, cable swap, slot change) is
    directly comparable to that baseline:

        elapsed s | replays | GPU util %
        0         | 822     | 1
        60        | 1,026   | 98
        181       | 1,478   | 98
        302       | 2,150   | 98
        513       | 3,173   | 98
        2,351 replays in 513 s = 4.58/s sustained under match load.

    Usage:
        powershell -ExecutionPolicy Bypass -File tools\monitor_pcie_health.ps1
        powershell -ExecutionPolicy Bypass -File tools\monitor_pcie_health.ps1 -DurationSeconds 600 -IntervalSeconds 15

    No admin rights required: nvidia-smi and a System-log Get-WinEvent read
    both work as a standard user.
#>
[CmdletBinding()]
param(
    [int]$DurationSeconds = 300,
    [int]$IntervalSeconds = 15
)

$ErrorActionPreference = 'Stop'

function Get-ReplayCount {
    $line = (nvidia-smi -q | Select-String "Replays Since Reset")
    if (-not $line) { throw "nvidia-smi did not report a replay counter" }
    return [int]($line -replace '.*:\s*', '')
}

function Get-GpuUtil {
    return [int](nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
}

function Get-WheaCountSince([datetime]$since) {
    $events = Get-WinEvent -FilterHashtable @{LogName = 'System'; ProviderName = 'Microsoft-Windows-WHEA-Logger' } `
        -ErrorAction SilentlyContinue
    if (-not $events) { return 0 }
    return ($events | Where-Object { $_.TimeCreated -gt $since }).Count
}

$startTime = Get-Date
$startReplays = Get-ReplayCount
$rows = @()

Write-Host "[monitor] start $startTime  baseline replays=$startReplays" -ForegroundColor Cyan
Write-Host ("{0,10} {1,10} {2,8} {3,10}" -f "elapsed s", "replays", "util pct", "whea total")

$elapsed = 0
while ($elapsed -le $DurationSeconds) {
    $replays = Get-ReplayCount
    $util = Get-GpuUtil
    $whea = Get-WheaCountSince($startTime)
    $rows += [pscustomobject]@{
        ElapsedS = $elapsed
        Replays  = $replays
        UtilPct  = $util
        WheaTot  = $whea
    }
    Write-Host ("{0,10} {1,10} {2,8} {3,10}" -f $elapsed, $replays, $util, $whea)
    Start-Sleep -Seconds $IntervalSeconds
    $elapsed += $IntervalSeconds
}

$endReplays = Get-ReplayCount
$deltaReplays = $endReplays - $startReplays
$actualDuration = ((Get-Date) - $startTime).TotalSeconds
$rate = 0
if ($actualDuration -gt 0) { $rate = $deltaReplays / $actualDuration }
$rateRounded = [math]::Round($rate, 3)
$durationRounded = [math]::Round($actualDuration, 1)
$wheaTotal = Get-WheaCountSince($startTime)

Write-Host ""
Write-Host "[monitor] $deltaReplays replays in $durationRounded s = $rateRounded per s sustained" -ForegroundColor Cyan
Write-Host "[monitor] $wheaTotal WHEA-Logger corrected-hardware-error events during the window" -ForegroundColor Cyan
Write-Host "[monitor] P7 baseline (faulty link, under match load): 4.58 replays per s" -ForegroundColor DarkGray

if ($rate -lt 0.1 -and $wheaTotal -eq 0) {
    Write-Host "[monitor] VERDICT: link looks healthy over this window (near-zero replay rate, no WHEA events)." -ForegroundColor Green
} elseif ($rate -ge 0.1) {
    Write-Host "[monitor] VERDICT: replay rate is non-trivial ($rateRounded per s) -- fault likely still present." -ForegroundColor Yellow
} else {
    Write-Host "[monitor] VERDICT: replay rate is low but WHEA events fired -- possibly an intermittent fault; rerun over a longer window." -ForegroundColor Yellow
}
