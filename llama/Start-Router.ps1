<#
  Manual control of the foremankit gateway (port 8081, foremankit/gateway.py) used by the Claude Code VS Code extension.
    .\Start-Router.ps1            start (no-op if already running)
    .\Start-Router.ps1 -Status    show whether it runs + last log lines
    .\Start-Router.ps1 -Stop      stop it (the extension then cannot reach the cloud until you start it again,
                                  or remove claudeCode.environmentVariables from the VS Code settings)
#>
param([switch]$Stop, [switch]$Status, [switch]$Force, [int]$Port = 8081)

$here   = $PSScriptRoot
$python = Join-Path $here '..\.venv\Scripts\python.exe'
$log    = Join-Path $env:TEMP 'router.log'
$err    = Join-Path $env:TEMP 'router.err'

function Get-Router {
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*foremankit.gateway*' }
}

if ($Stop) {
    $p = Get-Router
    if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Write-Host 'Router stopped.' }
    else    { Write-Host 'Router was not running.' }
    return
}

if ($Status) {
    $p = Get-Router
    if ($p) { Write-Host "Router RUNNING (PID $($p.ProcessId -join ', '))" -ForegroundColor Green }
    else    { Write-Host 'Router NOT running.' -ForegroundColor Yellow }
    if (Test-Path $log) { Write-Host "`nLast log lines:"; Get-Content $log -Tail 10 }
    return
}

if (Get-Router) { Write-Host "Router already running on port $Port."; return }
# direct mode (ForemanKit panel): the router is off on purpose, so the Task Scheduler must not bring it back
$offFlag = Join-Path (Join-Path $env:USERPROFILE '.config\foremankit') 'router_off'
if ((Test-Path $offFlag) -and -not $Force) { Write-Host 'Direct mode: router stays off (use -Force, or the panel: Start router).'; return }

# `-c` instead of the foremankit-gateway.exe console script: the exe is locked while the MCP server of the same venv runs
Start-Process -WindowStyle Hidden -FilePath $python -ArgumentList '-c', '"from foremankit.gateway import main; main()"' `
    -WorkingDirectory $here -RedirectStandardOutput $log -RedirectStandardError $err

for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        # any HTTP answer (even 401 from the cloud) proves the proxy is up
        Invoke-WebRequest -Uri "http://127.0.0.1:$Port/v1/models" -UseBasicParsing -TimeoutSec 3 | Out-Null
        Write-Host "Router started on http://127.0.0.1:$Port" -ForegroundColor Green; return
    } catch {
        if ($_.Exception.Response) { Write-Host "Router started on http://127.0.0.1:$Port" -ForegroundColor Green; return }
    }
}
Write-Warning "Router did not answer within 7 s. See $err"
if (Test-Path $err) { Get-Content $err -Tail 10 }
