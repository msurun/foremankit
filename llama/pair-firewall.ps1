<#
.SYNOPSIS
  Open (or close) the Windows firewall for ForemanKit computer pairing on Private AND Public networks.

.DESCRIPTION
  Pairing listens on UDP 8766 (discovery) and TCP 8767 (key hand-over). The Windows "allow access" prompt offers only
  Private networks by default, and a prompt that was dismissed leaves a block rule behind that beats every allow rule.
  This script creates inbound allow rules (UDP 8766 and TCP 8767) for every given python.exe on the Private and Public profiles, restricted to the
  local subnet (a computer outside your own network cannot reach the ports even on a public network), and removes the
  inbound BLOCK rules Windows created for those same programs. Pass BOTH the venv's python.exe and the base interpreter
  it starts: a venv python.exe is only a launcher, the socket is opened by the base python.exe, and a rule matches the file. Needs an elevated PowerShell (the panel starts it with the
  UAC prompt). The ports only answer while a pairing session is open (two minutes).

    .\pair-firewall.ps1 -Python "C:\path\.venv\Scripts\python.exe;C:\path\Python310\python.exe"
    .\pair-firewall.ps1 -Python ... -DryRun     # print what would happen, change nothing (no admin needed)
    .\pair-firewall.ps1 -Remove                 # delete the two rules again
#>
param(
    [string]$Python = '',          # one or more python.exe paths separated by ';' (an array does not survive -File)
    [switch]$DryRun,
    [switch]$Remove
)

$ports = @(@('UDP', 8766), @('TCP', 8767))
$ruleFilter = 'ForemanKit pairing*'
$profiles = 'Private', 'Public'

function Test-Administrator {
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-BlockingRules {
    # inbound block rules that apply to this exact program (the leftovers of a dismissed "allow access" prompt)
    param([string]$Program)
    Get-NetFirewallApplicationFilter -Program $Program -ErrorAction SilentlyContinue |
        Get-NetFirewallRule -ErrorAction SilentlyContinue |
        Where-Object { $_.Direction -eq 'Inbound' -and $_.Action -eq 'Block' -and $_.Enabled -eq 'True' }
}

if ($Remove) {
    Write-Host "remove rules: $ruleFilter"
    if (-not $DryRun) { Remove-NetFirewallRule -DisplayName $ruleFilter -ErrorAction SilentlyContinue }
    return
}

$programs = @(($Python -split ';') | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique)
if ($programs.Count -eq 0) { throw "-Python must name the python.exe files that run ForemanKit (got '$Python')." }
if (-not $DryRun -and -not (Test-Administrator)) { throw 'Run this in an elevated (administrator) PowerShell, or use the panel button, which asks for it.' }

if (-not $DryRun) { Remove-NetFirewallRule -DisplayName $ruleFilter -ErrorAction SilentlyContinue }
foreach ($program in $programs) {
    foreach ($entry in $ports) {
        $protocol, $port = $entry
        $name = "ForemanKit pairing $protocol $port ($(Split-Path (Split-Path $program -Parent) -Parent | Split-Path -Leaf))"
        Write-Host ("allow inbound {0} {1} for {2} on {3} (local subnet only)" -f $protocol, $port, $program, ($profiles -join '+'))
        if ($DryRun) { continue }
        New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow -Protocol $protocol -LocalPort $port `
            -Program $program -Profile $profiles -RemoteAddress LocalSubnet | Out-Null
    }
    foreach ($rule in @(Get-BlockingRules $program)) {
        $verb = if ($DryRun) { 'would be removed' } else { 'removed' }
        Write-Host ("block rule for the same program found: '{0}' ({1}) - {2}, it overrides every allow rule" -f $rule.DisplayName, $rule.Profile, $verb)
        if (-not $DryRun) { Remove-NetFirewallRule -Name $rule.Name }
    }
}
Write-Host 'Done.'
