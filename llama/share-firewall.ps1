<#
.SYNOPSIS
  Open the llama-server port for the OTHER laptop only, and only on private (home) networks. Run once, as administrator.

.DESCRIPTION
    .\share-firewall.ps1 -PeerIp 192.168.1.50            # allow that one address
    .\share-firewall.ps1 -PeerIp 192.168.1.50,192.168.1.51 # several machines
    .\share-firewall.ps1 -Remove                          # delete the rule again

  The rule is Inbound / TCP / the llama-server port / Profile = Private only / RemoteAddress = the peer(s).
  The corporate network is a Domain (or Public) profile, so nothing is reachable there even while the server is running.
  Windows firewall rules take IP addresses, not host names: set a DHCP reservation for the peer in the router (or update the
  rule when its address changes). Nothing else on the network, and nobody without the API key, can use the server.
#>
[CmdletBinding()]
param(
    [string[]]$PeerIp,
    [int]$Port = 8080,
    [switch]$Remove
)
$ErrorActionPreference = 'Stop'
$name = 'ForemanKit llama-server (peer)'
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw 'Run this script from an elevated (administrator) PowerShell.' }
if ($Remove) {
    Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    Write-Host "Rule '$name' removed."; return
}
if (-not $PeerIp) { throw 'Give the other laptop address:  -PeerIp 192.168.1.50' }
Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Remove-NetFirewallRule   # replace, never duplicate
New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port `
    -Profile Private -RemoteAddress $PeerIp | Out-Null
Write-Host "Rule '$name': TCP $Port from $($PeerIp -join ', ') on Private networks only." -ForegroundColor Green
Get-NetConnectionProfile | ForEach-Object { Write-Host ("  network '{0}' is {1}" -f $_.Name, $_.NetworkCategory) }
