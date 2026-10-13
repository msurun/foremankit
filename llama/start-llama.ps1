<#
.SYNOPSIS
  Start llama.cpp's llama-server with any GGUF model from this folder (menu in a console; default Qwen3.6-35B-A3B Q4_K_XL) in the configuration tuned for this machine (llama/profiles.json).

.DESCRIPTION
  Everything is relative to this folder (LLAMA.CPP\llama-server.exe + the *.gguf files next to this script).

    .\start-llama.ps1                # start (blocks; Ctrl+C stops it)
    .\start-llama.ps1                # in a console: menu of every .gguf in this folder (Enter = default); -NoMenu skips it
    .\start-llama.ps1 -NoThink       # thinking off: faster, fewer tokens (good for simple tasks, foremankit "quick")
    .\start-llama.ps1 -ThinkBudget 1024   # thinking on, capped at 1024 tokens per answer (llama-server --reasoning-budget)
    .\start-llama.ps1 -PrintOnly     # show the command and the pre-flight checks, start nothing
    .\start-llama.ps1 -List          # show every llama-server running on this machine (PID, model, bind address, start time)
    .\start-llama.ps1 -Replace       # stop the llama-server(s) already running here, then start this one
    .\start-llama.ps1 -Force         # start anyway (e.g. a deliberate second server on another port; each holds ~25 GB of RAM)

  ONE SERVER PER MACHINE. Several sessions used to start copies (even on the same port with another bind address, which
  Windows allows). The launcher now: reuses an identical running server (nothing started), refuses when a different one
  runs (names its PID, model and address), and holds a machine-wide lock so two sessions cannot start at the same moment.
  Start servers only through this script, never by calling llama-server.exe directly.

  Endpoint: http://127.0.0.1:8080/v1  (OpenAI-compatible; model name = -Alias). foremankit finds it by itself
  (provider llm_local, endpoint "llamacpp", env LLAMACPP_URL).

  Why this configuration (llama-bench, same build b11228, on AC power, RTX A500 4 GB + 63 GB RAM, 2026-10-08):
    --n-cpu-moe 40   all routed-expert weights stay in RAM; everything else (attention, SSM layers, shared experts,
                     output, KV cache) goes to the GPU (~2 GiB VRAM). Static placement, no on-demand paging.
                     Q5_K_M: 16.6 tok/s generation / 136 tok/s prompt (38 gives the same; 36 collapses to 5.8 because
                     the experts no longer fit in VRAM). Q4_K_XL control: 16.2 / 154. Run-to-run noise is ~10%.
    --load-mode none loads the whole model into RAM instead of mmap. With expert overrides mmap can fall to
                     1-2 tok/s when free RAM is low (page thrashing); measured on the author's machines.
    -ngl 99 -fa on   all non-expert layers on GPU, flash attention.
    --jinja          chat template embedded in the GGUF (ChatML + <think> + XML tool calls).
    sampling         top_k 20 / top_p 0.95 are embedded in the GGUF; temperature lowered to 0.6 for code/data work.
    -c 32768         hybrid attention: only 10 of 40 layers keep a KV cache, so a long context is cheap.
                     Context vs speed (RTX 3070 8 GB, Q4_K_XL, q8_0 KV, ncmoe 34, 2026-10-12): 32k 29.3, 64k 27.8, 128k 21-22 tok/s;
                     262k does not start. Above ~64k attention/buffers cost more speed than the extra context is worth: the
                     KV cache is allocated at start, so check VRAM stays below ~7.5 GB (nvidia-smi) when raising -Ctx.

  Measured 2026-10-09 (llama-bench b11228, 3 repetitions, AC, results in benchmarks\RESULTS.md):
    pinned to the 6 P-cores (-C 0xFFC03 --cpu-strict 1 -t 12): 18.4 tok/s vs 17.0 unpinned; E-cores HURT (18 threads P+E: 11.4).
    Q4_K_XL beats Q5_K_M: 19.1 vs 18.4 tok/s generation, 192 vs 167 prompt; tool calling 48/48 vs 46/48 correct.
    No gain from: -fa off, q8_0 KV cache, batch sizes, --poll, --prio, mmap/mlock. --cache-reuse is useless here (hybrid model).
    Prompt processing is GPU-bound (~155-195 tok/s) and does not depend on threads: a 10k-token cold prompt takes ~60 s,
    20k ~130 s; an unchanged prefix is reused (0.7 s). Keep the system prompt/tools stable. The cache is lost whenever
    anything before the end of the cached text changes.
    These values live in profiles.json (key = "cpu | gpu | vram"), not in this script: the launcher recognises the machine and
    applies its profile. On an unknown machine it warns and uses safe defaults; run `python tune_llama.py` there (15 min, AC power)
    and the next start uses the measured values. -IgnoreProfile forces the safe defaults; explicit parameters always win.

  Re-tune on another machine: `python tune_llama.py` measures n-cpu-moe for a model.
  Ask for ~22 GB of free RAM before starting (the check below warns when it is lower).
#>
[CmdletBinding()]
param(
    [string]$Model = '',                       # empty: menu of all *.gguf here (interactive), else Qwen3.6 Q4_K_XL / Q5_K_M / first *.gguf
    [switch]$NoMenu,                           # skip the model menu and take the default (scripts, foremankit auto-start)
    [string]$Alias = '',                       # model name served on /v1; default = the GGUF file name without extension
    [int]$NCpuMoe  = -1,                       # -1 = from the machine profile, else llama-server's own --fit
    [int]$Ctx      = 0,                        # 0 = from the profile, else 32768
    [int]$Port     = 8080,
    [int]$Threads  = -1,                       # -1 = from the machine profile, else llama.cpp auto; 0 = auto
    [string]$CpuMask = '',                     # from the machine profile unless given; '' = no pinning
    [switch]$IgnoreProfile,                    # skip profiles.json (safe defaults: --fit, auto threads)
    [double]$Temp  = 0.6,
    [switch]$NoThink,
    [int]$ThinkBudget = 0,                     # >0: thinking on, at most this many thinking tokens per answer (not with -NoThink)
    [string]$KvCache = '',                     # e.g. q8_0: KV cache quantisation (overrides the profile); halves KV memory
    [switch]$NoKvOffload,                      # keep the KV cache in system RAM instead of VRAM (long context on a small GPU)
    [switch]$Share,                           # lend this server to the other laptop(s) on the LAN (config/peers.yaml): key required
    [switch]$PrintOnly,
    [switch]$List,                             # list running llama-servers and exit
    [switch]$Replace,                          # stop the servers running here first
    [switch]$Force                             # skip the one-server-per-machine guard
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot

# ---- one server per machine (see .DESCRIPTION)
function Get-LlamaServers {
    Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" | ForEach-Object {
        $line = [string]$_.CommandLine
        [pscustomobject]@{
            Pid     = [int]$_.ProcessId
            Started = $_.CreationDate
            Model   = $(if ($line -match '--model\s+"?([^"]+?\.gguf)') { Split-Path $matches[1] -Leaf } else { '?' })
            Bind    = $(if ($line -match '--host\s+(\S+)') { $matches[1] } else { '127.0.0.1' })
            Port    = $(if ($line -match '--port\s+(\d+)') { [int]$matches[1] } else { 8080 })
        }
    }
}
function Get-StartDecision {
    # 'start' = nothing runs; 'reuse' = exactly one server runs and it is the one asked for; 'refuse' = anything else
    param($Running, [string]$ModelFile, [int]$WantedPort, [bool]$Shared)
    $all = @($Running)
    if ($all.Count -eq 0) { return 'start' }
    $same = @($all | Where-Object { $_.Model -eq $ModelFile -and $_.Port -eq $WantedPort -and (($_.Bind -eq '0.0.0.0') -eq $Shared) })
    if ($all.Count -eq 1 -and $same.Count -eq 1) { return 'reuse' }
    return 'refuse'
}
function Show-LlamaServers {
    param($Servers)
    $all = @($Servers)
    if ($all.Count -eq 0) { Write-Host 'No llama-server is running on this machine.'; return }
    foreach ($s in $all) {
        Write-Host ("  PID {0}  {1}  {2}:{3}  started {4:yyyy-MM-dd HH:mm}" -f $s.Pid, $s.Model, $s.Bind, $s.Port, $s.Started)
    }
}
function Get-LaunchLock {
    # machine-wide lock held for the whole life of this script: a second launcher cannot start at the same moment
    try {
        $lock = New-Object System.Threading.Mutex($false, 'Global\foremankit-llama-server-launcher')
        try { $got = $lock.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $got = $true }
        if ($got) { return $lock } else { return $null }
    } catch {
        Write-Warning "Could not create the launcher lock ($($_.Exception.Message)); relying on the process check only."
        return $false
    }
}
if ($List) { Show-LlamaServers (Get-LlamaServers); return }
$exe  = Join-Path $here 'LLAMA.CPP\llama-server.exe'
if (-not (Test-Path $exe)) { throw "llama-server.exe not found: $exe" }
function Get-Fingerprint {
    $cpu = ((Get-CimInstance Win32_Processor | Select-Object -First 1).Name -replace '\s+', ' ').Trim().ToLower()
    $gpu = 'none'; $vram = 0
    try {
        $line = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        if ($line) { $n, $m = $line -split ','; $gpu = ($n -replace '\s+', ' ').Trim().ToLower(); $vram = [int][math]::Round([double]$m.Trim() / 1024) }
    } catch {}
    "$cpu | $gpu | ${vram}gb"
}
function Select-ModelInteractively {
    # numbered menu of every .gguf next to this script (and in $env:FOREMANKIT_MODELS_DIR); Enter = the default model
    param([string]$Default)
    $dirs = @($here, $env:FOREMANKIT_MODELS_DIR) | Where-Object { $_ -and (Test-Path $_) }
    $files = @($dirs | ForEach-Object { Get-ChildItem -Path $_ -Filter '*.gguf' -File -ErrorAction SilentlyContinue } |
              Where-Object { $_.Name -notmatch 'mmproj' } | Sort-Object Name)
    if ($files.Count -eq 0) { return $null }
    $machine = Get-Fingerprint
    $profiles = $null
    $file = Join-Path $here 'profiles.json'
    if (Test-Path $file) { $profiles = Get-Content $file -Raw | ConvertFrom-Json }
    Write-Host "`nWhich model to start?" -ForegroundColor Cyan
    for ($i = 0; $i -lt $files.Count; $i++) {
        $f = $files[$i]
        $hit = if ($profiles) { $profiles.PSObject.Properties["$machine || $([IO.Path]::GetFileNameWithoutExtension($f.Name).ToLower())"] } else { $null }
        $info = if ($hit) { '{0} tok/s, ctx {1}' -f $hit.Value.tg_tok_s, $hit.Value.ctx } else { 'not tuned (safe defaults)' }
        $mark = if ($f.Name -eq (Split-Path $Default -Leaf)) { '*' } else { ' ' }
        Write-Host ('  {0}{1,2}) {2,-58} {3,5:N1} GB   {4}' -f $mark, ($i + 1), $f.Name, ($f.Length / 1GB), $info)
    }
    $answer = Read-Host "Number (Enter = default marked with *)"
    if (-not $answer) { return $null }
    $n = 0
    if (-not [int]::TryParse($answer, [ref]$n) -or $n -lt 1 -or $n -gt $files.Count) { throw "Not a valid choice: '$answer'" }
    return $files[$n - 1].FullName
}
if (-not $Model) {
    # default model: Qwen3.6 Q4_K_XL (best speed/quality here, see benchmarks/COMPARISON.md); Q5_K_M only if Q4 is absent
    $dirs = @($here, $env:FOREMANKIT_MODELS_DIR) | Where-Object { $_ -and (Test-Path $_) }
    $found = $null
    foreach ($pat in 'Qwen3.6-35B-A3B*Q4_K_XL*.gguf', 'Qwen3.6-35B-A3B*Q5_K_M*.gguf') {
        $found = $dirs | ForEach-Object { Get-ChildItem -Path $_ -Filter $pat -File -ErrorAction SilentlyContinue } | Select-Object -First 1
        if ($found) { break }
    }
    if (-not $found) { $found = Get-ChildItem -Path $here -Filter '*.gguf' -File | Sort-Object Name | Select-Object -First 1 }
    if (-not $found) { throw "No .gguf found; pass -Model <path>." }
    $Model = $found.FullName
    $interactive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected
    if (-not $NoMenu -and $interactive) {
        $chosen = Select-ModelInteractively $Model
        if ($chosen) { $Model = $chosen }
    }
}
if (-not [IO.Path]::IsPathRooted($Model)) {
    # the launcher changes directory to the exe's folder before starting, so a bare file name must become a full path now
    $inHere = Join-Path $here $Model
    if (Test-Path $inHere) { $Model = $inHere } elseif (Test-Path $Model) { $Model = (Resolve-Path $Model).Path }
}
if (-not $PSBoundParameters.ContainsKey('Alias')) { $Alias = [IO.Path]::GetFileNameWithoutExtension($Model).ToLower() }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

# ---- guard: never a second server on this machine (checked before the slow profile and RAM steps)
$runningServers = @(Get-LlamaServers)
$decision = Get-StartDecision $runningServers (Split-Path $Model -Leaf) $Port ([bool]$Share)
if ($Replace -and $runningServers.Count -gt 0 -and -not $PrintOnly) {
    Write-Host 'Stopping the running llama-server(s):' -ForegroundColor Yellow
    Show-LlamaServers $runningServers
    $runningServers | ForEach-Object { Stop-Process -Id $_.Pid -Force }
    Start-Sleep -Seconds 3
    $runningServers = @(Get-LlamaServers)
    $decision = Get-StartDecision $runningServers (Split-Path $Model -Leaf) $Port ([bool]$Share)
}
if (-not $Force) {
    if ($decision -eq 'reuse') {
        Write-Host 'This llama-server is already running here, so nothing was started (use -Replace to restart it):' -ForegroundColor Green
        Show-LlamaServers $runningServers
        return
    }
    if ($decision -eq 'refuse') {
        Write-Host 'A different llama-server is already running on this machine:' -ForegroundColor Red
        Show-LlamaServers $runningServers
        $message = 'Not starting a second server (each holds ~25 GB of RAM). Use the running one, or -Replace to swap it, or -Force.'
        if ($PrintOnly) { Write-Warning $message } else { throw $message }
    }
    if ($decision -eq 'start' -and (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
        $message = "Port $Port is held by a program that is not a llama-server. Free it, use -Port <other>, or -Force."
        if ($PrintOnly) { Write-Warning $message } else { throw $message }
    }
    if (-not $PrintOnly) {
        $script:launchLock = Get-LaunchLock
        if ($null -eq $script:launchLock) { throw 'Another launcher holds the machine-wide lock: a server is being started (or runs) right now. Run -List.' }
    }
}

# ---- machine profile: tuned by tune_llama.py, looked up by "cpu | gpu | vram || model" (same text the tuner writes)
$fp = (Get-Fingerprint) + ' || ' + ([IO.Path]::GetFileNameWithoutExtension($Model)).ToLower()
$prof = $null
$profFile = Join-Path $here 'profiles.json'
if (-not $IgnoreProfile -and (Test-Path $profFile)) {
    $all = Get-Content $profFile -Raw | ConvertFrom-Json
    $hit = $all.PSObject.Properties[$fp]    # $fp already contains the model: "machine || model"
    if ($hit) { $prof = $hit.Value }
}
if ($prof) {
    Write-Host ("Profile: matched this machine (tuned {0}, {1}: {2} tok/s generation, {3} tok/s prompt)" -f $prof.tuned, $prof.note, $prof.tg_tok_s, $prof.pp_tok_s) -ForegroundColor Green
    if (-not $PSBoundParameters.ContainsKey('NCpuMoe') -and $null -ne $prof.ncmoe)  { $NCpuMoe = [int]$prof.ncmoe }
    if (-not $PSBoundParameters.ContainsKey('Threads') -and $null -ne $prof.threads) { $Threads = [int]$prof.threads }
    if (-not $PSBoundParameters.ContainsKey('CpuMask')) { $CpuMask = [string]$prof.cpu_mask }
    if (-not $PSBoundParameters.ContainsKey('Ctx') -and $prof.ctx) { $Ctx = [int]$prof.ctx }
    if (-not $PSBoundParameters.ContainsKey('Temp') -and $prof.temp) { $Temp = [double]$prof.temp }
} else {
    Write-Warning "No tuned profile for this machine + model ('$fp'). Using safe defaults (llama-server --fit, automatic threads, no CPU pinning)."
    Write-Warning "For the best speed run:  python `"$here\tune_llama.py`"  (about 15 min, on AC power); the next start picks the result up."
}
if ($Ctx -le 0) { $Ctx = 32768 }

$argv = @(
    '--model', $Model,
    '--alias', $Alias,
    '--host', $(if ($Share) { '0.0.0.0' } else { '127.0.0.1' }),   # localhost only unless -Share (then a key is required)
    '--port', $Port,
    '-c', $Ctx,
    '-ngl', 99,
    '--load-mode', 'none',
    '--jinja',
    '-fa', 'on',
    '-np', 1,
    '--temp', $Temp, '--top-p', 0.95, '--top-k', 20, '--min-p', 0,
    '--no-webui'
)
if ($NCpuMoe -ge 0) { $argv += @('--n-cpu-moe', $NCpuMoe) }          # otherwise --fit (default on) decides the offload
if ($Threads -gt 0) { $argv += @('-t', $Threads, '-tb', $Threads) }
if ($CpuMask)       { $argv += @('-C', $CpuMask, '--cpu-strict', 1) }
if ($prof -and $prof.kv_cache) { $argv += @('-ctk', $prof.kv_cache, '-ctv', $prof.kv_cache) }   # e.g. Gemma needs q8_0 KV to fit 4 GB at 32k
if ($KvCache)       { $argv += @('-ctk', $KvCache, '-ctv', $KvCache) }
if ($NoKvOffload)   { $argv += '--no-kv-offload' }
if ($prof -and $prof.ubatch)  { $argv += @('-ub', $prof.ubatch) }
if ($NoThink -and $ThinkBudget -gt 0) { throw '-NoThink and -ThinkBudget exclude each other.' }
if ($NoThink -or ($prof -and $prof.reasoning -eq 'off' -and $ThinkBudget -le 0)) { $argv += @('--reasoning', 'off') }   # profile can default thinking off; -ThinkBudget wins
if ($ThinkBudget -le 0 -and -not $NoThink -and $prof -and $prof.think_budget) { $ThinkBudget = [int]$prof.think_budget }   # profile default; -ThinkBudget / -NoThink win
if ($ThinkBudget -gt 0) { $argv += @('--reasoning', 'on', '--reasoning-budget', $ThinkBudget) }
$argv = $argv | ForEach-Object { [string]$_ }

# ---- sharing with the other laptop (-Share): listens on the LAN, every request must carry the shared key
if ($Share) {
    $key = $env:FOREMANKIT_PEER_KEY
    if (-not $key) {
        $envFile = Join-Path (Split-Path $here -Parent) '.env'
        if (Test-Path $envFile) {
            $line = Get-Content $envFile | Where-Object { $_ -match '^FOREMANKIT_PEER_KEY=.+' } | Select-Object -First 1
            if ($line) { $key = ($line -split '=', 2)[1].Trim() }
        }
    }
    if (-not $key) { throw "-Share needs FOREMANKIT_PEER_KEY (run: foremankit peers key - then copy that .env line to the other laptop)." }
    $env:LLAMA_API_KEY = $key                  # llama-server reads it from the environment: the key never appears in the process list
    $bad = Get-NetConnectionProfile | Where-Object { $_.NetworkCategory -ne 'Private' }
    if ($bad) { Write-Warning ("Network '{0}' is {1}: the firewall rule only opens the port on PRIVATE networks (home). Set the home network to Private, or the other laptop cannot connect." -f ($bad[0].Name), $bad[0].NetworkCategory) }
    if (-not (Get-NetFirewallRule -DisplayName 'ForemanKit llama-server (peer)' -ErrorAction SilentlyContinue)) {
        Write-Warning "No firewall rule yet: run (as administrator)  .\share-firewall.ps1 -PeerIp <the other laptop's IP>"
    }
    $listen = $argv[[array]::IndexOf($argv, '--host') + 1]
    Write-Host ("Sharing: listening on {0}:{1}, API key required (not shown). Priority rules: config/peers.yaml." -f $listen, $Port) -ForegroundColor Yellow
    if ($listen -eq '127.0.0.1') { Write-Warning 'The server still listens on localhost only, so the other laptop cannot reach it (bind address for -Share not enabled yet).' }
}

# ---- pre-flight checks (warnings only)
$modelGb = [math]::Round((Get-Item $Model).Length / 1GB, 1)
$freeGb  = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
Write-Host ("Model: {0} ({1} GB), free RAM: {2} GB" -f (Split-Path $Model -Leaf), $modelGb, $freeGb)
if ($freeGb -lt ($modelGb + 3)) {
    Write-Warning "Free RAM ($freeGb GB) is lower than model + 3 GB. Close other programs, or it will swap and crawl."
}
$bat = Get-CimInstance Win32_Battery | Select-Object -First 1
if ($bat -and $bat.BatteryStatus -eq 1) { Write-Warning 'Running on battery: expect a throttled CPU and a lower speed.' }

if (-not (Test-Path $Model)) { throw "Model not found: $Model" }
$shown = ($argv | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
Write-Host "Command:`n  llama-server.exe $shown" -ForegroundColor Cyan
Write-Host "Endpoint: http://127.0.0.1:$Port/v1   model name: $Alias`n"
if ($PrintOnly) { Write-Host 'PrintOnly: nothing started.' -ForegroundColor Yellow; return }

Set-Location (Split-Path $exe)                 # so the exe finds its CUDA/ggml DLLs
try {
    & $exe @argv
} finally {
    # an interactive terminal outlives this script: without this the lock stayed owned after the server stopped
    if ($script:launchLock -is [System.Threading.Mutex]) { try { $script:launchLock.ReleaseMutex() } catch {}; $script:launchLock.Dispose() }
}
