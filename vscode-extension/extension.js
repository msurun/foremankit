// ForemanKit Control: a sidebar panel (activity bar button) for the router, the local llama-server and model tuning.
// A thin shell: all logic lives in the Python module foremankit.panel; this file runs it, shows its JSON and opens terminals.
const vscode = require('vscode');
const cp = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.resolve(fs.realpathSync(__dirname), '..');
const PYTHON = process.platform === 'win32'
  ? path.join(ROOT, '.venv', 'Scripts', 'python.exe') : path.join(ROOT, '.venv', 'bin', 'python');
const POLL_MS = 4000;
const WORKERS_POLL_MS = 12000;
const LIVE_POLL_MS = 1000;
const DELEGATIONS_POLL_MS = 8000;
const MODEL_NAMES_POLL_MS = 6 * 3600 * 1000;   // which model names the providers list now (renamed models show up as a problem)
const LIVE_FRESH_MS = 4000;
const http = require('http');
const STATE_DIR = process.env.FOREMANKIT_STATE_DIR || path.join(os.homedir(), '.config', 'foremankit');
const RELOAD_BUTTON = 'Switch and reload window';
const CONFIRM = {
  'download-llama': 'Downloads llama.cpp (about 150-650 MB from GitHub, sha256-checked) into llama/LLAMA.CPP.',
  tune: 'Tuning stops the running llama-server, then tunes the offload and finds the largest fast context. '
    + 'It takes about 25 minutes: keep the laptop on AC power.',
};
const RELOAD_WARNING = ' The VS Code window reloads immediately after the switch: a running Claude Code answer, '
  + 'subagents and background tasks in this window are stopped (the conversation can be resumed). '
  + 'Claude Code reads this setting only when a window starts.';
const CONFIRM_QUICK = {
  'router-direct': 'Direct mode stops the router and removes it from the VS Code settings, so Claude Code talks straight to '
    + 'Anthropic and the local models stop working.' + RELOAD_WARNING,
  'router-normal': 'Router mode starts the router and puts it back into the VS Code settings.' + RELOAD_WARNING,
};
const NEEDS_NO_MODEL = ['download-llama'];
const NEEDS_RELOAD = ['router-direct', 'router-normal'];
const FILES = {
  profiles: path.join(ROOT, 'llama', 'profiles.json'),
  orchestrator: path.join(ROOT, 'config', 'orchestrator.yaml'),
  routerlog: path.join(os.tmpdir(), 'router.log'),
};

function readLive() {
  try {
    const live = JSON.parse(fs.readFileSync(path.join(STATE_DIR, 'gateway_live.json'), 'utf8'));
    return { ...live, fresh: Date.now() - live.time * 1000 < LIVE_FRESH_MS };
  } catch (e) { return null; }
}

// The llama-server demands the shared key while it is lent to the LAN (-Share): same key as the peers use, from the
// environment or the repository's .env. Without one the request goes out bare, as before.
function peerKey() {
  if (process.env.FOREMANKIT_PEER_KEY) return process.env.FOREMANKIT_PEER_KEY;
  try {
    const line = fs.readFileSync(path.join(ROOT, '.env'), 'utf8').split(/\r?\n/).find((row) => row.startsWith('FOREMANKIT_PEER_KEY='));
    return line ? line.slice('FOREMANKIT_PEER_KEY='.length).trim() : '';
  } catch (e) { return ''; }
}

function readSlots() {
  return new Promise((resolve) => {
    const key = peerKey();
    const headers = key ? { Authorization: 'Bearer ' + key } : {};
    const request = http.get({ host: '127.0.0.1', port: 8080, path: '/slots', headers, timeout: 800 }, (answer) => {
      let text = '';
      answer.on('data', (chunk) => { text += chunk; });
      answer.on('end', () => {
        try {
          const slot = JSON.parse(text)[0];
          resolve(slot ? { busy: !!slot.is_processing, promptTokens: slot.n_prompt_tokens, promptDone: slot.n_prompt_tokens_processed } : null);
        } catch (e) { resolve(null); }
      });
    });
    request.on('error', () => resolve(null));
    request.on('timeout', () => { request.destroy(); resolve(null); });
  });
}

function workspaceDir() {
  const folders = vscode.workspace.workspaceFolders;
  return folders && folders.length ? folders[0].uri.fsPath : '';
}

function backend(args) {
  return new Promise((resolve) => {
    cp.execFile(PYTHON, ['-m', 'foremankit.panel', ...args], { cwd: ROOT, timeout: 60000 }, (error, stdout, stderr) => {
      if (error) { resolve({ error: (stderr || error.message).trim().split('\n').pop() }); return; }
      try { resolve(JSON.parse(stdout)); } catch (e) { resolve({ error: 'bad answer: ' + stdout.slice(0, 120) }); }
    });
  });
}

function terminal() {
  const existing = vscode.window.terminals.find((t) => t.name === 'ForemanKit');
  return existing || vscode.window.createTerminal({ name: 'ForemanKit', cwd: ROOT });
}

class Panel {
  constructor() { this.view = null; this.timer = null; }

  resolveWebviewView(view) {
    this.view = view;
    view.webview.options = { enableScripts: true };
    view.webview.html = html();
    view.webview.onDidReceiveMessage((message) => this.handle(message));
    this.timer = setInterval(() => this.pushStatus(), POLL_MS);
    this.workersTimer = setInterval(() => this.pushWorkers(), WORKERS_POLL_MS);
    this.liveTimer = setInterval(() => this.pushLive(), LIVE_POLL_MS);
    this.delegationsTimer = setInterval(() => this.pushDelegations(), DELEGATIONS_POLL_MS);
    this.pushDelegations();
    view.onDidDispose(() => { clearInterval(this.timer); clearInterval(this.workersTimer); clearInterval(this.liveTimer); clearInterval(this.delegationsTimer); clearInterval(this.modelNamesTimer); clearInterval(this.pairTimer); });
    this.refreshModelNames();
    this.modelNamesTimer = setInterval(() => this.refreshModelNames(), MODEL_NAMES_POLL_MS);
    this.pushStatus();
    this.pushWorkers();
    this.pushFirewall();
    this.pushModels();
  }

  async refreshModelNames() { await backend(['refresh-models']); await this.pushStatus(); }

  post(message) { if (this.view) this.view.webview.postMessage(message); }
  async pushStatus() { if (this.view && this.view.visible && !this.busy) this.post({ type: 'status', data: await backend(['status']) }); }
  async pushDelegations() { if (this.view && this.view.visible) this.post({ type: 'delegations', data: await backend(['delegations']) }); }
  async pushFirewall() { this.post({ type: 'firewall', data: await backend(['pair-firewall-status']) }); }
  async pushLive() { if (this.view && this.view.visible) this.post({ type: 'live', data: { gateway: readLive(), slot: await readSlots() } }); }
  async pushWorkers() { if (this.view && this.view.visible) this.post({ type: 'workers', data: await backend(['workers']) }); }
  async pushModels() { this.post({ type: 'models', data: await backend(['models']) }); }

  async handle(message) {
    if (message.type === 'refresh') { await this.pushStatus(); await this.pushModels(); await this.pushWorkers(); await this.pushDelegations(); return; }
    if (message.type === 'config') { await this.runConfig(message); return; }
    if (message.type === 'preset') { await this.runPreset(message.name); return; }
    if (message.type === 'pair') { await this.runPair(message); return; }
    if (message.type === 'open' && FILES[message.file]) { await vscode.window.showTextDocument(vscode.Uri.file(FILES[message.file])); return; }
    if (message.type === 'thinking') { await this.quickAnswer(['thinking', '--levels', message.levels]); return; }
    if (message.type === 'quick') await this.runQuick(message.action, message.model, message.share);
    if (message.type === 'terminal') await this.runLong(message);
  }

  async confirm(text, button) {
    return (await vscode.window.showWarningMessage(text, { modal: true }, button)) === button;
  }

  async quickAnswer(args) {
    const answer = await backend(args);
    setTimeout(() => this.pushStatus(), 1500);
    return answer;
  }

  async runQuick(action, model, share) {
    const needsReload = NEEDS_RELOAD.includes(action);
    if (CONFIRM_QUICK[action] && !(await this.confirm(CONFIRM_QUICK[action], RELOAD_BUTTON))) return;
    const answer = await this.quickAnswer(model ? [action, '--model', model].concat(share ? ['--share'] : []) : [action]);
    const text = 'ForemanKit: ' + (answer.message || answer.error);
    vscode.window.setStatusBarMessage(text, 6000);
    if (needsReload && !answer.error) vscode.commands.executeCommand('workbench.action.reloadWindow');
    if (needsReload && answer.error) vscode.window.showErrorMessage(text);
  }

  async runPreset(name) {
    if (this.busy) return;
    this.busy = true; this.post({ type: 'busy', value: true });
    try { await this.applyPreset(name); } finally {
      this.busy = false; this.post({ type: 'busy', value: false }); await this.pushStatus();
    }
  }

  async applyPreset(name) {
    const cwd = workspaceDir();
    const plan = await backend(['preset-plan', '--name', name, '--cwd', cwd]);
    if (plan.error) { await this.explainRefusal(name, plan); return; }
    if (!plan.steps.length) { vscode.window.setStatusBarMessage('ForemanKit: already active', 6000); return; }
    const note = plan.needs_server ? ' The local model server is not running: start it afterwards.' : '';
    const ask = plan.summary.join('. ') + '.' + (plan.reload ? RELOAD_WARNING : '') + note;
    if (plan.reload && !(await this.confirm(ask, RELOAD_BUTTON))) return;
    const answer = await backend(['apply-preset', '--name', name, '--cwd', cwd]);
    if (answer.error) { vscode.window.showErrorMessage('ForemanKit: ' + answer.error); return; }
    vscode.window.setStatusBarMessage('ForemanKit: ' + answer.message + (answer.reload ? '' : ' - restart the Claude Code chat'), 8000);
    if (answer.reload) vscode.commands.executeCommand('workbench.action.reloadWindow');
  }

  // A refusal must be loud: a status-bar line that fades after seconds looks like "nothing happens".
  async explainRefusal(name, plan) {
    if (!plan.mark) { vscode.window.showErrorMessage('ForemanKit: ' + plan.error); return; }
    const button = 'Mark this folder as ' + plan.mark;
    const ask = plan.error + ' If this folder holds no data above "' + plan.mark + '", you can declare that (writes .foremankit/sensitivity); '
      + 'the orchestrator then reads files in it. Nothing was changed.';
    if ((await vscode.window.showWarningMessage(ask, { modal: true }, button)) !== button) return;
    const done = await backend(['mark-folder', '--name', plan.mark, '--cwd', workspaceDir()]);
    if (done.error) { vscode.window.showErrorMessage('ForemanKit: ' + done.error); return; }
    await this.applyPreset(name);
  }

  configArgs({ action, role, name, url, accept, priority, model, fields, enabled, category, period, value }) {
    const table = {
      orchestrator: ['orchestrator', '--name', name || 'this'],
      budgetSet: ['budget-set', period, '--name', category, '--value', String(value)],
      fixModel: ['fix-model', '--name', name, '--model', model],
      orchestratorMode: ['orchestrator-mode', '--name', name, '--cwd', workspaceDir()],
      delegationSet: ['delegation-set', String(!!enabled), '--name', name],
      escalationSet: ['escalation-set', '--name', name, '--model', model],
      workerTest: ['worker-test', '--name', name],
      workerEnable: ['worker-enable', String(!!enabled), '--name', name],
      role: ['role', role],
      workerAdd: ['worker-add', '--name', name, '--url', url, '--accept', accept, '--priority', String(priority || 10)],
      workerRemove: ['worker-remove', '--name', name],
      profileSet: ['profile-set', '--model', model, '--fields', fields],
    };
    return table[action];
  }

  async runConfig(message) {
    const args = this.configArgs(message);
    if (!args) return;
    const answer = await backend(args);
    const text = 'ForemanKit: ' + (answer.message || answer.error);
    if (message.action === 'workerTest') vscode.window.showInformationMessage(text);
    else vscode.window.setStatusBarMessage(text, 6000);
    await this.pushStatus();
    await this.pushWorkers();
    await this.pushModels();
  }

  async runPair({ action, ip, port, code, replace }) {
    if (action === 'host') {
      const answer = await backend(['pair-host']);
      if (answer.error) { vscode.window.showErrorMessage('ForemanKit: ' + answer.error); return; }
      this.post({ type: 'pairState', data: { state: 'waiting', code: answer.code, expires: answer.expires } });
      this.watchPairing();
    } else if (action === 'firewall') {
      const answer = await backend(['pair-firewall']);
      vscode.window.showInformationMessage('ForemanKit: ' + (answer.message || answer.error));
      await this.pushFirewall();
    } else if (action === 'find') {
      this.post({ type: 'pairFound', data: await backend(['pair-find']) });
    } else if (action === 'join') {
      const args = ['pair-join', '--ip', ip, '--port', String(port), '--code', code].concat(replace ? ['--replace'] : []);
      const answer = await backend(args);
      const text = answer.message || answer.error;
      if (/different lab key/.test(text) && !replace) {
        const ok = await vscode.window.showWarningMessage('This computer already has another lab key. Replace it with the key of the computer you are pairing with? '
          + 'Do this only if that computer belongs to the lab you want to join.', { modal: true }, 'Replace key');
        if (ok === 'Replace key') return this.runPair({ action, ip, port, code, replace: true });
      }
      vscode.window.showInformationMessage('ForemanKit: ' + text);
      await this.pushWorkers();
    }
  }

  watchPairing() {
    clearInterval(this.pairTimer);
    this.pairTimer = setInterval(async () => {
      const state = await backend(['pair-state']);
      this.post({ type: 'pairState', data: state });
      if (state.state !== 'waiting') { clearInterval(this.pairTimer); this.pushWorkers(); }
    }, 2000);
  }

  async runLong({ action, model }) {
    if (!model && !NEEDS_NO_MODEL.includes(action)) return;
    if (CONFIRM[action] && !(await this.confirm(CONFIRM[action], 'Continue'))) return;
    const answer = await backend(['command', action, '--model', model]);
    if (answer.error) { vscode.window.showErrorMessage('ForemanKit: ' + answer.error); return; }
    const term = terminal();
    term.show(true);
    term.sendText(answer.command);
    setTimeout(() => this.pushStatus(), 20000);
  }
}

function html() {
  const nonce = crypto.randomBytes(16).toString('hex');   // CSP nonces must be unpredictable, Math.random() is not
  const csp = "default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-" + nonce + "'";
  return PAGE.replace('__CSP__', csp).replace('__NONCE__', nonce);
}

const PAGE = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="__CSP__">
<style>
 body{font-family:var(--vscode-font-family);font-size:var(--vscode-font-size);color:var(--vscode-foreground);padding:8px}
 h3{margin:14px 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;opacity:.7}
 .row{display:flex;align-items:center;gap:6px;margin:4px 0;flex-wrap:wrap} .dot{width:9px;height:9px;border-radius:50%;background:#888}
 .dot.up{background:#3fb950} .dot.down{background:#f85149} .dot.wait{background:#d29922;animation:blink 1s infinite}
 @keyframes blink{50%{opacity:.3}}
 button{background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0;padding:5px 10px;cursor:pointer;border-radius:2px}
 button.secondary{background:var(--vscode-button-secondaryBackground);color:var(--vscode-button-secondaryForeground)}
 button:hover{filter:brightness(1.1)} select{width:100%;background:var(--vscode-dropdown-background);color:var(--vscode-dropdown-foreground);border:1px solid var(--vscode-dropdown-border);padding:4px}
 .bar{height:6px;background:rgba(128,128,128,.3);border-radius:3px}
 .limit{display:flex;align-items:center;gap:6px;margin:3px 0;font-size:12px} .limit span{min-width:150px}
 .limit input{width:80px!important;box-sizing:border-box;margin:0} .fill{height:6px;border-radius:3px;background:var(--vscode-progressBar-background)}
 .info,.err{font-size:12px;opacity:.8;margin:4px 0} .err{color:var(--vscode-errorForeground);opacity:1}
 .levels label{margin-right:8px;font-size:12px} .grid{display:grid;grid-template-columns:1fr;gap:4px;font-size:12px;margin:6px 0}
 .grid label{display:flex;flex-direction:column;gap:2px} input[type=number],input:not([type]),select{width:100%;box-sizing:border-box}
 input:not([type=checkbox]):not([type=radio]){background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border,transparent);padding:3px}
 .days{display:flex;align-items:flex-end;gap:3px;height:34px;margin:4px 0} .days div{flex:1;background:var(--vscode-progressBar-background);min-height:2px;border-radius:2px;opacity:.8}
 #delegClaude{font-style:italic}
 .call{font-size:11px;margin:2px 0;display:flex;gap:6px} .call .t{opacity:.6;min-width:34px} .call .bad{color:var(--vscode-errorForeground)}
 .code{font-size:20px;letter-spacing:.15em;font-weight:bold;margin:4px 0;user-select:all}
 .worker{display:flex;align-items:center;gap:6px;margin:3px 0;font-size:12px} .worker .x{margin-left:auto;cursor:pointer;opacity:.7} a{color:var(--vscode-textLink-foreground);cursor:pointer}
</style></head><body>
<div class="row head"><span class="dot" id="headDot"></span><b id="headText">ForemanKit</b></div>
<div id="liveLine" class="info" style="display:none"></div>
<div id="ctxWrap" style="display:none"><div class="bar"><div class="fill" id="ctxFill"></div></div><div class="info" id="ctxText"></div></div>
<div id="error" class="err"></div>
<div id="issues"></div>
<h3>Who orchestrates Claude Code</h3>
<div id="presetChoice"></div>
<div class="info" id="escalationRow" style="display:none"><b>Main loop tier</b><div id="escalationTiers"></div><span id="escalationNow"></span><br>auto = flash, moving to pro after repeated tool errors; a named tier is pinned until you switch back.</div>
<div class="info">Claude = your subscription, straight to Anthropic. External = the router is on; pick the provider in the Claude Code model picker (DeepSeek, Gemini, Kaggle, Local LLM) and set its tier with the switch above. External APIs need the key in .env; a folder above the provider's sensitivity ceiling is refused. The panel sets router, role and direct mode for you.</div>
<h3>Helpers (delegation)</h3>
<label class="info"><input type="checkbox" id="delegAll"> Delegation enabled (off = everything runs on the orchestrator)</label>
<details><summary class="info">Which helpers</summary>
<div class="info"><label><input type="checkbox" id="delegPaid"> paid clouds</label> <label><input type="checkbox" id="delegFree"> free clouds</label> <label><input type="checkbox" id="delegLocal"> local models</label></div>
</details>
<h3>Local model</h3>
<select id="models"></select><div class="info" id="modelInfo"></div>
<div class="row"><button id="startServer">Start server</button><button class="secondary" data-quick="stop-server">Stop</button></div>
<details><summary class="info" id="pricesSummary">Prices now</summary>
<div id="prices"></div>
</details>
<details><summary class="info" id="costSummary">Costs</summary>
<div id="budgetRows"></div>
<div id="delegSummary" class="info">no delegated calls yet</div>
<div id="delegDays" class="days"></div>
<div id="delegToday" class="info"></div>
<div id="delegClaude" class="info"></div>
<div id="delegRecent"></div>
</details>
<details><summary class="info">Advanced</summary>
<label class="info"><input type="checkbox" id="shareLab"> Share with the lab (-Share: other PCs can use this server; needs FOREMANKIT_PEER_KEY)</label>
<div class="row"><button class="secondary" id="tune">Tune model + max context</button></div>
<div class="info">Parameters of the selected model:</div>
<div class="grid">
<label>context <input id="p_ctx" type="number" min="2048" max="262144" step="1024"></label>
<label>experts on CPU (ncmoe) <input id="p_ncmoe" type="number" min="0" max="99"></label>
<label>threads <input id="p_threads" type="number" min="1" max="64"></label>
<label>KV cache <select id="p_kv_cache"><option value="">default</option><option>f16</option><option>q8_0</option><option>q4_0</option></select></label>
<label>thinking at start <select id="p_reasoning"><option value="">default</option><option>off</option><option>on</option></select></label>
<label>thinking budget <input id="p_think_budget" type="number" min="0" max="32768"></label>
<label>temperature <input id="p_temp" type="number" min="0" max="2" step="0.1"></label>
</div>
<div class="row"><button class="secondary" id="saveProfile">Save to profile</button></div>
<div class="info">Empty field = launcher default. Applies at the next server start. Tuning overwrites ctx, ncmoe and threads.</div>
<h3>Thinking by effort level</h3>
<div class="levels" id="levels"></div>
<div class="info">Ticked levels make the local model think; the others answer directly.</div>
<h3>Computers of the lab</h3>
<div id="workers"></div>
<div class="row"><button class="secondary" id="workerAddToggle">Add worker...</button></div>
<div id="workerForm" style="display:none"><div class="grid">
<label>name <input id="w_name" placeholder="garage"></label>
<label>URL <input id="w_url" placeholder="http://192.168.1.50:8080"></label>
<label>accepts data up to <select id="w_accept"><option>normal</option><option selected>internal</option><option>private</option></select></label>
<label>priority (lower = first) <input id="w_priority" type="number" value="10" min="1" max="99"></label></div>
<div class="row"><button id="workerSave">Save worker</button><button class="secondary" id="workerCancel">Cancel</button></div></div>
<div class="info">Local orchestrator runs on:</div>
<div id="orchChoice"></div>
<div class="info" style="margin-top:8px">Pair another PC of yours (it gets the lab key without copying files):</div>
<div id="fwLine" class="info"></div>
<div class="row"><button class="secondary" id="pairFirewall">Open firewall (Private + Public)</button></div>
<div class="row"><button class="secondary" id="pairHost">Show pairing code</button><button class="secondary" id="pairFind">Find computers</button></div>
<div id="pairBox" class="info"></div>
<div id="pairFound"></div>
<h3>Manual controls</h3>
<div class="info">Normally the choice above does this. Role of the local server:</div>
<div class="row"><label><input type="radio" name="role" value="orchestrator"> Orchestrator</label>
<label><input type="radio" name="role" value="worker"> Worker</label><label><input type="radio" name="role" value="both"> Both</label></div>
<div class="row"><button id="routerStart">Router on (reloads window)</button><button class="secondary" id="routerDirect">Router off, direct (reloads window)</button></div>
<div class="info"><a data-open="profiles">profiles.json</a> &middot; <a data-open="orchestrator">orchestrator.yaml</a> &middot; <a data-open="routerlog">router.log</a></div>
</details>
<div class="row"><button class="secondary" id="refresh">Refresh</button></div>
<script nonce="__NONCE__">
 const vscode = acquireVsCodeApi(); let models = []; let startingUntil = 0; let lastStatus = null; let headCost = ''; let busy = false; const LEVELS = ['low', 'medium', 'high', 'xhigh', 'max'];
 const $ = (id) => document.getElementById(id);
 const send = (m) => vscode.postMessage(m);
 const current = () => models.find((m) => m.file === $('models').value);
 function dot(id, up) { $(id).className = 'dot ' + (up ? 'up' : 'down'); }
 function showLevels(thinking) {
  if ($('levels').children.length === 0) {
   for (const level of LEVELS) { const l = document.createElement('label'); const c = document.createElement('input');
    c.type = 'checkbox'; c.id = 'lvl_' + level; c.onchange = sendLevels; l.appendChild(c); l.appendChild(document.createTextNode(' ' + level)); $('levels').appendChild(l); } }
  for (const level of LEVELS) if (document.activeElement !== $('lvl_' + level)) $('lvl_' + level).checked = !!thinking[level];
 }
 function sendLevels() { send({ type: 'thinking', levels: LEVELS.map((l) => l + '=' + $('lvl_' + l).checked).join(',') }); }
 function showLive(live) {
  const line = $('liveLine'); const g = live.gateway; const slot = live.slot;
  let text = '';
  if (g && g.fresh && g.active) {
   text = g.phase === 'reading prompt' ? 'reading the prompt... ' + Math.round(Date.now() / 1000 - g.since) + ' s'
    : g.phase + ' - ' + g.tps.toFixed(1) + ' tok/s (' + g.tokens + ' tokens)';
  } else if (slot && slot.busy) {
   const pct = slot.promptTokens ? Math.round(100 * slot.promptDone / slot.promptTokens) : 0;
   text = pct < 100 ? 'working on a delegated task: reading the prompt ' + pct + '%' : 'working on a delegated task';
  } else if (g && g.tokens > 0) {
   text = 'idle - last answer ' + g.tps.toFixed(1) + ' tok/s (' + g.tokens + ' tokens)';
  }
  line.style.display = text ? 'block' : 'none'; line.textContent = text;
 }
 const money = (v) => '$' + (v < 0.01 && v > 0 ? v.toFixed(4) : v.toFixed(2));
 function showDelegations(d) {
  if (d.error) return;
  $('delegSummary').textContent = 'today: ' + d.today.calls + ' delegated calls, ' + money(d.today.cost) + ' | last 7 days: ' + money(d.week_cost);
  const max = Math.max(...d.days.map((x) => x.cost), 0.000001); const bars = $('delegDays'); bars.innerHTML = '';
  for (const x of d.days) { const b = document.createElement('div'); b.style.height = Math.max(2, Math.round(34 * x.cost / max)) + 'px'; b.title = x.day + ': ' + money(x.cost); bars.appendChild(b); }
  const q = d.claude_equivalent;
  $('delegClaude').textContent = 'NOT spent - for comparison: the same tokens on Claude would cost Haiku 5.5 ' + money(q.today['haiku-5.5']) + ' today, ' + money(q.week['haiku-5.5']) + ' in 7 days; Sonnet 5.5 ' + money(q.today['sonnet-5.5']) + ' today, ' + money(q.week['sonnet-5.5']) + ' in 7 days (list prices, no cache discount).';
  const targets = Object.entries(d.today.by_target).sort((a, b) => b[1].cost - a[1].cost);
  $('delegToday').textContent = targets.length ? 'today by target: ' + targets.map(([n, v]) => n + ' ' + v.calls + 'x ' + money(v.cost)).join('; ') : '';
  const box = $('delegRecent'); box.innerHTML = '';
  for (const c of d.recent) {
   const row = document.createElement('div'); row.className = 'call';
   const t = document.createElement('span'); t.className = 't'; const when = new Date(c.ts * 1000); t.textContent = when.toTimeString().slice(0, 5); row.appendChild(t);
   const text = document.createElement('span'); if (!c.ok) text.className = 'bad';
   text.textContent = c.kind + ' -> ' + c.target + (c.escalated ? ' (escalated)' : '') + ' | ' + c.tokens + ' tok, ' + money(c.cost) + (c.seconds ? ', ' + c.seconds + ' s' : '') + (c.ok ? '' : ' FAILED');
   row.appendChild(text); box.appendChild(row);
  }
 }
 function showPairState(s) {
  const box = $('pairBox'); box.innerHTML = '';
  if (!s || s.state === 'none') return;
  const note = document.createElement('div');
  if (s.state === 'waiting') {
   const code = document.createElement('div'); code.className = 'code'; code.textContent = s.code; box.appendChild(code);
   note.textContent = 'Enter this code on the other computer (Find computers, then Pair). Open for ' + Math.max(0, Math.round(s.expires - Date.now() / 1000)) + ' s.';
  } else if (s.state === 'paired') {
   note.textContent = 'Paired with ' + (s.peer ? s.peer.name + ' (' + s.peer.ip + ')' : '') + ' - added to the workers list.';
  } else {
   note.className = 'err'; note.textContent = s.state === 'expired' ? 'Session expired. Show the code again.' : (s.detail || 'Pairing failed.') ;
  }
  box.appendChild(note);
 }
 function showFound(list) {
  const box = $('pairFound'); box.innerHTML = '';
  if (list.error) { box.textContent = list.error; return; }
  if (list.length === 0) { box.className = 'info'; box.textContent = 'No computer is showing a pairing code. Press "Show pairing code" on the PC that has the lab key, then search again.'; return; }
  for (const host of list) {
   const row = document.createElement('div'); row.className = 'worker';
   const label = document.createElement('span'); label.textContent = host.name + ' (' + host.ip + ')'; row.appendChild(label);
   const input = document.createElement('input'); input.placeholder = 'code'; input.style.width = '90px'; row.appendChild(input);
   const go = document.createElement('button'); go.textContent = 'Pair'; row.appendChild(go);
   go.onclick = () => send({ type: 'pair', action: 'join', ip: host.ip, port: host.port, code: input.value });
   box.appendChild(row);
  }
 }
 function showDelegationSwitches(d) {
  if (!d) return;
  const any = d.paid || d.free || d.local;
  for (const [id, key] of [['delegPaid', 'paid'], ['delegFree', 'free'], ['delegLocal', 'local']]) if (document.activeElement !== $(id)) $(id).checked = d[key];
  if (document.activeElement !== $('delegAll')) $('delegAll').checked = any;
 }
 function showFirewall(f) {
  const open = f && f.open;
  $('fwLine').textContent = f && f.open === null ? '' : open ? 'Firewall: open for pairing on Private and Public networks (this subnet only).'
   : 'Firewall: not open yet. Other computers cannot reach pairing until you open it (Windows asks for administrator consent once).';
  $('pairFirewall').style.display = f && f.open === null ? 'none' : (open ? 'none' : 'inline-block');
 }
 function showWaiting() { renderHead(); }
 function showStatus(s) {
  $('error').textContent = s.error || '';
  if (s.error) return;
  lastStatus = s; if (s.server.up) startingUntil = 0;
  showLevels(s.thinking); showPrices(s.prices); showBudget(s.budget); showPresets(s.preset); showEscalation(s); showIssues(s.preset.issues); showRole(s.role); showDelegationSwitches(s.delegation); renderHead();
  $('ctxWrap').style.display = s.context ? 'block' : 'none';
  if (s.context) { const pct = Math.min(100, Math.round(100 * s.context.tokens / s.context.window));
   $('ctxFill').style.width = pct + '%'; $('ctxText').textContent = 'context ' + pct + '% (' + s.context.tokens + ' / ' + s.context.window + ')'; }
 }
 function showModels(list) {
  if (list.error) { $('error').textContent = list.error; return; }
  models = list; const keep = $('models').value; $('models').innerHTML = '';
  for (const m of list) { const o = document.createElement('option'); o.value = m.file; o.textContent = m.file; $('models').appendChild(o); }
  if (keep) $('models').value = keep; showInfo();
 }
 function showInfo() {
  const m = current(); if (!m) return;
  fillAdvanced(m);
  $('modelInfo').textContent = m.size_gb + ' GB, ' + (m.tuned ? m.tg_tok_s + ' tok/s, ctx ' + m.ctx
   + (m.reasoning === 'off' ? ', thinking off' : '') : 'not tuned yet (use Tune model)');
 }

 const PARAMS = ['ctx', 'ncmoe', 'threads', 'kv_cache', 'reasoning', 'think_budget', 'temp'];
 function fillAdvanced(m) { for (const name of PARAMS) $('p_' + name).value = m[name] == null ? '' : m[name]; }
 function showRole(role) { for (const r of document.querySelectorAll('input[name=role]')) if (document.activeElement !== r) r.checked = r.value === role; }
 function showWorkers(list) {
  if (list.error) return; const box = $('workers'); box.innerHTML = ''; showOrchestratorChoice(list);
  if (list.length === 0) { box.textContent = 'none configured'; box.className = 'info'; return; }
  for (const w of list) {
   const row = document.createElement('div'); row.className = 'worker';
   const d = document.createElement('span'); d.className = 'dot ' + (w.up ? 'up' : 'down'); row.appendChild(d);
   const t = document.createElement('span'); t.textContent = w.name + ' ' + w.url + (w.up ? (w.model ? ' ' + w.model : '') : ' (down)') + ' [' + w.accept + ']'; row.appendChild(t);
   const on = document.createElement('input'); on.type = 'checkbox'; on.checked = w.enabled; on.title = 'router may send tasks here';
   on.onchange = () => send({ type: 'config', action: 'workerEnable', name: w.name, enabled: on.checked }); row.insertBefore(on, row.firstChild);
   const ts = document.createElement('button'); ts.className = 'secondary'; ts.textContent = 'Test';
   ts.onclick = () => send({ type: 'config', action: 'workerTest', name: w.name }); row.appendChild(ts);
   if (w.editable) {
    const ed = document.createElement('button'); ed.className = 'secondary'; ed.textContent = 'Edit';
    ed.onclick = () => { $('w_name').value = w.name; $('w_url').value = w.url; $('w_accept').value = w.accept; $('w_priority').value = w.priority || 10; $('workerForm').style.display = 'block'; };
    const rm = document.createElement('button'); rm.className = 'secondary'; rm.textContent = 'Remove';
    rm.onclick = () => send({ type: 'config', action: 'workerRemove', name: w.name }); row.appendChild(ed); row.appendChild(rm);
   } else { const f = document.createElement('span'); f.className = 'info'; f.textContent = 'nodes.yaml'; row.appendChild(f); }
   box.appendChild(row);
  }
 }
 function showPresets(info) {
  const box = $('presetChoice'); if (!info || box.contains(document.activeElement)) return; box.innerHTML = '';
  for (const name of info.names.concat(info.current === 'custom' ? ['custom'] : [])) {
   const l = document.createElement('label'); l.className = 'info'; l.style.marginRight = '10px';
   const r = document.createElement('input'); r.type = 'radio'; r.name = 'preset'; r.checked = name === info.current; r.disabled = busy || name === 'custom';
   r.onchange = () => send({ type: 'preset', name });
   l.appendChild(r); l.appendChild(document.createTextNode(' ' + (name === 'custom' ? 'custom (set by hand below)' : name))); box.appendChild(l);
  }
 }
 function showEscalation(s) {
  const prices = s.prices || {};
  const mode = (s.preset.state || {}).mode || '';
  $('escalationRow').style.display = mode === 'claude' || mode === 'local' ? 'none' : 'block';
  const box = $('escalationTiers');
  const providers = prices.escalation || [];
  const key = providers.map((p) => p.provider + ':' + p.control).join(',');
  if (box.dataset.key !== key) {
   box.innerHTML = '';
   for (const p of providers) {
    const label = document.createElement('label'); label.className = 'info'; label.style.marginRight = '10px';
    label.appendChild(document.createTextNode(p.provider + ' '));
    const sel = document.createElement('select');
    sel.dataset.provider = p.provider;
    for (const m of p.modes || []) { const o = document.createElement('option'); o.value = m.name; o.textContent = m.label; sel.appendChild(o); }
    sel.value = p.control || 'auto';
    sel.onchange = () => send({ type: 'config', action: 'escalationSet', name: sel.dataset.provider, model: sel.value });
    label.appendChild(sel); box.appendChild(label);
   }
   box.dataset.key = key;
  }
  const t = prices.tier;
  $('escalationNow').textContent = t ? 'now: ' + t.alias + ' (' + t.tier + ')' : 'no request yet';
 }
 function showIssues(list) {
  const box = $('issues'); box.innerHTML = '';
  for (const issue of list || []) {
   const row = document.createElement('div'); row.className = 'err'; row.textContent = issue.text + ' ';
   if (issue.fix) { const b = document.createElement('button'); b.textContent = 'Fix'; b.onclick = () => runFix(issue.fix); row.appendChild(b); }
   box.appendChild(row);
  }
 }
 function runFix(fix) {
  if (fix.startsWith('apply-preset ')) send({ type: 'preset', name: fix.slice('apply-preset '.length) });
  else if (fix.startsWith('fix-model ')) { const [, target, id] = fix.split(' '); send({ type: 'config', action: 'fixModel', name: target, model: id }); }
  else if (fix === 'start-server') $('startServer').click();
  else if (fix === 'download-llama') send({ type: 'terminal', action: 'download-llama', model: '' });
 }
 const perM = (p) => '$' + p.input_miss + ' in / $' + p.input_hit + ' cached / $' + p.output + ' out per 1M';
 function untilText(ts) {
  const hours = Math.max(0, Math.round((ts * 1000 - Date.now()) / 3600000));
  return new Date(ts * 1000).toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' }) + ' (in ' + hours + ' h)';
 }
 const money2 = (v) => '$' + (v >= 1 ? v.toFixed(2) : v.toFixed(3));
 function capText(cap) { return cap === null || cap === undefined ? 'no cap' : money2(cap); }
 function pct(spent, cap) { return cap === null || cap === undefined || cap <= 0 ? 0 : Math.min(100, Math.round(100 * spent / cap)); }
 function limitRow(category, label, period, spent, cap) {
  const row = document.createElement('div'); row.className = 'limit';
  const text = document.createElement('span');
  const p = pct(spent, cap);
  text.textContent = (period === 'daily' ? 'daily' : 'monthly') + ': ' + money2(spent) + ' / ' + capText(cap) + (cap === null ? '' : '  ' + p + '%');
  if (p >= 90) text.className = 'err';
  row.appendChild(text);
  const bar = document.createElement('div'); bar.className = 'bar'; bar.style.flex = '1';
  const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = p + '%';
  if (p >= 90) fill.style.background = 'var(--vscode-errorForeground)';
  bar.appendChild(fill); row.appendChild(bar);
  const input = document.createElement('input'); input.type = 'number'; input.min = '0'; input.step = '0.25';
  input.value = cap === null || cap === undefined ? '' : cap; input.placeholder = 'no cap';
  input.title = 'Limit for ' + label + ' (' + period + '); leave empty and press Enter for no cap';
  row.appendChild(input);
  input.onchange = () => send({ type: 'config', action: 'budgetSet', category, period, value: input.value });
  return row;
 }
 function budgetLine(category, label, row) {
  const box = document.createElement('div');
  const head = document.createElement('div'); head.className = 'info';
  head.innerHTML = '<b>' + label + '</b> &mdash; type a limit and press Enter (empty = no cap)';
  box.appendChild(head);
  box.appendChild(limitRow(category, label, 'daily', row.day_usd, row.daily_cap));
  box.appendChild(limitRow(category, label, 'monthly', row.month_usd, row.monthly_cap));
  return box;
 }
 function showBudget(b) {
  const box = $('budgetRows'); if (!b) return; box.innerHTML = '';
  box.appendChild(budgetLine('orchestrator', 'Orchestrator', b.orchestrator || {}));
  box.appendChild(budgetLine('delegation', 'Agents (delegation)', b.delegation || {}));
  const short = (r) => money2(r.day_usd) + '/' + capText(r.daily_cap);
  $('costSummary').textContent = 'Costs (' + short(b.orchestrator || {}) + ' now)';
  headCost = 'orch ' + short(b.orchestrator || {}) + ' | agents ' + short(b.delegation || {}); renderHead();
 }
 function showPrices(info) {
  const box = $('prices'); if (!info) return; box.innerHTML = '';
  const add = (text, cls) => { const d = document.createElement('div'); d.className = cls || 'info'; d.textContent = text; box.appendChild(d); };
  if (info.tier) add('orchestrator now: ' + info.tier.mode + ' / ' + info.tier.alias + ' (' + info.tier.tier + ' tier, switch ' + (info.control || 'auto') + ')');
  let cheap = '';
  for (const p of info.providers) {
   if (p.phase) {
    cheap = cheap || (p.phase === 'off-peak' ? 'cheap' : 'expensive');
    add(p.provider + ': ' + (p.phase === 'off-peak' ? 'OFF-PEAK = CHEAP (half price)' : 'PEAK = EXPENSIVE') + ', changes ' + (p.next_change ? untilText(p.next_change) : 'never'));
   } else add(p.provider + ': flat price, no peak hours');
   for (const [alias, m] of Object.entries(p.models)) {
    add('  ' + alias + ' ' + m.id + ': ' + perM(m.price) + (m.phase === 'off-peak' ? ' (peak: $' + m.peak_price.input_miss + ' / $' + m.peak_price.input_hit + ' / $' + m.peak_price.output + ')' : ''));
    if (m.upcoming) add('  ! ' + alias + ' from ' + m.upcoming.from + ': ' + perM(m.upcoming.price));
   }
  }
  $('pricesSummary').textContent = 'Prices now' + (cheap ? ' (' + cheap + ')' : '');
 }
 function renderHead() {
  const s = lastStatus; if (!s) return;
  const server = s.server.up ? s.server.model + ' ctx ' + s.server.n_ctx : (Date.now() < startingUntil || s.server.starting ? 'server starting...' : 'server off');
  const bad = (s.preset.issues || []).length > 0;
  $('headDot').className = 'dot ' + (bad ? 'down' : 'up');
  $('headText').textContent = s.preset.current + ' | router ' + (s.router.up ? 'on' : 'off') + ' | ' + server + (headCost ? ' | ' + headCost : '');
 }
 function showOrchestratorChoice(list) {
  const box = $('orchChoice'); box.innerHTML = '';
  const options = [{ name: 'this', label: 'This PC', checked: !list.some((w) => w.orchestrator) }]
   .concat(list.map((w) => ({ name: w.name, label: w.name + (w.up ? '' : ' (down)'), checked: w.orchestrator })));
  for (const o of options) {
   const l = document.createElement('label'); l.className = 'info'; l.style.marginRight = '10px';
   const r = document.createElement('input'); r.type = 'radio'; r.name = 'orch'; r.checked = o.checked;
   r.onchange = () => send({ type: 'config', action: 'orchestrator', name: o.name });
   l.appendChild(r); l.appendChild(document.createTextNode(' ' + o.label)); box.appendChild(l);
  }
 }
 $('pairFirewall').onclick = () => send({ type: 'pair', action: 'firewall' });
 $('pairHost').onclick = () => send({ type: 'pair', action: 'host' });
 $('pairFind').onclick = () => { $('pairFound').textContent = 'searching...'; send({ type: 'pair', action: 'find' }); };
 $('delegAll').onchange = () => send({ type: 'config', action: 'delegationSet', name: 'all', enabled: $('delegAll').checked });
 for (const [id, key] of [['delegPaid', 'paid'], ['delegFree', 'free'], ['delegLocal', 'local']]) $(id).onchange = () => send({ type: 'config', action: 'delegationSet', name: key, enabled: $(id).checked });
 $('saveProfile').onclick = () => { const fields = {}; for (const n of PARAMS) fields[n] = $('p_' + n).value;
  send({ type: 'config', action: 'profileSet', model: $('models').value, fields: JSON.stringify(fields) }); };
 for (const r of document.querySelectorAll('input[name=role]')) r.onchange = () => send({ type: 'config', action: 'role', role: r.value });
 $('workerAddToggle').onclick = () => { $('workerForm').style.display = 'block'; };
 $('workerCancel').onclick = () => { $('workerForm').style.display = 'none'; };
 $('workerSave').onclick = () => { send({ type: 'config', action: 'workerAdd', name: $('w_name').value.trim(), url: $('w_url').value.trim(),
  accept: $('w_accept').value, priority: Number($('w_priority').value) || 10 }); $('workerForm').style.display = 'none'; };
 $('models').onchange = showInfo;
 $('startServer').onclick = () => { startingUntil = Date.now() + 180000; showWaiting();
  send({ type: 'quick', action: 'start-server', model: $('models').value, share: $('shareLab').checked }); };
 $('tune').onclick = () => send({ type: 'terminal', action: 'tune', model: $('models').value });
 $('routerStart').onclick = () => send({ type: 'quick', action: 'router-normal' });
 $('routerDirect').onclick = () => send({ type: 'quick', action: 'router-direct' });
 $('refresh').onclick = () => send({ type: 'refresh' });
 document.querySelectorAll('[data-quick]').forEach((b) => b.onclick = () => send({ type: 'quick', action: b.dataset.quick }));
 document.querySelectorAll('[data-open]').forEach((a) => a.onclick = () => send({ type: 'open', file: a.dataset.open }));
 window.addEventListener('message', (e) => { if (e.data.type === 'status') showStatus(e.data.data); if (e.data.type === 'busy') { busy = e.data.value; if (lastStatus) showPresets(lastStatus.preset); } if (e.data.type === 'models') showModels(e.data.data); if (e.data.type === 'workers') showWorkers(e.data.data); if (e.data.type === 'live') showLive(e.data.data); if (e.data.type === 'delegations') showDelegations(e.data.data);
  if (e.data.type === 'firewall') showFirewall(e.data.data); if (e.data.type === 'pairState') showPairState(e.data.data); if (e.data.type === 'pairFound') showFound(e.data.data); });
</script></body></html>`;

function activate(context) {
  const panel = new Panel();
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('foremankit.panel', panel),
    vscode.commands.registerCommand('foremankit.refresh', () => panel.handle({ type: 'refresh' })));
}

module.exports = { activate };
