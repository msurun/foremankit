// ForemanKit Control: a sidebar panel (activity bar button) for the router, the local llama-server and model tuning.
// A thin shell: all logic lives in the Python module foremankit.panel; this file runs it, shows its JSON and opens terminals.
const vscode = require('vscode');
const cp = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.resolve(fs.realpathSync(__dirname), '..');
const PYTHON = process.platform === 'win32'
  ? path.join(ROOT, '.venv', 'Scripts', 'python.exe') : path.join(ROOT, '.venv', 'bin', 'python');
const POLL_MS = 4000;
const WORKERS_POLL_MS = 12000;
const RELOAD = 'Reload Window';
const CONFIRM = {
  tune: 'Tuning stops the running llama-server, then tunes the offload and finds the largest fast context. '
    + 'It takes about 25 minutes: keep the laptop on AC power.',
};
const CONFIRM_QUICK = {
  'router-direct': 'Direct mode stops the router and removes it from the VS Code settings, so Claude Code talks straight to '
    + 'Anthropic and the local models stop working. You must RELOAD THE VS CODE WINDOW afterwards (Claude Code reads the '
    + 'setting only when a window starts). Until you reload, Claude Code has no connection at all.',
  'router-normal': 'Router mode starts the router and puts it back into the VS Code settings. You must RELOAD THE VS CODE '
    + 'WINDOW afterwards.',
};
const NEEDS_RELOAD = ['router-direct', 'router-normal'];
const FILES = {
  profiles: path.join(ROOT, 'llama', 'profiles.json'),
  orchestrator: path.join(ROOT, 'config', 'orchestrator.yaml'),
  routerlog: path.join(os.tmpdir(), 'router.log'),
};

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
    view.onDidDispose(() => { clearInterval(this.timer); clearInterval(this.workersTimer); });
    this.pushStatus();
    this.pushWorkers();
    this.pushModels();
  }

  post(message) { if (this.view) this.view.webview.postMessage(message); }
  async pushStatus() { if (this.view && this.view.visible) this.post({ type: 'status', data: await backend(['status']) }); }
  async pushWorkers() { if (this.view && this.view.visible) this.post({ type: 'workers', data: await backend(['workers']) }); }
  async pushModels() { this.post({ type: 'models', data: await backend(['models']) }); }

  async handle(message) {
    if (message.type === 'refresh') { await this.pushStatus(); await this.pushModels(); await this.pushWorkers(); return; }
    if (message.type === 'config') { await this.runConfig(message); return; }
    if (message.type === 'open' && FILES[message.file]) { await vscode.window.showTextDocument(vscode.Uri.file(FILES[message.file])); return; }
    if (message.type === 'thinking') { await this.quickAnswer(['thinking', '--levels', message.levels]); return; }
    if (message.type === 'quick') await this.runQuick(message.action);
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

  async runQuick(action) {
    if (CONFIRM_QUICK[action] && !(await this.confirm(CONFIRM_QUICK[action], 'Continue'))) return;
    const answer = await this.quickAnswer([action]);
    const text = 'ForemanKit: ' + (answer.message || answer.error);
    if (NEEDS_RELOAD.includes(action) && !answer.error) {
      const pick = await vscode.window.showInformationMessage(text, RELOAD);
      if (pick === RELOAD) vscode.commands.executeCommand('workbench.action.reloadWindow');
    } else {
      vscode.window.setStatusBarMessage(text, 4000);
    }
  }

  configArgs({ action, role, name, url, accept, priority, model, fields }) {
    const table = {
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
    vscode.window.setStatusBarMessage('ForemanKit: ' + (answer.message || answer.error), 6000);
    await this.pushStatus();
    await this.pushWorkers();
    await this.pushModels();
  }

  async runLong({ action, model }) {
    if (!model) return;
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
  const nonce = Math.random().toString(36).slice(2);
  const csp = "default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-" + nonce + "'";
  return PAGE.replace('__CSP__', csp).replace('__NONCE__', nonce);
}

const PAGE = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="__CSP__">
<style>
 body{font-family:var(--vscode-font-family);font-size:var(--vscode-font-size);color:var(--vscode-foreground);padding:8px}
 h3{margin:14px 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;opacity:.7}
 .row{display:flex;align-items:center;gap:6px;margin:4px 0;flex-wrap:wrap} .dot{width:9px;height:9px;border-radius:50%;background:#888}
 .dot.up{background:#3fb950} .dot.down{background:#f85149}
 button{background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0;padding:5px 10px;cursor:pointer;border-radius:2px}
 button.secondary{background:var(--vscode-button-secondaryBackground);color:var(--vscode-button-secondaryForeground)}
 button:hover{filter:brightness(1.1)} select{width:100%;background:var(--vscode-dropdown-background);color:var(--vscode-dropdown-foreground);border:1px solid var(--vscode-dropdown-border);padding:4px}
 .bar{height:6px;background:rgba(128,128,128,.3);border-radius:3px} .fill{height:6px;border-radius:3px;background:var(--vscode-progressBar-background)}
 .info,.err{font-size:12px;opacity:.8;margin:4px 0} .err{color:var(--vscode-errorForeground);opacity:1}
 .levels label{margin-right:8px;font-size:12px} .grid{display:grid;grid-template-columns:1fr;gap:4px;font-size:12px;margin:6px 0}
 .grid label{display:flex;flex-direction:column;gap:2px} input[type=number],input:not([type]),select{width:100%;box-sizing:border-box}
 input:not([type=checkbox]):not([type=radio]){background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border,transparent);padding:3px}
 .worker{display:flex;align-items:center;gap:6px;margin:3px 0;font-size:12px} .worker .x{margin-left:auto;cursor:pointer;opacity:.7} a{color:var(--vscode-textLink-foreground);cursor:pointer}
</style></head><body>
<h3>Status</h3>
<div class="row"><span class="dot" id="routerDot"></span><span id="routerText">router</span></div>
<div class="row"><span class="dot" id="serverDot"></span><span id="serverText">llama-server</span></div>
<div id="ctxWrap" style="display:none"><div class="bar"><div class="fill" id="ctxFill"></div></div><div class="info" id="ctxText"></div></div>
<div id="error" class="err"></div>
<div id="routerWarn" class="err" style="display:none">Router stopped: Claude Code in VS Code cannot reach any model, also not Anthropic. Start it, or use direct mode and reload the window.</div>
<div id="directInfo" class="info" style="display:none">Direct mode: Claude Code talks straight to Anthropic; local models are unavailable.</div>
<h3>Router</h3>
<div class="row"><button id="routerStart">Start router</button><button class="secondary" id="routerDirect">Stop (direct mode)...</button></div>
<label class="info"><input type="checkbox" id="localOn"> Local models enabled (off = cloud only, router keeps running)</label>
<div class="info">Stop and start change the VS Code settings: reload the window afterwards.</div>
<h3>Local model</h3>
<select id="models"></select><div class="info" id="modelInfo"></div>
<div class="row"><button id="startServer">Start server</button><button class="secondary" data-quick="stop-server">Stop</button></div>
<div class="info">Role of this server:</div>
<div class="row"><label><input type="radio" name="role" value="orchestrator"> Orchestrator</label>
<label><input type="radio" name="role" value="worker"> Worker</label><label><input type="radio" name="role" value="both"> Both</label></div>
<div class="info">Orchestrator answers Claude Code only; worker takes delegated tasks only.</div>
<div class="row"><button class="secondary" id="tune">Tune model + max context</button></div>
<details><summary class="info">Advanced parameters of the selected model</summary>
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
</details>
<h3>Extra workers</h3>
<div id="workers"></div>
<div class="row"><button class="secondary" id="workerAddToggle">Add worker...</button></div>
<div id="workerForm" style="display:none"><div class="grid">
<label>name <input id="w_name" placeholder="garage"></label>
<label>URL <input id="w_url" placeholder="http://192.168.1.50:8080"></label>
<label>accepts data up to <select id="w_accept"><option>normal</option><option selected>internal</option><option>private</option></select></label>
<label>priority (lower = first) <input id="w_priority" type="number" value="10" min="1" max="99"></label></div>
<div class="row"><button id="workerSave">Save worker</button><button class="secondary" id="workerCancel">Cancel</button></div></div>
<h3>Thinking by effort level</h3>
<div class="levels" id="levels"></div>
<div class="info">Ticked levels make the local model think; the others answer directly.</div>
<h3>Files</h3>
<div class="info"><a data-open="profiles">profiles.json</a> &middot; <a data-open="orchestrator">orchestrator.yaml</a> &middot; <a data-open="routerlog">router.log</a></div>
<div class="row"><button class="secondary" id="refresh">Refresh</button></div>
<script nonce="__NONCE__">
 const vscode = acquireVsCodeApi(); let models = []; const LEVELS = ['low', 'medium', 'high', 'xhigh', 'max'];
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
 function showStatus(s) {
  $('error').textContent = s.error || '';
  if (s.error) return;
  $('routerWarn').style.display = !s.router.up && !s.direct_mode ? 'block' : 'none';
  $('directInfo').style.display = s.direct_mode ? 'block' : 'none';
  $('localOn').checked = s.local_enabled; showLevels(s.thinking); showRole(s.role);
  dot('routerDot', s.router.up); $('routerText').textContent = 'router :' + s.router.port + (s.router.up ? ' running' : ' stopped');
  dot('serverDot', s.server.up);
  $('serverText').textContent = s.server.up ? s.server.model + ' (ctx ' + s.server.n_ctx + ')' : 'llama-server stopped';
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
  if (list.error) return; const box = $('workers'); box.innerHTML = '';
  if (list.length === 0) { box.textContent = 'none configured'; box.className = 'info'; return; }
  for (const w of list) {
   const row = document.createElement('div'); row.className = 'worker';
   const d = document.createElement('span'); d.className = 'dot ' + (w.up ? 'up' : 'down'); row.appendChild(d);
   const t = document.createElement('span'); t.textContent = w.name + ' ' + w.url + (w.up ? (w.model ? ' ' + w.model : '') : ' (down)') + ' [' + w.accept + ']'; row.appendChild(t);
   if (w.editable) { const x = document.createElement('span'); x.className = 'x'; x.textContent = 'remove';
    x.onclick = () => send({ type: 'config', action: 'workerRemove', name: w.name }); row.appendChild(x); }
   box.appendChild(row);
  }
 }
 $('saveProfile').onclick = () => { const fields = {}; for (const n of PARAMS) fields[n] = $('p_' + n).value;
  send({ type: 'config', action: 'profileSet', model: $('models').value, fields: JSON.stringify(fields) }); };
 for (const r of document.querySelectorAll('input[name=role]')) r.onchange = () => send({ type: 'config', action: 'role', role: r.value });
 $('workerAddToggle').onclick = () => { $('workerForm').style.display = 'block'; };
 $('workerCancel').onclick = () => { $('workerForm').style.display = 'none'; };
 $('workerSave').onclick = () => { send({ type: 'config', action: 'workerAdd', name: $('w_name').value.trim(), url: $('w_url').value.trim(),
  accept: $('w_accept').value, priority: Number($('w_priority').value) || 10 }); $('workerForm').style.display = 'none'; };
 $('models').onchange = showInfo;
 $('startServer').onclick = () => send({ type: 'terminal', action: 'start-server', model: $('models').value });
 $('tune').onclick = () => send({ type: 'terminal', action: 'tune', model: $('models').value });
 $('routerStart').onclick = () => send({ type: 'quick', action: 'router-normal' });
 $('routerDirect').onclick = () => send({ type: 'quick', action: 'router-direct' });
 $('localOn').onchange = () => send({ type: 'quick', action: $('localOn').checked ? 'local-on' : 'local-off' });
 $('refresh').onclick = () => send({ type: 'refresh' });
 document.querySelectorAll('[data-quick]').forEach((b) => b.onclick = () => send({ type: 'quick', action: b.dataset.quick }));
 document.querySelectorAll('[data-open]').forEach((a) => a.onclick = () => send({ type: 'open', file: a.dataset.open }));
 window.addEventListener('message', (e) => { if (e.data.type === 'status') showStatus(e.data.data); if (e.data.type === 'models') showModels(e.data.data); if (e.data.type === 'workers') showWorkers(e.data.data); });
</script></body></html>`;

function activate(context) {
  const panel = new Panel();
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('foremankit.panel', panel),
    vscode.commands.registerCommand('foremankit.refresh', () => panel.handle({ type: 'refresh' })));
}

module.exports = { activate };
