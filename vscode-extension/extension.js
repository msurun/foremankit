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
const NEEDS_NO_MODEL = ['install', 'uninstall'];
const CONFIRM = {
  tune: 'Tuning stops the running llama-server and takes about 15 minutes (keep the laptop on AC power).',
  sweep: 'The context sweep stops the running llama-server and takes about 10 minutes.',
  uninstall: 'Uninstall removes the ForemanKit agents, hooks and VS Code entries.',
};
const CONFIRM_QUICK = {
  'stop-router': 'Stopping the router cuts Claude Code off from Anthropic (it talks to the router) until you start it again. '
    + 'To switch off only the local models use "Local models" instead.',
};
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
    view.onDidDispose(() => clearInterval(this.timer));
    this.pushStatus();
    this.pushModels();
  }

  post(message) { if (this.view) this.view.webview.postMessage(message); }
  async pushStatus() { if (this.view && this.view.visible) this.post({ type: 'status', data: await backend(['status']) }); }
  async pushModels() { this.post({ type: 'models', data: await backend(['models']) }); }

  async handle(message) {
    if (message.type === 'refresh') { await this.pushStatus(); await this.pushModels(); return; }
    if (message.type === 'open' && FILES[message.file]) { await vscode.window.showTextDocument(vscode.Uri.file(FILES[message.file])); return; }
    if (message.type === 'quick') {
      if (CONFIRM_QUICK[message.action]) {
        const ok = await vscode.window.showWarningMessage(CONFIRM_QUICK[message.action], { modal: true }, 'Stop anyway');
        if (ok !== 'Stop anyway') return;
      }
      const answer = await backend([message.action]);
      vscode.window.setStatusBarMessage('ForemanKit: ' + (answer.message || answer.error), 4000);
      setTimeout(() => this.pushStatus(), 1500);
      return;
    }
    if (message.type === 'terminal') await this.runLong(message);
  }

  async runLong({ action, model, dense }) {
    if (!model && !NEEDS_NO_MODEL.includes(action)) return;
    if (CONFIRM[action]) {
      const ok = await vscode.window.showWarningMessage(CONFIRM[action], { modal: true }, 'Continue');
      if (ok !== 'Continue') return;
    }
    const args = ['command', action, '--model', model || ''].concat(dense ? ['--dense'] : []);
    const answer = await backend(args);
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
 .row{display:flex;align-items:center;gap:6px;margin:4px 0} .dot{width:9px;height:9px;border-radius:50%;background:#888}
 .dot.up{background:#3fb950} .dot.down{background:#f85149}
 button{background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0;padding:5px 10px;cursor:pointer;border-radius:2px}
 button.secondary{background:var(--vscode-button-secondaryBackground);color:var(--vscode-button-secondaryForeground)}
 button:hover{filter:brightness(1.1)} select{width:100%;background:var(--vscode-dropdown-background);color:var(--vscode-dropdown-foreground);border:1px solid var(--vscode-dropdown-border);padding:4px}
 .bar{height:6px;background:rgba(128,128,128,.3);border-radius:3px} .fill{height:6px;border-radius:3px;background:var(--vscode-progressBar-background)}
 .info,.err{font-size:12px;opacity:.8;margin:4px 0} .err{color:var(--vscode-errorForeground);opacity:1}
 a{color:var(--vscode-textLink-foreground);cursor:pointer}
</style></head><body>
<h3>Status</h3>
<div class="row"><span class="dot" id="routerDot"></span><span id="routerText">router</span></div>
<div class="row"><span class="dot" id="serverDot"></span><span id="serverText">llama-server</span></div>
<div id="ctxWrap" style="display:none"><div class="bar"><div class="fill" id="ctxFill"></div></div><div class="info" id="ctxText"></div></div>
<div id="error" class="err"></div>
<div id="routerWarn" class="err" style="display:none">Router stopped: Claude Code in VS Code cannot reach any model, also not Anthropic. Start it.</div>
<h3>Router (gateway)</h3>
<div class="row"><button data-quick="start-router">Start</button><button class="secondary" data-quick="stop-router">Stop...</button></div>
<label class="info"><input type="checkbox" id="localOn"> Local models enabled (off = cloud only, the router keeps running)</label>
<h3>Local model</h3>
<select id="models"></select><div class="info" id="modelInfo"></div>
<div class="row"><button id="startServer">Start server</button><button class="secondary" data-quick="stop-server">Stop</button></div>
<h3>Tuning</h3>
<div class="row"><button class="secondary" id="tune">Tune model</button><button class="secondary" id="sweep">Max context</button></div>
<label class="info"><input type="checkbox" id="dense"> dense model (no experts)</label>
<h3>Install</h3>
<div class="row"><button class="secondary" id="install">Install</button><button class="secondary" id="uninstall">Uninstall</button></div>
<h3>Files</h3>
<div class="info"><a data-open="profiles">profiles.json</a> &middot; <a data-open="orchestrator">orchestrator.yaml</a> &middot; <a data-open="routerlog">router.log</a></div>
<div class="row"><button class="secondary" id="refresh">Refresh</button></div>
<script nonce="__NONCE__">
 const vscode = acquireVsCodeApi(); let models = [];
 const $ = (id) => document.getElementById(id);
 const send = (m) => vscode.postMessage(m);
 const current = () => models.find((m) => m.file === $('models').value);
 function dot(id, up) { $(id).className = 'dot ' + (up ? 'up' : 'down'); }
 function showStatus(s) {
  $('error').textContent = s.error || '';
  if (s.error) return;
  $('routerWarn').style.display = s.router.up ? 'none' : 'block';
  $('localOn').checked = s.local_enabled;
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
  $('dense').checked = !m.moe;
  $('modelInfo').textContent = m.size_gb + ' GB, ' + (m.tuned ? m.tg_tok_s + ' tok/s, ctx ' + m.ctx
   + (m.reasoning === 'off' ? ', thinking off' : '') : 'not tuned yet (use Tune model)');
 }
 $('models').onchange = showInfo;
 $('startServer').onclick = () => send({ type: 'terminal', action: 'start-server', model: $('models').value });
 $('tune').onclick = () => send({ type: 'terminal', action: 'tune', model: $('models').value, dense: $('dense').checked });
 $('sweep').onclick = () => send({ type: 'terminal', action: 'sweep', model: $('models').value });
 $('install').onclick = () => send({ type: 'terminal', action: 'install' });
 $('uninstall').onclick = () => send({ type: 'terminal', action: 'uninstall' });
 $('localOn').onchange = () => send({ type: 'quick', action: $('localOn').checked ? 'local-on' : 'local-off' });
 $('refresh').onclick = () => send({ type: 'refresh' });
 document.querySelectorAll('[data-quick]').forEach((b) => b.onclick = () => send({ type: 'quick', action: b.dataset.quick }));
 document.querySelectorAll('[data-open]').forEach((a) => a.onclick = () => send({ type: 'open', file: a.dataset.open }));
 window.addEventListener('message', (e) => { if (e.data.type === 'status') showStatus(e.data.data); if (e.data.type === 'models') showModels(e.data.data); });
</script></body></html>`;

function activate(context) {
  const panel = new Panel();
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('foremankit.panel', panel),
    vscode.commands.registerCommand('foremankit.refresh', () => panel.handle({ type: 'refresh' })));
}

module.exports = { activate };
