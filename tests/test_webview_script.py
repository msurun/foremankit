# The panel's webview script lives inside the PAGE template literal of vscode-extension/extension.js, where escapes
# like `\/` lose their backslash. `node --check extension.js` cannot see that; this test evaluates the template the
# way the extension does and parses every <script> of the resulting page, so a broken panel fails here, not in VS Code.
import shutil
import subprocess
from pathlib import Path

import pytest

EXTENSION = Path(__file__).resolve().parents[1] / "vscode-extension" / "extension.js"
CHECKER = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[1], 'utf8');
const marker = 'const PAGE = `';
const start = src.indexOf(marker) + marker.length;
const page = eval('`' + src.slice(start, src.indexOf('`;', start)) + '`');
const scripts = [...page.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (!scripts.length) throw new Error('no <script> found in PAGE');
scripts.forEach(body => new Function(body));
console.log(scripts.length);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_webview_script_parses():
    """Every script of the rendered panel page is valid JavaScript (a syntax error blanks the whole panel)."""
    done = subprocess.run(["node", "-e", CHECKER, str(EXTENSION)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
