"""Tests for the 'Agent runs' section of the ForemanKit VS Code extension panel.

Validates the structure, configuration dispatch table, section ordering, element IDs,
and Node.js syntax integrity in `vscode-extension/extension.js`.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

EXTENSION_PATH = Path(__file__).resolve().parent.parent / "vscode-extension" / "extension.js"
REQUIRED_ELEMENT_IDS = (
    "arAgyResume",
    "arOpencodeResume",
    "arCodexResume",
    "arAll",
    "arAgy",
    "arOpencode",
    "arCodex",
    "arCodexRow",
    "arCodexInfo",
    "arRecent",
    "arReviews",
    "arUsageRefresh",
    "arUsageMark",
)
REQUIRED_BACKEND_ACTIONS = (
    "'agent-runs'",
    "'agent-run-set'",
    "'agent-run-resume'",
    "'usage-refresh'",
    "'usage-mark'",
)
CONFIG_TABLE_KEYS = (
    "agentRunSet",
    "agentRunResume",
    "usageRefresh",
    "usageMark",
)

# The stub models DOM text replacement so refresh tests catch accumulated old rows.
MODEL_REVIEW_DOM = """
class Element {
    constructor() { this.children = []; this.style = {}; this.text = ''; }
    set textContent(value) { this.text = String(value); this.children = []; }
    get textContent() {
        return this.text + this.children.map(child => child.textContent).join('');
    }
    set innerHTML(value) {
        if (value !== '') throw new Error('Render payloads with textContent');
        this.textContent = '';
    }
    appendChild(child) { this.children.push(child); }
}
const elements = {};
const $ = name => elements[name] || (elements[name] = new Element());
const document = { createElement: () => new Element(), activeElement: null };
"""
MODEL_REVIEW = {
    "branch": "agent-run/<feature>", "reviewer": "codex<&>", "status": "ok",
    "counts_by_severity": {"high": 2, "medium": 3, "low": 4},
}


def render_model_review_updates(updates: list[dict]) -> list[str]:
    """Run real Agent runs rendering in Node and return the final visible review lines."""
    node_exe = shutil.which("node")
    assert node_exe, "Install Node.js and add it to PATH to test model review rendering"
    source = load_extension_source()
    start = source.index(" function showAgentRunner(")
    end = source.index(" function showSupervisor(", start)
    script = MODEL_REVIEW_DOM + source[start:end]
    script += f"\n{json.dumps(updates)}.forEach(update => showAgentRuns(update));"
    script += """
const box = $('arModelReviews');
const lines = box.children.length
    ? box.children.map(child => child.textContent) : [box.textContent];
console.log(JSON.stringify(lines));
"""
    result = subprocess.run(
        [node_exe, "-e", script], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"Model review rendering failed: {result.stderr}"
    return json.loads(result.stdout)


def test_model_reviews_are_placed_next_to_review_verdicts() -> None:
    """Model reviews have their own labeled container before usage in Agent runs."""
    source = load_extension_source()
    assert source.index('<h3>Agent runs') < source.index('id="arReviews"')
    assert source.index('id="arReviews"') < source.index('<b>Model reviews</b>')
    assert source.index('<b>Model reviews</b>') < source.index('id="arModelReviews"')
    assert source.index('id="arModelReviews"') < source.index('<b>Claude usage</b>')


def test_agent_runs_calls_the_model_review_renderer() -> None:
    """Each valid Agent runs update passes model reviews to the dedicated helper."""
    source = load_extension_source()
    start = source.index(' function showAgentRuns(')
    end = source.index(' function showSupervisor(', start)
    assert 'showModelReviews(d.model_reviews);' in source[start:end]
    assert "$('arReviews').textContent = d.review_text" in source[start:end]


def test_model_reviews_display_branch_reviewer_and_labeled_counts() -> None:
    """A completed review displays literal identifiers, severity labels and status."""
    lines = render_model_review_updates([{"model_reviews": [MODEL_REVIEW]}])
    assert lines == ['agent-run/<feature> | codex<&> | H: 2 / M: 3 / L: 4 | ok']


@pytest.mark.parametrize('status', ['failed', 'skipped', 'unparsable'])
def test_model_reviews_display_unsuccessful_status(status: str) -> None:
    """Unsuccessful attempts remain visible with their actual status and zero counts."""
    review = {**MODEL_REVIEW, 'status': status, 'counts_by_severity': {}}
    lines = render_model_review_updates([{"model_reviews": [review]}])
    assert lines == [f'agent-run/<feature> | codex<&> | H: 0 / M: 0 / L: 0 | {status}']


def test_empty_model_reviews_show_none_yet() -> None:
    """An empty model review list explicitly reports that no attempts exist."""
    assert render_model_review_updates([{"model_reviews": []}]) == ['none yet']


def test_absent_model_reviews_show_none_yet() -> None:
    """Older payloads without model reviews receive the same empty state."""
    assert render_model_review_updates([{}]) == ['none yet']


def test_refresh_replaces_stale_model_reviews() -> None:
    """A subsequent populated update replaces all prior review rows."""
    replacement = {**MODEL_REVIEW, 'branch': 'agent-run/new'}
    lines = render_model_review_updates([
        {'model_reviews': [MODEL_REVIEW, MODEL_REVIEW]}, {'model_reviews': [replacement]},
    ])
    assert lines == ['agent-run/new | codex<&> | H: 2 / M: 3 / L: 4 | ok']


@pytest.mark.parametrize('empty_update', [{'model_reviews': []}, {}])
def test_refresh_to_no_reviews_clears_stale_entries(empty_update: dict) -> None:
    """Empty or absent review data removes populated rows from the previous poll."""
    updates = [{'model_reviews': [MODEL_REVIEW]}, empty_update]
    assert render_model_review_updates(updates) == ['none yet']


def load_extension_source() -> str:
    """Read the extension.js source file using UTF-8 encoding."""
    # In plain words: read the entire VS Code extension script into memory so tests
    # can inspect its HTML templates, handlers, and configuration dispatch dictionary.
    return EXTENSION_PATH.read_text(encoding="utf-8")


def test_codex_row_wires_status_and_checkbox():
    """The third row renders status, dims with the master switch and sends the Codex toggle."""
    source = load_extension_source()
    assert "showAgentRunner('arCodex', 'arCodexInfo', runners.codex)" in source
    assert "$('arCodexRow').style.opacity = d.enabled" in source
    assert "$('arCodex').onchange = () => send({ type: 'config', action: 'agentRunSet', name: 'codex'," in source


def test_required_element_ids_are_present():
    """Verify all required HTML element IDs exist in extension.js."""
    # Arrange: load the source and enumerate the UI element identifiers.
    source = load_extension_source()

    # Act & Assert: every required element ID must appear in the webview markup or script.
    for element_id in REQUIRED_ELEMENT_IDS:
        assert element_id in source, f"Missing element ID: {element_id}"


def test_backend_action_names_appear_in_source():
    """Verify backend CLI command arguments are referenced in extension.js."""
    # Arrange: load the source code.
    source = load_extension_source()

    # Act & Assert: each backend command token must be wired to backend calls.
    for action in REQUIRED_BACKEND_ACTIONS:
        assert action in source, f"Missing backend action argument: {action}"


def test_config_table_has_agent_run_keys():
    """Verify agent run and usage keys are configured in the configArgs dispatch table."""
    # Arrange: find the configArgs method body inside extension.js.
    source = load_extension_source()
    config_args_start = source.find("configArgs(")
    assert config_args_start != -1, "configArgs method not found"
    config_args_end = source.find("return table[action];", config_args_start)
    table_snippet = source[config_args_start:config_args_end]

    # Act & Assert: all three configuration actions must be present as keys in the table.
    for key in CONFIG_TABLE_KEYS:
        assert f"{key}:" in table_snippet, f"Missing configArgs table key: {key}"


def test_agent_runs_section_is_ordered_between_delegation_and_night_run():
    """Verify the Agent runs section appears after Helpers (delegation) and before Night run."""
    # In plain words: the layout order in the panel sidebar must follow the specified
    # sequence so users encounter autonomous agent runs right after delegation settings.
    source = load_extension_source()
    delegation_pos = source.find("<h3>Helpers (delegation)</h3>")
    agent_runs_pos = source.find("<h3>Agent runs")
    night_run_pos = source.find("<h3>Night run")

    assert delegation_pos != -1, "Helpers (delegation) section missing"
    assert agent_runs_pos != -1, "Agent runs section missing"
    assert night_run_pos != -1, "Night run section missing"
    assert delegation_pos < agent_runs_pos < night_run_pos, (
        f"Expected order delegation ({delegation_pos}) < agent_runs ({agent_runs_pos}) "
        f"< night_run ({night_run_pos})"
    )


def test_agent_runs_section_has_no_cost_or_money_helper_calls():
    """Verify the Agent runs section does not contain money helper calls or dollar cost fields."""
    # In plain words: agent runs cost no money (subscription or free models),
    # so no cost fields or `money(...)` helper calls should appear in this section.
    source = load_extension_source()
    section_start = source.find("<h3>Agent runs")
    next_section_start = source.find("<h3>", section_start + 4)
    assert section_start != -1 and next_section_start != -1
    section_text = source[section_start:next_section_start]

    assert "money(" not in section_text, "Found unexpected money() helper call in Agent runs"
    assert "$" not in section_text, "Found unexpected dollar sign in Agent runs section"


def test_extension_javascript_syntax_with_node():
    """Verify extension.js has valid JavaScript syntax using node --check when node is present."""
    # In plain words: run the official Node.js syntax checker on the extension file
    # to catch any syntax error or malformed template literal before runtime.
    node_exe = shutil.which("node")
    if not node_exe:
        pytest.skip("Node.js is not available on PATH")

    result = subprocess.run(
        [node_exe, "--check", str(EXTENSION_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Node syntax check failed: {result.stderr}"


def test_runner_pause_display_and_resume_handlers():
    """Each runner can clear a pause and displays local retry time and probe status safely."""
    source = load_extension_source()
    for ident, runner in (("arAgy", "agy"), ("arOpencode", "opencode"), ("arCodex", "codex")):
        assert f"$('{ident}Resume').onclick = () => send({{ type: 'config', action: 'agentRunResume', name: '{runner}' }})" in source
    block = source[source.index(' function showAgentRunner('):source.index(' function showAgentRecent(')]
    assert "r.retry_at * 1000 > Date.now()" in block
    assert "paused until " in block and " (limit)" in block and "testing after pause" in block
    assert "getHours()" in block and "getMinutes()" in block
    assert " (soft)" in block and "r.daily_limit_hard" in block
    assert "textContent" in block and "innerHTML" not in block
