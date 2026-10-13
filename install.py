#!/usr/bin/env python3
"""Idempotent installer for foremankit.

  python install.py --dry-run     # show every action/command, change nothing
  python install.py               # venv + package + Claude Code agents/rules/MCP registration
  python install.py --local-orchestrator   # also: Claude Code plugin -> gateway -> local LLM (VS Code settings + gateway autostart at logon)
  python install.py --uninstall   # remove what foremankit added (agents, rules block, hooks, MCP); files that
                                  # existed before the first install come back, files it created go away
  python install.py --restore-original   # exact bytes of CLAUDE.md, settings.json and the agent files
                                         # as they were before the first install (legacy: earliest backup)

Only touches: ./.venv, ~/.claude/agents/{opus-advisor,delegator}.md, the foremankit block in
~/.claude/CLAUDE.md, the notify hook entries in ~/.claude/settings.json, and the user-scope MCP entry
"foremankit". With --local-orchestrator also the VS Code user settings.json / tasks.json. Every modified file is
backed up first.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import venv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLAUDE_HOME = Path.home() / ".claude"
BEGIN, END = "<!-- BEGIN foremankit -->", "<!-- END foremankit -->"
AGENTS = ("opus-advisor.md", "delegator.md")
MCP_NAME = "foremankit"
KEYS = ("DEEPSEEK_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
        "MISTRAL_API_KEY", "CEREBRAS_API_KEY", "LLAMACPP_URL", "OLLAMA_LOCAL_URL",
        "LLM_REMOTE_LAN_URL")
HOOK_TOOL = "mcp__foremankit__delegate"
HOOK_MARK = "notify_delegate.py"
SESSION_HOOK_MARK = "session_length_reminder.py"   # UserPromptSubmit: one reminder when the transcript gets very long
NUDGE_HOOK_MARK = "delegation_nudge.py"             # UserPromptSubmit + PostToolUse(Write|Edit|Bash): delegation counter, silent when delegation is off
NUDGE_TOOLS = "Write|Edit|Bash"                     # the tools whose output the counter measures
CHECKPOINT_HOOK_MARK = "auto_checkpoint.py"         # PostToolUse on delegate: auto-checkpoint after agent tasks
GATEWAY_ENV = "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_MODEL_OPTION", "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"
PICKER_NAME = "Local LLM (foremankit)"
TASK_ROUTER, TASK_SERVER = "ForemanKit: router 8081", "ForemanKit: llama-server 8080"   # VS Code tasks of earlier installs
LOGON_TASK = "ForemanKit router"          # Windows Task Scheduler entry that starts the gateway at logon (replaces the VS Code task)
EXTENSIONS_DIR_ENV = "FOREMANKIT_VSCODE_EXTENSIONS"   # tests point it at a tmp folder: the real extensions folder stays untouched
SCHEDULER_OFF_ENV = "FOREMANKIT_NO_SCHEDULER"   # tests set it: Task Scheduler calls become an in-memory flag
_fake_scheduler = {"registered": False}
REVIVE_MINUTES = 5                        # a stopped gateway cuts Claude Code off from Anthropic: bring it back quickly
LOGON_DELAY = "PT30S"                     # give OneDrive and the network time to come up
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
# STATE_DIR holds the "before" picture of every file the installer touches, so --uninstall can put the
# user's files back exactly as they were. TRACKED lists those files, relative to CLAUDE_HOME; the
# forward slashes are part of the snapshot keys and of the copied layout under STATE_DIR/"files".
STATE_DIR = Path.home() / ".config" / "foremankit" / "install-state"
TRACKED = ("CLAUDE.md", "settings.json", "agents/opus-advisor.md", "agents/delegator.md")
# Runtime leftovers of the gateway: its context counter sits next to the install snapshot, its logs in the temp
# folder (llama/Start-Router.ps1 writes router.log and router.err there). --purge deletes them; TEMP_DIR is a
# constant so that tests never touch the real temp folder.
TEMP_DIR = Path(tempfile.gettempdir())
GATEWAY_RUNTIME_FILES = ("gateway_context.json", "local_role", "orchestrator_target.json", "workers.json", "pairing.json", "gateway_live.json")
GATEWAY_LOG_FILES = ("router.log", "router.err", "llama-server.log", "llama-server.log.err")


# Extra tracked files outside ~/.claude: keys starting with "vscode/" resolve under the VS Code
# user folder instead of CLAUDE_HOME, so the same snapshot machinery covers both tools.
VSCODE_TRACKED = ("vscode/settings.json", "vscode/tasks.json")


# Command-line markers of processes foremankit starts (gateway, llama-server, their launchers).
# Matched against the whole command line, lower-cased, so a process of another tool is not killed.
FOREMANKIT_PROCESS_MARKERS = ("gateway", "llama-server", "start-orchestrator",
                            "Start-Router", "orchestrator")

# Marker file recording that the installer ADDED task.allowAutomaticTasks (rather than the user).
AUTO_TASKS_KEY = "task.allowAutomaticTasks"
AUTO_TASKS_MARKER = "vscode-added.json"


def select_foremankit_processes(rows: list[tuple[int, str]], root: Path,
                              own_pid: int) -> list[tuple[int, str]]:
    """Return the (pid, command line) rows started from `root`, excluding own_pid.

    A row matches when its command line, with backslashes normalised to forward slashes and
    lower-cased, contains `root` (same normalisation) AND one of FOREMANKIT_PROCESS_MARKERS.
    Pure: no process is touched, so it is easy to test.
    """
    wanted_root = str(root).replace("\\", "/").lower()
    markers = tuple(marker.lower() for marker in FOREMANKIT_PROCESS_MARKERS)
    kept = []
    for pid, command in rows:
        normalised = command.replace("\\", "/").lower()
        if pid == own_pid:
            continue
        if wanted_root in normalised and any(m in normalised for m in markers):
            kept.append((pid, command))
    return kept


def list_processes() -> list[tuple[int, str]]:
    """Return (pid, full command line) for every process, or [] when listing fails.

    Both systems print one `<pid> <command line>` per line: `ps -eo pid=,args=` on POSIX and a
    PowerShell Get-CimInstance query on Windows (a space is used, not a tab, because a tab inside
    single quotes is not an escape in PowerShell). A line that does not parse is skipped.
    """
    if os.name == "nt":
        query = "Get-CimInstance Win32_Process | ForEach-Object { '{0} {1}' -f $_.ProcessId, $_.CommandLine }"
        cmd = ["powershell", "-NoProfile", "-Command", query]
    else:
        cmd = ["ps", "-eo", "pid=,args="]
    try:
        done = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except OSError:
        return []
    if done.returncode != 0:
        return []
    rows = (line.strip().split(None, 1) for line in done.stdout.splitlines())
    return [(int(parts[0]), parts[1]) for parts in rows if len(parts) == 2 and parts[0].isdigit()]


# Legacy names of the previous "agentkit" installation. A machine that still carries them is
# migrated by the new installer: the config folder is moved, the old rules block is stripped and
# the old VS Code tasks are dropped. Values are read at call time where the tests monkeypatch them.
LEGACY_BEGIN, LEGACY_END = "<!-- BEGIN agentkit -->", "<!-- END agentkit -->"
LEGACY_CONFIG_DIR = Path.home() / ".config" / "agentkit"
LEGACY_TASK_LABELS = ("AgentKit: router 8081", "AgentKit: llama-server 8080")
LEGACY_BACKUP_MARK = ".bak-agentkit-"
LEGACY_MCP_NAME = "agentkit"


def find_claude() -> str | None:
    """The `claude` CLI: on PATH, else the newest binary bundled with the VS Code extension (it is usually not on PATH)."""
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    name = "claude.exe" if os.name == "nt" else "claude"
    bundled = sorted((Path.home() / ".vscode" / "extensions").glob(f"anthropic.claude-code-*/resources/native-binary/{name}"))
    return str(bundled[-1]) if bundled else None


def _hook_command(py: Path, script_name: str) -> str:
    """Shell command that runs one of our hook scripts with the Python interpreter `py`."""
    script = (ROOT / "hooks" / script_name).as_posix()
    return f'"{py.as_posix()}" "{script}"'


def _tool_entry(command: str) -> dict:
    """A hook entry that fires for the delegate tool (matcher HOOK_TOOL)."""
    return {"matcher": HOOK_TOOL, "hooks": [{"type": "command", "command": command}]}


def _without_marked(entries: list, mark: str) -> list:
    """The hook entries that do NOT run a command containing `mark`, i.e. everything that is not ours."""
    return [e for e in entries if not any(mark in h.get("command", "") for h in e.get("hooks", []))]


def _store_event(hooks: dict, event: str, kept: list) -> None:
    """Write the entries back under `event`; an empty list drops the key so settings.json stays clean."""
    if kept:
        hooks[event] = kept
    else:
        hooks.pop(event, None)


def _update_event(hooks: dict, event: str, mark: str, entry: dict | None) -> None:
    """Replace our old entries under `event` (matched by `mark`) with `entry`; None only removes them."""
    kept = _without_marked(hooks.get(event, []), mark)
    if entry is not None:
        kept.append(entry)
    _store_event(hooks, event, kept)


def _apply_notify(hooks: dict, py: Path, remove: bool) -> None:
    """Notify hook: PreToolUse and PostToolUse on the delegate tool."""
    command = _hook_command(py, "notify_delegate.py")
    for event in ("PreToolUse", "PostToolUse"):
        _update_event(hooks, event, HOOK_MARK, None if remove else _tool_entry(command))


def _apply_session(hooks: dict, py: Path, remove: bool) -> None:
    """Session-length reminder: UserPromptSubmit, no matcher."""
    command = _hook_command(py, "session_length_reminder.py")
    entry = {"hooks": [{"type": "command", "command": command}]}
    _update_event(hooks, "UserPromptSubmit", SESSION_HOOK_MARK, None if remove else entry)


def _apply_checkpoint(hooks: dict, py: Path, remove: bool) -> None:
    """Auto-checkpoint: PostToolUse on the delegate tool, writes .agent-out/checkpoint-<time>.md."""
    command = _hook_command(py, "auto_checkpoint.py")
    _update_event(hooks, "PostToolUse", CHECKPOINT_HOOK_MARK, None if remove else _tool_entry(command))


def _apply_nudge(hooks: dict, py: Path, remove: bool) -> None:
    """Delegation counter: UserPromptSubmit, and PostToolUse on the tools whose output it measures."""
    command = _hook_command(py, "delegation_nudge.py")
    for event, matcher in (("UserPromptSubmit", None), ("PostToolUse", NUDGE_TOOLS)):
        entry = {"hooks": [{"type": "command", "command": command}]}
        if matcher:
            entry = {"matcher": matcher, **entry}
        _update_event(hooks, event, NUDGE_HOOK_MARK, None if remove else entry)


def _apply_hooks(hooks: dict, py: Path, remove: bool, only: str | None) -> None:
    """Add or remove, in `hooks`, every hook group that `only` selects (None selects all of them)."""
    groups = (("notify", _apply_notify), ("session", _apply_session),
              ("checkpoint", _apply_checkpoint), ("nudge", _apply_nudge))
    for name, apply in groups:
        if only in (None, name):
            apply(hooks, py, remove)


class Installer:
    def __init__(self, dry: bool):
        self.dry = dry

    def say(self, msg: str) -> None:
        print(("[dry-run] " if self.dry else "") + msg)

    def run(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess | None:
        self.say("$ " + subprocess.list2cmdline(cmd))
        if self.dry:
            return None
        return subprocess.run(cmd, check=check, text=True, capture_output=True)

    def backup(self, path: Path) -> None:
        if path.exists():
            dst = path.with_name(f"{path.name}.bak-foremankit-{STAMP}")
            self.say(f"backup {path} -> {dst.name}")
            if not self.dry:
                shutil.copy2(path, dst)

    def write(self, path: Path, text: str) -> None:
        if path.exists() and path.read_text(encoding="utf-8") == text:
            self.say(f"unchanged {path}")
            return
        self.backup(path)
        self.say(f"write {path}")
        if not self.dry:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    SNAPSHOT_FILE = "snapshot.json"
    SNAPSHOT_FILES_DIR = "files"
    BACKUP_MARK = ".bak-foremankit-"
    BLANK_WHEN_EMPTY = ("CLAUDE.md", "settings.json")   # settings.json counts as blank when it is {}

    def _snapshot_record(self) -> Path:
        """Return the path of snapshot.json inside the current STATE_DIR."""
        return STATE_DIR / self.SNAPSHOT_FILE

    def take_snapshot(self, extra: tuple[str, ...] = ()) -> None:
        """Record which tracked files exist and copy them aside, once per key.

        `extra` adds keys that are only relevant to some install modes (for example the VS Code
        files). An already recorded key is never overwritten, so a re-run cannot lose the original
        bytes; a key seen for the first time is appended to the existing record. Does nothing in
        dry-run mode.
        """
        keys = tuple(TRACKED) + tuple(extra)
        record = self._snapshot_record()
        if self.dry:
            self.say(f"would snapshot {len(keys)} tracked files into {STATE_DIR}")
            return
        if record.exists():
            self._extend_snapshot(record, keys)
            return
        existed = self._copy_tracked_files(keys, {})
        self._write_snapshot_record(record, existed)

    def _copy_tracked_files(self, keys: tuple[str, ...], existed: dict[str, bool]) -> dict[str, bool]:
        """Copy every key that exists into STATE_DIR/"files" and return the extended flags."""
        for key in keys:
            source = self.tracked_path(key)
            existed[key] = source.exists()
            if source.exists():
                copy = STATE_DIR / self.SNAPSHOT_FILES_DIR / key
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, copy)
                self.say(f"snapshot {source} -> {copy}")
        return existed

    def _write_snapshot_record(self, record: Path, existed: dict[str, bool]) -> None:
        """Write snapshot.json with the existence flags of every tracked key."""
        record.parent.mkdir(parents=True, exist_ok=True)
        data = {"taken": datetime.now().isoformat(timespec="seconds"), "existed": existed}
        record.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _extend_snapshot(self, record: Path, keys: tuple[str, ...]) -> None:
        """Add keys that the record does not know yet; recorded keys keep their flags and bytes."""
        snapshot = json.loads(record.read_text(encoding="utf-8"))
        missing = tuple(key for key in keys if key not in snapshot["existed"])
        if not missing:
            self.say(f"snapshot already covers {len(keys)} tracked files: {record}")
            return
        snapshot["existed"] = self._copy_tracked_files(missing, snapshot["existed"])
        self._write_snapshot_record(record, snapshot["existed"])

    def _earliest_backup(self, key: str) -> Path | None:
        """Return the oldest backup next to the live file, or None.

        Both marks are considered, then sorted together by name: the stamp is ``%Y%m%d-%H%M%S``, so
        sorting the names sorts them by time and a pre-rename ``.bak-agentkit-*`` file is found too.
        """
        name = Path(key).name
        parent = self.tracked_path(key).parent
        marks = (self.BACKUP_MARK, LEGACY_BACKUP_MARK)
        backups = [p for mark in marks for p in parent.glob(f"{name}{mark}*")]
        return min(backups) if backups else None

    def _restore_one(self, key: str, existed: bool, snapshot_bytes: bytes | None) -> None:
        """Put one tracked file back to the snapshot state, or delete it if it did not exist before."""
        target = self.tracked_path(key)
        if not existed:
            if target.exists():
                self.backup(target)
                self.say(f"remove {target} (did not exist before foremankit)")
                if not self.dry:
                    target.unlink()
            return
        if not target.exists() or target.read_bytes() != snapshot_bytes:
            self.backup(target)
            self.say(f"restore {target} from snapshot")
            if not self.dry:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(snapshot_bytes or b"")

    def restore_tracked(self, key: str) -> None:
        """Bring back ONE tracked file that existed before the installation (no-op otherwise)."""
        record = self._snapshot_record()
        if not record.exists():
            return
        if json.loads(record.read_text(encoding="utf-8"))["existed"].get(key):
            copy = STATE_DIR / self.SNAPSHOT_FILES_DIR / key
            self._restore_one(key, True, copy.read_bytes())

    def restore_original(self) -> None:
        """Undo the file changes: rewrite the snapshot bytes, or fall back to the earliest backup.

        With a snapshot every TRACKED file is restored byte for byte. Without one (a legacy install)
        the oldest `*.bak-foremankit-*` file next to the target is used; a file with neither snapshot nor
        backup is left untouched.
        """
        record = self._snapshot_record()
        if record.exists():
            self._restore_from_snapshot(json.loads(record.read_text(encoding="utf-8")))
        else:
            self.say(f"no snapshot in {STATE_DIR}: falling back to backups")
            self._restore_from_backups()

    def _restore_from_snapshot(self, snapshot: dict) -> None:
        """Restore every tracked file using the bytes copied under STATE_DIR/"files"."""
        for key, existed in snapshot["existed"].items():
            copy = STATE_DIR / self.SNAPSHOT_FILES_DIR / key
            self._restore_one(key, existed, copy.read_bytes() if copy.exists() else None)

    def _restore_from_backups(self) -> None:
        """Restore every TRACKED file from its earliest backup, if there is one.

        Only the Claude files are covered: a legacy install predates the VS Code tracking, so a
        VS Code key would have no backup to fall back on.
        """
        for key in TRACKED:
            backup = self._earliest_backup(key)
            if backup is None:
                self.say(f"no backup for {self.tracked_path(key)}: left as it is")
                continue
            self._restore_one(key, True, backup.read_bytes())

    def remove_empty_created_files(self) -> None:
        """Delete created files that the installer left blank.

        Guards against leaving empty files behind: CLAUDE.md counts as blank when it holds only
        whitespace, a JSON file when is_blank_json accepts its content. Without a snapshot nothing
        is removed.
        """
        record = self._snapshot_record()
        if not record.exists():
            self.say(f"no snapshot in {STATE_DIR}: nothing to clean up")
            return
        existed = json.loads(record.read_text(encoding="utf-8"))["existed"]
        for key in self.BLANK_WHEN_EMPTY + VSCODE_TRACKED:
            target = self.tracked_path(key)
            if existed.get(key, True) or not target.exists() or not self._is_blank(target):
                continue
            self.say(f"remove {target} (created blank by foremankit)")
            if not self.dry:
                target.unlink()

    @classmethod
    def _is_blank(cls, path: Path) -> bool:
        """Return True when the file is whitespace-only, or JSON that still holds no setting."""
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            try:
                return cls.is_blank_json(json.loads(text))
            except ValueError:
                return False
        return not text.strip()

    def tracked_path(self, key: str) -> Path:
        """Resolve a snapshot key to its absolute path.

        Keys starting with "vscode/" live under the VS Code user folder, every other key under
        CLAUDE_HOME. The prefix stays part of the key so STATE_DIR/"files" keeps one flat layout.
        """
        relative = key.removeprefix("vscode/")
        base = self.vscode_user_dir() if key.startswith("vscode/") else CLAUDE_HOME
        return base / relative

    @staticmethod
    def is_blank_json(data: dict) -> bool:
        """Return True when a settings/tasks JSON holds nothing foremankit cares about.

        `{}` is blank; so is the VS Code tasks skeleton, which only carries the format version and
        an empty task list and therefore proves the installer created the file.
        """
        without_version = {key: value for key, value in data.items() if key != "version"}
        if not without_version:
            return True
        return without_version == {"tasks": []}

    def leftovers(self) -> list[str]:
        """Return human-readable traces of foremankit that remain on this machine.

        Checks the agent files, the CLAUDE.md block, the settings.json hooks, the VS Code
        gateway env entries, the autostart tasks and the traces of a pre-rename agentkit
        installation. A missing or unparsable file is skipped, and STATE_DIR is deliberately never
        reported. [] means a clean setup.
        """
        vscode = Installer.vscode_user_dir()
        found: list[str] = []
        found += self._leftover_agents()
        found += self._leftover_rules()
        found += self._leftover_hooks()
        found += self._leftover_gateway_env(vscode / "settings.json")
        found += self._leftover_tasks(vscode / "tasks.json")
        found += [f"Task Scheduler entry {LOGON_TASK} still present"] if self.logon_task_registered() else []
        link = self.extension_dir()
        found += [f"VS Code panel extension still linked: {link}"] if (link.is_symlink() or link.exists()) else []
        found += self._leftover_legacy_rules()
        found += self._leftover_legacy_tasks(vscode / "tasks.json")
        return found

    def _leftover_agents(self) -> list[str]:
        """Return one message per foremankit agent file still under CLAUDE_HOME/agents."""
        found = []
        for name in AGENTS:
            path = CLAUDE_HOME / "agents" / name
            if path.exists():
                found.append(f"agent file still present: {path}")
        return found

    def _leftover_rules(self) -> list[str]:
        """Return a message when the BEGIN marker is still in CLAUDE.md, else []."""
        path = CLAUDE_HOME / "CLAUDE.md"
        if path.exists() and BEGIN in path.read_text(encoding="utf-8"):
            return [f"rules block {BEGIN} still present in {path}"]
        return []

    def _leftover_hooks(self) -> list[str]:
        """Return a message when settings.json still holds an foremankit hook command, else []."""
        path = CLAUDE_HOME / "settings.json"
        data = self._read_json(path)
        if data is None:
            return []
        text = json.dumps(data)
        marks = (HOOK_MARK, SESSION_HOOK_MARK, CHECKPOINT_HOOK_MARK, NUDGE_HOOK_MARK)
        found = [mark for mark in marks if mark in text]
        return [f"hook {', '.join(found)} still registered in {path}"] if found else []

    def _leftover_gateway_env(self, path: Path) -> list[str]:
        """Return a message when the VS Code settings still point the plugin at the gateway."""
        data = self._read_json(path)
        if data is None:
            return []
        entries = data.get("claudeCode.environmentVariables", [])
        names = {entry.get("name") for entry in entries if isinstance(entry, dict)}
        found = sorted(names & set(GATEWAY_ENV))
        return [f"VS Code env {', '.join(found)} still set in {path}"] if found else []

    def _leftover_tasks(self, path: Path) -> list[str]:
        """Return a message when the VS Code tasks.json still holds an foremankit autostart task."""
        data = self._read_json(path)
        if data is None:
            return []
        labels = {task.get("label") for task in data.get("tasks", [])}
        found = sorted(label for label in labels & {TASK_ROUTER, TASK_SERVER})
        return [f"autostart task {', '.join(found)} still in {path}"] if found else []

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        """Return the parsed JSON object at `path`, or None when the file is missing or invalid."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def gateway_references(self) -> list[str]:
        """Places that still point Claude Code at the local gateway (stopping it would cut the cloud off).

        Claude Code reads ANTHROPIC_BASE_URL at start, so a reference left in the VS Code settings, in
        ~/.claude/settings.json ("env") or in the environment of this shell means a dead gateway port
        and no connection to the Anthropic cloud at all.
        """
        port = (yaml_load(ROOT / "config" / "orchestrator.yaml") or {}).get("port", 8081)
        needle = f"127.0.0.1:{port}"
        found = [t for t in self._leftover_gateway_env(self.vscode_user_dir() / "settings.json")]
        env_of_claude = (self._read_json(CLAUDE_HOME / "settings.json") or {}).get("env", {})
        if needle in str(env_of_claude.get("ANTHROPIC_BASE_URL", "")):
            found.append(f"ANTHROPIC_BASE_URL in {CLAUDE_HOME / 'settings.json'} (env)")
        if needle in os.environ.get("ANTHROPIC_BASE_URL", ""):
            found.append("ANTHROPIC_BASE_URL in the environment of this shell")
        return found

    def purge_state(self) -> None:
        """Delete STATE_DIR with everything inside it, and nothing else.

        In dry-run mode it only reports. A missing STATE_DIR is a no-op, so calling it twice is safe.
        """
        self._purge_gateway_files()
        if not STATE_DIR.exists():
            self.say(f"nothing to purge: {STATE_DIR} does not exist")
            return
        self.say(f"remove state folder {STATE_DIR}")
        if not self.dry:
            shutil.rmtree(STATE_DIR)

    def _purge_gateway_files(self) -> None:
        """Delete the gateway's context counter and its logs; missing files are fine."""
        paths = [STATE_DIR.parent / name for name in GATEWAY_RUNTIME_FILES]
        paths += [TEMP_DIR / name for name in GATEWAY_LOG_FILES]
        for path in paths:
            if path.exists():
                self.say(f"remove {path}")
                if not self.dry:
                    path.unlink()

    def stop_services(self) -> int:
        """Stop the foremankit processes started from this folder; return how many matched.

        Only processes whose command line mentions this repository and an foremankit marker are
        stopped. In dry-run mode it reports them without sending any signal.
        """
        matches = select_foremankit_processes(list_processes(), ROOT, os.getpid())
        for pid, command in matches:
            self.say(f"stop pid {pid}: {command.strip()}")
            if self.dry:
                continue
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
            else:
                os.kill(pid, signal.SIGTERM)
        return len(matches)

    def _record_auto_tasks_marker(self, settings_path: Path) -> None:
        """Write STATE_DIR/AUTO_TASKS_MARKER when the installer is the one adding the key.

        Records allowAutomaticTasks=true only when the user file has no such key yet and the
        marker is missing, so a value the user had is never attributed to foremankit.
        """
        marker = STATE_DIR / AUTO_TASKS_MARKER
        if self.dry or marker.exists():
            return
        settings = self._read_json(settings_path) or {}
        if AUTO_TASKS_KEY in settings:
            return
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"allowAutomaticTasks": True}) + "\n", encoding="utf-8")
        self.say(f"recorded installer-added {AUTO_TASKS_KEY} in {marker}")

    def _without_auto_tasks(self, data: dict) -> dict:
        """Return a copy of `data` without AUTO_TASKS_KEY when the installer added it.

        The key is kept when the marker is absent (a user value) or when the current value is not
        the "auto" the installer wrote, so uninstall never touches a user setting.
        """
        marker = STATE_DIR / AUTO_TASKS_MARKER
        added_by_us = marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("allowAutomaticTasks")
        if not added_by_us or data.get(AUTO_TASKS_KEY) != "auto":
            return data
        return {key: value for key, value in data.items() if key != AUTO_TASKS_KEY}

    def migrate_config_dir(self) -> bool:
        """Move ~/.config/agentkit to the foremankit folder; return whether it was there.

        When both folders exist the legacy files are copied without overwriting anything the user
        already has in the new folder. In dry-run mode nothing is moved or copied. Raises OSError
        when the filesystem refuses the move.
        """
        if not LEGACY_CONFIG_DIR.exists():
            return False
        self.say(f"migrate {LEGACY_CONFIG_DIR} -> {STATE_DIR.parent}")
        if self.dry:
            return True
        target = STATE_DIR.parent
        if target.exists() and any(target.iterdir()):
            self._copy_missing_legacy_files()
            return True
        if target.exists():
            target.rmdir()   # shutil.move would put the folder INSIDE an existing empty target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(LEGACY_CONFIG_DIR), str(target))
        return True

    def _copy_missing_legacy_files(self) -> None:
        """Copy legacy config files that are missing in the new folder; never overwrite.

        Both folders exist, so the user already ran the new installer: only files the new folder
        does not have yet are brought over, and the legacy folder is left in place.
        """
        for source in LEGACY_CONFIG_DIR.rglob("*"):
            if not source.is_file():
                continue
            target = STATE_DIR.parent / source.relative_to(LEGACY_CONFIG_DIR)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            self.say(f"copied legacy {source} -> {target}")

    def remove_legacy_rules_block(self) -> None:
        """Strip the old BEGIN/END agentkit block from CLAUDE.md, keeping the surrounding text.

        The current foremankit block, if any, is not touched; the block of the previous name only
        existed before the rename. Does nothing when CLAUDE.md is missing or carries no such block.
        """
        target = CLAUDE_HOME / "CLAUDE.md"
        if not target.exists():
            return
        current = target.read_text(encoding="utf-8")
        if LEGACY_BEGIN not in current or LEGACY_END not in current:
            return
        head, rest = current.split(LEGACY_BEGIN, 1)
        _, tail = rest.split(LEGACY_END, 1)
        self.write(target, self.merged_claude_md(f"{head}\n{tail}", None))

    def remove_legacy_tasks(self) -> None:
        """Drop the old AgentKit autostart tasks from the VS Code tasks.json; user tasks stay.

        Uses write_json so the file is backed up first and left alone when it is not plain JSON.
        Does nothing when the file does not exist.
        """
        path = self.vscode_user_dir() / "tasks.json"
        if not path.exists():
            return
        self.write_json(path, lambda data: {**data, "tasks": [
            task for task in data.get("tasks", []) if task.get("label") not in LEGACY_TASK_LABELS]})

    def remove_legacy_mcp(self) -> None:
        """Unregister the old "agentkit" MCP server from the user scope when it is registered.

        Silence is the expected outcome: without the `claude` CLI, or when the old server was never
        added, there is nothing to undo. Mirrors register_mcp(remove=True), only with the old name.
        """
        claude = find_claude()
        if not claude:
            return
        existing = subprocess.run([claude, "mcp", "get", LEGACY_MCP_NAME],
                                  text=True, capture_output=True, check=False)
        if existing.returncode != 0:
            return
        self.run([claude, "mcp", "remove", LEGACY_MCP_NAME, "--scope", "user"], check=False)

    def migrate_legacy(self) -> None:
        """Run every step that undoes an old "agentkit" installation in one call.

        Order: the config folder first (the keys live there), then the file edits. Safe to call on
        a machine that never saw the old name: every step checks for its own traces.
        """
        self.migrate_config_dir()
        self.remove_legacy_rules_block()
        self.remove_legacy_tasks()
        self.remove_legacy_mcp()

    def _leftover_legacy_rules(self) -> list[str]:
        """Return a message when the old agentkit BEGIN marker is still in CLAUDE.md, else []."""
        path = CLAUDE_HOME / "CLAUDE.md"
        if path.exists() and LEGACY_BEGIN in path.read_text(encoding="utf-8"):
            return [f"legacy rules block {LEGACY_BEGIN} still present in {path}"]
        return []

    def _leftover_legacy_tasks(self, path: Path) -> list[str]:
        """Return a message when the VS Code tasks.json still holds an old AgentKit task."""
        data = self._read_json(path)
        if data is None:
            return []
        labels = {task.get("label") for task in data.get("tasks", [])}
        found = sorted(label for label in labels & set(LEGACY_TASK_LABELS))
        return [f"legacy autostart task {', '.join(found)} still in {path}"] if found else []

    # ------------------------------------------------------------ steps
    def venv_python(self) -> Path:
        sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        return ROOT / ".venv" / sub

    def make_venv(self) -> Path:
        py = self.venv_python()
        if py.exists():
            self.say(f"venv exists: {py.parent.parent}")
        else:
            self.say(f"create venv {ROOT / '.venv'}")
            if not self.dry:
                venv.create(ROOT / ".venv", with_pip=True)
        if self.already_installed(py):
            # Re-running pip while the MCP server is up fails on the locked foremankit-mcp.exe and leaves a
            # half-uninstalled package behind. An editable install needs no refresh for code changes.
            self.say("package already installed (editable) from this folder: skipping pip")
            return py
        self.run([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
        self.run([str(py), "-m", "pip", "install", "--quiet", "-e", f"{ROOT}[dev]"])
        return py

    def already_installed(self, py: Path) -> bool:
        if self.dry or not py.exists():
            return False
        probe = "import foremankit,pathlib;print(pathlib.Path(foremankit.__file__).resolve())"
        done = subprocess.run([str(py), "-c", probe], cwd=Path.home(), text=True, capture_output=True, check=False)
        return done.returncode == 0 and Path(done.stdout.strip()).is_relative_to(ROOT)

    def install_agents(self) -> None:
        for name in AGENTS:
            src = ROOT / "claude" / "agents" / name
            self.write(CLAUDE_HOME / "agents" / name, src.read_text(encoding="utf-8"))

    @staticmethod
    def merged_claude_md(current: str, block: str | None) -> str:
        """Insert/replace/remove the BEGIN/END block; idempotent."""
        if BEGIN in current and END in current:
            head, rest = current.split(BEGIN, 1)
            _, tail = rest.split(END, 1)
            head, tail = head.rstrip("\n"), tail.lstrip("\n")
        else:
            head, tail = current.rstrip("\n"), ""
        parts = [p for p in (head, f"{BEGIN}\n{block.strip()}\n{END}" if block else "", tail) if p]
        return "\n\n".join(parts) + "\n"

    def install_rules(self, remove: bool = False) -> None:
        target = CLAUDE_HOME / "CLAUDE.md"
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        block = None if remove else (ROOT / "claude" / "CLAUDE.global.md").read_text(encoding="utf-8")
        new = self.merged_claude_md(current, block)
        if new.strip() == current.strip():
            self.say(f"unchanged {target}")
            return
        self.write(target, new)

    def mcp_command(self, py: Path) -> Path:
        """Console script, not `python -m`: -m puts the cwd on sys.path, and a project folder
        named `foremankit/` would then shadow the installed package (CONNECTION_CLOSED)."""
        return py.with_name("foremankit-mcp.exe" if os.name == "nt" else "foremankit-mcp")

    def register_mcp(self, py: Path, remove: bool = False) -> None:
        claude = find_claude()
        cmd = self.mcp_command(py)
        if not claude:
            print("! `claude` CLI not found on PATH; register manually:")
            print(f"  claude mcp add --scope user {MCP_NAME} -- \"{cmd}\"")
            return
        existing = None if self.dry else subprocess.run(
            [claude, "mcp", "get", MCP_NAME], text=True, capture_output=True, check=False)
        present = bool(existing and existing.returncode == 0)
        if present and not remove and str(cmd).lower() in existing.stdout.lower():
            self.say(f"MCP '{MCP_NAME}' already registered with {cmd}")
            return
        if present or self.dry:
            self.run([claude, "mcp", "remove", MCP_NAME, "--scope", "user"], check=False)
        if not remove:
            # No -e KEY=... on purpose: keys would be written into ~/.claude.json.
            self.run([claude, "mcp", "add", "--scope", "user", MCP_NAME, "--", str(cmd)])

    def install_hooks(self, py: Path, remove: bool = False, only: str | None = None) -> None:
        """Merge the hooks into ~/.claude/settings.json: the notify hook (Pre/PostToolUse on the delegate tool) and the
        session-length reminder (UserPromptSubmit), the auto-checkpoint and the delegation counter (UserPromptSubmit +
        PostToolUse on Write|Edit|Bash). `only` = "notify" | "session" | "checkpoint" | "nudge" limits it to one of them."""
        target = CLAUDE_HOME / "settings.json"
        try:
            data = json.loads(target.read_text(encoding="utf-8-sig")) if target.exists() else {}
        except ValueError:
            print(f"! {target} is not valid JSON: hook not installed, fix the file and re-run")
            return
        hooks = data.setdefault("hooks", {})
        _apply_hooks(hooks, py, remove, only)
        if not hooks:
            data.pop("hooks")
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        if target.exists() and json.loads(target.read_text(encoding="utf-8-sig")) == data:
            self.say(f"unchanged {target}")
            return
        self.backup(target)
        self.say(f"{'remove' if remove else 'write'} hooks in {target}")
        if not self.dry:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")   # no BOM: Node's JSON.parse rejects it

    # ------------------------------------------------------------ local orchestrator (VS Code)
    @staticmethod
    def vscode_user_dir() -> Path:
        """Per-OS folder of the VS Code user settings."""
        if sys.platform == "win32":
            return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Code" / "User"
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "Code" / "User"
        return Path.home() / ".config" / "Code" / "User"

    @staticmethod
    def merged_vscode_settings(data: dict, base_url: str, model_id: str, remove: bool = False, tasks: bool = False) -> dict:
        """Add (or remove) the three gateway entries in claudeCode.environmentVariables; other entries stay."""
        values = dict(zip(GATEWAY_ENV, (base_url, model_id, PICKER_NAME)))
        entries = [e for e in data.get("claudeCode.environmentVariables", []) if e.get("name") not in values]
        if not remove:
            entries += [{"name": name, "value": value} for name, value in values.items()]
        merged = {**data, "claudeCode.environmentVariables": entries}
        if not entries:
            merged.pop("claudeCode.environmentVariables")
        if tasks and not remove:
            merged.setdefault("task.allowAutomaticTasks", "auto")   # only needed when a folder-open task is added
        return merged

    @staticmethod
    def merged_vscode_tasks(data: dict, tasks: list[dict], remove: bool = False, drop_labels: tuple = ()) -> dict:
        """Replace the tasks with the same labels (or just remove them); `drop_labels` are removed too. User tasks stay."""
        labels = {task["label"] for task in tasks} | set(drop_labels)
        kept = [t for t in data.get("tasks", []) if t.get("label") not in labels]
        merged = {"version": "2.0.0", **data, "tasks": kept if remove else kept + tasks}
        return merged

    def autostart_tasks(self, py: Path) -> list[dict]:
        """VS Code folder-open tasks: only the gateway, and only off Windows (Windows uses the logon task)."""
        if sys.platform == "win32":
            return []
        router = f'powershell -NoProfile -ExecutionPolicy Bypass -File "{ROOT / "llama" / "Start-Router.ps1"}"'
        return [{"label": TASK_ROUTER, "type": "shell", "command": router, "runOptions": {"runOn": "folderOpen"},
                 "presentation": {"reveal": "silent", "panel": "dedicated", "close": True}}]

    def register_logon_task(self) -> None:
        """Windows: start the gateway at logon through Task Scheduler (no-op script when it already runs)."""
        if sys.platform != "win32":
            print("! no autostart here: start the gateway with `foremankit-gateway` or let VS Code run the task")
            return
        script = ROOT / "llama" / "Start-Router.ps1"
        # conhost --headless: no console window flashes every few minutes; the repeating trigger restarts a dead gateway
        command = (f"$a = New-ScheduledTaskAction -Execute 'conhost.exe' -Argument '--headless powershell.exe -NoProfile "
                   f"-ExecutionPolicy Bypass -File \"{script}\"'; "
                   f"$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME; $logon.Delay = '{LOGON_DELAY}'; "
                   f"$again = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes {REVIVE_MINUTES}) "
                   f"-RepetitionDuration (New-TimeSpan -Days 3650); "
                   f"$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew "
                   f"-ExecutionTimeLimit (New-TimeSpan -Minutes 5); "
                   f"Register-ScheduledTask -TaskName '{LOGON_TASK}' -Action $a -Trigger @($logon, $again) -Settings $s -Force | Out-Null; "
                   f"Start-ScheduledTask -TaskName '{LOGON_TASK}'")
        self.say(f"register Task Scheduler entry '{LOGON_TASK}' (gateway at logon)")
        if os.environ.get(SCHEDULER_OFF_ENV):
            _fake_scheduler["registered"] = not self.dry
        elif not self.dry:
            subprocess.run(["powershell", "-NoProfile", "-Command", command], check=False)

    def unregister_logon_task(self) -> None:
        """Windows: remove the Task Scheduler entry; missing is fine."""
        if sys.platform != "win32":
            return
        self.say(f"remove Task Scheduler entry '{LOGON_TASK}'")
        if os.environ.get(SCHEDULER_OFF_ENV):
            _fake_scheduler["registered"] = _fake_scheduler["registered"] and self.dry
        elif not self.dry:
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"Unregister-ScheduledTask -TaskName '{LOGON_TASK}' -Confirm:$false -ErrorAction SilentlyContinue"],
                           check=False)

    @staticmethod
    def extension_dir() -> Path:
        """Where VS Code looks for the ForemanKit panel: <publisher>.<name>-<version> under ~/.vscode/extensions."""
        manifest = json.loads((ROOT / "vscode-extension" / "package.json").read_text(encoding="utf-8"))
        folder = Path(os.environ.get(EXTENSIONS_DIR_ENV) or Path.home() / ".vscode" / "extensions")
        return folder / f"{manifest['publisher']}.{manifest['name']}-{manifest['version']}"

    def link_extension(self) -> None:
        """Make the panel visible in VS Code: a directory link to vscode-extension/ (updates with the repository)."""
        target, link = ROOT / "vscode-extension", self.extension_dir()
        self.say(f"link VS Code extension {link} -> {target}")
        if self.dry or link.exists():
            return
        link.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False)
        else:
            link.symlink_to(target, target_is_directory=True)

    def unlink_extension(self) -> None:
        """Remove only the link, never the repository folder it points to."""
        link = self.extension_dir()
        self.say(f"remove VS Code extension link {link}")
        if self.dry or not (link.is_symlink() or link.exists()):
            return
        link.unlink() if link.is_symlink() else os.rmdir(link)

    @staticmethod
    def logon_task_registered() -> bool:
        """True when the Task Scheduler entry exists (always False off Windows)."""
        if os.environ.get(SCHEDULER_OFF_ENV):
            return _fake_scheduler["registered"]
        if sys.platform != "win32":
            return False
        query = subprocess.run(["powershell", "-NoProfile", "-Command",
                                f"Get-ScheduledTask -TaskName '{LOGON_TASK}' -ErrorAction Stop | Out-Null"],
                               capture_output=True, check=False)
        return query.returncode == 0

    def write_json(self, path: Path, build) -> None:
        """Read a JSON file ({} if absent), apply `build(data) -> data`, write it back if changed (backup first)."""
        try:
            current = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
        except ValueError:
            print(f"! {path} is not plain JSON (comments?): not touched, edit it by hand (see README)")
            return
        data = build(current)
        if data == current:
            self.say(f"unchanged {path}")
            return
        self.write(path, json.dumps(data, indent=4, ensure_ascii=False) + "\n")

    def ensure_llama_server(self, download: bool) -> None:
        """llama/LLAMA.CPP/llama-server is not in git: download it on request (--with-llama), else say how."""
        server = ROOT / "llama" / "LLAMA.CPP" / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if server.exists():
            self.say(f"llama-server present: {server}")
        elif download and not self.dry:
            subprocess.run([sys.executable, str(ROOT / "llama" / "get_llama.py")], check=False)
        else:
            print("! llama-server is missing (no local models without it): run  python install.py --local-orchestrator "
                  "--with-llama  or  python llama/get_llama.py  (downloads about 150-650 MB from GitHub, sha256-checked)")

    def enable_router_mode(self, py: Path) -> None:
        """Router mode: Claude Code in VS Code goes through the gateway (the local models become available).

        Writes the three gateway entries into the VS Code settings and makes sure the gateway runs (Task Scheduler logon
        entry on Windows, a folder-open VS Code task elsewhere). Claude Code reads the settings only when a window starts."""
        cfg = yaml_load(ROOT / "config" / "orchestrator.yaml")
        base_url = f"http://127.0.0.1:{cfg.get('port', 8081)}"
        model_id = (cfg.get("local_models") or ["claude-sonnet-5"])[0]
        folder = self.vscode_user_dir()
        wanted = self.autostart_tasks(py)
        if wanted:
            self._record_auto_tasks_marker(folder / "settings.json")   # before the write: afterwards the key always exists
        self.write_json(folder / "settings.json", lambda d: self.merged_vscode_settings(d, base_url, model_id, False, bool(wanted)))
        if wanted:
            self.write_json(folder / "tasks.json", lambda d: self.merged_vscode_tasks(d, wanted, False, (TASK_SERVER,)))
        self.register_logon_task()

    def disable_router_mode(self) -> None:
        """Direct mode (the default): nothing of the router is left in VS Code or in the Task Scheduler.

        Removes the gateway entries from the settings, the folder-open tasks (current and from earlier installs) and the
        logon entry. Nothing is created when VS Code files are missing."""
        cfg = yaml_load(ROOT / "config" / "orchestrator.yaml")
        base_url = f"http://127.0.0.1:{cfg.get('port', 8081)}"
        model_id = (cfg.get("local_models") or ["claude-sonnet-5"])[0]
        self.unregister_logon_task()
        folder = self.vscode_user_dir()
        settings, tasks = folder / "settings.json", folder / "tasks.json"
        if settings.exists():
            self.write_json(settings, lambda d: self.merged_vscode_settings(d, base_url, model_id, True))
            self.write_json(settings, self._without_auto_tasks)
        if tasks.exists():
            self.write_json(tasks, lambda d: self.merged_vscode_tasks(d, [], True, (TASK_SERVER, TASK_ROUTER)))

    def router_mode_active(self) -> bool:
        """True when the VS Code user settings still point Claude Code at the local gateway."""
        settings = self.vscode_user_dir() / "settings.json"
        if not settings.exists():
            return False
        port = yaml_load(ROOT / "config" / "orchestrator.yaml").get("port", 8081)
        return f"127.0.0.1:{port}" in settings.read_text(encoding="utf-8-sig", errors="replace")

    def install_local_orchestrator(self, py: Path, remove: bool = False, router: bool = False) -> None:
        """The ForemanKit panel in VS Code. Direct mode stays the default: the router is only set up with `router=True`
        (or later with the panel's "Router mode" switch), and removed again by `remove` or the panel's "Direct mode"."""
        if remove:
            self.disable_router_mode()
            self.unlink_extension()
            return
        self.link_extension()
        if router:
            self.enable_router_mode(py)
        elif self.router_mode_active():
            print("! VS Code is still in ROUTER mode (gateway entries in its settings): open the ForemanKit panel and click "
                  "\"Stop (direct mode)\" - or run  python install.py --local-orchestrator --uninstall  - to talk to Anthropic directly")
        print("  ForemanKit panel installed: reload the VS Code window once and open the helmet icon in the sidebar. "
              "Claude Code talks straight to Anthropic (direct mode); switch to router mode in the panel to use the local models. "
              "The llama-server is not started automatically. Undo: python install.py --uninstall")

    def remove_agents(self) -> None:
        """Remove the foremankit agents; an agent file the user had before the installation comes back."""
        for name in AGENTS:
            path = CLAUDE_HOME / "agents" / name
            if path.exists():
                self.backup(path)
                self.say(f"remove {path}")
                if not self.dry:
                    path.unlink()
            self.restore_tracked(f"agents/{name}")

    @staticmethod
    def check_subagent_override() -> None:
        found = bool(os.environ.get("CLAUDE_CODE_SUBAGENT_MODEL"))
        settings = CLAUDE_HOME / "settings.json"
        if settings.exists():
            try:
                found |= "CLAUDE_CODE_SUBAGENT_MODEL" in json.loads(settings.read_text(encoding="utf-8")).get("env", {})
            except (ValueError, AttributeError):
                pass
        if found:
            print("! CLAUDE_CODE_SUBAGENT_MODEL is set: it overrides `model:` in subagents. Remove it.")


def yaml_load(path: Path) -> dict:
    """orchestrator.yaml needs PyYAML, which the venv has; fall back to defaults when run outside it."""
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (ImportError, OSError):
        return {}


def report_keys() -> None:
    print("\nKeys / URLs (set in env or in .env / ~/.config/foremankit/.env):")
    for key in KEYS:
        print(f"  {key:<26} {'set' if os.environ.get(key) else 'NOT SET'}")


def build_parser() -> argparse.ArgumentParser:
    """The command line of the installer: install, uninstall, restore, check."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="show actions only")
    ap.add_argument("--uninstall", action="store_true",
                    help="remove only what foremankit added; keeps later edits of your files")
    ap.add_argument("--stop-services", action="store_true",
                    help="with --uninstall: also stop the running gateway and llama-server of this folder")
    ap.add_argument("--purge", action="store_true",
                    help="with --uninstall: also delete the install snapshot (--restore-original is then impossible)")
    ap.add_argument("--check", action="store_true",
                    help="list the traces of foremankit that are still installed (exit code 1 when any)")
    ap.add_argument("--restore-original", action="store_true",
                    help="put CLAUDE.md, settings.json, the agent files and the VS Code files back byte for byte "
                         "as they were before the first install (later edits are lost)")
    ap.add_argument("--router", action="store_true",
                    help="with --local-orchestrator: also switch Claude Code to router mode (local models); default is direct mode")
    ap.add_argument("--with-llama", action="store_true",
                    help="with --local-orchestrator: download llama-server (llama/get_llama.py) when it is missing")
    ap.add_argument("--local-orchestrator", action="store_true",
                    help="also route the Claude Code plugin through the gateway (local Qwen as orchestrator, VS Code only)")
    ap.add_argument("--skip-claude", action="store_true", help="only venv + package, do not touch ~/.claude")
    return ap


def report_leftovers(inst: Installer) -> int:
    """Print what is still installed and return how many traces there are."""
    traces = inst.leftovers()
    for trace in traces:
        print(f"  left: {trace}")
    print("No trace of foremankit in Claude Code and VS Code." if not traces else f"{len(traces)} trace(s) left.")
    return len(traces)


def stop_services_safely(inst: Installer, args: argparse.Namespace) -> None:
    """Stop the gateway and llama-server only when nothing points Claude Code at the gateway any more.

    The order matters: the settings are cleaned first. A gateway stopped while ANTHROPIC_BASE_URL still
    targets it leaves the plugin without any model, the cloud included.
    """
    if not args.stop_services:
        return
    blockers = [] if args.dry_run else inst.gateway_references()
    if blockers:
        print("NOT stopping the gateway: Claude Code would lose the connection to the cloud because")
        for blocker in blockers:
            print(f"  still points at the gateway: {blocker}")
        print("Remove those entries (or restart VS Code after fixing them) and run --uninstall --stop-services again.")
        return
    if not inst.stop_services():
        print("No running foremankit gateway or llama-server of this folder.")


def run_uninstall(inst: Installer, args: argparse.Namespace) -> int:
    """Undo everything the installer did: services, agents, rules, MCP, hooks, VS Code, empty files."""
    py = inst.venv_python()
    inst.migrate_legacy()   # an install made under the old name "agentkit" is cleaned up as well
    inst.remove_agents()
    inst.install_rules(remove=True)
    inst.register_mcp(py, remove=True)
    inst.install_hooks(py, remove=True)
    inst.install_local_orchestrator(py, remove=True)
    inst.remove_empty_created_files()
    stop_services_safely(inst, args)
    if args.purge:
        inst.purge_state()
    if args.dry_run:
        print("\nDry run: nothing was changed.")
        return 0
    left = report_leftovers(inst)
    print("\nUninstalled from Claude Code and VS Code. Restart VS Code so the plugin drops the gateway settings. "
          "Left on purpose: the venv, the API keys in ~/.config/foremankit/.env and the usage log"
          + ("" if args.purge else "; --restore-original still works (snapshot kept, --purge deletes it)") + ".")
    if not args.stop_services:
        print("The gateway and llama-server, if running, keep running: add --stop-services to stop them.")
    return 1 if left else 0


def run_install(inst: Installer, args: argparse.Namespace) -> None:
    """Install: venv and package, then (unless skipped) the Claude Code parts, then the optional orchestrator."""
    py = inst.make_venv()
    inst.migrate_legacy()   # moves ~/.config/agentkit (API keys, state) and removes the old entries
    if not args.skip_claude or args.local_orchestrator:
        # Before the first change, so --restore-original has a baseline; VS Code files only when they will change.
        inst.take_snapshot(VSCODE_TRACKED if args.local_orchestrator else ())
    if not args.skip_claude:
        inst.install_agents()
        inst.install_rules()
        inst.register_mcp(py)
        inst.install_hooks(py)
        inst.check_subagent_override()
    if args.local_orchestrator:
        inst.install_local_orchestrator(py, router=args.router)
        inst.ensure_llama_server(args.with_llama)
    report_keys()
    print(f"\nNext: `{py} -m foremankit.cli status`  (or activate .venv and run `foremankit status`)")


def main() -> int:
    """Entry point: dispatch to check, restore, uninstall or install."""
    args = build_parser().parse_args()
    inst = Installer(args.dry_run)
    if args.check:
        return 1 if report_leftovers(inst) else 0
    if args.restore_original:
        inst.restore_original()
        print("\nRestored the files from before the first install (backups of the replaced files "
              "sit next to them). Run --uninstall too to unregister the MCP server.")
        return 0
    if args.uninstall:
        return run_uninstall(inst, args)
    run_install(inst, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
