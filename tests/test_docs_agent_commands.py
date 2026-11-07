"""Keep all registered agent commands visible in the maintained guides.
Derive command names from the parser so additions cannot silently miss documentation.
"""
from pathlib import Path

import pytest

from foremankit.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
PARSER = build_parser()
SUBCOMMANDS = next(action for action in PARSER._actions if action.dest == "cmd").choices
AGENT_COMMANDS = tuple(name for name in SUBCOMMANDS if name.startswith("agent-"))


@pytest.mark.parametrize("command", AGENT_COMMANDS)
@pytest.mark.parametrize("document", ["README.md", "README.pl.md", "claude/foremankit-reference.md"])
def test_every_agent_command_is_documented(command: str, document: str) -> None:
    """Each user guide includes every registered agent command."""
    assert command in (ROOT / document).read_text(encoding="utf-8")


@pytest.mark.parametrize("command", AGENT_COMMANDS)
def test_short_agent_guidance_names_each_command(command: str) -> None:
    """At least one injected guide names each command."""
    guidance = (ROOT / "config/orchestrator.md").read_text(encoding="utf-8")
    guidance += (ROOT / "claude/CLAUDE.global.md").read_text(encoding="utf-8")
    assert command in guidance


@pytest.mark.parametrize("command", AGENT_COMMANDS)
@pytest.mark.parametrize("document", ["README.md", "README.pl.md"])
def test_readmes_document_every_agent_flag(command: str, document: str) -> None:
    """The command syntax covers every option supported by the actual parser."""
    options = {option for action in SUBCOMMANDS[command]._actions
               for option in action.option_strings if option not in ("-h", "--help")}
    text = (ROOT / document).read_text(encoding="utf-8")
    assert all(option in text for option in options)
