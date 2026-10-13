"""MCP stdio server exposing exactly two tools (short descriptions: they cost context in every session)."""
from __future__ import annotations

import traceback
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

import anyio

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:  # mcp 2.x renamed FastMCP -> MCPServer (same decorator API)
    from mcp.server.mcpserver import MCPServer as _Server

from .config import state_dir
from .formatting import format_result, format_status
from .records import DEFAULT_MAX_OUTPUT_TOKENS
from .router import Router

mcp = _Server("foremankit")
# Plain text only: structured output would duplicate the result in Claude's context.
TOOL = {"structured_output": False}
_router: Router | None = None
ERROR_LOG_NAME = "mcp-errors.log"


def router() -> Router:
    """The one Router of this MCP server process, created on first use."""
    global _router
    if _router is None:
        _router = Router()
    return _router


def error_log_path() -> Path:
    """File that receives the traceback of every failed tool call."""
    return state_dir() / ERROR_LOG_NAME


def _log_failure(tool_name: str) -> Path:
    """Append the current exception's traceback to the error log; returns the log path."""
    path = error_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as log:
            log.write(f"--- {datetime.now().isoformat(timespec='seconds')} {tool_name}\n{traceback.format_exc()}\n")
    except OSError:
        pass  # a full disk must not hide the original error from the caller
    return path


async def _run_guarded(tool_name: str, work: Callable[[], str]) -> str:
    """Run `work` in a thread; on any error return a readable message instead of the bare
    'Error executing tool' that FastMCP produces, and keep the traceback in the log."""
    try:
        return await anyio.to_thread.run_sync(work)
    except Exception as error:  # noqa: BLE001 - the MCP boundary must report every failure as text
        log_path = _log_failure(tool_name)
        return f"ERROR in {tool_name}: {type(error).__name__}: {error} (traceback: {log_path})"


# The 11 parameters mirror the MCP tool schema, so the 4-parameter limit does not apply here.
@mcp.tool(**TOOL)
# codecheck: allow too-many-parameters (every option is a field of the MCP tool schema that clients fill in)
async def delegate(task: str, kind: str, sensitivity: str = "normal", context_files: list[str] | None = None,
                   system_hint: str = "", max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, escalate: bool = False,
                   allow_cost: bool = False, verify: str = "", verify_target: str = "", review: bool = False) -> str:
    """Run task on a cheap model. kind: bulk_code|data_script|hard_code|long_context|quick|summarize.
    sensitivity: normal | internal (process data: paid no-train Gemini or the local LLM) | private (local LLM only).
    Result saved to .agent-out/; returns path + preview. Keep the answer under ~4000 tokens (one file/part per call).
    verify + verify_target: write the answer into that file and run the check (e.g. "python -m pytest tests/x.py -q");
    failures go back to the same model; the file stays changed only if the check passes.
    allow_cost: accept a paid call above the per-call cost cap.
    review: a different model (own local LLM or a paid one) reviews the finished code; grounded blockers go back to the author once."""
    fn = partial(router().delegate, task, kind, sensitivity, context_files=context_files or [],
                 system_hint=system_hint, max_output_tokens=max_output_tokens, escalate=escalate,
                 allow_cost=allow_cost, verify=verify, verify_target=verify_target, review=review)
    return await _run_guarded("delegate", lambda: format_result(fn()))


@mcp.tool(**TOOL)
async def router_status() -> str:
    """Active providers, remaining limits, spend today/month, local LLM servers."""
    return await _run_guarded("router_status", lambda: format_status(router().status()))


def main() -> None:
    """Start the MCP server (stdio)."""
    mcp.run()


if __name__ == "__main__":
    main()
