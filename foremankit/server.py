"""MCP stdio server exposing exactly two tools (short descriptions: they cost context in every session)."""
from __future__ import annotations

from functools import partial

import anyio

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:  # mcp 2.x renamed FastMCP -> MCPServer (same decorator API)
    from mcp.server.mcpserver import MCPServer as _Server

from .formatting import format_result, format_status
from .records import DEFAULT_MAX_OUTPUT_TOKENS
from .router import Router

mcp = _Server("foremankit")
# Plain text only: structured output would duplicate the result in Claude's context.
TOOL = {"structured_output": False}
_router: Router | None = None


def router() -> Router:
    """The one Router of this MCP server process, created on first use."""
    global _router
    if _router is None:
        _router = Router()
    return _router


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
    return format_result(await anyio.to_thread.run_sync(fn))


@mcp.tool(**TOOL)
async def router_status() -> str:
    """Active providers, remaining limits, spend today/month, local LLM servers."""
    return format_status(await anyio.to_thread.run_sync(router().status))


def main() -> None:
    """Start the MCP server (stdio)."""
    mcp.run()


if __name__ == "__main__":
    main()
