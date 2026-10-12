"""End-to-end over stdio: spawn the real MCP server (dry run) and call both tools."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import anyio
import pytest

mcp = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_stdio_server_exposes_two_tools(tmp_path):
    # Regression: a project folder named `foremankit/` in the cwd must not shadow the package.
    (tmp_path / "foremankit").mkdir()
    scripts = Path(sys.executable).parent
    exe = next((p for p in (scripts / "foremankit-mcp.exe", scripts / "foremankit-mcp") if p.exists()), None)
    if exe is None:
        pytest.skip("foremankit-mcp script not installed")
    env = {**os.environ, "FOREMANKIT_DRY_RUN": "1", "FOREMANKIT_STATE_DIR": str(tmp_path / "state"),
           "FOREMANKIT_OUT_DIR": str(tmp_path / "out"), "LLM_LOCAL_ENABLED": "on"}
    params = StdioServerParameters(command=str(exe), args=[], env=env, cwd=str(tmp_path))

    async def run():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            res = await session.call_tool("delegate", {"task": "hi", "kind": "quick", "sensitivity": "private"})
            status = await session.call_tool("router_status", {})
            return names, res, status

    names, res, status = anyio.run(run)
    assert names == ["delegate", "router_status"]
    assert len(res.content) == 1 and "DRY RUN" in res.content[0].text and "file:" in res.content[0].text
    assert "providers:" in status.content[0].text
