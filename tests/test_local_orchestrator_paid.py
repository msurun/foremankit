"""While the local orchestrator drives Claude Code, paid sources are refused unless the caller passes allow_cost."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from foremankit.records import CallContext

PRICE = {"input": 3.0, "output": 15.0}   # USD per 1M tokens, a paid model
RULE = "local orchestrator"


def _set_mode(monkeypatch, tmp_path: Path, mode: str) -> None:
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    (tmp_path / "orchestrator_mode").write_text(mode, encoding="utf-8")


def _cost_problem(router, allow_cost: bool = False, paid_flag: bool | None = None) -> str | None:
    if paid_flag is None:
        router.routing["budget"].pop("local_orchestrator_paid", None)
    else:
        router.routing["budget"]["local_orchestrator_paid"] = paid_flag
    ctx = CallContext(call_id="id", kind="quick", sensitivity="normal", escalate=False, est_in=10,
                      max_out=10, allow_cost=allow_cost)
    candidate = SimpleNamespace(price=PRICE, send_max=10, provider=None)
    return router.candidate_builder._cost_problem(candidate, ctx, {}, 0.0)


def test_local_mode_refuses_a_paid_source(monkeypatch, make_router, tmp_path):
    _set_mode(monkeypatch, tmp_path, "local")
    reason = _cost_problem(make_router(), paid_flag=False)
    assert reason and RULE in reason


def test_local_mode_with_allow_cost_is_not_refused_by_this_rule(monkeypatch, make_router, tmp_path):
    _set_mode(monkeypatch, tmp_path, "local")
    reason = _cost_problem(make_router(), allow_cost=True, paid_flag=False)
    assert not (reason and RULE in reason)


def test_claude_mode_is_not_refused_by_this_rule(monkeypatch, make_router, tmp_path):
    _set_mode(monkeypatch, tmp_path, "claude")
    reason = _cost_problem(make_router(), paid_flag=False)
    assert not (reason and RULE in reason)


def test_flag_true_is_not_refused_by_this_rule(monkeypatch, make_router, tmp_path):
    _set_mode(monkeypatch, tmp_path, "local")
    reason = _cost_problem(make_router(), paid_flag=True)
    assert not (reason and RULE in reason)


def test_missing_flag_is_not_refused_by_this_rule(monkeypatch, make_router, tmp_path):
    _set_mode(monkeypatch, tmp_path, "local")
    reason = _cost_problem(make_router(), paid_flag=None)
    assert not (reason and RULE in reason)
