# -*- coding: utf-8 -*-
"""
Utility helpers for Foremankit.

This module centralises small, pure‑Python helpers that are used across the
code‑base. Keeping them in a single place reduces duplication and makes the
behaviour easy to test and mock.

* ``ping_endpoint`` – performs a GET request with optional bearer token and a
  configurable retry/back‑off strategy. Returns ``True`` on HTTP 200 and
  ``False`` otherwise.
* ``HookResult`` – tiny data class that represents the decision made by a hook.
  It mirrors the JSON structure that the gateway expects (``decision`` and
  ``reason``). The ``as_json`` method serialises the object for printing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

__all__ = ["ping_endpoint", "HookResult"]


def ping_endpoint(url: str, token: str | None = None, *, retries: int = 3, backoff: float = 0.2) -> bool:
    """Ping a remote HTTP endpoint.

    In the night‑run flow we need to verify that the Cloudflare tunnel to Kaggle is
    still alive before we decide to continue a task.  The function performs a
    ``GET`` request with a short timeout (2 seconds).  If the request fails, it
    retries ``retries`` times, sleeping ``backoff`` seconds multiplied by the
    attempt index (exponential‑like growth but bounded).  The function never
    raises: any exception is caught and interpreted as *unreachable*.

    Parameters
    ----------
    url:
        Full URL to query – typically the upstream address stored in the
        ``FOREMANKIT_NODE_KAGGLE_LAN`` environment variable.
    token:
        Optional Bearer token to place in the ``Authorization`` header.
    retries:
        Number of additional attempts after the first failure.
    backoff:
        Base sleep time in seconds; the actual delay is ``backoff * (i + 1)``.

    Returns
    -------
    bool
        ``True`` if a **200** response is received, ``False`` otherwise.
    """
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries + 1):
        try:
            resp = httpx.get(url, headers=headers, timeout=2.0)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        if attempt < retries:
            time.sleep(backoff * (attempt + 1))
    return False


@dataclass(slots=True)
class HookResult:
    """Simple container for a hook decision.

    The gateway expects JSON with ``decision`` (always ``"block"`` for our
    hooks) and a ``reason`` string that explains why the model turn should be
    blocked.  ``HookResult`` provides a convenient ``as_json`` method that
    returns the dictionary ready for ``json.dumps``.
    """

    decision: str = "block"
    reason: str = ""
    payload: Mapping[str, Any] | None = None

    def as_json(self) -> dict[str, Any]:
        base = {"decision": self.decision, "reason": self.reason}
        if self.payload:
            base.update(self.payload)
        return base
