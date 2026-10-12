"""Learn how fast a node really is from the calls it served.

`speed` in nodes.yaml is what we EXPECT from a node. A node that is not ours (someone else's laptop, a shared server) has
a speed that floats with its load and its network, so the router keeps a moving average of what the node actually
delivered and blends it with the expected value. A node whose speed was measured carefully can be pinned with
`learn: false` and then always uses its written `speed`.

Numbers are tokens per second for reading the prompt (`prompt_tps`) and for writing the answer (`gen_tps`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SPEED_KEYS = ("prompt_tps", "gen_tps")
DEFAULT_SETTINGS = {
    "enabled": True,        # learn at all
    "alpha": 0.3,           # weight of the newest measurement in the moving average
    "min_samples": 3,       # measurements needed before the learned value fully replaces the written one
    "max_age_h": 72,        # a node not used for this long falls back to its written speed
}
MIN_GENERATED_TOKENS = 16       # shorter answers say nothing about generation speed
MIN_LATENCY_ANSWER_TOKENS = 64  # without server timings a rough estimate needs a longer answer
MIN_PROMPT_TOKENS = 200         # a short prompt is read too fast to time
MIN_GENERATION_SHARE = 0.25     # at least this share of the latency is attributed to writing the answer
SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class Observation:
    """What one finished call said about a node's speed."""
    node: str
    measured: dict[str, float]
    now: float


@dataclass(frozen=True)
class DeclaredSpeed:
    """The speed written for a node, and whether the router may learn a better value."""
    speed: dict[str, Any] | None
    node: str
    learn: bool = True


def settings(nodes_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """The `settings.speed_learning` block of nodes.yaml on top of the defaults."""
    given = ((nodes_cfg or {}).get("settings") or {}).get("speed_learning") or {}
    return {**DEFAULT_SETTINGS, **given}


def measure(result: dict[str, Any], latency_s: float, assumed_prompt_tps: float) -> dict[str, float]:
    """What one finished call says about the node's speed: {"gen_tps": ..., "prompt_tps": ...} (keys only when known).
    A llama-server reports exact `timings` in its answer; otherwise generation speed is estimated from the latency
    after subtracting the time the prompt should have taken at `assumed_prompt_tps`."""
    timings = result.get("timings") or {}
    measured: dict[str, float] = {}
    if timings:
        if timings.get("predicted_n", 0) >= MIN_GENERATED_TOKENS and timings.get("predicted_per_second"):
            measured["gen_tps"] = float(timings["predicted_per_second"])
        if timings.get("prompt_n", 0) >= MIN_PROMPT_TOKENS and timings.get("prompt_per_second"):
            measured["prompt_tps"] = float(timings["prompt_per_second"])
        return measured
    usage = result.get("usage") or {}
    answer_tokens = usage.get("completion_tokens", 0)
    if answer_tokens >= MIN_LATENCY_ANSWER_TOKENS and latency_s > 0:
        prompt_time = usage.get("prompt_tokens", 0) / max(assumed_prompt_tps, 1.0)
        generation_time = max(latency_s - prompt_time, MIN_GENERATION_SHARE * latency_s)
        measured["gen_tps"] = answer_tokens / generation_time
    return measured


def observe(state: dict[str, Any], observation: Observation, alpha: float) -> None:
    """Fold a measurement into the node's moving average kept in the state file."""
    entry = state.setdefault("node_speed", {}).setdefault(observation.node, {})
    for key in SPEED_KEYS:
        value = observation.measured.get(key, 0.0)
        if value <= 0:
            continue
        previous = entry.get(key)
        entry[key] = round(value if previous is None else alpha * value + (1 - alpha) * previous, 3)
        entry[f"n_{key}"] = entry.get(f"n_{key}", 0) + 1
    entry["ts"] = observation.now


def effective(declared: DeclaredSpeed, state: dict[str, Any], config: dict[str, Any], now: float) -> dict[str, Any] | None:
    """The speed the router should plan with: the written `speed`, moved towards what the node delivered lately.
    The learned value takes over gradually: with n of `min_samples` measurements it has weight n / min_samples."""
    if not declared.learn or not config["enabled"]:
        return declared.speed
    entry = (state.get("node_speed") or {}).get(declared.node)
    if not entry or now - entry.get("ts", 0) > float(config["max_age_h"]) * SECONDS_PER_HOUR:
        return declared.speed
    blended = dict(declared.speed or {})
    for key in SPEED_KEYS:
        if key in entry:
            blended[key] = _blend(blended.get(key), entry, key, int(config["min_samples"]))
    return blended


def _blend(written: Any, entry: dict[str, Any], key: str, min_samples: int) -> float:
    """Written value moved towards the learned one by n / min_samples (full weight from min_samples on)."""
    weight = min(1.0, entry.get(f"n_{key}", 0) / max(1, min_samples))
    return entry[key] if written is None else float(written) * (1 - weight) + entry[key] * weight
