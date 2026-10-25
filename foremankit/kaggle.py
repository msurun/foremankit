"""Kaggle free-tier budget: hours left this week and what a job queue will cost in them.

Kaggle grants about 30 GPU-hours and a separate ~20 TPU-hours per week, and one session dies at ~9 h.
The GPU tier runs llama.cpp (Qwen and any GGUF); the TPU tier runs Gemma natively on JetStream+MaxText.
Finished sessions go to a JSONL ledger in the state directory, so `remaining` survives restarts.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import ConfigError, state_dir

WEEKLY_HOURS = {"gpu": 30.0, "tpu": 20.0}   # verify in the Kaggle UI; overridable constants
SESSION_CAP_HOURS = 9.0
# Boot and warm-up before the first token, billed as wall-clock: the TPU lane also builds
# the serving stack and loads a dense 27B checkpoint before it can answer anything.
OVERHEAD_HOURS = {"gpu": 0.25, "tpu": 0.75}
SESSIONS_FILE = "kaggle_sessions.jsonl"
METADATA_FILE = "kernel-metadata.json"
KERNEL_SLUG = "foremankit-batch"
PLACEHOLDER_OWNER = "REPLACE_WITH"
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_WEEK = 7 * 24 * 3600.0
# Placeholders until the Stage-0 measurement lands; TPU is JetStream+MaxText with bf16 weights.
DEFAULT_SPEED = {
    "gpu": {"prompt_tps": 150.0, "gen_tps": 25.0},
    "tpu": {"prompt_tps": 900.0, "gen_tps": 60.0},
}


def quota(accelerator: str) -> float:
    """Weekly hours for `accelerator` (gpu|tpu); raises ValueError on an unknown name."""
    try:
        return WEEKLY_HOURS[accelerator]
    except KeyError:
        raise ValueError(f"unknown accelerator {accelerator!r}; use gpu or tpu") from None


def week_start(now: float | None = None) -> float:
    """Epoch seconds of the Monday 00:00 UTC that starts the week containing `now`."""
    moment = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc)
    midnight = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp() - midnight.weekday() * 86400.0


def sessions_path() -> Path:
    """The session ledger (JSONL) inside the state directory."""
    return state_dir() / SESSIONS_FILE


def read_sessions() -> list[dict[str, Any]]:
    """Every usable ledger record; malformed lines are skipped and a missing file reads as empty."""
    path = sessions_path()
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = _parse_record(line)
        if record is not None:
            records.append(record)
    return records


def _parse_record(line: str) -> dict[str, Any] | None:
    """One ledger line as a dict, or None when it is blank, broken or lacks a numeric `started`."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or not isinstance(record.get("started"), (int, float)):
        return None
    return record


def log_session(seconds: dict[str, float] | None = None, *, jobs_done: int = 0,
                note: str = "", now: float | None = None) -> dict[str, Any]:
    """Append one finished session to the ledger and return the record as written.

    `seconds` maps an accelerator name ("gpu", "tpu") to the session time to charge.
    """
    moment = time.time() if now is None else now
    record: dict[str, Any] = {"started": moment}
    for accelerator in WEEKLY_HOURS:
        record[f"{accelerator}_seconds"] = float((seconds or {}).get(accelerator, 0.0))
    record["jobs_done"] = int(jobs_done)
    record["note"] = note
    path = sessions_path()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def week_used(accelerator: str = "gpu", now: float | None = None) -> float:
    """Seconds of `accelerator` logged in the week that contains `now`."""
    key = f"{accelerator}_seconds"
    start = week_start(now)
    return sum(float(record.get(key, 0.0)) for record in read_sessions()
               if start <= record["started"] < start + SECONDS_PER_WEEK)


def remaining(accelerator: str = "gpu", now: float | None = None) -> float:
    """Hours of `accelerator` still available this week, never negative."""
    used_hours = week_used(accelerator, now) / SECONDS_PER_HOUR
    return max(0.0, quota(accelerator) - used_hours)


def estimate_hours(jobs: Iterable[dict[str, Any]], speed: dict[str, float] | None = None,
                   accelerator: str = "gpu") -> float:
    """Hours a non-empty `jobs` list will take on `accelerator` at `speed`, prefill and decode apart.

    Each job may carry tokens_in|prompt_tokens and tokens_out|max_tokens; a job without
    token counts adds nothing, so pass the planned counts, not guesses. Kaggle charges
    session wall-clock time, so the accelerator's fixed start-up overhead is included once.
    """
    quota(accelerator)   # a typo in the name must fail loudly, not fall back to the GPU rates
    rate = {**DEFAULT_SPEED[accelerator], **(speed or {})}
    seconds = 0.0
    for job in jobs:
        seconds += _tokens(job, "tokens_in", "prompt_tokens") / rate["prompt_tps"]
        seconds += _tokens(job, "tokens_out", "max_tokens") / rate["gen_tps"]
    if seconds <= 0:
        return 0.0
    return OVERHEAD_HOURS[accelerator] + seconds / SECONDS_PER_HOUR


def _tokens(job: dict[str, Any], *keys: str) -> float:
    """The first positive token count among `keys` in `job`, else 0."""
    for key in keys:
        value = job.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return 0.0


def sessions_needed(hours: float) -> int:
    """How many <=9 h Kaggle sessions `hours` of work fits into (0 for no work)."""
    if hours <= 0:
        return 0
    return int(-(-hours // SESSION_CAP_HOURS))   # ceiling division


def fits(accelerator: str, hours: float, now: float | None = None) -> bool:
    """Say whether `hours` of work still fit in this week's `accelerator` budget."""
    return hours <= remaining(accelerator, now)


def _read_metadata(directory: Path) -> tuple[Path, dict[str, Any]]:
    """Read kernel-metadata.json from `directory`, returning its path and parsed content."""
    path = directory / METADATA_FILE
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{path} is missing; keep the kernel folder intact") from None
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path} is not valid JSON: {error}") from None


def accelerator_of(directory: Path) -> str:
    """The free tier a kernel folder asks for: "tpu" when enable_tpu is set, else "gpu"."""
    _, metadata = _read_metadata(directory)
    return "tpu" if metadata.get("enable_tpu") else "gpu"


def prepare_metadata(directory: Path, datasets: list[str] | None = None, user: str = "") -> dict[str, Any]:
    """Rewrite kernel-metadata.json with the real owner and attached datasets, returning it.

    The placeholder owner from the template is replaced only when `user` is given; the
    datasets listed here become the `/kaggle/input` folders the notebook reads.
    """
    path, metadata = _read_metadata(directory)
    owner, _, slug = str(metadata.get("id", "")).partition("/")
    if owner.startswith(PLACEHOLDER_OWNER) and user:
        metadata["id"] = f"{user}/{slug or KERNEL_SLUG}"
    if datasets:
        metadata["dataset_sources"] = list(dict.fromkeys(datasets))
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def require_credentials() -> str:
    """Return the Kaggle username from the environment, refusing to push without credentials.

    The key is read only to prove it is set; it never reaches the repo or a log line.
    """
    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if not user or not key:
        raise ConfigError(
            "set KAGGLE_USERNAME and KAGGLE_KEY in the environment first "
            "(kaggle.com -> Settings -> Create New Token); never commit them"
        )
    return user


def push(directory: Path, datasets: list[str] | None = None) -> str:
    """Push the kernel folder with the kaggle CLI and return its output line for the user."""
    metadata = prepare_metadata(directory, datasets, require_credentials())
    try:
        result = subprocess.run(["kaggle", "kernels", "push", "-p", str(directory)],
                                capture_output=True, text=True)
    except FileNotFoundError:
        raise ConfigError("the kaggle CLI is not installed: pip install kaggle") from None
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ConfigError(f"kaggle kernels push failed: {detail}")
    return f"{metadata['id']}: {result.stdout.strip() or 'pushed'}"
