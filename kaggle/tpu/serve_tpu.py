# Runs a batch of LLM jobs inside a Kaggle TPU notebook on Google's native serving stack
# (JetStream + MaxText, JAX). It is the TPU counterpart of kaggle/kernel/run_batch.py and
# writes the same results.jsonl contract, so downstream tools do not care which device ran it.
# The notebook is pushed headlessly: everything from probe to publish happens here, unattended.
"""Headless TPU batch runner: JetStream server plus JSONL job replay.

Reads jobs.jsonl, sends each job to a locally started JetStream server (MaxText backend),
and appends one result row per job to /kaggle/working/results.jsonl.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

JOBS_ENV = "FOREMANKIT_KAGGLE_JOBS"
MODEL_ENV = "FOREMANKIT_MAXTEXT_MODEL"
PARAMS_ENV = "FOREMANKIT_TPU_PARAMS"
PORT_ENV = "FOREMANKIT_JETSTREAM_PORT"
PEER_KEY_ENV = "FOREMANKIT_PEER_KEY"
MAX_RUN_HOURS_ENV = "FOREMANKIT_MAX_RUN_HOURS"
CTX_SIZE_ENV = "FOREMANKIT_CTX_SIZE"
JETSTREAM_ARGS_ENV = "FOREMANKIT_JETSTREAM_ARGS"
JETSTREAM_SPEC_ENV = "FOREMANKIT_JETSTREAM_SPEC"
MAXTEXT_SPEC_ENV = "FOREMANKIT_MAXTEXT_SPEC"
OUTPUT_DATASET_ENV = "FOREMANKIT_OUTPUT_DATASET"

JOBS_PATH = os.environ.get(JOBS_ENV, "")
MAXTEXT_MODEL = os.environ.get(MODEL_ENV, "gemma3-27b")
TPU_PARAMS = os.environ.get(PARAMS_ENV, "")
JETSTREAM_PORT = os.environ.get(PORT_ENV, "9000")
PEER_KEY = os.environ.get(PEER_KEY_ENV, "kaggle-local")
MAX_RUN_HOURS = os.environ.get(MAX_RUN_HOURS_ENV, "8.5")
CTX_SIZE = os.environ.get(CTX_SIZE_ENV, "32768")
JETSTREAM_ARGS = os.environ.get(JETSTREAM_ARGS_ENV, "")
OUTPUT_DATASET = os.environ.get(OUTPUT_DATASET_ENV, "")

# Must be pinned to a tested commit before real runs: "main" moves under us and a
# flag rename in JetStream breaks server_command silently.
JETSTREAM_SPEC = os.environ.get(JETSTREAM_SPEC_ENV, "git+https://github.com/google/JetStream.git@main")
MAXTEXT_SPEC = os.environ.get(
    MAXTEXT_SPEC_ENV,
    "git+https://github.com/AI-Hypercomputer/maxtext.git@main",
)

WORKING_DIR = Path("/kaggle/working")
RESULTS_PATH = WORKING_DIR / "results.jsonl"
KAGGLE_INPUT_DIR = Path("/kaggle/input")
DEFAULT_PROMPT = ""
DEFAULT_MAX_TOKENS = 256
TPU_PLATFORM = "tpu"

HEALTH_POLL_INTERVAL_S = 3
HEALTH_TIMEOUT_S = 900
HTTP_TIMEOUT_S = 300
SERVER_STOP_TIMEOUT_S = 30
SECONDS_PER_HOUR = 3600
INSTALL_ERROR_TAIL_LINES = 40


def log(message: str) -> None:
    """Print an info line tagged with the peer name and flushed immediately."""
    print(f"[peer] {message}", flush=True)


def warn(message: str) -> None:
    """Print a warning line tagged with the peer name to stderr."""
    print(f"[peer] {message}", file=sys.stderr, flush=True)


def probe_tpu() -> str:
    """Return a one-line summary of the visible JAX devices.

    Raises RuntimeError when no device reports platform "tpu", which almost always
    means the kernel was pushed without enable_tpu.
    """
    import jax

    devices = jax.devices()
    tpu_devices = [device for device in devices if device.platform == TPU_PLATFORM]
    if not tpu_devices:
        raise RuntimeError(
            "No TPU device is visible: the kernel was pushed without enable_tpu. "
            "Set accelerator 'tpu' in kernel-metadata.json and push again."
        )
    return f"{len(tpu_devices)} TPU device(s): {tpu_devices[0].device_kind}"


def install_stack() -> None:
    """Install JetStream and MaxText with pip, leaving jax/jaxlib untouched.

    The Kaggle TPU image ships a matched jax/jaxlib pair; reinstalling them breaks the
    libtpu binding, so only the two serving packages are requested.
    Raises RuntimeError with the tail of the install log when pip fails.
    """
    command = [sys.executable, "-m", "pip", "install", "--quiet", "--no-input",
               JETSTREAM_SPEC, MAXTEXT_SPEC]
    log(f"installing serving stack: {JETSTREAM_SPEC} {MAXTEXT_SPEC}")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        tail = (completed.stdout + completed.stderr).splitlines()[-INSTALL_ERROR_TAIL_LINES:]
        raise RuntimeError("pip install failed:\n" + "\n".join(tail))


def find_server_executable() -> Path:
    """Return the path of the JetStream server CLI.

    Raises RuntimeError when it is missing, usually because install_stack failed.
    """
    executable = shutil.which("jetstream_server")
    if executable is None:
        raise RuntimeError(
            "jetstream_server is not on PATH; check the install log above "
            f"and the pinned spec {JETSTREAM_SPEC}."
        )
    return Path(executable)


def find_checkpoint() -> Path:
    """Return the MaxText/Orbax checkpoint directory to serve.

    Uses FOREMANKIT_TPU_PARAMS when set, else the first directory under /kaggle/input
    holding a "checkpoint" file or a "params" subdirectory.
    Raises RuntimeError when nothing usable is found.
    """
    if TPU_PARAMS:
        path = Path(TPU_PARAMS)
        if not path.is_dir():
            raise RuntimeError(f"{PARAMS_ENV}={TPU_PARAMS} is not an existing directory.")
        return path
    for candidate in sorted(KAGGLE_INPUT_DIR.glob("*/*")) + sorted(KAGGLE_INPUT_DIR.glob("*/*/*")):
        if (candidate / "checkpoint").exists() or (candidate / "params").is_dir():
            return candidate
    raise RuntimeError(
        "No MaxText checkpoint found under /kaggle/input. Attach the Gemma MaxText "
        "checkpoint as a Kaggle dataset or set " + PARAMS_ENV + "."
    )


def server_command(executable: Path, params: Path) -> list[str]:
    """Build the JetStream server argv. The only place where CLI flags live.

    Flag names follow JetStream's MaxText server; a pinned JetStream version may rename
    them, so update this function when bumping JETSTREAM_SPEC.
    """
    command = [
        str(executable),
        "--jax_maxtext_model_name", MAXTEXT_MODEL,
        "--maxtext_config_path", str(params),
        "--port", JETSTREAM_PORT,
        "--host", "127.0.0.1",
        "--api_key", PEER_KEY,
    ]
    command.extend(shlex.split(JETSTREAM_ARGS))
    return command


def health_url() -> str:
    """Return the local JetStream health URL."""
    return f"http://127.0.0.1:{JETSTREAM_PORT}/health"


def completions_url() -> str:
    """Return the local JetStream generate URL."""
    return f"http://127.0.0.1:{JETSTREAM_PORT}/generate"


def redacted_command(command: list[str]) -> str:
    """Join `command` for the log, masking the value that follows --api_key."""
    shown = list(command)
    for index, part in enumerate(shown[:-1]):
        if part == "--api_key":
            shown[index + 1] = "***"
    return " ".join(shown)


def _probe_health() -> str:
    """Return "ok" when /health answers 200, else the last status or error as text."""
    try:
        with urllib.request.urlopen(health_url(), timeout=HEALTH_POLL_INTERVAL_S) as response:
            if response.status == 200:
                return "ok"
            return f"HTTP {response.status}"
    except (urllib.error.URLError, OSError) as error:
        return f"error: {error}"


def wait_until_healthy() -> None:
    """Poll /health until it answers ok. Raises RuntimeError after the timeout."""
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    status = "not contacted"
    while time.monotonic() < deadline:
        status = _probe_health()
        if status == "ok":
            return
        time.sleep(HEALTH_POLL_INTERVAL_S)
    raise RuntimeError(f"JetStream did not become healthy in {HEALTH_TIMEOUT_S}s: {status}")


def start_server(executable: Path, params: Path) -> subprocess.Popen:
    """Start the JetStream server and block until it is healthy.

    The argv is logged with the API key masked. Raises RuntimeError from wait_until_healthy.
    """
    command = server_command(executable, params)
    log("server argv: " + redacted_command(command))
    process = subprocess.Popen(command)
    wait_until_healthy()
    log("server healthy")
    return process


def concatenate_prompt(job: dict) -> str:
    """Join the optional system message and the prompt into one text blob.

    JetStream /generate takes plain text, so there is no structured role field; the
    single place where the prompt format can change.
    """
    system = job.get("system") or ""
    prompt = job.get("prompt") or DEFAULT_PROMPT
    return f"{system}\n\n{prompt}".strip() if system else prompt


def build_body(job: dict) -> dict:
    """Build the /generate request body for one job."""
    return {
        "prompt": concatenate_prompt(job),
        "max_tokens": int(job.get("max_tokens", DEFAULT_MAX_TOKENS)),
    }


def post_generation(body: dict) -> dict:
    """POST a generation request to JetStream and return the decoded JSON reply."""
    request = urllib.request.Request(
        completions_url(),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PEER_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def response_text(payload: dict) -> str:
    """Return the generated text from a JetStream reply.

    Handles {"response": <str>} and {"response": [<str>, ...]}. Raises KeyError with
    the payload keys when neither shape is present.
    """
    if "response" not in payload:
        raise KeyError(f"no 'response' in payload, keys are {sorted(payload)}")
    response = payload["response"]
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        return "".join(str(part) for part in response)
    raise KeyError(f"'response' has type {type(response).__name__}, keys are {sorted(payload)}")


def token_counts(payload: dict) -> tuple[int, int]:
    """Return (tokens_in, tokens_out) from usage-like fields, (0, 0) when absent.

    Never derives a count from string length; an absent count is honestly zero.
    """
    usage = payload.get("usage") or {}
    tokens_in = usage.get("prompt_tokens") or payload.get("num_prompt_tokens") or 0
    tokens_out = usage.get("completion_tokens") or payload.get("num_generated_tokens") or 0
    return int(tokens_in), int(tokens_out)


def success_result(job: dict, payload: dict, wall_s: float) -> dict:
    """Build the success row written for one job."""
    tokens_in, tokens_out = token_counts(payload)
    return {
        "id": job["id"],
        "kind": job.get("kind", ""),
        "success": True,
        "response": response_text(payload),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_s": wall_s,
    }


def failure_result(job: dict, error: str, wall_s: float) -> dict:
    """Build the failure row written for one job."""
    return {
        "id": job["id"],
        "kind": job.get("kind", ""),
        "success": False,
        "error": error,
        "wall_s": wall_s,
    }


def run_job(job: dict) -> dict:
    """Run one job against the server and return a result row, never raising."""
    started_at = time.monotonic()
    try:
        payload = post_generation(build_body(job))
        return success_result(job, payload, round(time.monotonic() - started_at, 3))
    except (urllib.error.URLError, OSError, ValueError, TypeError, KeyError, IndexError) as error:
        warn(f"job {job.get('id')} failed: {error}")
        return failure_result(job, str(error), round(time.monotonic() - started_at, 3))


def parse_job_line(raw_line: str, source: str, line_number: int) -> dict | None:
    """Parse one JSONL line into a job dict, or return None after warning.

    Skips blank lines, invalid JSON, non-object values and objects without "id".
    """
    if not raw_line.strip():
        return None
    try:
        job = json.loads(raw_line)
    except json.JSONDecodeError as error:
        warn(f"{source}:{line_number} skipped, invalid JSON: {error}")
        return None
    if not isinstance(job, dict) or "id" not in job:
        warn(f"{source}:{line_number} skipped, missing id or not an object")
        return None
    return job


def read_jobs_file(path: Path) -> list[dict]:
    """Read a JSONL jobs file, skipping bad lines with a warning."""
    jobs: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            job = parse_job_line(raw_line, str(path), line_number)
            if job is not None:
                jobs.append(job)
    return jobs


def jobs_file_path() -> Path:
    """Return the jobs file path from the env, else the first under /kaggle/input.

    Raises RuntimeError when nothing is found.
    """
    if JOBS_PATH:
        path = Path(JOBS_PATH)
        if not path.is_file():
            raise RuntimeError(f"{JOBS_ENV}={JOBS_PATH} is not an existing file.")
        return path
    candidates = sorted(KAGGLE_INPUT_DIR.glob("*/jobs.jsonl")) + sorted(
        KAGGLE_INPUT_DIR.glob("*/*/jobs.jsonl")
    )
    if not candidates:
        raise RuntimeError(
            "No jobs.jsonl found under /kaggle/input. Set " + JOBS_ENV + " or attach a dataset."
        )
    return candidates[0]


def load_jobs() -> list[dict]:
    """Locate and read the jobs file."""
    return read_jobs_file(jobs_file_path())


def start_results_file() -> None:
    """Truncate results.jsonl so a re-run never mixes rows with an older run."""
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8"):
        pass


def append_result(result: dict) -> None:
    """Append one result row to results.jsonl, flushed and fsync'd to disk."""
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def stop_server(process: subprocess.Popen) -> None:
    """Terminate the server, escalating to kill after the timeout."""
    process.terminate()
    try:
        process.wait(timeout=SERVER_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        warn(f"server did not stop in {SERVER_STOP_TIMEOUT_S}s, killing it")
        process.kill()
        process.wait()


def run_jobs_within_budget(jobs: list[dict], max_seconds: float) -> tuple[int, int]:
    """Run jobs until done or the budget is spent. Returns (ok, failed)."""
    deadline = time.monotonic() + max_seconds
    ok = 0
    failed = 0
    warned = False
    for job in jobs:
        if time.monotonic() >= deadline:
            if not warned:
                warn(f"budget of {max_seconds}s spent, skipping remaining jobs")
                warned = True
            break
        result = run_job(job)
        append_result(result)
        ok += 1 if result["success"] else 0
        failed += 0 if result["success"] else 1
    return ok, failed


def _result_row(line: str) -> dict | None:
    """Parse one results.jsonl line, returning None when it is blank or invalid."""
    if not line.strip():
        return None
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    return row if isinstance(row, dict) else None


def report_throughput() -> None:
    """Print the Stage-0 throughput line: total tokens over total wall time.

    This measurement decides whether the live-steering mode is worth building at all.
    """
    tokens_in = 0
    tokens_out = 0
    wall_s = 0.0
    with RESULTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = _result_row(line)
            if row is None:
                continue
            tokens_in += int(row.get("tokens_in", 0))
            tokens_out += int(row.get("tokens_out", 0))
            wall_s += float(row.get("wall_s", 0.0))
    total_tokens = tokens_in + tokens_out
    rate = total_tokens / wall_s if wall_s > 0 else 0.0
    print(
        f"throughput: {total_tokens} tokens in {wall_s:.2f}s = {rate:.2f} tok/s "
        f"({tokens_in} in, {tokens_out} out)",
        flush=True,
    )


def kaggle_cli_available() -> bool:
    """Return True when the kaggle CLI and KAGGLE_KEY are both present."""
    return shutil.which("kaggle") is not None and bool(os.environ.get("KAGGLE_KEY"))


def _ensure_dataset_metadata(dataset_dir: Path) -> None:
    """Write a minimal dataset-metadata.json when the output directory has none.

    Kaggle requires the file for a versioned push of a dataset the kernel owns.
    """
    metadata_path = dataset_dir / "dataset-metadata.json"
    if metadata_path.exists():
        return
    metadata = {"title": dataset_dir.name, "id": OUTPUT_DATASET, "licenses": []}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def publish_dataset() -> None:
    """Push results.jsonl as a new version of the output dataset.

    Raises RuntimeError when the kaggle CLI push fails.
    """
    dataset_dir = WORKING_DIR / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RESULTS_PATH, dataset_dir / RESULTS_PATH.name)
    _ensure_dataset_metadata(dataset_dir)
    completed = subprocess.run(
        ["kaggle", "datasets", "version", "-p", str(dataset_dir), "-m", "batch results"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("kaggle datasets version failed:\n" + completed.stderr.strip())


def maybe_publish_dataset() -> None:
    """Publish the results dataset when configured, warning instead of failing."""
    if not OUTPUT_DATASET:
        return
    if not kaggle_cli_available():
        warn("publish skipped: kaggle CLI or KAGGLE_KEY missing")
        return
    try:
        publish_dataset()
        log(f"published dataset {OUTPUT_DATASET}")
    except (RuntimeError, OSError, urllib.error.URLError) as error:
        warn(f"publish failed: {error}")


def run_jobs_phase() -> tuple[int, int, float]:
    """Replay every job within the run budget. Returns (ok, failed, elapsed seconds)."""
    started_at = time.monotonic()
    jobs = load_jobs()
    log(f"{len(jobs)} job(s) loaded")
    start_results_file()
    max_seconds = float(MAX_RUN_HOURS) * SECONDS_PER_HOUR
    ok, failed = run_jobs_within_budget(jobs, max_seconds)
    return ok, failed, round(time.monotonic() - started_at, 2)


def print_summary(ok: int, failed: int, elapsed_s: float) -> None:
    """Print the closing summary and the fetch hint for results.jsonl."""
    print(f"Total {ok + failed} | ok {ok} | failed {failed} | elapsed {elapsed_s}s", flush=True)
    print(
        "results: "
        f"{RESULTS_PATH} (fetch with: kaggle kernels output <owner>/<slug> -p ./out)",
        flush=True,
    )


def main() -> int:
    """Probe the TPU, install the stack, serve, replay jobs, publish. Returns exit code 0."""
    log(probe_tpu())
    install_stack()
    executable = find_server_executable()
    params = find_checkpoint()
    log(f"model {MAXTEXT_MODEL} from {params}, ctx {CTX_SIZE}")
    server = start_server(executable, params)
    try:
        ok, failed, elapsed_s = run_jobs_phase()
    finally:
        stop_server(server)
    print_summary(ok, failed, elapsed_s)
    report_throughput()
    maybe_publish_dataset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
