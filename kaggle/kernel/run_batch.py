"""Runs inside a Kaggle GPU notebook: serve a GGUF model on localhost, replay a job queue, exit.

The notebook is pushed headlessly with `kaggle kernels push` (no inbound tunnel), so the whole
run is self-contained: locate the model, ensure a llama-server binary, start it, run every job
from the queue, write results.jsonl, then kill the server.

Note: a dense 27B model exhausts 32 GB of T4 memory fast; pass KV-cache quantisation via
FOREMANKIT_LLAMA_ARGS (for example `--cache-type-k q8_0 --cache-type-v q8_0`) rather than
relying on a hard-coded flag here.
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
import zipfile
from pathlib import Path

WORKING_DIR = Path("/kaggle/working")
KAGGLE_INPUT_DIR = Path("/kaggle/input")
RESULTS_PATH = WORKING_DIR / "results.jsonl"
LOAD_TIMEOUT_S = 600
HEALTH_POLL_INTERVAL_S = 3
SECONDS_PER_HOUR = 3600
JOB_HTTP_TIMEOUT_S = 300
SERVER_STOP_TIMEOUT_S = 30
DEFAULT_MAX_TOKENS = 256
LLAMA_SERVER_NAME = "llama-server"
RELEASE_URL_TEMPLATE = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
)
# Vulkan first: it uses the T4 through the driver Kaggle already ships; the CPU zip is only a fallback.
RELEASE_ASSETS = ("bin-ubuntu-vulkan-x64", "bin-ubuntu-x64")
PEER_KEY = os.environ.get("FOREMANKIT_PEER_KEY", "kaggle-local")
PORT = int(os.environ.get("FOREMANKIT_PORT", os.environ.get("PORT", "8080")))
CTX_SIZE = int(os.environ.get("FOREMANKIT_CTX_SIZE", "32768"))
MAX_RUN_HOURS = float(os.environ.get("FOREMANKIT_MAX_RUN_HOURS", "8.5"))
MODEL_GGUF = os.environ.get("FOREMANKIT_MODEL_GGUF", "")
JOBS_PATH = os.environ.get("FOREMANKIT_KAGGLE_JOBS", "")
OUTPUT_DATASET = os.environ.get("FOREMANKIT_OUTPUT_DATASET", "")
LLAMACPP_VERSION = os.environ.get("FOREMANKIT_LLAMACPP_VERSION", "b4567")
LLAMA_ARGS = os.environ.get("FOREMANKIT_LLAMA_ARGS", "")


def log(message: str) -> None:
    """Print a peer-tagged progress line, flushing so notebook logs appear live."""
    print(f"[peer] {message}", flush=True)


def warn(message: str) -> None:
    """Print a peer-tagged warning line to stderr, flushing so notebook logs appear live."""
    print(f"[peer] WARN {message}", file=sys.stderr, flush=True)


def locate_model() -> Path:
    """Return the GGUF model path from the env var or the first one found under /kaggle/input."""
    if MODEL_GGUF:
        model_path = Path(MODEL_GGUF)
        if not model_path.is_file():
            raise RuntimeError(
                f"FOREMANKIT_MODEL_GGUF points to {model_path}, which is not a file. "
                "Fix the path or attach the model as a Kaggle dataset."
            )
        return model_path
    for candidate in KAGGLE_INPUT_DIR.rglob("*.gguf"):
        return candidate
    raise RuntimeError(
        "no *.gguf found under /kaggle/input. Attach the model as a Kaggle dataset "
        "or set FOREMANKIT_MODEL_GGUF to its path."
    )


def find_existing_server() -> Path | None:
    """Return an executable llama-server already present under WORKING_DIR, else None."""
    for candidate in WORKING_DIR.rglob(LLAMA_SERVER_NAME):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def download_release_server() -> Path:
    """Download the first pinned release asset that works and return its llama-server."""
    failures = []
    for asset in RELEASE_ASSETS:
        name = f"llama-{LLAMACPP_VERSION}-{asset}.zip"
        try:
            return _fetch_release(name)
        except (urllib.error.URLError, OSError, zipfile.BadZipFile, RuntimeError) as error:
            warn(f"{name} unusable: {error}")
            failures.append(str(error))
    raise RuntimeError(f"no usable release asset for {LLAMACPP_VERSION}: {'; '.join(failures)}")


def _fetch_release(name: str) -> Path:
    """Download one release zip, extract it and return the llama-server binary inside."""
    archive_path = WORKING_DIR / name
    log(f"downloading llama.cpp release {name}")
    url = f"{RELEASE_URL_TEMPLATE}{LLAMACPP_VERSION}/{name}"
    with urllib.request.urlopen(url, timeout=JOB_HTTP_TIMEOUT_S) as response:
        archive_path.write_bytes(response.read())
    release_dir = WORKING_DIR / "llamacpp_release"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(release_dir)
    for candidate in release_dir.rglob(LLAMA_SERVER_NAME):
        candidate.chmod(candidate.stat().st_mode | 0o111)
        return candidate
    raise RuntimeError(f"{name} did not contain {LLAMA_SERVER_NAME}")


def clone_llama_cpp() -> Path:
    """Clone llama.cpp at the pinned tag, falling back to the default branch if the tag is gone."""
    source_dir = WORKING_DIR / "llama.cpp"
    log(f"cloning llama.cpp at {LLAMACPP_VERSION}")
    tagged = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", LLAMACPP_VERSION,
         "https://github.com/ggml-org/llama.cpp.git", str(source_dir)],
        capture_output=True,
        text=True,
    )
    if tagged.returncode != 0:
        warn(f"tag {LLAMACPP_VERSION} clone failed; retrying default branch")
        shutil.rmtree(source_dir, ignore_errors=True)
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp.git", str(source_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    return source_dir


def build_from_source() -> Path:
    """Clone and build llama.cpp with CUDA enabled, returning the built llama-server path."""
    source_dir = clone_llama_cpp()
    build_dir = source_dir / "build"
    _run_cmake(build_dir, source_dir)
    for candidate in build_dir.rglob(LLAMA_SERVER_NAME):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("build succeeded but llama-server binary was not found")


def _run_cmake(build_dir: Path, source_dir: Path) -> None:
    """Configure and build llama.cpp with CUDA, raising on a non-zero cmake exit."""
    subprocess.run(
        ["cmake", "-B", str(build_dir), "-DGGML_CUDA=ON"],
        cwd=source_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release", "-j"],
        cwd=source_dir,
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_llama_server() -> Path:
    """Return a llama-server path, trying reuse, release download, then source build in turn."""
    strategies = [
        ("existing binary", find_existing_server, ()),
        ("release download", download_release_server, ()),
        ("source build", build_from_source, ()),
    ]
    for label, strategy, arguments in strategies:
        try:
            result = strategy(*arguments)
        except (RuntimeError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
            warn(f"{label} failed: {error}")
            continue
        if result is not None:
            log(f"using llama-server from {label}: {result}")
            return result
    raise RuntimeError(
        "could not obtain llama-server: no binary in /kaggle/working, release download "
        "failed, and source build failed. Check GPU, network and cmake on the Kaggle image."
    )


def server_command(executable: Path, model_path: Path) -> list[str]:
    """Return the argv for launching llama-server with the configured port, key and args."""
    command = [
        str(executable),
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-ngl", "99",
        "--api-key", PEER_KEY,
        "--ctx-size", str(CTX_SIZE),
        "--no-webui",
    ]
    if LLAMA_ARGS:
        command.extend(shlex.split(LLAMA_ARGS))
    return command


def health_url() -> str:
    """Return the localhost health-check URL for the running server."""
    return f"http://127.0.0.1:{PORT}/health"


def _probe_health() -> str:
    """Run one health check: 'ok' when the server is ready, else the last observed status."""
    try:
        with urllib.request.urlopen(health_url(), timeout=HEALTH_POLL_INTERVAL_S) as response:
            return "ok" if response.status == 200 else f"HTTP {response.status}"
    except (urllib.error.URLError, OSError) as error:
        return str(error)


def wait_until_healthy() -> None:
    """Poll the health endpoint until it reports ready or LOAD_TIMEOUT_S elapses."""
    deadline = time.monotonic() + LOAD_TIMEOUT_S
    last_status = "no response"
    while time.monotonic() < deadline:
        last_status = _probe_health()
        if last_status == "ok":
            log("server is healthy")
            return
        time.sleep(HEALTH_POLL_INTERVAL_S)
    raise RuntimeError(
        f"server did not become healthy within {LOAD_TIMEOUT_S}s; last status: {last_status}"
    )


def start_server(executable: Path, model_path: Path) -> subprocess.Popen:
    """Launch llama-server and return the process once it answers health checks."""
    log(f"starting llama-server on port {PORT}")
    process = subprocess.Popen(server_command(executable, model_path))
    wait_until_healthy()
    return process


def parse_job_line(raw_line: str, source: Path, line_number: int) -> dict | None:
    """Parse one JSONL line into a job dict, or return None when blank or malformed."""
    stripped = raw_line.strip()
    if not stripped:
        return None
    try:
        job = json.loads(stripped)
    except json.JSONDecodeError as error:
        warn(f"skipping malformed line {line_number} in {source}: {error}")
        return None
    if not isinstance(job, dict) or "id" not in job:
        warn(f"skipping line {line_number} in {source}: not a job object")
        return None
    return job


def read_jobs_file(path: Path) -> list[dict]:
    """Read a JSONL jobs file, skipping blank and malformed lines with a warning."""
    jobs = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            job = parse_job_line(raw_line, path, line_number)
            if job is not None:
                jobs.append(job)
    return jobs


def jobs_file_path() -> Path:
    """Return the jobs file from the env var or the first jobs.jsonl under /kaggle/input."""
    if JOBS_PATH:
        path = Path(JOBS_PATH)
        if not path.is_file():
            raise RuntimeError(
                f"FOREMANKIT_KAGGLE_JOBS points to {path}, which is not a file."
            )
        return path
    for candidate in KAGGLE_INPUT_DIR.rglob("jobs.jsonl"):
        return candidate
    raise RuntimeError(
        "no jobs.jsonl found under /kaggle/input. Attach the jobs dataset "
        "or set FOREMANKIT_KAGGLE_JOBS to its path."
    )


def load_jobs() -> list[dict]:
    """Return all valid jobs from the configured JSONL file, skipping malformed lines."""
    path = jobs_file_path()
    jobs = read_jobs_file(path)
    log(f"loaded {len(jobs)} jobs from {path}")
    return jobs


def build_request_body(job: dict) -> dict:
    """Return the chat-completions request body for a job's system and user messages."""
    messages = []
    if job.get("system"):
        messages.append({"role": "system", "content": job["system"]})
    messages.append({"role": "user", "content": job.get("prompt", "")})
    return {"messages": messages, "max_tokens": int(job.get("max_tokens", DEFAULT_MAX_TOKENS))}


def completions_url() -> str:
    """Return the localhost chat-completions URL for the running server."""
    return f"http://127.0.0.1:{PORT}/v1/chat/completions"


def post_completion(body: dict) -> dict:
    """POST a chat-completions body and return the decoded JSON response."""
    request = urllib.request.Request(
        completions_url(),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PEER_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=JOB_HTTP_TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def success_result(job: dict, payload: dict, wall_s: float) -> dict:
    """Build a success result dict from the API payload."""
    choice = payload["choices"][0]
    usage = payload.get("usage", {})
    return {
        "id": job["id"],
        "kind": job.get("kind"),
        "success": True,
        "response": choice["message"]["content"],
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "wall_s": wall_s,
    }


def failure_result(job: dict, error: str, wall_s: float) -> dict:
    """Build a failure result dict from a caught error message."""
    return {
        "id": job.get("id"),
        "kind": job.get("kind"),
        "success": False,
        "error": error,
        "wall_s": wall_s,
    }


def run_job(job: dict) -> dict:
    """Run one job against the local server, returning success or failure details."""
    started = time.monotonic()
    try:
        payload = post_completion(build_request_body(job))
        return success_result(job, payload, time.monotonic() - started)
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as error:
        warn(f"job {job.get('id')} failed: {error}")
        return failure_result(job, str(error), time.monotonic() - started)


def start_results_file() -> None:
    """Truncate results.jsonl so a re-run in a warm working dir never mixes old rows in."""
    RESULTS_PATH.write_text("", encoding="utf-8")


def append_result(result: dict) -> None:
    """Append a single result as one JSON line to results.jsonl and flush to disk."""
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def stop_server(process: subprocess.Popen) -> None:
    """Terminate the server, waiting briefly before escalating to a kill."""
    process.terminate()
    try:
        process.wait(timeout=SERVER_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        warn("server did not stop in time; killing")
        process.kill()


def run_jobs_within_budget(jobs: list[dict], max_seconds: float) -> tuple[int, int]:
    """Run jobs until the time budget expires, returning (ok_count, failed_count)."""
    start = time.monotonic()
    ok_count = 0
    failed_count = 0
    for job in jobs:
        if time.monotonic() - start >= max_seconds:
            warn(f"time budget of {MAX_RUN_HOURS}h reached; stopping")
            break
        result = run_job(job)
        append_result(result)
        if result["success"]:
            ok_count += 1
        else:
            failed_count += 1
    return ok_count, failed_count


def kaggle_cli_available() -> bool:
    """Return True when both the kaggle CLI and a KAGGLE_KEY are available for publishing."""
    return shutil.which("kaggle") is not None and "KAGGLE_KEY" in os.environ


def publish_dataset() -> None:
    """Publish a new dataset version from the working dir, logging any failure."""
    _ensure_dataset_metadata()
    try:
        subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(WORKING_DIR), "-m",
             "foremankit batch results", "--dir-mode", "zip"],
            check=True,
            capture_output=True,
            text=True,
        )
        log(f"published dataset version for {OUTPUT_DATASET}")
    except (subprocess.CalledProcessError, OSError) as error:
        warn(f"dataset publish failed: {error}")


def _ensure_dataset_metadata() -> None:
    """Write dataset-metadata.json, which the kaggle CLI refuses to version without."""
    path = WORKING_DIR / "dataset-metadata.json"
    if path.exists():
        return
    record = {"title": OUTPUT_DATASET.split("/")[-1], "id": OUTPUT_DATASET}
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def maybe_publish_dataset() -> None:
    """Publish results when an output dataset is set and the Kaggle CLI is ready."""
    if OUTPUT_DATASET and kaggle_cli_available():
        publish_dataset()


def _result_row(line: str) -> dict | None:
    """One line of results.jsonl as a dict, or None when it is blank or broken."""
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    return row if isinstance(row, dict) else None


def report_throughput() -> None:
    """Print tokens and wall time for the run: the Stage-0 number that gates live steering.

    wall_s includes prefill and decode together, so the rate here is a lower bound on
    prefill_tps and never a substitute for a single-prompt measurement.
    """
    counts = {"tokens_in": 0.0, "tokens_out": 0.0, "wall_s": 0.0}
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        row = _result_row(line)
        for key in counts:
            counts[key] += float(row.get(key) or 0.0) if row else 0.0
    wall = counts["wall_s"]
    if not wall:
        return
    total = counts["tokens_in"] + counts["tokens_out"]
    print(f"throughput: {total:,.0f} tokens in {wall:.0f}s = {total / wall:.1f} tok/s "
          f"({counts['tokens_in']:,.0f} in, {counts['tokens_out']:,.0f} out)", flush=True)


def main() -> int:
    """Run the full headless batch and return the process exit code."""
    started = time.monotonic()
    model_path = locate_model()
    executable = ensure_llama_server()
    server_process = start_server(executable, model_path)
    try:
        jobs = load_jobs()
        start_results_file()
        ok_count, failed_count = run_jobs_within_budget(jobs, MAX_RUN_HOURS * SECONDS_PER_HOUR)
    finally:
        stop_server(server_process)
    elapsed = int(time.monotonic() - started)
    total = ok_count + failed_count
    print(f"Total {total} | ok {ok_count} | failed {failed_count} | elapsed {elapsed}s", flush=True)
    report_throughput()
    print(f"results: {RESULTS_PATH} (fetch with: kaggle kernels output <owner>/<slug> -p ./out)",
          flush=True)
    maybe_publish_dataset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
