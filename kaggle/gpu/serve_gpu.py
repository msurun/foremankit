# Kaggle GPU serve entry point: llama-server behind a Cloudflare quick tunnel.
# Runs inside an interactive notebook via `!python serve_gpu.py`; standard library only.
# Config comes from FOREMANKIT_* environment variables; see main() for the flow.
"""Run inside an interactive Kaggle GPU notebook: serve a GGUF model with llama-server behind a Cloudflare tunnel."""
import glob
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping

PORT = 8080
HEALTH_POLL_INTERVAL_S = 3
LOAD_TIMEOUT_S = 1200
TUNNEL_TIMEOUT_S = 120
SECONDS_PER_HOUR = 3600
HEARTBEAT_INTERVAL_S = 600
KEEP_ALIVE_POLL_S = 30
TAIL_LINES = 30
SERVER_LOG = Path("/kaggle/working/llama-server.log")
ENDPOINT_FILE = Path("/kaggle/working/endpoint.txt")
INPUT_DIR = Path("/kaggle/input")
BUILD_DIR = Path("/kaggle/working/llama.cpp")
BUILD_LOG = Path("/kaggle/working/build.log")
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CLOUDFLARED_PATH = Path("/tmp/cloudflared")
MODEL_DIR = "/tmp/models"
DEFAULT_HF_REPO = "unsloth/Qwen3.8-27B-GGUF"
DEFAULT_HF_FILE = "Qwen3.8-27B-UD-Q4_K_XL.gguf"


def log_tail(path: Path) -> str:
    """Return the last TAIL_LINES lines of a file, or a note when it is missing."""
    if not path.exists():
        return f"(no log at {path})"
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-TAIL_LINES:])

def log(message: str) -> None:
    """Print a GPU-tagged progress line, flushing so notebook logs appear live."""
    print(f"[gpu] {message}", flush=True)

def warn(message: str) -> None:
    """Print a GPU-tagged warning line to stderr, flushing so notebook logs appear live."""
    print(f"[gpu] WARN {message}", file=sys.stderr, flush=True)

def server_command(binary: Path, model: Path, key: str, env: Mapping[str, str]) -> list[str]:
    """Return the argv for launching llama-server with the configured port, key and args."""
    return [
        str(binary),
        "--model", str(model),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-ngl", "99",
        "--ctx-size", env.get("FOREMANKIT_CTX_SIZE", "262144"),   # the model's native maximum
        "--cache-type-k", env.get("FOREMANKIT_KV_TYPE", "q8_0"),   # halves the KV cache: room for 262k on 2x T4
        "--cache-type-v", env.get("FOREMANKIT_KV_TYPE", "q8_0"),
        "--parallel", env.get("FOREMANKIT_PARALLEL", "2"), "--kv-unified",
        "-fa", "on", "--jinja", "--cache-reuse", "256",
        "-b", "2048", "-ub", "1024",
        "--api-key", key,
        "--alias", env.get("FOREMANKIT_ALIAS") or model.stem,
        "--no-webui",
        "--metrics",
    ] + shlex.split(env.get("FOREMANKIT_LLAMA_ARGS", ""))

def tunnel_url(line: str) -> str | None:
    """Return the first https://<something>.trycloudflare.com found in a cloudflared log line."""
    match = re.search(r"https://([a-z0-9-]+)\.trycloudflare\.com", line)
    if not match or match.group(1) == "api":
        return None
    return f"https://{match.group(1)}.trycloudflare.com"

def new_key(env: Mapping[str, str]) -> str:
    """Return the configured key or a new random one."""
    return env.get("FOREMANKIT_SERVE_KEY") or f"sk-{secrets.token_urlsafe(24)}"

def find_model(env: Mapping[str, str], input_dir: Path) -> Path | None:
    """Return the configured model path or the first *.gguf under input_dir."""
    if "FOREMANKIT_MODEL_GGUF" in env:
        path = Path(env["FOREMANKIT_MODEL_GGUF"])
        if not path.is_file():
            raise FileNotFoundError(f"FOREMANKIT_MODEL_GGUF points to {path}, which is not a file")
        return path
    candidates = sorted(
        path for path in input_dir.rglob("*.gguf")
        if "mmproj" not in path.name
    )
    return candidates[0] if candidates else None

def banner(url: str, key: str, alias: str) -> str:
    """Return the multi-line READY text with the root URL, key and alias."""
    return (
        f"READY: {url}\n"
        f"Key: {key}\n"
        f"Alias: {alias}\n"
        "Paste address and key into the ForemanKit panel: Kaggle notebook -> Save -> Test"
    )

def check_gpus() -> None:
    """Print the GPUs listed by nvidia-smi -L and raise if none are found."""
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if not result.stdout.strip():
        raise RuntimeError("no GPUs found")
    log(f"GPUs:\n{result.stdout}")

def find_llama_binary(env: Mapping[str, str]) -> Path | None:
    """Return the configured llama-server path or the first one under /kaggle/input, copied to /tmp as executable
    (an attached dataset is read-only, so it cannot be chmod-ed in place)."""
    configured = Path(env["FOREMANKIT_LLAMA_BIN"]) if env.get("FOREMANKIT_LLAMA_BIN") else None
    found = [configured] if configured and configured.is_file() else sorted(INPUT_DIR.rglob("llama-server"))
    for candidate in (path for path in found if path.is_file()):
        target = Path("/tmp/llama-server")
        shutil.copy2(candidate, target)
        target.chmod(target.stat().st_mode | 0o111)
        return target
    return None

def clone_llama() -> None:
    """Shallow-clone llama.cpp into BUILD_DIR, at FOREMANKIT_LLAMACPP_REF when it is set (kept from an earlier run)."""
    if (BUILD_DIR / ".git").is_dir():
        return
    command = ["git", "clone", "--depth", "1"]
    if os.environ.get("FOREMANKIT_LLAMACPP_REF"):
        command += ["--branch", os.environ["FOREMANKIT_LLAMACPP_REF"]]
    subprocess.run(command + [REPO_URL, str(BUILD_DIR)], check=True, capture_output=True, text=True)


def cuda_driver_flags() -> list[str]:
    """Point CMake at libcuda when the toolkit has no stubs/ copy (Kaggle image: "CUDA::cuda_driver not found")."""
    patterns = [str(CUDA_HOME / "lib64" / "stubs" / "libcuda.so"),
                str(CUDA_HOME / "targets" / "x86_64-linux" / "lib" / "stubs" / "libcuda.so"),
                "/usr/lib/x86_64-linux-gnu/libcuda.so*", "/usr/local/nvidia/lib64/libcuda.so*",
                "/usr/lib64-nvidia/libcuda.so*", "/usr/lib64/libcuda.so*"]
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if found:
            log(f"libcuda for CMake: {found[0]}")
            return [f"-DCUDA_cuda_driver_LIBRARY={found[0]}"]
    warn("libcuda.so not found; the CUDA build may fail at CUDA::cuda_driver")
    return []


def run_cmake() -> int:
    """Configure and build the llama-server target, logging to BUILD_LOG; return the build exit code."""
    build = BUILD_DIR / "build"
    cmake_flags = [
        "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
    ]
    nvcc = CUDA_HOME / "bin" / "nvcc"
    if nvcc.is_file():   # Kaggle ships the toolkit in /usr/local/cuda, not always on PATH
        cmake_flags.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
    cmake_flags += cuda_driver_flags()
    env = {**os.environ, "PATH": f"{CUDA_HOME / 'bin'}:{os.environ.get('PATH', '')}"}
    steps = (["cmake", "-B", str(build)] + cmake_flags,
             ["cmake", "--build", str(build), "--target", "llama-server", "-j", str(os.cpu_count() or 2)])
    with BUILD_LOG.open("w") as handle:
        for step in steps:
            code = subprocess.run(step, cwd=BUILD_DIR, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
            if code != 0:
                return code
    return 0


def build_llama() -> Path:
    """Clone, build and return the llama-server path, printing the build log tail on failure."""
    clone_llama()
    binary = BUILD_DIR / "build" / "bin" / "llama-server"
    if run_cmake() != 0 or not binary.is_file():
        log(f"build failed; tail of {BUILD_LOG}:\n{log_tail(BUILD_LOG)}")
        raise RuntimeError("llama-server build failed")
    log("save /kaggle/working/llama.cpp/build/bin/llama-server as a Kaggle dataset and attach it to skip this ~10 min build")
    return binary

def ensure_llama_binary() -> Path:
    """Return a llama-server path, trying reuse, then source build."""
    binary = find_llama_binary(os.environ)
    if binary is not None:
        log(f"using existing llama-server: {binary}")
        return binary
    binary = BUILD_DIR / "build" / "bin" / "llama-server"
    if binary.is_file():
        log(f"using built llama-server: {binary}")
        return binary
    return build_llama()

def download_model(env: Mapping[str, str]) -> Path:
    """Download a model with huggingface_hub and return its path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=env.get("FOREMANKIT_HF_REPO", DEFAULT_HF_REPO),
        filename=env.get("FOREMANKIT_HF_FILE", DEFAULT_HF_FILE),
        local_dir=MODEL_DIR,
    ))

def ensure_model() -> Path:
    """Return a model path, trying reuse, then download."""
    model = find_model(os.environ, INPUT_DIR)
    if model is not None:
        log(f"using model: {model}")
        return model
    log("downloading model")
    return download_model(os.environ)

def start_server(binary: Path, model: Path, key: str) -> subprocess.Popen:
    """Launch llama-server and return the process."""
    log(f"starting llama-server on port {PORT}")
    return subprocess.Popen(
        server_command(binary, model, key, os.environ),
        stdout=SERVER_LOG.open("w"),
        stderr=subprocess.STDOUT,
    )

def health_status() -> str:
    """Return "ok" when the health endpoint answers 200, otherwise a short status string."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=HEALTH_POLL_INTERVAL_S) as response:
            return "ok" if response.status == 200 else f"HTTP {response.status}"
    except (urllib.error.URLError, OSError) as error:
        return str(error)


def wait_until_healthy(server: subprocess.Popen) -> None:
    """Poll the health endpoint until it reports ready; RuntimeError (with the log tail) when the server exits
    first or LOAD_TIMEOUT_S elapses."""
    deadline = time.monotonic() + LOAD_TIMEOUT_S
    last_status = "no response"
    while time.monotonic() < deadline and server.poll() is None:
        last_status = health_status()
        if last_status == "ok":
            log("server is healthy")
            return
        time.sleep(HEALTH_POLL_INTERVAL_S)
    why = f"exited with code {server.returncode}" if server.poll() is not None else f"not healthy after {LOAD_TIMEOUT_S}s"
    log(f"llama-server {why}; last status: {last_status}; tail of {SERVER_LOG}:\n{log_tail(SERVER_LOG)}")
    raise RuntimeError(f"llama-server {why}")

def self_test(key: str) -> None:
    """POST a test request and print the prompt and generation tok/s."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 16}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            timings = payload.get("timings", {})
            log(f"self-test: prompt {timings.get('prompt_per_second', 0):.1f} tok/s, generation {timings.get('predicted_per_second', 0):.1f} tok/s")
    except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
        warn(f"self-test failed: {error}")

def download_cloudflared() -> Path:
    """Download cloudflared to /tmp unless it exists or is on PATH."""
    on_path = shutil.which("cloudflared")
    if on_path:
        return Path(on_path)
    path = CLOUDFLARED_PATH
    if path.is_file():
        return path
    log("downloading cloudflared")
    with urllib.request.urlopen(CLOUDFLARED_URL, timeout=60) as response:
        path.write_bytes(response.read())
    path.chmod(path.stat().st_mode | 0o111)
    return path

def start_tunnel() -> tuple[subprocess.Popen, str]:
    """Start cloudflared and return the process and the tunnel URL."""
    binary = download_cloudflared()
    process = subprocess.Popen(
        [str(binary), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + TUNNEL_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("cloudflared exited early")
        line = process.stderr.readline()
        if not line:
            continue
        url = tunnel_url(line)
        if url is not None:
            return process, url
    raise RuntimeError("cloudflared did not provide a tunnel URL within 120s")

def drain_tunnel(process: subprocess.Popen) -> None:
    """Drain cloudflared's stderr in a daemon thread."""
    def _drain() -> None:
        while process.poll() is None:
            if process.stderr is not None:
                process.stderr.readline()
    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()

def keep_alive(server: subprocess.Popen, tunnel: subprocess.Popen, max_seconds: float) -> int:
    """Keep alive until the time budget expires or either process exits."""
    start = time.monotonic()
    heartbeat = time.monotonic()
    while time.monotonic() - start < max_seconds:
        if server.poll() is not None:
            log(f"llama-server exited; tail of {SERVER_LOG}:\n{log_tail(SERVER_LOG)}")
            return 1
        if tunnel.poll() is not None:
            log("cloudflared exited")
            return 1
        if time.monotonic() - heartbeat > HEARTBEAT_INTERVAL_S:
            log(f"heartbeat: {int(time.monotonic() - start)}s elapsed")
            heartbeat = time.monotonic()
        time.sleep(KEEP_ALIVE_POLL_S)
    log(f"time budget of {max_seconds / SECONDS_PER_HOUR:.1f}h reached; stopping")
    return 0

def main() -> int:
    """Run the full interactive serve and return the process exit code."""
    server = None
    tunnel = None
    try:
        server, key, alias = prepare_server()
        wait_until_healthy(server)
        self_test(key)
        tunnel, url = start_tunnel()
        drain_tunnel(tunnel)
        log(banner(url, key, alias))
        ENDPOINT_FILE.write_text(f"{url}\n{key}\n", encoding="utf-8")
        return keep_alive(server, tunnel, float(os.environ.get("FOREMANKIT_MAX_RUN_HOURS", "8.5")) * SECONDS_PER_HOUR)
    except KeyboardInterrupt:
        log("interrupted")
        return 1
    finally:
        for process in (server, tunnel):
            if process is not None:
                process.terminate()
                process.wait()


def prepare_server() -> tuple[subprocess.Popen, str, str]:
    """Check GPUs, resolve binary and model, start llama-server; return process, key and alias."""
    check_gpus()
    binary = ensure_llama_binary()
    model = ensure_model()
    key = new_key(os.environ)
    server = start_server(binary, model, key)
    return server, key, os.environ.get("FOREMANKIT_ALIAS") or model.stem

if __name__ == "__main__":
    sys.exit(main())
