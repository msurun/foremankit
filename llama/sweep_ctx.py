"""Find the largest context a model keeps fast on THIS machine and store it in profiles.json.

    python sweep_ctx.py MODEL.gguf [MODEL2.gguf ...]      # models must already have a profile (python tune_llama.py)

For each model it starts the real llama-server at growing -c values (KV cache q8_0, ncmoe/threads from the profile),
reads the VRAM in use and the generation speed of one short answer, and stops at the first start failure. The context
allocated at start is what costs speed, so an empty prompt is enough to see the effect (measured 2026-10-12: Qwen 35B).
The chosen value is the largest context whose speed is >= MIN_SPEED_FRACTION of the 32k baseline and whose VRAM stays
below VRAM_LIMIT_MIB; it is written to the profile (ctx, kv_cache, tg_tok_s) and picked up by start-llama.ps1.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = HERE / "profiles.json"
LAUNCHER = HERE / "start-llama.ps1"
CONTEXTS = (32768, 65536, 98304, 131072, 196608, 262144)
MIN_SPEED_FRACTION = 0.85
VRAM_LIMIT_MIB = 7700
START_TIMEOUT_S = 240
PROMPT = "Opisz historię Polski szczegółowo."


def run_powershell(command):
    return subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True).stdout.strip()


def stop_servers():
    run_powershell("Get-Process llama-server -EA SilentlyContinue | Stop-Process -Force")
    time.sleep(3)


def start_server(model, ncmoe, threads, ctx):
    """Launch through the official launcher (never llama-server.exe directly) with an explicit context."""
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCHER), "-Model", model,
           "-Ctx", str(ctx), "-KvCache", "q8_0", "-NoThink", "-NoMenu", "-Force"]   # -Force: servers are stopped here first
    if ncmoe is not None:
        cmd += ["-NCpuMoe", str(ncmoe)]
    if threads:
        cmd += ["-Threads", str(threads)]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_healthy():
    deadline = time.time() + START_TIMEOUT_S
    while time.time() < deadline:
        try:
            if b"ok" in urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2).read():
                return True
        except OSError:
            pass
        time.sleep(4)
    return False


def generation_speed():
    """tok/s of the best of two 128-token answers (the first one includes warm-up)."""
    best = 0.0
    for _ in range(2):
        body = json.dumps({"prompt": PROMPT, "n_predict": 128, "temperature": 0.6}).encode()
        request = urllib.request.Request("http://127.0.0.1:8080/completion", body, {"Content-Type": "application/json"})
        timings = json.load(urllib.request.urlopen(request, timeout=300))["timings"]
        best = max(best, timings["predicted_per_second"])
    return best


def vram_used_mib():
    return int(run_powershell("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()[0])


def measure(model, profile, ctx):
    stop_servers()
    process = start_server(model, profile.get("ncmoe"), profile.get("threads"), ctx)
    try:
        if not wait_healthy():
            return None
        return generation_speed(), vram_used_mib()
    finally:
        process.terminate()
        stop_servers()


def pick(results):
    """Largest context that is still fast and leaves VRAM headroom; results = [(ctx, tok_s, vram_mib)]."""
    baseline = results[0][1]
    good = [r for r in results if r[1] >= MIN_SPEED_FRACTION * baseline and r[2] <= VRAM_LIMIT_MIB]
    return good[-1] if good else results[0]


def find_profile_key(profiles, model):
    stem = Path(model).stem.lower()
    keys = [k for k in profiles if k.endswith("|| " + stem) and "rtx 3070" in k]
    if not keys:
        raise SystemExit(f"No profile for {model} on this machine; run: python tune_llama.py --model {model}")
    return keys[0]


def sweep_model(model):
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    key = find_profile_key(profiles, model)
    results = []
    for ctx in CONTEXTS:
        measured = measure(model, profiles[key], ctx)
        print(f"{model} ctx={ctx}: {'FAILED TO START' if measured is None else '%.1f tok/s, %d MiB' % measured}", flush=True)
        if measured is None:
            break
        results.append((ctx, *measured))
    if not results:
        return
    ctx, speed, _ = pick(results)
    profiles[key].update(ctx=ctx, kv_cache="q8_0", tg_tok_s=round(speed, 1))
    PROFILES.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"==> {model}: ctx {ctx} stored ({speed:.1f} tok/s)", flush=True)


if __name__ == "__main__":
    for gguf in sys.argv[1:]:
        sweep_model(gguf)
