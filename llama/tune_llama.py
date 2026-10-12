"""Tune llama.cpp for THIS machine and save the result as a profile that start-qwen3.6.ps1 picks up by itself.

    python tune_llama.py                      # model: Q4_K_XL, else Q5_K_M (here / $FOREMANKIT_MODELS_DIR), else first *.gguf
    python tune_llama.py --model PATH         # a specific GGUF
    python tune_llama.py --show               # print this machine's fingerprint and its stored profile, run nothing

What it does (about 15 minutes, keep the laptop on AC power and do not use it meanwhile):
  1. profile key = "CPU name | GPU name | VRAM GiB || model file name"   (same text the launcher computes; one profile per machine AND model)
  2. MoE offload: walk --n-cpu-moe down (44, 42, ...) until the model stops loading (VRAM full, with Windows
     "Prefer No Sysmem Fallback") or generation collapses (< 70 % of the best, i.e. spill to shared memory).
     Chosen value = smallest working one + 2 layers of VRAM headroom for the KV cache / compute buffers.
  3. threads: on a hybrid CPU (P + E cores) compare "P-cores only, SMT on", "P-cores one thread per core", and
     "no pinning"; on a uniform CPU compare physical vs logical thread counts. 3 repetitions each, best mean wins.
  3b. validates the chosen ncmoe by starting the real llama-server at the target context (default 32768) and raises it by 2
     until it starts (llama-bench alone uses a short context and can pick a value that later runs out of VRAM).
  4. writes llama/profiles.json (key = fingerprint). Results of every step are printed.
Measured on the HP ZBook Firefly 14 G11 (see benchmarks/RESULTS.md): this procedure ends at ncmoe 40, P-core mask, t=12.
Machines without an NVIDIA GPU get no ncmoe (the launcher then relies on llama-server --fit) and only the thread search.
"""
import os
import argparse
import csv
import ctypes
import io
import json
import re
import subprocess
import sys
import time
import urllib.request
from ctypes import wintypes
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "LLAMA.CPP" / "llama-bench.exe"
PROFILES = HERE / "profiles.json"
SERVER = HERE / "LLAMA.CPP" / "llama-server.exe"
NO_SPILL = 0.70          # tg below this fraction of the best = spilled to shared memory
HEADROOM_LAYERS = 2      # extra expert layers kept in RAM so a 32k context still fits in VRAM


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()


def fingerprint():
    """Must stay identical to Get-Fingerprint in start-qwen3.6.ps1."""
    cpu = re.sub(r"\s+", " ", ps("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")).strip().lower()
    gpu, vram = "none", 0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.strip().splitlines()[0]
        name, mib = [x.strip() for x in out.split(",")]
        gpu, vram = re.sub(r"\s+", " ", name).lower(), round(float(mib) / 1024)
    except Exception:
        pass
    ram = round(float(ps("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")) / 2**30)
    return f"{cpu} | {gpu} | {vram}gb", dict(cpu=cpu, gpu=gpu, vram_gib=vram, ram_gib=ram, host=ps("hostname"))


def cpu_sets():
    """[(logical index, core index, efficiency class)] via GetSystemCpuSetInformation."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = wintypes.ULONG(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buf = (ctypes.c_ubyte * size.value)()
    if not k32.GetSystemCpuSetInformation(buf, size.value, ctypes.byref(size), None, 0):
        return []
    off, rows = 0, []
    while off < size.value:
        sz = int.from_bytes(bytes(buf[off:off + 4]), "little")
        if int.from_bytes(bytes(buf[off + 4:off + 8]), "little") == 0:
            d = bytes(buf[off + 8:off + 24])
            rows.append((d[6], d[7], d[10]))
        off += sz
    return rows


def thread_candidates():
    """List of (label, threads, mask or None)."""
    rows = cpu_sets()
    logical = len(rows) or 1
    classes = sorted({r[2] for r in rows}, reverse=True)
    if len(classes) > 1:
        top = [r for r in rows if r[2] == classes[0]]
        mask = lambda items: "0x%X" % sum(1 << r[0] for r in items)
        one_per_core = {}
        for r in top:
            one_per_core.setdefault(r[1], r)
        return [(f"P-cores SMT, t={len(top)}", len(top), mask(top)),
                (f"P-cores 1 thread/core, t={len(one_per_core)}", len(one_per_core), mask(list(one_per_core.values()))),
                (f"no pinning, t={logical}", logical, None)]
    cores = len({r[1] for r in rows}) or logical
    cands = [(f"physical cores, t={cores}", cores, None)]
    if logical != cores:
        cands.append((f"logical threads, t={logical}", logical, None))
    return cands


def bench(model, ncmoe, threads, mask, reps, p=512, n=128, has_gpu=True, extra=()):
    cmd = [str(BENCH), "-m", str(model), "-fa", "1", "-lm", "none", "-t", str(threads), "-p", str(p), "-n", str(n),
           "-r", str(reps), "-o", "csv", "--delay", "3"]
    if has_gpu:
        cmd += ["-ngl", "99"] + (["-ncmoe", str(ncmoe)] if ncmoe is not None else [])
    else:
        cmd += ["-ngl", "0"]
    if mask:
        cmd += ["-C", mask, "--cpu-strict", "1"]
    cmd += list(extra)
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    if "build_commit" not in out:
        return None                       # did not load (out of VRAM or file problem)
    res = {}
    for r in csv.DictReader(io.StringIO(out[out.find("build_commit"):])):
        res["pp" if int(r["n_prompt"]) else "tg"] = (float(r["avg_ts"]), float(r["stddev_ts"]))
    return res if "tg" in res else None


def server_loads(model, ncmoe, threads, mask, ctx, extra=(), limit=240):
    """Start the real llama-server with the target context; True if it comes up (/health 200). Always stops it again."""
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
        sys.exit("port 8080 already serves something - stop that server first")
    except OSError:
        pass
    cmd = [str(SERVER), "-m", str(model), "--host", "127.0.0.1", "--port", "8080", "-c", str(ctx), "-ngl", "99", "--load-mode", "none",
           "--jinja", "-fa", "on", "-np", "1", "-t", str(threads), "--no-webui"]
    if ncmoe is not None:
        cmd += ["--n-cpu-moe", str(ncmoe)]
    if mask:
        cmd += ["-C", mask, "--cpu-strict", "1"]
    cmd += list(extra)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while time.time() - t0 < limit:
            if proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3) as r:
                    if r.status == 200:
                        return True
            except Exception:
                time.sleep(3)
        return False
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except Exception:
            proc.kill()
        time.sleep(3)


def find_model(arg):
    if arg:
        return Path(arg)
    extra = Path(os.environ["FOREMANKIT_MODELS_DIR"]) if os.environ.get("FOREMANKIT_MODELS_DIR") else None
    folders = [f for f in (HERE, extra) if f is not None and f.exists()]
    for pat in ("Qwen3.6-35B-A3B*Q4_K_XL*.gguf", "Qwen3.6-35B-A3B*Q5_K_M*.gguf"):   # Q4_K_XL is the default model
        for folder in folders:
            found = sorted(folder.glob(pat))
            if found:
                return found[0]
    found = sorted(HERE.glob("*.gguf"))
    if found:
        return found[0]
    sys.exit("No .gguf found; pass --model PATH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--ncmoe", type=int, help="skip the search (step 1), start validation from this value")
    ap.add_argument("--dense", action="store_true", help="dense (non-MoE) model: no --n-cpu-moe, the whole model goes to the GPU")
    a = ap.parse_args()
    machine, info = fingerprint()
    model = find_model(a.model)
    key = f"{machine} || {model.stem.lower()}"          # one profile per machine AND model
    print("profile key:", key, f"| RAM {info['ram_gib']} GiB | host {info['host']}")
    profiles = json.loads(PROFILES.read_text(encoding="utf-8")) if PROFILES.exists() else {}
    if a.show:
        print(json.dumps(profiles.get(key, "no profile for this machine + model"), indent=2))
        return
    if not BENCH.exists():
        sys.exit(f"llama-bench.exe not found: {BENCH}")
    battery = ps("(Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus")
    if battery == "1":
        sys.exit("Running on battery: results would be meaningless. Plug in the charger and retry.")
    has_gpu = info["gpu"] != "none"
    print(f"model: {model.name} ({model.stat().st_size / 2**30:.1f} GiB)\n")

    ncmoe = None
    best_tg = 0.0
    extra_flags, extra_profile = [], {}
    if a.dense:
        print("-- step 1 skipped: dense model (no experts), everything goes to the GPU")
        print()
    elif has_gpu and a.ncmoe is not None:
        ncmoe = a.ncmoe
        print(f"-- step 1 skipped: starting from --ncmoe {ncmoe}")
        print()
    elif has_gpu:
        print("-- step 1: --n-cpu-moe (descending; first loadable value wins the search)")
        first_threads, first_mask = thread_candidates()[0][1:]
        works = []
        for n in range(44, -1, -2):                      # down to 0: a small quant may fit in VRAM completely
            r = bench(model, n, first_threads, first_mask, reps=1, p=128, n=32)
            if r is None:
                print(f"  ncmoe={n}: does not load (VRAM full)")
                break
            tg = r["tg"][0]
            print(f"  ncmoe={n}: tg {tg:.1f} tok/s")
            if best_tg and tg < NO_SPILL * best_tg:
                print("  generation collapsed: spill to shared memory, stop")
                break
            best_tg = max(best_tg, tg)
            works.append(n)
        if not works:
            sys.exit("Not even ncmoe=44 loads; the GPU has too little free VRAM. Close GPU programs and retry.")
        lowest = min(works)
        # one more probe for the odd value between the lowest working and the first failure
        odd = lowest - 1
        r = bench(model, odd, first_threads, first_mask, reps=1, p=128, n=32) if odd >= 0 else None
        if r is not None and r["tg"][0] >= NO_SPILL * best_tg:
            lowest = odd
        ncmoe = min(lowest + HEADROOM_LAYERS, max(works))
        print(f"  smallest working: {lowest}  -> chosen with {HEADROOM_LAYERS} layers of headroom: {ncmoe}\n")

    if has_gpu:
        print(f"-- step 1b: validate with the real llama-server at ctx {a.ctx} (llama-bench ran a short context; KV + buffers need more VRAM)")
        first = thread_candidates()[0]
        # memory savers in increasing cost; within a level try a few more expert layers on the CPU first
        levels = [("f16 KV", [], {}), ("KV q8_0", ["-ctk", "q8_0", "-ctv", "q8_0"], {"kv_cache": "q8_0"}),
                  ("KV q8_0 + ubatch 256", ["-ctk", "q8_0", "-ctv", "q8_0", "-ub", "256"], {"kv_cache": "q8_0", "ubatch": 256})]
        chosen = None
        for label, flags, prof_extra in levels:
            for add in ((0, 2, 4) if ncmoe is not None else (0,)):      # dense: nothing to move to the CPU
                n = None if ncmoe is None else ncmoe + add
                ok = server_loads(model, n, first[1], first[2], a.ctx, extra=flags)
                print(f"  ncmoe={n}, {label}: {'server starts' if ok else 'does not start (VRAM)'}")
                if ok:
                    chosen = (n, flags, prof_extra)
                    break
            if chosen:
                break
        if not chosen:
            sys.exit("llama-server does not start even with q8_0 KV and ubatch 256; lower --ctx")
        ncmoe, extra_flags, extra_profile = chosen
        print(f"  validated ncmoe: {ncmoe}")
        print()

    print("-- step 2: threads / CPU pinning (3 repetitions, pp512 + tg128)")
    results = []
    for label, threads, mask in thread_candidates():
        r = bench(model, ncmoe, threads, mask, reps=3, has_gpu=has_gpu, extra=extra_flags)
        if r is None:
            print(f"  {label}: failed")
            continue
        print(f"  {label}: tg {r['tg'][0]:.2f} +- {r['tg'][1]:.2f}  pp {r.get('pp', (0, 0))[0]:.0f}")
        results.append((r["tg"][0], -r["tg"][1], label, threads, mask, r))
    if not results:
        sys.exit("All thread configurations failed.")
    results.sort(reverse=True)
    _, _, label, threads, mask, r = results[0]
    print(f"  best: {label}\n")

    profiles[key] = dict(
        host=info["host"], tuned=str(date.today()), model=model.name, ncmoe=ncmoe, threads=threads, cpu_mask=mask,
        ctx=a.ctx, tg_tok_s=round(r["tg"][0], 1), pp_tok_s=round(r.get("pp", (0, 0))[0]), ram_gib=info["ram_gib"],
        note=label, **extra_profile)
    PROFILES.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved profile -> {PROFILES}\n{json.dumps(profiles[key], indent=2)}")


if __name__ == "__main__":
    main()
