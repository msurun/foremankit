"""llama-bench harness: one change at a time, N repetitions, telemetry per run, one CSV table.

Usage:  python bench_suite.py PHASE [PHASE ...]      (see PHASES below)
        python bench_suite.py --list

Every run appends rows to results.csv (config, test, depth, mean t/s, stddev, CPU temp max, perf mean, VRAM, notes).
A cooldown guard waits before a run while the CPU thermal zone is above COOL_BELOW_C.
"""
import csv
import io
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA = HERE.parent / "llama"
BENCH = LLAMA / "LLAMA.CPP" / "llama-bench.exe"
Q5 = LLAMA / "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
Q4 = Path(os.environ.get("FOREMANKIT_Q4_MODEL", str(LLAMA / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")))
RESULTS = HERE / "results.csv"
COLS = ["time", "phase", "config", "model", "args", "test", "depth", "mean_ts", "stddev_ts", "reps",
        "cpu_temp_max_C", "cpu_perf_mean_pct", "vram_max_MiB", "shared_max_MiB", "notes"]
COOL_BELOW_C = 75          # wait until the CPU zone is below this before a run
ABORT_ABOVE_C = 98         # sampled during a run: flag as thermally suspect

# Baseline = what start-qwen3.6.ps1 does (+ explicit threads, since llama-bench defaults to 16)
BASE_OLD = ["-ngl", "99", "-ncmoe", "40", "-fa", "1", "-lm", "none", "-t", "16", "-b", "2048", "-ub", "512"]
# after phase 1: P-cores only (logical 0,1,10-19), SMT on, pinned
BASE = BASE_OLD + ["-C", "0xFFC03", "--cpu-strict", "1"]
BASE[BASE.index("-t") + 1] = "12"


def with_(base, **over):
    """Replace flag values: with_(BASE, ncmoe=38) -> '-ncmoe 38'."""
    out, flags = list(base), {"ncmoe": "-ncmoe", "t": "-t", "fa": "-fa", "b": "-b", "ub": "-ub", "lm": "-lm",
                              "ngl": "-ngl", "ctk": "-ctk", "ctv": "-ctv", "C": "-C", "strict": "--cpu-strict",
                              "poll": "--poll", "prio": "--prio", "nopo": "-nopo", "repack": "--repack"}
    for k, v in over.items():
        f = flags[k]
        if f in out:
            out[out.index(f) + 1] = str(v)
        else:
            out += [f, str(v)]
    return out


def sample_once():
    """(cpu temp C, cpu performance %, shared GPU MiB) from typeperf + nvidia-smi VRAM."""
    t = perf = shared = None
    try:
        r = subprocess.run(["typeperf", r"\Thermal Zone Information(\_TZ.CPUZ)\Temperature",
                            r"\Processor Information(_Total)\% Processor Performance",
                            r"\GPU Adapter Memory(*)\Shared Usage", "-sc", "1"], capture_output=True, text=True, timeout=20)
        rows = [x for x in csv.reader(io.StringIO(r.stdout)) if x]
        vals = [float(x) for x in rows[1] if x.replace(".", "", 1).isdigit()] if len(rows) > 1 else []
        if len(vals) >= 3:
            t, perf, shared = vals[0] - 273.15, vals[1], sum(vals[2:]) / 2**20
    except Exception:
        pass
    vram = None
    try:
        v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        vram = float(v)
    except Exception:
        pass
    return t, perf, shared, vram


def cooldown():
    waited = 0
    while True:
        t = sample_once()[0]
        if t is None or t < COOL_BELOW_C or waited > 240:
            return waited
        time.sleep(15)
        waited += 15


class Telemetry(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop, self.rows = threading.Event(), []

    def run(self):
        while not self.stop.is_set():
            self.rows.append(sample_once())

    def summary(self):
        col = lambda i: [r[i] for r in self.rows if r[i] is not None]
        mx = lambda xs: round(max(xs), 1) if xs else ""
        perf = col(1)
        return mx(col(0)), round(sum(perf) / len(perf)) if perf else "", mx(col(3)), mx(col(2))


def run(phase, label, model, args, reps=3, tests=("-p", "512", "-n", "128"), notes="", delay=3):
    waited = cooldown()
    cmd = [str(BENCH), "-m", str(model), *args, *tests, "-r", str(reps), "-o", "csv", "--delay", str(delay)]
    tel = Telemetry()
    tel.start()
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    tel.stop.set()
    tel.join(timeout=25)
    tmax, perf, vram, shared = tel.summary()
    rows = list(csv.DictReader(io.StringIO(p.stdout[p.stdout.find("build_commit"):] if "build_commit" in p.stdout else "")))
    note = notes + (f" cooled {waited}s;" if waited else "")
    if isinstance(tmax, float) and tmax >= ABORT_ABOVE_C:
        note += " THERMAL-SUSPECT;"
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLS)
        if new:
            w.writeheader()
        if not rows:
            w.writerow(dict(time=datetime.now().strftime("%H:%M:%S"), phase=phase, config=label, model=model.name,
                            args=" ".join(args), test="FAILED", notes=(p.stderr or p.stdout)[-300:].replace("\n", " ")))
        for r in rows:
            depth = r.get("n_depth", "0")
            ts = float(r["avg_ts"])
            test = (f"pp{r['n_prompt']}" if int(r["n_prompt"]) else f"tg{r['n_gen']}")
            w.writerow(dict(time=datetime.now().strftime("%H:%M:%S"), phase=phase, config=label, model=model.name,
                            args=" ".join(args), test=test, depth=depth, mean_ts=round(ts, 2),
                            stddev_ts=round(float(r["stddev_ts"]), 2), reps=reps, cpu_temp_max_C=tmax,
                            cpu_perf_mean_pct=perf, vram_max_MiB=vram, shared_max_MiB=shared, notes=note.strip()))
            print(f"[{phase}] {label:34s} {test:6s} d={depth:>5s}  {ts:7.2f} +- {float(r['stddev_ts']):5.2f}  "
                  f"Tmax={tmax}°C perf~{perf}% VRAMmax={vram} ({time.time() - t0:.0f}s)", flush=True)


def P(name):
    return PHASES[name]


def ph_baseline():
    run("baseline", "base (ncmoe40 fa1 t16 lm none)", Q5, BASE)


def ph_old():
    run("recheck", "OLD base t16 no mask (after cooling change)", Q5, BASE_OLD)
    run("recheck", "NEW base P-mask t12", Q5, BASE)


def ph_ncmoe():
    for n in (42, 40, 38, 37, 36):
        run("ncmoe", f"ncmoe={n}", Q5, with_(BASE, ncmoe=n))


def ph_threads():
    for t in (4, 6, 8, 10, 12, 14, 16, 20, 22):
        run("threads", f"t={t} (no mask)", Q5, with_(BASE, t=t))


def ph_masks():
    # P-cores only: logical 0,1,10-19 (6 cores/12 threads) = 0xFFC03; one thread per P-core: 0,10,12,14,16,18 = 0x57C01
    for label, mask, t in (("P-cores SMT on, t=12", "0xFFC03", 12), ("P-cores SMT on, t=6", "0xFFC03", 6),
                           ("P-cores 1 thr/core, t=6", "0x57C01", 6), ("P+E (all but LP-E), t=18", "0xFFFFF", 18),
                           ("P 1thr/core + 8 E, t=14", "0x57FFD", 14)):
        run("masks", label, Q5, with_(BASE, t=t, C=mask, strict=1))


def ph_misc():
    run("fa", "fa=0", Q5, with_(BASE, fa=0))
    run("kv", "ctk/ctv q8_0 (needs fa)", Q5, with_(BASE, ctk="q8_0", ctv="q8_0"))
    for b, ub in ((512, 512), (1024, 512), (2048, 1024), (2048, 2048), (4096, 2048)):
        run("batch", f"b={b} ub={ub}", Q5, with_(BASE, b=b, ub=ub))
    run("load", "lm=mmap", Q5, with_(BASE, lm="mmap"))
    run("load", "lm=mmap+mlock", Q5, with_(BASE, lm="mmap+mlock"))
    run("load", "lm=auto", Q5, with_(BASE, lm="auto"))
    run("poll", "poll=0", Q5, with_(BASE, poll=0))
    run("poll", "poll=100", Q5, with_(BASE, poll=100))
    run("prio", "prio=2 (high)", Q5, with_(BASE, prio=2))
    run("opoff", "no-op-offload=1", Q5, with_(BASE, nopo=1))


def ph_quant():
    for n in (40, 38):
        run("quant", f"Q4_K_XL ncmoe={n}", Q4, with_(BASE, ncmoe=n))


def ph_depth():
    run("depth", "base depth 0/4096/16384", Q5, BASE, reps=2, tests=("-p", "512", "-n", "64", "-d", "0,4096,16384"))


PHASES = {"old": ph_old, "baseline": ph_baseline, "ncmoe": ph_ncmoe, "threads": ph_threads, "masks": ph_masks, "misc": ph_misc,
          "quant": ph_quant, "depth": ph_depth}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        print("phases:", ", ".join(PHASES))
        sys.exit(0)
    for name in sys.argv[1:]:
        PHASES[name]()
