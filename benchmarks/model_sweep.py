"""Sweep every GGUF in a folder: tune -> serve with its own profile -> task suite + agent task -> one report.

python model_sweep.py [--dir ../llama] [--only IQ2,Q3] [--skip PATTERN] [--dense PATTERN[,PATTERN]] [--order big|small]
                      [--reps-short 3] [--reps-long 5] [--agent-reps 2] [--no-tune] [--resume] [--report-only]

For each model file (mmproj files are skipped):
  1. llama/tune_llama.py (unless a profile for this machine + model already exists, or --no-tune);
     --dense marks files whose name matches a pattern as dense (no --n-cpu-moe);
  2. the launcher llama/start-qwen3.6.ps1 -Model <file> serves it with the tuned profile on port 8080;
  3. the quality_bench task suite (6 short + 3 long tasks with automatic checks, thinking OFF) with --reps-short / --reps-long
     repetitions per task, and agent_task.py (fix 4 bugs with tools, visible + hidden tests) --agent-reps times;
  4. the server is stopped and the next model starts.
Output: sweep_results.jsonl (every run) and SWEEP_RESULTS.md (table per model + pass rate per task + ranking).
Port 8080 must be free: stop any shared server (start-qwen3.6.ps1 -Share) first and restart it afterwards.
Typical cost: 20-30 min per model (tuning ~10-15 min, tests ~10 min).
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA = HERE.parent / "llama"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LLAMA))
import agent_task as at
import quality_bench as qb
import tune_llama as tl

OUT = HERE / "sweep_results.jsonl"
REPORT = HERE / "SWEEP_RESULTS.md"
LAUNCHER = LLAMA / "start-qwen3.6.ps1"
HEALTH = "http://127.0.0.1:8080/health"


def log(row: dict) -> None:
    row["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with OUT.open("a", encoding="utf-8") as fh:
        print(json.dumps(row, ensure_ascii=False), file=fh)


def port_busy() -> bool:
    try:
        urllib.request.urlopen(HEALTH, timeout=2)
        return True
    except OSError:
        return False


def stop(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    for _ in range(30):
        if not port_busy():
            return
        time.sleep(2)


def profile_of(model: Path) -> dict | None:
    machine, _ = tl.fingerprint()
    path = LLAMA / "profiles.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return data.get(f"{machine} || {model.stem.lower()}")


def run_model(model: Path, dense: bool, a: argparse.Namespace) -> None:
    stem = model.stem
    size = round(model.stat().st_size / 2**30, 1)
    prof = profile_of(model)
    if prof is None and not a.no_tune:
        cmd = [sys.executable, str(LLAMA / "tune_llama.py"), "--model", str(model)] + (["--dense"] if dense else [])
        print(f"== tuning {stem} ...", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=LLAMA)
        prof = profile_of(model)
        if r.returncode != 0 or prof is None:
            log(dict(kind="model_failed", model=stem, size_gib=size, stage="tune", detail=f"exit {r.returncode}"))
            print(f"!! tuning failed for {stem}", flush=True)
            return
        print(f"   tuned in {time.time() - t0:.0f} s: {prof}", flush=True)
    log(dict(kind="profile", model=stem, size_gib=size, dense=dense, profile=prof))

    serve_log = (HERE / f"sweep_server_{stem}.log").open("w")
    proc = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCHER),
                             "-Model", str(model)], stdout=serve_log, stderr=subprocess.STDOUT)
    try:
        ready = False
        t0 = time.time()
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(HEALTH, timeout=3) as r:
                    ready = r.status == 200
            except OSError:
                pass
            if ready:
                break
            time.sleep(3)
        if not ready:
            log(dict(kind="model_failed", model=stem, size_gib=size, stage="serve", detail=f"see {serve_log.name}"))
            print(f"!! server for {stem} did not come up", flush=True)
            return
        print(f"== {stem}: server ready in {time.time() - t0:.0f} s, VRAM {qb.vram()} MiB", flush=True)
        for cat, name, prompt, check in qb.TASKS:
            for rep in range(a.reps_short if cat == "short" else a.reps_long):
                r = qb.one(stem, "nothink", cat, name, prompt, check)
                r.update(kind="task", rep=rep + 1, size_gib=size)
                log(r)
                print(f"   {name[:34]:34s} r{rep + 1} {'OK ' if r['correct'] else 'BAD'} {r.get('wall_s', '-')}s "
                      f"{r.get('gen_tps', '-')} t/s {'TRUNC' if r.get('truncated') else ''}", flush=True)
        for rep in range(1, a.agent_reps + 1):
            r = at.agent("nothink", seed=rep, label=f"{stem}_agent{rep}", sampling=at.SAMPLING)
            r.update(kind="agent", model=stem, rep=rep, size_gib=size)
            log(r)
            print(f"   agent r{rep}: {'SUCCESS' if r['success'] else 'FAIL'} {r['wall_s']} s, turns {r['turns']}, "
                  f"visible {r['visible_ok']} hidden {r['hidden_ok']}", flush=True)
        log(dict(kind="model_done", model=stem, size_gib=size))
    finally:
        stop(proc)
        serve_log.close()


def report() -> None:
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()] if OUT.exists() else []
    models = list(dict.fromkeys(r["model"] for r in rows if r.get("kind") in {"task", "agent"}))
    stats = []
    for m in models:
        t = [r for r in rows if r.get("kind") == "task" and r["model"] == m]
        ag = [r for r in rows if r.get("kind") == "agent" and r["model"] == m]
        prof = next((r["profile"] for r in rows if r.get("kind") == "profile" and r["model"] == m), None) or {}
        sh = [r for r in t if r["cat"] == "short"]
        lo = [r for r in t if r["cat"] == "long"]
        gen = [r["gen_tps"] for r in t if r.get("completion_tokens", 0) > 30 and r.get("gen_tps")]
        rate = lambda x: sum(r["correct"] for r in x) / len(x) if x else 0.0
        stats.append(dict(
            model=m, size=next((r["size_gib"] for r in t), 0), ncmoe=prof.get("ncmoe"), kv=prof.get("kv_cache") or "f16",
            tuner_tg=prof.get("tg_tok_s"), tuner_pp=prof.get("pp_tok_s"),
            short=rate(sh), short_n=f"{sum(r['correct'] for r in sh)}/{len(sh)}", long=rate(lo),
            long_n=f"{sum(r['correct'] for r in lo)}/{len(lo)}", agent=f"{sum(r['success'] for r in ag)}/{len(ag)}",
            agent_rate=rate([dict(correct=r["success"]) for r in ag]) if ag else 0.0,
            tshort=sum(r["wall_s"] for r in sh) / len(sh) if sh else 0, tlong=sum(r["wall_s"] for r in lo) / len(lo) if lo else 0,
            tps=sum(gen) / len(gen) if gen else 0,
            tagent=sum(r["wall_s"] for r in ag) / len(ag) if ag else 0))
    for s in stats:
        s["quality"] = (s["short"] + s["long"] + s["agent_rate"]) / 3
    ranked = sorted(stats, key=lambda s: (-round(s["quality"], 2), -s["tps"]))
    out = ["# Model sweep (Legion/Firefly: one machine, one run per file)", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M}. Quality = mean of the short-task, long-task and agent-task pass rates (thinking off).",
           "Speed = measured generation tok/s over the answers longer than 30 tokens. Small samples: differences of one task are noise.", "",
           "## Ranking", "",
           "| # | model | size GiB | ncmoe | KV | tuner tok/s (gen / prompt) | measured gen tok/s | short | long | agent | quality | s/short | s/long | s/agent |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(ranked, 1):
        out.append(f"| {i} | {s['model']} | {s['size']} | {s['ncmoe']} | {s['kv']} | {s['tuner_tg']} / {s['tuner_pp']} | {s['tps']:.1f} | "
                   f"{s['short_n']} | {s['long_n']} | {s['agent']} | {s['quality']:.2f} | {s['tshort']:.1f} | {s['tlong']:.1f} | {s['tagent']:.0f} |")
    tasks = list(dict.fromkeys(r["task"] for r in rows if r.get("kind") == "task"))
    out += ["", "## Pass rate per task (correct / runs)", "", "| model | " + " | ".join(t[:2] for t in tasks) + " |",
            "|---|" + "---|" * len(tasks)]
    for s in ranked:
        cells = []
        for t in tasks:
            x = [r for r in rows if r.get("kind") == "task" and r["model"] == s["model"] and r["task"] == t]
            cells.append(f"{sum(r['correct'] for r in x)}/{len(x)}" if x else "-")
        out.append(f"| {s['model']} | " + " | ".join(cells) + " |")
    out += ["", "Task ids: " + "; ".join(t for t in tasks), ""]
    failed = [r for r in rows if r.get("kind") == "model_failed"]
    if failed:
        out += ["## Failed to run", ""] + [f"- {r['model']} ({r['size_gib']} GiB): stage {r['stage']} - {r['detail']}" for r in failed] + [""]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(LLAMA))
    ap.add_argument("--only", default="", help="comma-separated substrings; only files whose name contains one of them")
    ap.add_argument("--skip", default="", help="comma-separated substrings of files to skip")
    ap.add_argument("--dense", default="", help="comma-separated substrings marking dense (non-MoE) models")
    ap.add_argument("--order", choices=["big", "small"], default="big")
    ap.add_argument("--reps-short", type=int, default=3)
    ap.add_argument("--reps-long", type=int, default=5)
    ap.add_argument("--agent-reps", type=int, default=2)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip models that already have a model_done row")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        return report()
    split = lambda s: [x.strip().lower() for x in s.split(",") if x.strip()]
    only, skip, dense = split(a.only), split(a.skip), split(a.dense)
    # absolute paths: the launcher changes into llama\LLAMA.CPP before it starts the exe, so a relative --dir would break
    files = [f.resolve() for f in Path(a.dir).resolve().glob("*.gguf") if "mmproj" not in f.name.lower()]
    files = [f for f in files if (not only or any(o in f.name.lower() for o in only)) and not any(s in f.name.lower() for s in skip)]
    files.sort(key=lambda f: f.stat().st_size, reverse=(a.order == "big"))
    done = {json.loads(l)["model"] for l in OUT.read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l).get("kind") == "model_done"} if (a.resume and OUT.exists()) else set()
    print("models:", ", ".join(f"{f.name} ({f.stat().st_size / 2**30:.1f} GiB)" for f in files), flush=True)
    if port_busy():
        sys.exit("port 8080 already serves something: stop the shared server first (and restart it with -Share afterwards)")
    for f in files:
        if f.stem in done:
            print(f"== {f.stem}: already done, skipped", flush=True)
            continue
        run_model(f, any(d in f.name.lower() for d in dense), a)
    report()


if __name__ == "__main__":
    main()
