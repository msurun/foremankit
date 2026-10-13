"""End-to-end latency per model: seconds from sending the prompt to the finished answer (not tokens/s).

python latency_bench.py [--dir ../llama] [--only Q4,IQ1] [--reps 3] [--report-only]

For every GGUF in the folder (mmproj skipped): serve it through llama/start-llama.ps1 with its tuned profile, send fixed
scenarios with streaming and thinking OFF, and record time to first token (ttft_s) and time to the complete answer (total_s):
  quick      ~30-token question, short answer
  summ2k     ~2k-token input, 3-bullet summary
  code       ~60-token spec, a function with docstring and doctests
  long10k    ~10k-token input, one-number answer (cold: unique prefix, cache_prompt off)
  cached10k  the same 10k prefix again with a different question (cache_prompt on, prefix primed beforehand)
  decode200  forced 200-token answer (counting), max_tokens 200
Every request starts with a unique nonce, so no result benefits from a leftover cache except cached10k on purpose.
Port 8080 must be free (stop a shared server first). Output: latency_results.jsonl and LATENCY_RESULTS.md (medians).
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model_sweep as ms
import quality_bench as qb

OUT = HERE / "latency_results.jsonl"
REPORT = HERE / "LATENCY_RESULTS.md"
URL = "http://127.0.0.1:8080/v1/chat/completions"
PREFIX10K = qb.DOC
SCENARIOS = [
    ("quick", "Explain in two sentences the difference between a Python list and a tuple.", 150),
    ("summ2k", qb.lines_filler(46) + "\n\nSummarize what these functions have in common in 3 short bullet points.", 250),
    ("code", ("Write a Python function merge_intervals(intervals) that merges overlapping [start, end] intervals. "
              "Include a docstring and three doctest examples."), 500),
    ("long10k", PREFIX10K + "\n\nWhat is the value of retry_limit? Answer with the number only.", 60),
    ("cached10k", None, 60),
    ("decode200", "Count from 1 to 400, separated by single spaces, and write nothing else.", 200),
]
ORDER = [name for name, _, _ in SCENARIOS]


def log(row: dict) -> None:
    row["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with OUT.open("a", encoding="utf-8") as fh:
        print(json.dumps(row, ensure_ascii=False), file=fh)


def ask(prompt: str, max_tokens: int, cache: bool, sampling: dict) -> dict:
    body = dict(model="q", messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, cache_prompt=cache,
                stream=True, stream_options={"include_usage": True}, chat_template_kwargs={"enable_thinking": False}, **sampling)
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, usage, timings = None, {}, {}
    with urllib.request.urlopen(req, timeout=1500) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            usage = d.get("usage") or usage
            timings = d.get("timings") or timings
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {"total_s": round(total, 2), "ttft_s": round(ttft, 2) if ttft is not None else None,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": timings.get("cache_n")}


def nonce() -> str:
    return f"[request {uuid.uuid4().hex[:10]}]\n"


def serve(model: Path):
    logf = (HERE / f"latency_server_{model.stem}.log").open("w")
    proc = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ms.LAUNCHER),
                             "-Model", str(model)], stdout=logf, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(ms.HEALTH, timeout=3) as r:
                if r.status == 200:
                    return proc, logf
        except OSError:
            pass
        time.sleep(3)
    ms.stop(proc)
    logf.close()
    return None, None


def run_model(model: Path, reps: int) -> None:
    size = round(model.stat().st_size / 2**30, 1)
    proc, logf = serve(model)
    if proc is None:
        log({"kind": "model_failed", "model": model.stem, "size_gib": size, "detail": "server did not come up"})
        print(f"!! server for {model.stem} did not come up", flush=True)
        return
    sampling = qb.GEMMA_SAMPLING if "gemma" in model.name.lower() else qb.SAMPLING
    try:
        ask(nonce() + "Say hi.", 20, False, sampling)          # warm-up, not recorded
        for name, prompt, max_tokens in SCENARIOS:
            for rep in range(1, reps + 1):
                try:
                    if name == "cached10k":
                        n = nonce()
                        ask(n + PREFIX10K + "\n\nWhat is shard_count? Answer with the number only.", 20, True, sampling)   # prime
                        r = ask(n + PREFIX10K + "\n\nWhat is cache_ttl_seconds? Answer with the number only.", max_tokens, True, sampling)
                    else:
                        r = ask(nonce() + prompt, max_tokens, False, sampling)
                except Exception as exc:                        # noqa: BLE001
                    r = {"error": str(exc)[:120]}
                r.update(kind="run", model=model.stem, size_gib=size, scenario=name, rep=rep)
                log(r)
                print(f"   {name:10s} r{rep} total {r.get('total_s', '-')}s ttft {r.get('ttft_s', '-')}s "
                      f"in {r.get('prompt_tokens', '-')} out {r.get('completion_tokens', '-')} cached {r.get('cached_tokens', '-')}", flush=True)
        log({"kind": "model_done", "model": model.stem, "size_gib": size})
    finally:
        ms.stop(proc)
        logf.close()


def report() -> None:
    rows = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()] if OUT.exists() else []
    runs = [r for r in rows if r.get("kind") == "run" and not r.get("error")]
    models = list(dict.fromkeys(r["model"] for r in runs))

    def med(m, sc, key):
        v = [r[key] for r in runs if r["model"] == m and r["scenario"] == sc and r.get(key) is not None]
        return statistics.median(v) if v else None

    cell = lambda x: "-" if x is None else f"{x:.1f}"
    out = ["# Latency: seconds from prompt to the finished answer", "",
           (f"Generated {datetime.now():%Y-%m-%d %H:%M}. Median of the repetitions per scenario, thinking OFF, streaming, unique prompt nonce "
            "(cold cache) except cached10k."), "",
           "| model | GiB | " + " | ".join(f"{s} (s)" for s in ORDER) + " | long10k first token (s) |",
           "|---|---|" + "---|" * (len(ORDER) + 1)]
    for m in sorted(models, key=lambda m: med(m, "long10k", "total_s") or 1e9):
        size = next(r["size_gib"] for r in runs if r["model"] == m)
        out.append(f"| {m} | {size} | " + " | ".join(cell(med(m, s, "total_s")) for s in ORDER) + f" | {cell(med(m, 'long10k', 'ttft_s'))} |")
    toks = {s: (med(models[0], s, "prompt_tokens"), med(models[0], s, "completion_tokens")) for s in ORDER} if models else {}
    out += ["", "Typical size (first model): " + "; ".join(f"{s}: in {int(i) if i else '-'} / out {int(o) if o else '-'} tokens"
                                                         for s, (i, o) in toks.items()), ""]
    failed = [r for r in rows if r.get("kind") == "model_failed"]
    if failed:
        out += ["## Failed to run", ""] + [f"- {r['model']}: {r['detail']}" for r in failed] + [""]
    errors = [r for r in rows if r.get("kind") == "run" and r.get("error")]
    if errors:
        out += [f"{len(errors)} request(s) ended with an error (excluded from the medians)."]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ms.LLAMA))
    ap.add_argument("--only", default="", help="comma-separated substrings of file names")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        return report()
    only = [x.strip().lower() for x in a.only.split(",") if x.strip()]
    files = [f for f in Path(a.dir).resolve().glob("*.gguf") if "mmproj" not in f.name.lower()
             and (not only or any(o in f.name.lower() for o in only))]
    files.sort(key=lambda f: f.stat().st_size, reverse=True)
    print("models:", ", ".join(f"{f.name} ({f.stat().st_size / 2**30:.1f} GiB)" for f in files), flush=True)
    if ms.port_busy():
        sys.exit("port 8080 already serves something: stop the shared server first (and restart it with -Share afterwards)")
    for f in files:
        print(f"== {f.stem}", flush=True)
        run_model(f, a.reps)
    report()


if __name__ == "__main__":
    main()
