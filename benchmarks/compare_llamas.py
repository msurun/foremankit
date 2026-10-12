"""Same tasks sent DIRECTLY to the two llama-servers (Firefly local, Legion over the LAN): time and correctness.

python compare_llamas.py [--reps 2] [--think-tasks S2,L3]

Bypasses the router. Same sampling, cache_prompt off, thinking off for every task plus thinking on for --think-tasks.
Both machines work on the SAME task AT THE SAME TIME (two independent computers, one thread each): half the wall time of a
sequential run, and each timing still measures only that machine (the client threads are negligible).
Tasks and automatic checks come from quality_bench.py. Output: compare_llamas.jsonl and a summary table on stdout.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from quality_bench import MAXTOK, SAMPLING, TASKS

from foremankit.config import load_env

load_env()
OUT = HERE / "compare_llamas.jsonl"
KEY = os.environ.get("FOREMANKIT_PEER_KEY", "")
MACHINES = {"firefly": ("http://127.0.0.1:8080", {}),
            "legion": ("http://192.168.4.83:8080", {"Authorization": f"Bearer {KEY}"})}


def ask(machine: str, prompt: str, cat: str, think: bool) -> dict:
    url, auth = MACHINES[machine]
    body = dict(model="q", messages=[{"role": "user", "content": prompt}], max_tokens=MAXTOK[cat], cache_prompt=False,
                chat_template_kwargs={"enable_thinking": think}, **SAMPLING)
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **auth})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1500) as r:
        data = json.loads(r.read())
    return data, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--think-tasks", default="S2,L3")
    a = ap.parse_args()
    think_ids = set(a.think_tasks.split(",")) if a.think_tasks else set()
    with ThreadPoolExecutor(len(MACHINES)) as pool:        # warm-up: first request after idle should not distort a timing
        list(pool.map(lambda m: ask(m, "Reply with OK.", "short", False), MACHINES))
    plan = [(cat, name, prompt, check, False) for cat, name, prompt, check in TASKS]
    plan += [(cat, name, prompt, check, True) for cat, name, prompt, check in TASKS if name[:2] in think_ids]
    for cat, name, prompt, check, think in plan:
        reps = a.reps if cat == "short" and not think else (1 if think else a.reps)
        for rep in range(reps):
            with ThreadPoolExecutor(len(MACHINES)) as pool:        # both machines at once
                futures = {m: pool.submit(ask, m, prompt, cat, think) for m in MACHINES}
                results = {m: f.result() for m, f in futures.items()}
            for m, (data, wall) in results.items():
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                ok, why = check(content)
                t = data.get("timings", {})
                row = dict(machine=m, task=name, cat=cat, think=think, rep=rep + 1, correct=bool(ok), why=why,
                           wall_s=round(wall, 2), prompt_tokens=data["usage"]["prompt_tokens"],
                           completion_tokens=data["usage"]["completion_tokens"], prompt_ms=round(t.get("prompt_ms", 0)),
                           gen_tps=round(t.get("predicted_per_second", 0), 1), prompt_tps=round(t.get("prompt_per_second", 0), 1),
                           truncated=data["choices"][0]["finish_reason"] == "length", answer=" ".join(content[:70].split()),
                           time=datetime.now().strftime("%H:%M:%S"))
                with OUT.open("a", encoding="utf-8") as fh:
                    print(json.dumps(row, ensure_ascii=False), file=fh)
                print(f"{name[:30]:30s} {'think' if think else 'nothink':7s} r{rep + 1} {m:7s} {'OK ' if ok else 'BAD'} "
                      f"{wall:6.1f}s {row['completion_tokens']:5d}tok {row['gen_tps']:5.1f}t/s {'TRUNC' if row['truncated'] else ''}",
                      flush=True)


if __name__ == "__main__":
    main()
