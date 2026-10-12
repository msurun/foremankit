"""Cloud vs local on the bulk_code tasks: the same code tasks and the same automatic tests as quality_bench.

python cloud_code_bench.py [--providers mistral,groq,deepseek:flash] [--reps 3] [--report-only]

Each provider is forced by putting it alone in the bulk_code chain, so the real foremankit path is used (bulk_code system prompt,
reasoning level from routing.yaml, budget, call log). Tasks: S3, S4, S5 (short functions, --reps repetitions) and L2 (TTL cache,
5 repetitions). The prompts are synthetic public code tasks, safe for any provider. Run from this folder.
Output: cloud_code_results.jsonl and CLOUD_VS_LOCAL.md (cloud medians next to the local rows of sweep_results.jsonl).
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quality_bench as qb

from foremankit.router import Router, RouterEnv, RouterError

OUT = HERE / "cloud_code_results.jsonl"
SWEEP = HERE / "sweep_results.jsonl"
REPORT = HERE / "CLOUD_VS_LOCAL.md"
PICK = ("S3", "S4", "S5", "L2")
GAP_S = {"mistral": 32.0, "groq": 4.0, "deepseek": 3.0}      # min seconds between calls (Mistral free tier: 2 rpm)


def log(row: dict) -> None:
    row["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with OUT.open("a", encoding="utf-8") as fh:
        print(json.dumps(row, ensure_ascii=False), file=fh)


def run_provider(entry: str, reps_short: int, reps_long: int) -> None:
    router = Router(env=RouterEnv(out_dir=HERE / ".agent-out"))
    router.routing.setdefault("chains", {})["bulk_code"] = [entry]
    gap = GAP_S.get(entry.partition(":")[0], 3.0)
    tasks = [t for t in qb.TASKS if t[1][:2] in PICK]
    last = 0.0
    for cat, name, prompt, check in tasks:
        for rep in range(1, (reps_long if cat == "long" else reps_short) + 1):
            time.sleep(max(0.0, last + gap - time.time()))
            row = {"kind": "run", "provider": entry, "task": name, "rep": rep}
            for attempt in (1, 2):
                try:
                    r = router.delegate(prompt, kind="bulk_code", sensitivity="normal", max_output_tokens=3000)
                    ok, why = check(Path(r["path"]).read_text(encoding="utf-8"))
                    row.update(correct=bool(ok), why=why, model=r["model"], latency_s=r["latency_s"], tokens_in=r["tokens_in"],
                               tokens_out=r["tokens_out"], cost_usd=r["cost_usd"], actual_provider=r["provider"])
                    break
                except RouterError as exc:
                    row.update(error=str(exc)[:160], correct=False)
                    if attempt == 1:
                        time.sleep(65)                 # rate-limit cooldown, then one retry
                except Exception as exc:               # noqa: BLE001
                    row.update(error=f"{exc.__class__.__name__}: {exc}"[:160], correct=False)
                    break
            last = time.time()
            log(row)
            print(f"{entry:15s} {name[:24]:24s} r{rep} {'OK ' if row.get('correct') else 'BAD'} {row.get('latency_s', '-')}s "
                  f"out {row.get('tokens_out', '-')} ${row.get('cost_usd', 0)} {row.get('error', '')}", flush=True)


def report() -> None:
    rows = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()] if OUT.exists() else []
    cloud = [r for r in rows if r.get("kind") == "run"]
    sweep = [json.loads(line) for line in SWEEP.read_text(encoding="utf-8").splitlines() if line.strip()] if SWEEP.exists() else []
    local = [r for r in sweep if r.get("kind") == "task" and r["task"][:2] in PICK and not r.get("error")]
    tasks = sorted({r["task"] for r in local} | {r["task"] for r in cloud}, key=lambda t: t[:2])

    def line(label, runs, time_key, cost=False):
        cells = []
        for t in tasks:
            x = [r for r in runs if r["task"] == t]
            cells.append(f"{sum(bool(r.get('correct')) for r in x)}/{len(x)}" if x else "-")
        tot = sum(bool(r.get("correct")) for r in runs)
        times = [r[time_key] for r in runs if r.get(time_key) is not None and not r.get("error")]
        spent = sum(r.get("cost_usd", 0) or 0 for r in runs)
        return (f"| {label} | " + " | ".join(cells) + f" | {tot}/{len(runs)} | "
                f"{statistics.median(times):.1f}" if times else f"| {label} | " + " | ".join(cells) + f" | {tot}/{len(runs)} | -") + \
               (f" | ${spent:.4f} |" if cost else " | $0 (local) |")

    out = ["# bulk_code: cloud providers vs local models (same tasks, same automatic tests)", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M}. Tasks: " + "; ".join(tasks) + ". Cloud rows go through foremankit's bulk_code path "
           "(system prompt, minimal reasoning); local rows are the raw prompt on llama-server (thinking off). Seconds = median "
           "request time (cloud latency_s, local wall_s). Small samples: one task more or less is noise.", "",
           "| source | " + " | ".join(t[:2] for t in tasks) + " | total | median s | cost |", "|---|" + "---|" * (len(tasks) + 3)]
    for p in dict.fromkeys(r["provider"] for r in cloud):
        out.append(line(f"cloud {p}", [r for r in cloud if r["provider"] == p], "latency_s", cost=True))
    for m in dict.fromkeys(r["model"] for r in local):
        out.append(line(f"local {m.replace('Qwen3.6-35B-A3B-UD-', '').replace('Qwen3.5-9B-UD-', '9B-')}",
                        [r for r in local if r["model"] == m], "wall_s"))
    errs = [r for r in cloud if r.get("error")]
    if errs:
        out += ["", f"{len(errs)} cloud request(s) failed (counted as incorrect): " + "; ".join(sorted({r['error'][:60] for r in errs}))]
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="mistral,groq,deepseek:flash")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        for entry in [p.strip() for p in a.providers.split(",") if p.strip()]:
            run_provider(entry, a.reps, 5)
    report()


if __name__ == "__main__":
    main()
