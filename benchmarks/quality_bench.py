"""Q4_K_XL vs Q5_K_M: speed AND correctness on short and long tasks, thinking on and off.

python quality_bench.py [--models q4,q5] [--reps-short 2] [--only short|long]

One llama-server per model (own start/stop, port 8080 must be free), each at its tuned settings (see llama/profiles.json:
Q4 ncmoe 40, Q5 ncmoe 41, P-core mask, 12 threads). Both thinking modes go through the same server via
chat_template_kwargs.enable_thinking. cache_prompt is OFF so every request pays the full prompt cost (fair, no cache luck).
Every answer is checked automatically (numbers/JSON against known values, code executed against unit tests).
Output: quality_results.jsonl (one line per run) and QUALITY_RESULTS.md (summary).
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from server_tests import MODELS, SERVER, URL, post, vram, wait_ready

OUT = HERE / "quality_results.jsonl"
MODELS["gemma"] = HERE.parent / "llama" / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
NCMOE = {"q4": "40", "q5": "41", "gemma": "31"}   # from llama/profiles.json (tuner)
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}   # model card "precise coding"
GEMMA_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0, "presence_penalty": 0.0}   # Google defaults (Unsloth docs)
MAXTOK = {"short": 3000, "long": 6000}


# ---------------------------------------------------------------- tasks
def lines_filler(n):
    return "\n".join(f"def handler_{i}(x, y): return (x * {i * 7 + 3}) + y - {zlib.crc32(str(i).encode()) % 97}" for i in range(n))


def make_long_doc():
    lines = lines_filler(230).split("\n")
    for pos, text in ((20, "# CONFIG: retry_limit = 17"), (70, "# CONFIG: retry_limit_old = 9  (deprecated)"),
                      (120, "# CONFIG: cache_ttl_seconds = 340"), (160, "# CONFIG: shard_count = 12"),
                      (200, "# CONFIG: shard_count_old = 4  (deprecated)")):
        lines.insert(pos, text)
    return "\n".join(lines)


def make_table():
    import random
    for seed in range(1, 200):
        rnd = random.Random(seed)
        vals = [rnd.randint(20, 80) for _ in range(30)]
        runs, cur = [], []
        for d, v in enumerate(vals, 1):
            if v > 50:
                cur.append((d, v))
            else:
                if cur:
                    runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        best = max(len(r) for r in runs)
        if sum(len(r) == best for r in runs) == 1 and best >= 4:
            r = next(r for r in runs if len(r) == best)
            exp = {"length": best, "start_day": r[0][0], "mean": round(sum(v for _, v in r) / best, 2)}
            csv = "day,value\n" + "\n".join(f"{d},{v}" for d, v in enumerate(vals, 1))
            return csv, exp
    raise RuntimeError("no table")


def last_json(text):
    for m in reversed(re.findall(r"\{[^{}]*\}", text or "", re.DOTALL)):
        try:
            return json.loads(m)
        except ValueError:
            continue
    return None


def code_of(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.DOTALL)
    return max(blocks, key=len) if blocks else None


def run_tests(text, tests):
    code = code_of(text)
    if not code:
        return False, "no code block"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.py"
        f.write_text(code + "\n\n" + tests + "\nprint('OK')\n", encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], capture_output=True, text=True, timeout=15, cwd=d, check=False)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return (r.returncode == 0 and "OK" in r.stdout), (r.stderr.strip().splitlines() or [""])[-1][:120]


def chk_num(expected):
    return lambda t: (bool(re.search(rf"(?<![\d./]){re.escape(str(expected))}(?!\d|\.\d|/)", t or "")), "")


def chk_word(word):
    last = lambda t: ((t or "").strip().splitlines() or [""])[-1].lower()
    return lambda t: (word.lower() in last(t), "")


TESTS_BALANCED = """
assert is_balanced('') is True
assert is_balanced('([]{})') is True
assert is_balanced('(]') is False
assert is_balanced('((') is False
assert is_balanced('a(b)c') is True
assert is_balanced('}{') is False
assert is_balanced('([)]') is False
"""
TESTS_DATES = """
assert extract_iso_dates('on 2024-02-29 and 2023-02-29 and 2023-13-01, 2021-12-31.') == ['2024-02-29', '2021-12-31']
assert extract_iso_dates('12024-01-01 and 2024-01-011') == []
assert extract_iso_dates('none here') == []
assert extract_iso_dates('1900-02-29 2000-02-29 2024-04-31') == ['2000-02-29']
"""
TESTS_MA = """
assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
assert moving_average([5], 1) == [5.0]
assert moving_average([1, 2], 3) == []
assert all(abs(a - b) < 1e-9 for a, b in zip(moving_average([1, 2, 6, 4, 9], 3), [3.0, 4.0, 19 / 3]))
try:
    moving_average([1, 2], 0)
    raise SystemExit('no ValueError')
except ValueError:
    pass
"""
TESTS_TTL = """
t = [0]
c = TTLCache(2, 10, clock=lambda: t[0])
c.put('a', 1); c.put('b', 2)
assert c.get('a') == 1
c.put('c', 3)
assert c.get('b') is None and len(c) == 2
t[0] = 11
assert c.get('a') is None and c.get('c') is None and len(c) == 0
c.put('x', 1); t[0] = 15; c.put('x', 2); t[0] = 24
assert c.get('x') == 2
t[0] = 26
assert c.get('x') is None
c.put('d', 4)
assert c.delete('d') is True and c.delete('d') is False
t = [0]
c2 = TTLCache(2, 10, clock=lambda: t[0])
c2.put('a', 1); t[0] = 5; c2.put('b', 2); t[0] = 12; c2.put('c', 3)
assert c2.get('b') == 2 and c2.get('c') == 3 and len(c2) == 2
t = [0]
c3 = TTLCache(3, 10, clock=lambda: t[0])
c3.put('a', 1); c3.put('b', 2); c3.put('c', 3); c3.get('a'); c3.put('d', 4)
assert c3.get('b') is None and c3.get('a') == 1
"""

CSV, TABLE_EXP = make_table()
DOC = make_long_doc()


def chk_json_log(t):
    j = last_json(t)
    ok = isinstance(j, dict) and j.get("ts") == "2026-10-09T08:15:22Z" and str(j.get("level", "")).lower() == "warn" \
        and j.get("user") == "j.doe" and j.get("latency_ms") == 1842 and j.get("ok") is False
    return ok, ""


def chk_doc(t):
    j = last_json(t)
    return (isinstance(j, dict) and j.get("retry_limit") == 17 and j.get("cache_ttl_seconds") == 340
            and j.get("shard_count") == 12 and j.get("sum") == 369), ""


def chk_table(t):
    j = last_json(t)
    return (isinstance(j, dict) and j.get("length") == TABLE_EXP["length"] and j.get("start_day") == TABLE_EXP["start_day"]
            and abs(float(j.get("mean", -1)) - TABLE_EXP["mean"]) < 0.011), ""


TASKS = [
    ("short", "S1 pipes (math)", ("Pipe A fills a tank in 6 hours, pipe B in 4 hours, and drain C empties a full tank in 12 hours. "
     "All three are open at once. How many hours until the empty tank is full? Give only the number of hours."), chk_num(3)),
    ("short", "S2 ordering (logic)", ("Alice is taller than Bob. Carol is shorter than Bob. Dave is taller than Alice. "
     "Who is the second tallest of the four? Answer with just the name."), chk_word("Alice")),
    ("short", "S3 balanced brackets (code)", ("Write a Python function is_balanced(s) that returns True if the brackets ()[]{} in "
     "the string s are correctly balanced and nested (other characters are ignored). Reply with one ```python code block only."),
     lambda t: run_tests(t, TESTS_BALANCED)),
    ("short", "S4 ISO dates (code)", ("Write a Python function extract_iso_dates(text) that returns the list, in order, of all valid "
     "calendar dates written as YYYY-MM-DD in text (leap years respected; month 01-12; a date must not be part of a longer digit "
     "sequence). Reply with one ```python code block only."), lambda t: run_tests(t, TESTS_DATES)),
    ("short", "S5 moving average (code)", ("Write a Python function moving_average(xs, k) returning the list of averages (floats) of every "
     "window of k consecutive elements; return [] if k > len(xs); raise ValueError if k <= 0. Reply with one ```python code block only."),
     lambda t: run_tests(t, TESTS_MA)),
    ("short", "S6 log to JSON (extraction)", ("Extract from this log line a JSON object with keys ts (string), level (lowercase string), "
     "user (string), latency_ms (integer), ok (boolean, true only if status is OK). Output only the JSON.\n"
     "2026-10-09T08:15:22Z [WARN] user=j.doe action=login took 1.842s status=FAILED retries=2"), chk_json_log),
    ("long", "L1 needle in 10k-token code (retrieval)", "Below is a large code file. Find the CURRENT (non-deprecated) values of the CONFIG keys "
     "retry_limit, cache_ttl_seconds and shard_count, and their sum. Answer with only JSON like "
     '{"retry_limit":0,"cache_ttl_seconds":0,"shard_count":0,"sum":0}.\n\n' + DOC, chk_doc),
    ("long", "L2 TTL cache (implementation)", ("Implement in Python a class TTLCache(capacity, ttl, clock=time.monotonic) with: "
     "put(key, value) (inserts or updates; sets expiry to clock()+ttl; marks the key most recently used; when the number of "
     "non-expired items would exceed capacity, first purge expired items and only then evict the least recently used one), "
     "get(key, default=None) (returns the value if present and not expired, else default; a hit marks the key most recently used "
     "but does NOT extend its expiry; an expired key is removed), delete(key) -> bool (True if a live key was removed), "
     "__len__ (number of non-expired items). An item is expired when clock() > its expiry. Include import time. "
     "Reply with one ```python code block only."), lambda t: run_tests(t, TESTS_TTL)),
    ("long", "L3 longest run in a table (multi-step reasoning)", "Daily readings follow (day,value). Find the longest run of consecutive "
     "days in which value > 50 (the longest run is unique). Report its length, its first day and the mean of its values rounded to "
     'two decimals. Do the calculation yourself, no code. Answer with only JSON like {"length":0,"start_day":0,"mean":0.00}.\n\n' + CSV,
     chk_table),
]


# ---------------------------------------------------------------- run
def start_server(key):
    cmd = [str(SERVER), "-m", str(MODELS[key]), "--alias", "q", "--host", "127.0.0.1", "--port", "8080", "-c", "32768",
           "-ngl", "99", "--n-cpu-moe", NCMOE[key], "--load-mode", "none", "--jinja", "-fa", "on", "-np", "1",
           "-t", "12", "-tb", "12", "-C", "0xFFC03", "--cpu-strict", "1", "--no-webui"]
    if key == "gemma":
        cmd += ["-ctk", "q8_0", "-ctv", "q8_0"]   # needed to fit 32k context in 4 GB VRAM (llama/profiles.json)
    lf = (HERE / f"quality_server_{key}.log").open("w")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    wait_ready(proc, lf.name)
    return proc


def one(model, mode, cat, name, prompt, check):
    body = dict(model="q", messages=[{"role": "user", "content": prompt}], max_tokens=MAXTOK[cat], cache_prompt=False,
                chat_template_kwargs={"enable_thinking": mode == "think"}, **(GEMMA_SAMPLING if model == "gemma" else SAMPLING))
    try:
        d, wall = post("/v1/chat/completions", body, timeout=1500)
    except Exception as e:  # noqa: BLE001 - any server or network failure is recorded as a failed task
        return {"model": model, "mode": mode, "cat": cat, "task": name, "error": str(e)[:100], "correct": False}
    m = d["choices"][0]["message"]
    content = m.get("content") or ""
    ok, why = check(content)
    t = d.get("timings", {})
    return {"model": model, "mode": mode, "cat": cat, "task": name, "correct": bool(ok), "why": why, "wall_s": round(wall, 1),
            "prompt_tokens": d["usage"]["prompt_tokens"], "completion_tokens": d["usage"]["completion_tokens"],
            "reasoning_chars": len(m.get("reasoning_content") or ""), "truncated": d["choices"][0]["finish_reason"] == "length",
            "prompt_ms": round(t.get("prompt_ms", 0)), "gen_tps": round(t.get("predicted_per_second", 0), 1),
            "answer": content[:80].replace("\n", " ")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="q4,q5")
    ap.add_argument("--reps-short", type=int, default=2)
    ap.add_argument("--only")
    a = ap.parse_args()
    try:
        urllib.request.urlopen(URL + "/health", timeout=2)
        sys.exit("port 8080 already serves something")
    except OSError:
        pass
    for key in a.models.split(","):
        proc = start_server(key)
        print(f"== {key} ready, VRAM {vram()} MiB", flush=True)
        try:
            for mode in ("nothink", "think"):
                for cat, name, prompt, check in TASKS:
                    if a.only and cat != a.only:
                        continue
                    for rep in range(a.reps_short if cat == "short" else 1):
                        r = one(key, mode, cat, name, prompt, check)
                        r["rep"] = rep + 1
                        r["time"] = datetime.now().strftime("%H:%M:%S")
                        with OUT.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                        print(f"{key} {mode:7s} {name[:34]:34s} r{rep + 1} {'OK ' if r['correct'] else 'BAD'} "
                              f"{r.get('wall_s', '-')}s {r.get('completion_tokens', '-')}tok {r.get('gen_tps', '-')}t/s "
                              f"{'TRUNC' if r.get('truncated') else ''} {r.get('why', '') or r.get('error', '')}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
