"""Real-conditions tests through llama-server: time to first token at 10-20k context, prefix/KV-shift reuse,
thinking-off mechanisms, clean tool-call JSON.

python server_tests.py --model q4|q5 [--reuse 256] [--reasoning-off] [--ncmoe 40] [--tests ttft,shift,think,tools]
Starts its own llama-server on port 8080 (refuses if the port is busy) and stops it at the end.
Appends one JSON line per test to server_results.jsonl.
"""
import argparse
import json
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA = HERE.parent / "llama"
SERVER = LLAMA / "LLAMA.CPP" / "llama-server.exe"
MODELS = {"q5": LLAMA / "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
          "q4": LLAMA / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"}   # the q5 file was deleted on 2026-10-09 (its results stay in the logs)
OUT = HERE / "server_results.jsonl"
URL = "http://127.0.0.1:8080"

# Qwen3.6 model card (Hugging Face): precise coding / thinking, and instruct (non-thinking)
CARD_CODING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
CARD_INSTRUCT = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data, time.time() - t0


def log(rec):
    rec["time"] = datetime.now().strftime("%H:%M:%S")
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def vram():
    try:
        return float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=10, check=False).stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def filler(tag, n_lines):
    """Unique pseudo-code lines, ~12 tokens each, so nothing repeats and nothing is cached by accident."""
    return "\n".join(f"def handler_{tag}_{i}(x_{i}, y_{i}): return (x_{i} * {i * 7 + 3}) + y_{i} - {tag.__hash__() % 97 + i}"
                     for i in range(n_lines)) + "\n"


def chat(content, max_tokens=8, extra=None, sampling=CARD_CODING, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body = dict(model="q", messages=msgs, max_tokens=max_tokens, **sampling, **(extra or {}))
    return post("/v1/chat/completions", body)


def timing(d):
    t = d.get("timings", {})
    return {"prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": round(t.get("prompt_ms", 0)),
            "prompt_tps": round(t.get("prompt_per_second", 0), 1), "gen_tps": round(t.get("predicted_per_second", 0), 1),
            "prompt_tokens": d["usage"]["prompt_tokens"]}


def test_ttft(tag):
    # 10k cold, same prefix + new question, 20k cold (different content)
    sys10 = "You are a coding agent. Reference code follows.\n" + filler("a", 230)
    d, s = chat("Question 1: name the first function. One word.", extra=NOTHINK, system=sys10)
    log(dict(test="ttft_10k_cold", cfg=tag, wall_s=round(s, 1), **timing(d)))
    d, s = chat("Question 2: name the second function. One word.", extra=NOTHINK, system=sys10)
    log(dict(test="ttft_10k_prefix_reuse", cfg=tag, wall_s=round(s, 1), **timing(d)))
    sys20 = "You are a coding agent. Reference code follows.\n" + filler("b", 460)
    d, s = chat("Name the first function. One word.", extra=NOTHINK, system=sys20)
    log(dict(test="ttft_20k_cold", cfg=tag, wall_s=round(s, 1), vram_MiB=vram(), **timing(d)))


def test_shift(tag):
    # context compaction scenario: A+B+C then A+C (B removed). With --cache-reuse C is shifted instead of recomputed.
    A, B, C = filler("sa", 120), filler("sb", 120), filler("sc", 120)
    chat("Q: say ok.", extra=NOTHINK, system="Ctx.\n" + A + B + C)
    d, s = chat("Q: say ok.", extra=NOTHINK, system="Ctx.\n" + A + C)
    log(dict(test="kv_shift_drop_middle", cfg=tag, wall_s=round(s, 1), **timing(d)))


def test_think(tag):
    q = "What is 17*23? Answer with just the number."
    for name, extra in (("default (no kwarg)", {}), ("enable_thinking=false", NOTHINK)):
        d, s = chat(q, max_tokens=1500, extra=extra, sampling=CARD_INSTRUCT)
        m = d["choices"][0]["message"]
        log({"test": "thinking", "cfg": tag, "mode": name, "wall_s": round(s, 1), "completion_tokens": d["usage"]["completion_tokens"],
             "has_reasoning": bool(m.get("reasoning_content")), "content": (m.get("content") or "")[:60]})


TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from disk", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write text to a file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search a regex in files", "parameters": {
        "type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in a file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
        "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "query_pi", "description": "Read a PI tag history", "parameters": {
        "type": "object", "properties": {"tag": {"type": "string"}, "hours": {"type": "integer"}, "interval_min": {"type": "integer"}},
        "required": ["tag", "hours"]}}},
]

# (prompt, expected tool or None, check(args)->bool)
TASKS = [
    ("Show me the contents of src/main.py", "read_file", lambda a: a.get("path", "").replace("\\", "/").endswith("src/main.py")),
    ("Run git status in the repository.", "run_shell", lambda a: a.get("command", "").strip().startswith("git status")),
    ("Find every TODO comment under the src folder.", "grep", lambda a: "TODO" in a.get("pattern", "")),
    ("Create notes.txt containing exactly: line one\\nline \"two\" with quotes and a backslash \\ char", "write_file",
     lambda a: a.get("path", "").endswith("notes.txt") and 'line "two"' in a.get("content", "")),
    ("In config.yaml replace the text `max_tokens: 1024` with `max_tokens: 4096`.", "edit_file",
     lambda a: "1024" in a.get("old", "") and "4096" in a.get("new", "")),
    ("Pull the last 36 hours of tag HPGR011.PRESS.SP at 5 minute resolution from PI.", "query_pi",
     lambda a: a.get("tag") == "HPGR011.PRESS.SP" and a.get("hours") == 36 and a.get("interval_min") == 5),
    ("What is the capital of Chile? Answer from your own knowledge, no tools.", None, lambda a: True),
    ("Read pyproject.toml and then tell me the project name.", "read_file", lambda a: "pyproject.toml" in a.get("path", "")),
]


def test_tools(tag, sampling_name, sampling, reps=3):
    sysmsg = "You are a coding assistant. Use the provided tools when they are needed; otherwise answer directly."
    ok = bad_json = n = 0
    fails = []
    secs = []
    for prompt, want, check in TASKS:
        for _ in range(reps):
            n += 1
            body = dict(model="q", messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": prompt}],
                        tools=TOOLS, max_tokens=400, **sampling, **NOTHINK)
            try:
                d, s = post("/v1/chat/completions", body)
            except (OSError, ValueError) as e:
                fails.append((prompt[:30], f"HTTP {e}"))
                continue
            secs.append(s)
            m = d["choices"][0]["message"]
            calls = m.get("tool_calls") or []
            if want is None:
                good = not calls and bool(m.get("content"))
            elif not calls:
                good = False
                fails.append((prompt[:30], "no tool call; content=" + (m.get("content") or "")[:60]))
            else:
                fn = calls[0]["function"]
                try:
                    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                except ValueError:
                    bad_json += 1
                    fails.append((prompt[:30], "invalid JSON args: " + str(fn["arguments"])[:60]))
                    continue
                good = fn["name"] == want and check(args)
                if not good:
                    fails.append((prompt[:30], f"{fn['name']} {json.dumps(args)[:90]}"))
            ok += good
    log({"test": "tool_calling", "cfg": tag, "sampling": sampling_name, "ok": ok, "of": n, "invalid_json": bad_json,
         "mean_s": round(sum(secs) / max(len(secs), 1), 1), "fails": fails[:6]})


def wait_ready(proc, log_path, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early (code {proc.returncode}); see {log_path}")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=3) as r:
                if r.status == 200:
                    return time.time() - t0
        except OSError:
            time.sleep(3)
    raise SystemExit("server did not become ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q4")
    ap.add_argument("--ncmoe", default="40")
    ap.add_argument("--reuse", default="0")
    ap.add_argument("--reasoning-off", action="store_true")
    ap.add_argument("--ctx", default="32768")
    ap.add_argument("--tests", default="ttft,shift,think,tools")
    a = ap.parse_args()
    try:
        urllib.request.urlopen(URL + "/health", timeout=2)
        raise SystemExit("port 8080 already serves something - refusing to start a second instance")
    except OSError:
        pass
    cmd = [str(SERVER), "-m", str(MODELS[a.model]), "--alias", "q", "--host", "127.0.0.1", "--port", "8080", "-c", a.ctx,
           "-ngl", "99", "--n-cpu-moe", a.ncmoe, "--load-mode", "none", "--jinja", "-fa", "on", "-np", "1",
           "-t", "12", "-C", "0xFFC03", "--cpu-strict", "1", "--no-webui", "--temp", "0.6", "--top-p", "0.95",
           "--top-k", "20", "--min-p", "0"]
    if a.reuse != "0":
        cmd += ["--cache-reuse", a.reuse]
    if a.reasoning_off:
        cmd += ["--reasoning", "off"]
    tag = f"{a.model} ncmoe={a.ncmoe} reuse={a.reuse} reasoning_off={a.reasoning_off}"
    log_path = HERE / f"server_{a.model}_{a.reuse}_{int(a.reasoning_off)}.log"
    print("CMD:", " ".join(cmd), flush=True)
    with log_path.open("w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            log({"test": "server_start", "cfg": tag, "ready_s": round(wait_ready(proc, log_path)), "vram_MiB": vram(), "cmd": " ".join(cmd)})
            for t in a.tests.split(","):
                if t == "ttft":
                    test_ttft(tag)
                elif t == "shift":
                    test_shift(tag)
                elif t == "think":
                    test_think(tag)
                elif t == "tools":
                    test_tools(tag, "card-coding t0.6 pp0", CARD_CODING)
                    test_tools(tag, "card-instruct t0.7 pp1.5", CARD_INSTRUCT)
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
