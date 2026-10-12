"""End-to-end agentic task, same for every model: fix a small buggy module until its tests pass, using tools.

python agent_task.py --models q4,q5 --modes nothink,think --reps 2 [--ncmoe q4=40,q5=41,gemma=NN] [--selftest]

The model gets a workspace with plantdata.py (4 buggy functions) and test_plantdata.py, and the tools
list_files / read_file / write_file / run_tests. It loops until it stops calling tools (or 20 turns / 15 min).
Afterwards the VISIBLE tests and a set of HIDDEN edge-case tests are run independently: success = both green.
Measured: wall time, turns, tool calls, prompt/completion tokens, time spent reading prompts vs generating.
Output: agent_results.jsonl.
"""
import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import server_tests as st  # noqa: E402
from server_tests import SERVER, URL, post, vram, wait_ready  # noqa: E402

OUT = HERE / "agent_results.jsonl"
st.MODELS["gemma"] = HERE.parent / "llama" / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
NCMOE_DEFAULT = {"q4": "40", "q5": "41", "gemma": "31"}   # from llama/profiles.json (tuner)
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0)
GEMMA_SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=64, min_p=0.0, presence_penalty=0.0)   # Google defaults (Unsloth docs)
MAX_TURNS, MAX_WALL = 20, 900

BUGGY = '''"""Small helpers for cleaning plant sensor series."""
from statistics import mean, pstdev


def moving_average(values, window):
    """Averages of every window of `window` consecutive values (len(values) - window + 1 items).
    Raises ValueError if window < 1; returns [] if window > len(values)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    out = []
    for i in range(len(values) - window):
        out.append(sum(values[i:i + window]) / window)
    return out


def fill_gaps(values):
    """Replace None entries that lie BETWEEN two known values by linear interpolation.
    None at the start or end of the series stay None."""
    out = list(values)
    for i, v in enumerate(out):
        if v is None and i > 0:
            out[i] = out[i - 1]
    return out


def remove_outliers(values, k=3.0):
    """Set to None every value farther than k robust sigmas from the median
    (robust sigma = 1.4826 * median absolute deviation). None entries are ignored and kept.
    If the MAD is 0 nothing is removed."""
    known = [v for v in values if v is not None]
    if not known:
        return list(values)
    m, s = mean(known), pstdev(known)
    return [None if v is not None and abs(v - m) > k * s else v for v in values]


def daily_summary(timestamps, values):
    """Group by calendar day (the YYYY-MM-DD prefix of each ISO timestamp) and return
    {day: {"min": .., "max": .., "mean": ..}} with mean rounded to 2 decimals.
    None values are ignored; days without any valid value are omitted.
    The days appear in the result in ascending date order."""
    days = {}
    for ts, v in zip(timestamps, values):
        days.setdefault(ts[:10], []).append(v)
    return {d: {"min": min(vs), "max": max(vs), "mean": mean(vs)} for d, vs in days.items()}
'''

FIXED = '''"""Small helpers for cleaning plant sensor series."""
from statistics import mean, median


def moving_average(values, window):
    if window < 1:
        raise ValueError("window must be >= 1")
    return [sum(values[i:i + window]) / window for i in range(len(values) - window + 1)]


def fill_gaps(values):
    out = list(values)
    known = [i for i, v in enumerate(out) if v is not None]
    for a, b in zip(known, known[1:]):
        for i in range(a + 1, b):
            out[i] = out[a] + (out[b] - out[a]) * (i - a) / (b - a)
    return out


def remove_outliers(values, k=3.0):
    known = [v for v in values if v is not None]
    if not known:
        return list(values)
    m = median(known)
    mad = median(abs(v - m) for v in known)
    if mad == 0:
        return list(values)
    s = 1.4826 * mad
    return [None if v is not None and abs(v - m) > k * s else v for v in values]


def daily_summary(timestamps, values):
    days = {}
    for ts, v in zip(timestamps, values):
        if v is not None:
            days.setdefault(ts[:10], []).append(v)
    return {d: {"min": min(vs), "max": max(vs), "mean": round(mean(vs), 2)} for d, vs in sorted(days.items())}
'''

TESTS = '''import unittest
from plantdata import moving_average, fill_gaps, remove_outliers, daily_summary


class MovingAverage(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])

    def test_full_window(self):
        self.assertEqual(moving_average([2, 4, 6], 3), [4.0])

    def test_window_larger_than_data(self):
        self.assertEqual(moving_average([1, 2], 3), [])

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            moving_average([1, 2], 0)


class FillGaps(unittest.TestCase):
    def test_linear_interpolation(self):
        self.assertEqual(fill_gaps([1.0, None, None, 4.0]), [1.0, 2.0, 3.0, 4.0])

    def test_edges_stay_none(self):
        self.assertEqual(fill_gaps([None, 2.0, None, 4.0, None]), [None, 2.0, 3.0, 4.0, None])


class RemoveOutliers(unittest.TestCase):
    def test_spike_removed(self):
        self.assertEqual(remove_outliers([10, 11, 10, 12, 11, 10, 500]), [10, 11, 10, 12, 11, 10, None])

    def test_none_kept_and_ignored(self):
        self.assertEqual(remove_outliers([10, None, 11, 10, 12, 11, 10, 500]), [10, None, 11, 10, 12, 11, 10, None])

    def test_clean_series_untouched(self):
        self.assertEqual(remove_outliers([5, 6, 5, 7, 6]), [5, 6, 5, 7, 6])


class DailySummary(unittest.TestCase):
    def test_two_days(self):
        ts = ["2026-10-08T23:00:00", "2026-10-09T00:00:00", "2026-10-09T12:00:00"]
        self.assertEqual(daily_summary(ts, [4.0, 1.0, 2.0]),
                         {"2026-10-08": {"min": 4.0, "max": 4.0, "mean": 4.0},
                          "2026-10-09": {"min": 1.0, "max": 2.0, "mean": 1.5}})

    def test_none_ignored(self):
        ts = ["2026-10-09T00:00:00", "2026-10-09T01:00:00", "2026-10-09T02:00:00"]
        self.assertEqual(daily_summary(ts, [1.0, None, 3.0])["2026-10-09"], {"min": 1.0, "max": 3.0, "mean": 2.0})

    def test_mean_rounded(self):
        ts = ["2026-10-09T00:00:00"] * 3
        self.assertEqual(daily_summary(ts, [1.0, 1.0, 2.0])["2026-10-09"]["mean"], 1.33)


if __name__ == "__main__":
    unittest.main()
'''

HIDDEN = '''import unittest
from plantdata import moving_average, fill_gaps, remove_outliers, daily_summary


class Hidden(unittest.TestCase):
    def test_ma_empty(self):
        self.assertEqual(moving_average([], 1), [])

    def test_fill_all_none(self):
        self.assertEqual(fill_gaps([None, None]), [None, None])

    def test_fill_uneven_positions(self):
        self.assertEqual(fill_gaps([0.0, None, 3.0, None, None, 9.0]), [0.0, 1.5, 3.0, 5.0, 7.0, 9.0])

    def test_fill_no_gaps_returns_copy(self):
        src = [1.0, 2.0]
        out = fill_gaps(src)
        self.assertEqual(out, [1.0, 2.0])
        self.assertIsNot(out, src)

    def test_outliers_mad_zero(self):
        self.assertEqual(remove_outliers([5, 5, 5, 5, 100]), [5, 5, 5, 5, 100])

    def test_outliers_negative_spike(self):
        self.assertEqual(remove_outliers([10, 11, 10, 12, 11, 10, -400]), [10, 11, 10, 12, 11, 10, None])

    def test_outliers_all_none(self):
        self.assertEqual(remove_outliers([None, None]), [None, None])

    def test_summary_all_none_day_omitted(self):
        ts = ["2026-10-08T00:00:00", "2026-10-09T00:00:00"]
        self.assertEqual(daily_summary(ts, [None, 2.0]), {"2026-10-09": {"min": 2.0, "max": 2.0, "mean": 2.0}})

    def test_summary_day_order(self):
        ts = ["2026-10-10T00:00:00", "2026-10-09T00:00:00"]
        self.assertEqual(list(daily_summary(ts, [1.0, 2.0])), ["2026-10-09", "2026-10-10"])


if __name__ == "__main__":
    unittest.main()
'''

TASK = ("The workspace contains plantdata.py (helpers for cleaning plant sensor series) and test_plantdata.py. "
        "Some tests fail. Find and fix the bugs in plantdata.py so that ALL tests pass; follow the docstrings (they are the "
        "specification), do not edit the tests, and do not change function names or signatures. Work with the tools: list_files, "
        "read_file, write_file, run_tests. When all tests pass, reply with a short summary of what you changed.")

TOOLS = [
    dict(type="function", function=dict(name="list_files", description="List files in the workspace", parameters=dict(type="object", properties={}))),
    dict(type="function", function=dict(name="read_file", description="Read a workspace file", parameters=dict(
        type="object", properties=dict(path=dict(type="string")), required=["path"]))),
    dict(type="function", function=dict(name="write_file", description="Create or overwrite a workspace file with the full new content",
                                        parameters=dict(type="object", properties=dict(path=dict(type="string"), content=dict(type="string")),
                                                        required=["path", "content"]))),
    dict(type="function", function=dict(name="run_tests", description="Run the unit tests (python -m unittest) and return the output",
                                        parameters=dict(type="object", properties={}))),
]


def run_unittest(ws, target=None):
    cmd = [sys.executable, "-m", "unittest", "-q"] + ([target] if target else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=ws)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return r.returncode == 0, (r.stdout + r.stderr)[-3000:]


def counts(out):
    """(tests run, tests failed) from unittest -q output."""
    ran = re.search(r"Ran (\d+) test", out or "")
    bad = sum(int(n) for n in re.findall(r"(?:failures|errors)=(\d+)", out or ""))
    return (int(ran.group(1)) if ran else 0), bad


def signatures(src):
    try:
        return {n.name: [a.arg for a in n.args.args] + [a.arg for a in n.args.kwonlyargs]
                for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    except SyntaxError:
        return None


def tool(ws, name, args):
    ws = Path(ws)
    try:
        if name == "list_files":
            return "\n".join(sorted(p.name for p in ws.iterdir() if p.is_file()))
        if name == "run_tests":
            return run_unittest(ws, "test_plantdata")[1]
        p = (ws / args["path"]).resolve()
        if ws.resolve() not in p.parents and p != ws.resolve():
            return "error: path outside workspace"
        if name == "read_file":
            return p.read_text(encoding="utf-8")
        if name == "write_file":
            if p.name.startswith("test_"):
                return "error: tests are read-only"
            p.write_text(args["content"], encoding="utf-8")
            return f"wrote {len(args['content'])} chars to {p.name}"
        return f"error: unknown tool {name}"
    except Exception as e:
        return f"error: {e}"


def agent(mode, seed, label, sampling):
    ws = Path(tempfile.mkdtemp(prefix="agent_ws_"))
    (ws / "plantdata.py").write_text(BUGGY, encoding="utf-8")
    (ws / "test_plantdata.py").write_text(TESTS, encoding="utf-8")
    msgs = [{"role": "system", "content": "You are a careful coding agent. Use the tools to inspect, edit and test code."},
            {"role": "user", "content": TASK}]
    stats = dict(turns=0, tool_calls=0, prompt_tokens=0, completion_tokens=0, prompt_s=0.0, gen_s=0.0, invalid_calls=0,
                 first_green_s=None, stopped="final")
    t0 = time.time()
    while True:
        if stats["turns"] >= MAX_TURNS:
            stats["stopped"] = "max_turns"
            break
        if time.time() - t0 > MAX_WALL:
            stats["stopped"] = "max_wall"
            break
        body = dict(model="q", messages=msgs, tools=TOOLS, max_tokens=6000, seed=seed, **sampling,
                    chat_template_kwargs={"enable_thinking": mode == "think"})
        try:
            d, _ = post("/v1/chat/completions", body, timeout=1200)
        except Exception as e:
            stats["stopped"] = f"error {str(e)[:60]}"
            break
        stats["turns"] += 1
        t = d.get("timings", {})
        stats["prompt_tokens"] += d["usage"]["prompt_tokens"]
        stats["completion_tokens"] += d["usage"]["completion_tokens"]
        stats["prompt_s"] += t.get("prompt_ms", 0) / 1000
        stats["gen_s"] += t.get("predicted_ms", 0) / 1000
        m = d["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content") or "", **({"tool_calls": calls} if calls else {})})
        if not calls:
            break
        for c in calls:
            stats["tool_calls"] += 1
            try:
                args = json.loads(c["function"]["arguments"]) if isinstance(c["function"]["arguments"], str) else c["function"]["arguments"]
                out = tool(ws, c["function"]["name"], args)
            except Exception as e:
                stats["invalid_calls"] += 1
                out = f"error: invalid arguments ({e})"
            if c["function"]["name"] == "run_tests" and stats["first_green_s"] is None and out.strip().endswith("OK"):
                stats["first_green_s"] = round(time.time() - t0, 1)
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": out[:6000]})
    wall = time.time() - t0
    (ws / "test_hidden.py").write_text(HIDDEN, encoding="utf-8")
    vis, vis_out = run_unittest(ws, "test_plantdata")
    hid, hid_out = run_unittest(ws, "test_hidden")
    final_src = (ws / "plantdata.py").read_text(encoding="utf-8")
    tests_intact = (ws / "test_plantdata.py").read_text(encoding="utf-8") == TESTS
    req, got = signatures(FIXED), signatures(final_src)
    sigs_ok = got is not None and all(got.get(n) == args for n, args in req.items())   # extra private helpers are fine
    (HERE / "agent_solutions").mkdir(exist_ok=True)
    (HERE / "agent_solutions" / f"{label}.py").write_text(final_src, encoding="utf-8")
    shutil.rmtree(ws, ignore_errors=True)
    vr, vb = counts(vis_out)
    hr, hb = counts(hid_out)
    stats.update(visible_ok=f"{vr - vb}/{vr}", hidden_ok=f"{hr - hb}/{hr}", tests_intact=tests_intact, signatures_ok=sigs_ok,
                 wall_s=round(wall, 1), visible_pass=vis, hidden_pass=hid, success=vis and hid and tests_intact and sigs_ok,
                 prompt_s=round(stats["prompt_s"], 1), gen_s=round(stats["gen_s"], 1), hidden_tail=("" if hid else hid_out[-160:]))
    return stats


def selftest():
    for name, src in (("BUGGY", BUGGY), ("FIXED", FIXED)):
        ws = Path(tempfile.mkdtemp())
        (ws / "plantdata.py").write_text(src, encoding="utf-8")
        (ws / "test_plantdata.py").write_text(TESTS, encoding="utf-8")
        (ws / "test_hidden.py").write_text(HIDDEN, encoding="utf-8")
        v = run_unittest(ws, "test_plantdata")
        h = run_unittest(ws, "test_hidden")
        print(name, "visible:", v[0], "| hidden:", h[0])
        if name == "BUGGY":
            print("  visible failures:", v[1].count("FAIL:") + v[1].count("ERROR:"), "| hidden failures:", h[1].count("FAIL:") + h[1].count("ERROR:"))
        else:
            print(v[1][-200:], h[1][-200:])
        shutil.rmtree(ws, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="q4,q5")
    ap.add_argument("--modes", default="nothink,think")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ncmoe", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    ncmoe = dict(NCMOE_DEFAULT, **dict(x.split("=") for x in a.ncmoe.split(",") if x))
    try:
        urllib.request.urlopen(URL + "/health", timeout=2)
        sys.exit("port 8080 already serves something")
    except OSError:
        pass
    for key in a.models.split(","):
        cmd = [str(SERVER), "-m", str(st.MODELS[key]), "--alias", "q", "--host", "127.0.0.1", "--port", "8080", "-c", "32768",
               "-ngl", "99", "--n-cpu-moe", ncmoe[key], "--load-mode", "none", "--jinja", "-fa", "on", "-np", "1",
               "-t", "12", "-tb", "12", "-C", "0xFFC03", "--cpu-strict", "1", "--no-webui"]
        if key == "gemma":
            cmd += ["-ctk", "q8_0", "-ctv", "q8_0"]   # needed to fit 32k context in 4 GB VRAM (llama/profiles.json)
        lf = (HERE / f"agent_server_{key}.log").open("w")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            wait_ready(proc, lf.name)
            print(f"== {key} ready, VRAM {vram()} MiB", flush=True)
            for mode in a.modes.split(","):
                for rep in range(1, a.reps + 1):
                    r = agent(mode, seed=rep, label=f"{key}_{mode}_rep{rep}", sampling=GEMMA_SAMPLING if key == "gemma" else SAMPLING)
                    r.update(model=key, mode=mode, rep=rep, time=datetime.now().strftime("%H:%M:%S"))
                    with OUT.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print(f"{key} {mode:7s} rep{rep} {'SUCCESS' if r['success'] else 'FAIL   '} {r['wall_s']}s turns={r['turns']} "
                          f"tools={r['tool_calls']} first_green={r['first_green_s']} tok={r['completion_tokens']} "
                          f"prompt_s={r['prompt_s']} gen_s={r['gen_s']} visible={r['visible_ok']} hidden={r['hidden_ok']} "
                          f"sig_ok={r['signatures_ok']} tests_intact={r['tests_intact']} stop={r['stopped']}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
