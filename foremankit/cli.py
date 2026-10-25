"""`foremankit` command line: status, models, stats, lab nodes and local-LLM switches (`nodes`, `llm`), profiles, manual delegate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import ConfigError
from .delegation import CATEGORIES, describe, set_switch
from .formatting import format_result, format_review, format_status
from .netguard import PrivacyError
from .passrates import format_pass_rates, summarize_pass_rates
from .records import DEFAULT_MAX_OUTPUT_TOKENS, ReviewJob
from .regression import regress
from .router import Router, RouterError
from .transcripts import find_transcripts, summarize_many

SENSITIVITY_CHOICES = ["normal", "internal", "private"]
MODELS_SHOWN_WITHOUT_ALL = 40
BUSY_WORDS = {True: "busy", False: "idle", None: "idle-state unknown"}


# ---------------------------------------------------------------- llm: the local-LLM switches
def _print_node(info: dict[str, Any]) -> None:
    where = "this machine" if info["self"] else "other machine"
    print(f"{info['name']} ({where}): enabled={info['enabled']} (source: {info['source']}), "
          f"{'reachable' if info.get('reachable') else 'not reachable'}, endpoint={info.get('endpoint') or '-'}, "
          f"accepts data up to: {info['accept']}")
    print(f"  configured endpoints: {', '.join(info['endpoints']) or '-'}")
    print(f"  models: {', '.join(info.get('models') or []) or '-'}  -> chosen: {info.get('chosen_model') or '-'}")
    for error in info.get("errors") or []:
        print(f"  ! {error}")


def _node_matches(node: Any, target: str) -> bool:
    """Whether a node is selected by `all`, its own name, `self` or `others`."""
    return (target in ("all", node.name) or (target == "self" and node.is_self)
            or (target == "others" and not node.is_self))


def _print_test_row(row: dict[str, Any]) -> None:
    print(f"[{row['node']} / {row['endpoint']}] {row['url']}")
    print(f"  netguard (private data): {'OK' if row['private_ok'] else 'REFUSED'} - {row['netguard']}")
    error = " error: " + row["error"] if row["error"] else ""
    print(f"  reachable: {row['reachable']} ({row['latency_ms']} ms){error}")
    print(f"  models: {', '.join(row['models']) or '-'}")
    generation = row.get("generation")
    if generation:
        outcome = (f"OK in {generation['seconds']}s -> {generation['reply']!r}" if generation["ok"]
                   else f"FAILED {generation['error']}")
        print(f"  generation [{generation['model']}]: " + outcome)


def _print_all_nodes(router: Router) -> None:
    """Print the status of every configured local-LLM node."""
    for node in router.nodes():
        _print_node(router.node_status(node))


def _print_matching_nodes(router: Router, target: str) -> None:
    """Print the status of the nodes selected by `target`."""
    for node in router.nodes():
        if _node_matches(node, target):
            _print_node(router.node_status(node))


def _run_node_tests(router: Router, target: str) -> None:
    """Test the selected nodes; `test` alone tests all of them."""
    for row in router.node_test(None if target == "test" else target):
        _print_test_row(row)


def _set_node_toggle(router: Router, target: str, action: str) -> None:
    """Store an on/off/auto toggle for the node and confirm it."""
    router.set_toggle(target, action)
    print(f"{target} -> {action} (stored in the state file; overrides env and config)")


def cmd_llm(router: Router, args: argparse.Namespace) -> int:
    """llm [status|test] | llm <node|self|others|all> [on|off|auto|status|test]"""
    target, action = args.target, args.action
    if target in (None, "status"):
        _print_all_nodes(router)
    elif target == "test" or action == "test":
        _run_node_tests(router, target)
    elif action in (None, "status"):
        _print_matching_nodes(router, target)
    else:
        _set_node_toggle(router, target, action)
    return 0


# ---------------------------------------------------------------- nodes: the lab overview
def _lab_node_state(entry: dict[str, Any]) -> str:
    if entry.get("reachable"):
        return "reachable"
    errors = f" ({'; '.join(entry['errors'])})" if entry.get("errors") else ""
    return "UNREACHABLE" + errors


def _speed_origin(entry: dict[str, Any]) -> str:
    if entry.get("speed_pinned"):
        return "pinned: written speed"
    return "learned" if entry.get("speed_learned") else "written, not learned yet"


def _print_lab_node(entry: dict[str, Any]) -> None:
    speed, written = entry.get("speed") or {}, entry.get("speed_written") or {}
    latency = f", {entry['latency_ms']} ms, {BUSY_WORDS[entry.get('busy')]}" if entry.get("reachable") else ""
    print(f"[{entry['name']}]{' (this machine)' if entry['self'] else ''} {entry.get('endpoint') or '-'}  "
          f"{_lab_node_state(entry)}{latency} | enabled={entry['enabled']}")
    print(f"    models: {', '.join(entry.get('models') or []) or '-'} | speed: "
          f"{speed.get('prompt_tps', 0):.0f} / {speed.get('gen_tps', 0):.0f} tok/s"
          f" (prompt/generation, {_speed_origin(entry)}; written {written.get('prompt_tps', '?')} / "
          f"{written.get('gen_tps', '?')}) | rule: {entry['rule']} | accepts data up to: {entry['accept']}")
    if entry.get("reachable"):
        print(f"    private data over this link: {'OK' if entry['private_ok'] else 'REFUSED'} ({entry['netguard']})")


def _create_lab_key() -> None:
    from .nodes import ensure_key
    path, created = ensure_key()
    print(f"FOREMANKIT_PEER_KEY {'created in' if created else 'already present in'} {path}")
    print("The same line must be in the .env of EVERY machine (copy it by hand; it is never printed or committed).")


def cmd_nodes(router: Router, args: argparse.Namespace) -> int:
    """nodes [status|key]: the lab's machines, their local LLM servers, or create the shared lab key."""
    if args.action == "key":
        _create_lab_key()
        return 0
    overview = router.nodes_status()
    print(f"this machine: {overview['hostname']} -> nodes.yaml entry: "
          f"{overview['self'] or 'NOT LISTED (used as the node `local`)'}")
    print(f"lab key (FOREMANKIT_PEER_KEY): {'set' if overview['key_set'] else 'MISSING - run `foremankit nodes key`'}"
          f" | tunnel: {'enabled' if overview['tunnel_enabled'] else 'disabled (LAN only)'}")
    if not overview["nodes"]:
        print("no nodes (no local server answers and no nodes.yaml)")
    for entry in overview["nodes"]:
        _print_lab_node(entry)
    return 0


# ---------------------------------------------------------------- check and review
def cmd_check(router: Router, args: argparse.Namespace) -> int:
    """Check python files against the measurable clean-code rules (config/conventions.md); exit 1 when any is broken."""
    from .codecheck import check_files, limits_from
    issues = check_files(args.files, limits_from(router.routing))
    for issue in issues:
        print(issue)
    print(f"{len(issues)} issue(s) in {len(args.files)} file(s)" if issues else "clean: no issues")
    return 1 if issues else 0


def _code_to_review(args: argparse.Namespace) -> tuple[str, str]:
    """(code, label): the `git diff` of the given paths, or the whole files."""
    if args.diff:
        diff = subprocess.run(["git", "diff", "--", *args.files], capture_output=True, text=True, encoding="utf-8", check=False)
        return diff.stdout, "git diff"
    code = "\n\n".join(f"# FILE: {name}\n{Path(name).read_text(encoding='utf-8')}" for name in args.files)
    return code, ", ".join(args.files)


def cmd_review(router: Router, args: argparse.Namespace) -> int:
    """Preliminary review of existing files (or of `git diff` with --diff) by a second model; prints the findings."""
    code, target = _code_to_review(args)
    if not code.strip():
        print("nothing to review")
        return 0
    report = router.review_code(ReviewJob(code, target, args.task, "", args.sensitivity, args.allow_cost))
    reply = report.pop("reply", None)
    report["findings"] = [dict(vars(finding)) for finding in report["findings"]]
    print("\n".join(format_review(report)))
    if reply:
        print(f"cost ${reply['cost_usd']:.4f}, {reply['latency_s']:.1f}s, full answer: {reply['path']}")
    return 1 if any(finding["severity"] == "blocker" and finding["grounded"] for finding in report["findings"]) else 0


# ---------------------------------------------------------------- models, stats, profile, delegate
def _print_models(name: str, info: dict[str, Any], show_all: bool) -> None:
    print(f"== {name}")
    if info.get("error"):
        print(f"  {info['error']}")
    print(f"  configured: {', '.join(info['configured']) or '-'}")
    if info.get("missing"):
        print(f"  NOT FOUND on endpoint (fix providers.yaml): {', '.join(info['missing'])}")
    if info["available"]:
        shown = info["available"] if show_all else info["available"][:MODELS_SHOWN_WITHOUT_ALL]
        print("  available: " + ", ".join(shown) + (" ..." if len(shown) < len(info["available"]) else ""))


def cmd_models(router: Router, args: argparse.Namespace) -> int:
    """List the models each active provider offers, next to the configured ones."""
    for name, info in router.models(args.provider).items():
        _print_models(name, info, args.all)
    return 0


def _print_outcomes(outcomes: dict[str, dict[str, Any]]) -> None:
    """Accepted results and cost per accepted result for each model (delegations that ran a check or review)."""
    print("accepted results per model:")
    for name, entry in sorted(outcomes.items()):
        per = entry["cost_per_accepted"]
        print(f"  {name:<34} accepted={int(entry['accepted'])}/{int(entry['delegations'])} "
              f"cost/accepted={'n/a' if per is None else f'${per:.4f}'}")


def cmd_stats(router: Router, args: argparse.Namespace) -> int:
    """Calls, tokens and cost per provider and per kind, and the saving against the reference model."""
    stats = router.stats(args.days)
    for title, key in (("per provider", "by_provider"), ("per kind", "by_kind")):
        print(f"{title}:")
        for name, counts in sorted(stats[key].items()):
            print(f"  {name:<14} calls={int(counts['calls']):>4} ok={int(counts['ok']):>4} "
                  f"in={int(counts['tokens_in']):>9} out={int(counts['tokens_out']):>8} ${counts['cost_usd']:.4f} "
                  f"cut={int(counts['truncated'])} cont={int(counts['continuations'])}")
    _print_outcomes(stats["outcomes"])
    cutoff = time.time() - args.days * 86400 if args.days else 0.0
    print("pass@k (checked delegations):")
    for line in format_pass_rates(summarize_pass_rates(router.log.read(), cutoff)).splitlines():
        print(f"  {line}")
    print(f"total cost ${stats['cost_usd']:.4f}; same tokens at {stats['reference']} API prices "
          f"~${stats['reference_usd']:.4f} -> saved ~${stats['saved_usd']:.4f} (rough: output quality/lengths differ)")
    return 0


def cmd_regress(router: Router, args: argparse.Namespace) -> int:
    """Run the REGRESSION.md commands whose module changed (git status); --list only shows them."""
    code, report = regress(args.ledger, os.getcwd(), run=not args.list)
    print(report)
    return code


def cmd_sessions(router: Router, args: argparse.Namespace) -> int:
    """Claude tokens, delegations per 100 requests and reads of .agent-out per session (from ~/.claude transcripts)."""
    summaries = summarize_many(find_transcripts(args.days))
    print(f"{'started':<17} {'requests':>8} {'tokens':>9} {'cache read':>11} {'deleg':>5} {'/100':>5} {'agents':>6} {'reads':>5}")
    for item in sorted(summaries, key=lambda row: row.started):
        print(f"{item.started[:16]:<17} {item.requests:>8} {item.total_tokens():>9} {item.cache_read_tokens:>11} "
              f"{item.delegations:>5} {item.delegations_per_100_requests():>5.1f} {item.agent_calls:>6} {item.agent_out_reads:>5}")
    delegations = sum(item.delegations for item in summaries)
    reads = sum(item.agent_out_reads for item in summaries)
    print(f"{len(summaries)} sessions, {delegations} delegations, {reads} reads of .agent-out results "
          "(tokens = input + cache creation + output; cache reads are listed apart)")
    return 0


def cmd_profile(router: Router, args: argparse.Namespace) -> int:
    """Show the current profile, or switch to the named one."""
    if not args.name:
        state = router.store.load()
        print(f"current profile: {state.get('profile') or '-'}; toggles: {state.get('toggles', {})}")
        print(f"available: {', '.join(sorted(router.profiles))}")
        return 0
    flags = router.apply_profile(args.name)
    print(f"profile {args.name}: {flags}")
    return 0


def cmd_delegate(router: Router, args: argparse.Namespace) -> int:
    """Delegate one task by hand and print the result."""
    result = router.delegate(args.task, args.kind, args.sensitivity, context_files=args.files,
                             system_hint=args.hint, max_output_tokens=args.max_output_tokens,
                             escalate=args.escalate, allow_cost=args.allow_cost, verify=args.verify,
                             verify_target=args.verify_target, review=args.review)
    print(format_result(result))
    return 0


def cmd_status(router: Router, args: argparse.Namespace) -> int:
    """Print the router status as text, or as JSON with --json."""
    status = router.status()
    print(json.dumps(status, indent=1) if args.json else format_status(status))
    return 0


def cmd_delegation(router: Router, args: argparse.Namespace) -> int:
    """Switch whole source categories (paid, free, local) on or off, or show their state."""
    if args.action in ("on", "off"):
        changed = set_switch(router.store, args.target, args.action)
        print(f"delegation {args.action}: {', '.join(changed)}")
    print(describe(router.store.load()))
    return 0


def cmd_orchestrator(router: Router, args: argparse.Namespace) -> int:
    """Show or switch who orchestrates Claude Code: claude, local, or an external upstream (deepseek, gemini)."""
    from foremankit import upstream
    from foremankit.gateway import settings
    cfg = settings()
    if args.action == "use":
        problem = upstream.check_switch(cfg, args.mode, Path.cwd())
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 2
        upstream.set_mode(args.mode)
    mode = upstream.active_mode()
    print(f"orchestrator mode: {mode}  (repo sensitivity here: {upstream.repo_sensitivity(Path.cwd())})")
    if args.action == "env" or (args.action == "use" and mode not in upstream.BUILTIN_MODES):
        print("start Claude Code with:")
        print(f"  ANTHROPIC_BASE_URL=http://127.0.0.1:{cfg['port']}  ANTHROPIC_AUTH_TOKEN=foremankit  (gateway: foremankit-gateway)")
    return 0


def cmd_kaggle(router: Router, args: argparse.Namespace) -> int:
    """Show the free Kaggle hours left this week, price a job queue in them, or log a finished session."""
    from foremankit import kaggle as budget
    handlers = {"status": _kaggle_status, "estimate": _kaggle_estimate,
                "log": _kaggle_log, "push": _kaggle_push}
    return handlers[args.action](budget, args)


def _kaggle_status(budget: Any, _args: argparse.Namespace) -> int:
    """Print used and left hours per accelerator, with how many <=9 h sessions that allows."""
    for accelerator in ("gpu", "tpu"):
        used = budget.week_used(accelerator) / budget.SECONDS_PER_HOUR
        left = budget.remaining(accelerator)
        print(f"{accelerator}: {used:5.1f} h used, {left:5.1f} h left of {budget.quota(accelerator):.0f} h "
              f"({budget.sessions_needed(left)} sessions of {budget.SESSION_CAP_HOURS:.0f} h)")
    print(f"ledger: {budget.sessions_path()} ({len(budget.read_sessions())} sessions)")
    return 0


def _kaggle_estimate(budget: Any, args: argparse.Namespace) -> int:
    """Print what the jobs file would cost in this week's hours and whether it still fits."""
    jobs = _read_jobs(_required_jobs(args))
    accelerator = args.accelerator
    hours = budget.estimate_hours(jobs, accelerator=accelerator)
    left = budget.remaining(accelerator)
    verdict = "fits" if hours <= left else "does NOT fit - split the queue or wait for Monday"
    print(f"{len(jobs)} jobs: {hours:.2f} {accelerator.upper()} h, "
          f"{budget.sessions_needed(hours)} session(s); {left:.1f} h left -> {verdict}")
    return 0


def _kaggle_log(budget: Any, args: argparse.Namespace) -> int:
    """Record a finished session so the weekly budget stays truthful."""
    seconds = {"gpu": args.gpu_hours * budget.SECONDS_PER_HOUR,
               "tpu": args.tpu_hours * budget.SECONDS_PER_HOUR}
    budget.log_session(seconds, jobs_done=args.jobs_done, note=args.note)
    print(f"logged {args.gpu_hours:.2f} GPU h, {args.tpu_hours:.2f} TPU h, {args.jobs_done} jobs; "
          f"left: gpu {budget.remaining('gpu'):.1f} h, tpu {budget.remaining('tpu'):.1f} h")
    return 0


def _kaggle_push(budget: Any, args: argparse.Namespace) -> int:
    """Push the kernel folder to Kaggle, then remind the user to log the session afterwards."""
    print(budget.push(Path(args.dir), args.datasets))
    accelerator = budget.accelerator_of(Path(args.dir))
    print(f"after the run: foremankit kaggle log --{accelerator}-hours <wall hours> --jobs-done <n>")
    return 0


def _required_jobs(args: argparse.Namespace) -> Path:
    """The queue path, or a ConfigError that says the estimate needs one."""
    if not args.jobs:
        raise ConfigError("estimate needs a jobs file: foremankit kaggle estimate jobs.jsonl")
    return Path(args.jobs)


def _read_jobs(path: Path) -> list[dict[str, Any]]:
    """The queue as dicts; a broken line is an error because the estimate would then lie."""
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ConfigError(f"{path.name} line {number}: {error}") from error
    return rows


COMMANDS = {"orchestrator": cmd_orchestrator, "delegation": cmd_delegation, "status": cmd_status, "models": cmd_models, "stats": cmd_stats, "llm": cmd_llm, "nodes": cmd_nodes,
            "check": cmd_check, "regress": cmd_regress, "review": cmd_review, "profile": cmd_profile, "delegate": cmd_delegate,
            "sessions": cmd_sessions, "kaggle": cmd_kaggle}


# ---------------------------------------------------------------- argument parser
def _add_basic_parsers(sub: Any) -> None:
    status = sub.add_parser("status", help="providers, limits, spend, local LLM servers")
    status.add_argument("--json", action="store_true")
    models = sub.add_parser("models", help="list /models of every active provider")
    models.add_argument("provider", nargs="?")
    models.add_argument("--all", action="store_true")
    stats = sub.add_parser("stats", help="cost/calls per provider and kind, savings vs Sonnet")
    stats.add_argument("--days", type=int)
    sessions = sub.add_parser("sessions", help="Claude tokens per session from the transcripts, delegations and result reads")
    sessions.add_argument("--days", type=int, default=7)
    profile = sub.add_parser("profile", help="switch profile (home|work|...)")
    profile.add_argument("name", nargs="?")


def _add_node_parsers(sub: Any) -> None:
    orchestrator = sub.add_parser("orchestrator", help="orchestrator [status|env] | orchestrator use <claude|local|deepseek|gemini>: who runs Claude Code")
    orchestrator.add_argument("action", nargs="?", default="status", choices=["status", "env", "use"])
    orchestrator.add_argument("mode", nargs="?", default="claude")
    delegation = sub.add_parser("delegation", help="delegation [paid|free|local|all] [on|off|status]: switch a whole source category")
    delegation.add_argument("target", nargs="?", default="all", choices=[*CATEGORIES, "all"])
    delegation.add_argument("action", nargs="?", default="status", choices=["on", "off", "status"])
    llm = sub.add_parser("llm", help="llm [status|test] | llm <node|self|others|all> [on|off|auto|status|test]")
    llm.add_argument("target", nargs="?", help="status | test | self | others | all | a node name from nodes.yaml")
    llm.add_argument("action", nargs="?", choices=["on", "off", "auto", "status", "test"])
    nodes = sub.add_parser(
        "nodes", help="nodes [status|key]: the lab's machines and their local LLM servers (config/nodes.yaml)")
    nodes.add_argument("action", nargs="?", choices=["status", "key"], default="status")


def _add_code_quality_parsers(sub: Any) -> None:
    review = sub.add_parser("review", help="review files (or `git diff`) with a second model: own local LLM or a paid one")
    review.add_argument("files", nargs="*", default=[])
    review.add_argument("--diff", action="store_true", help="review `git diff` of the given paths instead of whole files")
    review.add_argument("--task", default="Review this change.", help="what the code is supposed to do")
    review.add_argument("--sensitivity", default="normal", choices=SENSITIVITY_CHOICES)
    review.add_argument("--allow-cost", action="store_true")
    check = sub.add_parser("check", help="check python files against the clean-code rules of config/conventions.md")
    check.add_argument("files", nargs="+")
    regress_parser = sub.add_parser("regress", help="run the REGRESSION.md commands for the modules changed in the git tree")
    regress_parser.add_argument("--ledger", default="REGRESSION.md")
    regress_parser.add_argument("--list", action="store_true", help="only list the affected entries")


def _add_kaggle_parser(sub: Any) -> None:
    kaggle = sub.add_parser("kaggle", help="free Kaggle hours: status | estimate <jobs.jsonl> | log | push")
    kaggle.add_argument("action", nargs="?", default="status",
                        choices=["status", "estimate", "log", "push"])
    kaggle.add_argument("jobs", nargs="?", help="jobs JSONL for `estimate`")
    kaggle.add_argument("--gpu-hours", type=float, default=0.0)
    kaggle.add_argument("--tpu-hours", type=float, default=0.0)
    kaggle.add_argument("--jobs-done", type=int, default=0)
    kaggle.add_argument("--note", default="")
    kaggle.add_argument("--dir", default="kaggle/kernel",
                        help="kernel folder for `push` (kaggle/tpu for the TPU lane)")
    kaggle.add_argument("--accelerator", default="gpu", choices=["gpu", "tpu"],
                        help="which free tier `estimate` prices, and whose wall clock `push` logs")
    kaggle.add_argument("--datasets", nargs="*", default=None,
                        help="Kaggle datasets to attach (owner/slug), e.g. me/qwen27b-gguf me/foremankit-jobs")


def _add_delegate_parser(sub: Any) -> None:
    delegate = sub.add_parser("delegate", help="manual delegate (use FOREMANKIT_DRY_RUN=1 to test routing)")
    delegate.add_argument("task")
    delegate.add_argument("--kind", default="quick")
    delegate.add_argument("--sensitivity", default="normal", choices=SENSITIVITY_CHOICES)
    delegate.add_argument("--files", nargs="*", default=[])
    delegate.add_argument("--hint", default="")
    delegate.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    delegate.add_argument("--escalate", action="store_true")
    delegate.add_argument("--review", action="store_true",
                          help="a second model reviews the finished code (own LLM or paid)")
    delegate.add_argument("--allow-cost", action="store_true", help="accept a paid call above budget.per_call_usd")
    delegate.add_argument("--verify", default="",
                          help='check command, e.g. "python -m pytest tests/x.py -q" (needs --verify-target)')
    delegate.add_argument("--verify-target", default="",
                          help="the file the answer is written to and the check runs against")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the `foremankit` command with all its subcommands."""
    parser = argparse.ArgumentParser(prog="foremankit", description="Delegate work to cheaper/local models.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_basic_parsers(sub)
    _add_node_parsers(sub)
    _add_code_quality_parsers(sub)
    _add_kaggle_parser(sub)
    _add_delegate_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one `foremankit` command; returns the exit code (2 = the router refused the request)."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        return COMMANDS[args.cmd](Router(), args)
    except (RouterError, PrivacyError, ConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
