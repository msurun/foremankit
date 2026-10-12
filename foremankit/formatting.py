"""Plain-text rendering of router results, reviews and status (shared by the CLI and the MCP server)."""
from __future__ import annotations

from typing import Any

from . import review

MAX_FINDINGS_SHOWN = 8
MAX_MODELS_SHOWN = 8


def format_result(result: dict[str, Any]) -> str:
    """Render one delegation result: header, file, warnings, verify/review outcome and a preview."""
    lines = [_result_header(result), f"file: {result['path']}"]
    if result.get("truncated"):
        lines.append("WARNING: output truncated at max_output_tokens")
    lines += ["NOTE: " + note for note in result.get("notes") or []]
    if result.get("verify"):
        lines.append(_verify_line(result["verify"]))
    if result.get("review"):
        lines += format_review(result["review"])
    if result.get("skipped"):
        lines.append("not used (tried first): " + "; ".join(result["skipped"]))
    lines.append("preview:\n" + result["preview"])
    return "\n".join(lines)


def _result_header(result: dict[str, Any]) -> str:
    where = f"{result['provider']}/{result['model']}" + (f" via {result['endpoint']}" if result.get("endpoint") else "")
    dry = " [DRY RUN]" if result.get("dry_run") else ""
    return (f"← response from {where} in {result['latency_s']:.1f}s{dry}"
            f" | in={result['tokens_in']} (cache {result['cache_hit']}) out={result['tokens_out']} | ${result['cost_usd']:.4f}")


def _verify_line(verify: dict[str, Any]) -> str:
    rounds = f"after {verify['fix_rounds']} fix round(s)" if verify["fix_rounds"] else "on the first try"
    if verify["passed"]:
        return f"verify: PASSED {rounds}: `{verify['command']}`; {verify['target']} now holds the checked code"
    return (f"verify: FAILED {rounds}: `{verify['command']}`; {verify['target']} was restored. Last output:\n"
            + verify["output_tail"])


def format_review(report: dict[str, Any]) -> list[str]:
    """Text lines for a review report (the reviewer, the findings, whether the author fixed them)."""
    if report.get("skipped"):
        return [f"review: SKIPPED ({report['skipped']})"]
    findings = report["findings"]
    counts = {severity: sum(1 for finding in findings if finding["severity"] == severity) for severity in review.SEVERITIES}
    summary = ", ".join(f"{count} {severity}" for severity, count in counts.items() if count) or "no findings"
    fixed = "; blockers sent back to the author and fixed" if report.get("fixed") else ""
    lines = [f"review by {report['reviewer']}: {summary}{fixed}"]
    lines += [_finding_line(finding) for finding in findings[:MAX_FINDINGS_SHOWN]]
    return lines


def _finding_line(finding: dict[str, Any]) -> str:
    mark = "" if finding["grounded"] else " (quote not found in the code: unverified)"
    return f"  - {finding['severity'].upper()} {finding['where']}: {finding['problem']} -> {finding['fix']}{mark}"


def format_status(status: dict[str, Any]) -> str:
    """Render the router status: budget, providers and the local LLM nodes."""
    lines = [_status_header(status), "providers:"]
    lines += [_provider_line(provider) for provider in status["providers"]]
    lines.append("local LLM nodes (pool `llm`, fastest expected first):")
    for node in status["llm"]:
        lines += _node_lines(node)
    return "\n".join(lines)


def _status_header(status: dict[str, Any]) -> str:
    budget = status["budget"]
    dry = " [DRY RUN]" if status["dry_run"] else ""
    return (f"foremankit{dry} | profile={status.get('profile') or '-'}"
            f" | spend today ${budget['day_usd']:.4f}/{budget['daily_cap']} month ${budget['month_usd']:.4f}/{budget['monthly_cap']}")


def _provider_line(provider: dict[str, Any]) -> str:
    remaining = ", ".join(f"{key}={'∞' if value is None else value}" for key, value in provider["remaining"].items())
    flags = (("free" if provider["free"] else "PAID") + (", trains" if provider["trains_on_data"] else "")
             + (", internal-ok" if provider.get("internal_ok") else ""))
    state = "active" if provider["active"] else f"inactive ({provider['reason']})"
    cooldown = f", cooldown {provider['cooldown_s']}s" if provider["cooldown_s"] else ""
    return f"  {provider['name']:<11} {state:<22} [{flags}] left: {remaining}{cooldown}"


def _node_lines(node: dict[str, Any]) -> list[str]:
    reach = "reachable" if node.get("reachable") else "not reachable"
    who = "this machine" if node["self"] else "other machine"
    lines = [(f"  {node['name']:<12} {who}, enabled={node['enabled']} ({node['source']}) {reach}"
              f" endpoint={node.get('endpoint') or '-'} model={node.get('chosen_model') or '-'}"
              f" accepts<={node['accept']}")]
    if node.get("models"):
        lines.append(f"    models: {', '.join(node['models'][:MAX_MODELS_SHOWN])}")
    lines += [f"    ! {err}" for err in node.get("errors") or []]
    return lines
