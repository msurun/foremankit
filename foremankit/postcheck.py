"""After the answer: run the check command (feeding failures back to the author) and have another model review it."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import codecheck, netguard, review
from . import verify as verify_module
from .records import CallContext, ReviewJob, RouterError, TaskRequest, VerifySpec

if TYPE_CHECKING:
    from .router import Router

DEFAULT_VERIFY_ROUNDS = 2
DEFAULT_VERIFY_TIMEOUT_S = 300.0
OUTPUT_TAIL_CHARS = 1500
COST_FIELDS = ("tokens_in", "tokens_out", "cost_usd", "latency_s")
DEFAULT_REVIEW_MAX_TOKENS = 1500
REPLY_PREVIEW_CHARS = 300
CONVENTIONS_FAILURE_PREFIX = "The tests pass, but the file breaks the clean-code rules"


def fold_costs(result: dict[str, Any], other: dict[str, Any]) -> None:
    """Add the tokens, cost and time of another call to the result the caller sees."""
    for field_name in COST_FIELDS:
        result[field_name] += other[field_name]


def make_verify_spec(routing: dict[str, Any], request: TaskRequest) -> VerifySpec | None:
    """Validate the check before any model is paid; None when the request has no check."""
    if bool(request.verify) != bool(request.verify_target):
        raise RouterError("verify and verify_target go together: the check command and the file it checks")
    if not request.verify:
        return None
    allowed = (routing.get("verify") or {}).get("allowed_commands") or []
    tokens = verify_module.split_command(request.verify, allowed)
    path = verify_module.check_target(request.verify_target, Path.cwd())
    return VerifySpec(request.verify, request.verify_target, tokens, path)


class VerifyRun:
    """Writes the answer's code into the target, runs the check, and while it fails gives the output back to the SAME
    model (at most verify.max_rounds times). The original file is restored unless the check passes."""

    def __init__(self, router: Router, result: dict[str, Any], spec: VerifySpec):
        settings = router.routing.get("verify") or {}
        self.router, self.result, self.spec = router, result, spec
        self.max_rounds = int(settings.get("max_rounds", DEFAULT_VERIFY_ROUNDS))
        self.timeout_s = float(settings.get("timeout_s", DEFAULT_VERIFY_TIMEOUT_S))
        self.answer = result["_text"]
        self.ctx: CallContext = result["_ctx"]
        self.swap = verify_module.FileSwap(spec.path)
        self.report: dict[str, Any] = {"command": spec.command, "target": spec.target, "fix_rounds": 0,
                                       "passed": False, "output_tail": ""}

    def execute(self) -> None:
        """Adds result["verify"] and folds the cost of the fix calls into the result."""
        try:
            self._check_until_done()
        except BaseException:
            self.swap.restore()
            raise
        if not self.report["passed"]:
            self.swap.restore()
        self.result["verify"] = self.report

    def _check_until_done(self) -> None:
        for attempt in range(self.max_rounds + 1):
            failure = self._write_and_run()
            if failure is None or attempt == self.max_rounds:
                return
            if not self._ask_for_fix(failure, attempt + 1):
                return

    def _write_and_run(self) -> str | None:
        """Run the check on the answer's code. Returns the failure output, or None when there is nothing to fix:
        the check passed, the answer has no code block, or the check itself cannot run (the model is not to blame)."""
        code = verify_module.extract_code(self.answer, self.spec.target)
        if code is None:
            self.report["output_tail"] = "the answer has no code block to check"
            return None
        self.swap.write(code)
        try:
            passed, output = verify_module.run_command(self.spec.tokens, Path.cwd(), self.timeout_s)
        except verify_module.VerifyError as error:
            self.report["output_tail"] = str(error)
            return None
        if passed:
            output = self._convention_failure(code) or output
            passed = output is not None and not output.startswith(CONVENTIONS_FAILURE_PREFIX)
        self.report["output_tail"] = output[-OUTPUT_TAIL_CHARS:]
        self.report["passed"] = passed
        return None if passed else output

    def _convention_failure(self, code: str) -> str | None:
        """The clean-code issues of a python target as text for the author, or None when it is clean, not python, or
        the check is switched off (routing.yaml verify.conventions: false). Tests passing is not enough: the owner's
        standard (config/conventions.md) must hold too, so the cheap model fixes it instead of a Claude session."""
        settings = self.router.routing.get("verify") or {}
        if not settings.get("conventions", True) or not self.spec.target.endswith(".py"):
            return None
        limits = codecheck.limits_from(self.router.routing)
        issues = codecheck.check_source(code, self.spec.target, limits)
        if not issues:
            return None
        listing = "\n".join(str(issue) for issue in issues)
        return f"{CONVENTIONS_FAILURE_PREFIX} (config/conventions.md):\n{listing}"

    def _ask_for_fix(self, failure: str, round_number: int) -> bool:
        """One more call to the author with the failed output; False when that call failed."""
        request = (f"The check `{self.spec.command}` failed with this output:\n\n{failure}\n\n"
                   f"Fix the problem. Return the complete corrected {self.spec.target} as one code block.")
        messages = self.ctx.messages + [{"role": "assistant", "content": self.answer},
                                        {"role": "user", "content": request}]
        self.ctx = replace(self.ctx, messages=messages)
        call_ctx = replace(self.ctx, call_id=f"{self.result['id']}-fix{round_number}")
        fixed = self.router.caller.call(self.result["_cand"], call_ctx)
        if isinstance(fixed, str):
            self.report["output_tail"] += f"\n(the fix request failed: {fixed})"
            return False
        self.answer = fixed["_text"]
        self.report["fix_rounds"] += 1
        fold_costs(self.result, fixed)
        self.result.update(path=fixed["path"], preview=fixed["preview"], truncated=fixed["truncated"])
        return True


class ReviewStage:
    """Has a DIFFERENT model review the finished code and, once, sends grounded blocker findings back to the author."""

    def __init__(self, router: Router):
        self.router = router

    @property
    def settings(self) -> dict[str, Any]:
        """routing.yaml `review`."""
        return self.router.routing.get("review") or {}

    def review_code(self, job: ReviewJob, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
        """Ask a reviewer (routing.yaml chains.review) for findings. Never raises for a missing reviewer: the report
        then has `skipped` with the reason. Returns {reviewer, findings, reply} or {skipped, findings}."""
        text = review.build_task(job.task, job.code, job.target, job.check_summary)
        request = TaskRequest(text, "review", job.sensitivity, allow_cost=job.allow_cost,
                              max_output_tokens=int(self.settings.get("max_output_tokens", DEFAULT_REVIEW_MAX_TOKENS)))
        try:
            reply = self.router.run_chain(request, exclude)
        except (RouterError, netguard.PrivacyError) as error:
            return {"skipped": str(error)[:REPLY_PREVIEW_CHARS], "findings": []}
        where = f"{reply['provider']}/{reply['model']}" + (f" via {reply['endpoint']}" if reply.get("endpoint") else "")
        return {"reviewer": where, "findings": review.parse_findings(reply["_text"], job.code), "reply": reply}

    def run(self, result: dict[str, Any], request: TaskRequest, spec: VerifySpec | None) -> None:
        """Review the result's code; adds result["review"]."""
        report = self.review_code(self._job(result, request, spec), exclude=(result["_cand"].key,))
        reply = report.pop("reply", None)
        if reply:
            fold_costs(result, reply)
        actionable = review.to_fix(report["findings"], self.settings.get("fix_severities") or ["blocker"])
        report["fixed"] = False
        if actionable and int(self.settings.get("fix_rounds", 1)) > 0:
            report["fixed"] = self._fix_after_review(result, actionable, request, spec)
        report["findings"] = [dict(vars(finding)) for finding in report["findings"]]
        result["review"] = report

    @staticmethod
    def _job(result: dict[str, Any], request: TaskRequest, spec: VerifySpec | None) -> ReviewJob:
        if spec:
            code = spec.path.read_text(encoding="utf-8")
        else:
            code = verify_module.extract_code(result["_text"], "") or result["_text"]
        verify_report = result.get("verify")
        summary = ""
        if verify_report:
            summary = ("the check command passed" if verify_report["passed"]
                       else "the check command FAILED: " + verify_report["output_tail"])
        return ReviewJob(code, request.verify_target, request.task, summary, request.sensitivity, request.allow_cost)

    def _fix_after_review(self, result: dict[str, Any], findings: list[review.Finding], request: TaskRequest,
                          spec: VerifySpec | None) -> bool:
        """One round with the author: findings in, corrected code out; the check runs again when there is one."""
        ctx: CallContext = result["_ctx"]
        messages = ctx.messages + [{"role": "assistant", "content": result["_text"]},
                                   {"role": "user", "content": review.fix_request(findings, request.verify_target)}]
        call_ctx = replace(ctx, messages=messages, call_id=f"{result['id']}-review-fix")
        fixed = self.router.caller.call(result["_cand"], call_ctx)
        if isinstance(fixed, str):
            return False
        fold_costs(result, fixed)
        result.update(_text=fixed["_text"], path=fixed["path"], preview=fixed["preview"])
        if spec:
            result["verify_before_review"] = result.get("verify")
            VerifyRun(self.router, result, spec).execute()
        return True
