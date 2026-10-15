"""Where answers are stored: the .agent-out directory and the Markdown file of one answer."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .records import Answer, CallContext, Candidate

OUT_DIR_ENV = "FOREMANKIT_OUT_DIR"
DEFAULT_OUT_DIR_NAME = ".agent-out"
SIDECAR_SUFFIX = ".request.json"
FRONT_MATTER_END = "\n---\n\n"   # closes the front matter written by _front_matter


def relative_path(path: Path) -> str:
    """`path` relative to the working directory, with forward slashes (absolute when on another drive)."""
    try:
        return Path(os.path.relpath(path, Path.cwd())).as_posix()
    except ValueError:  # different drive on Windows
        return path.as_posix()


def resolve_out_dir(configured: Path | None) -> Path:
    """The configured directory, else $FOREMANKIT_OUT_DIR, else ./.agent-out."""
    if configured:
        return configured
    from_env = os.environ.get(OUT_DIR_ENV)
    return Path(from_env) if from_env else Path.cwd() / DEFAULT_OUT_DIR_NAME


def write_answer(out_dir: Path, candidate: Candidate, ctx: CallContext, answer: Answer) -> str:
    """Save the answer as <call id>.md with a short front matter; returns the path relative to the working dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignore(out_dir)
    path = out_dir / f"{ctx.call_id}.md"
    path.write_text(_front_matter(candidate, ctx, answer) + answer.text, encoding="utf-8")
    _write_request_sidecar(out_dir, candidate, ctx, answer)
    return relative_path(path)


def _write_request_sidecar(out_dir: Path, candidate: Candidate, ctx: CallContext, answer: Answer) -> None:
    """<call id>.request.json: what `continue_from` needs to finish a cut-off answer without the prompt being stored.
    Only a fingerprint of the prompt is kept; the caller sends the same task again and it must hash the same."""
    node = f"@{candidate.node}" if candidate.node else ""
    record = {"id": ctx.call_id, "kind": ctx.kind, "sensitivity": ctx.sensitivity, "prompt_sha256": ctx.prompt_sha,
              "chain_entry": f"{candidate.provider.name}{node}:{candidate.alias}", "model": candidate.model_id,
              "finish_reason": answer.finish_reason, "continuations": answer.continuations}
    (out_dir / f"{ctx.call_id}{SIDECAR_SUFFIX}").write_text(json.dumps(record, indent=1), encoding="utf-8")


def read_answer_text(path: Path) -> str:
    """The answer of an .agent-out file without its front matter."""
    text = path.read_text(encoding="utf-8")
    marker = FRONT_MATTER_END
    if text.startswith("---\n") and marker in text:
        return text.split(marker, 1)[1]
    return text



def _ensure_gitignore(out_dir: Path) -> None:
    """Answers may hold internal data: keep the whole directory out of git."""
    ignore = out_dir / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n", encoding="utf-8")


def _front_matter(candidate: Candidate, ctx: CallContext, answer: Answer) -> str:
    return (f"---\nid: {ctx.call_id}\nkind: {ctx.kind}\nsensitivity: {ctx.sensitivity}\n"
            f"provider: {candidate.key}\nmodel: {candidate.model_id}\n"
            f"tokens_in: {answer.usage['prompt_tokens']}\ntokens_out: {answer.usage['completion_tokens']}\n"
            f"cost_usd: {answer.cost_usd:.6f}\nfinish_reason: {answer.finish_reason}\n---\n\n")
