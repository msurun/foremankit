"""Where answers are stored: the .agent-out directory and the Markdown file of one answer."""
from __future__ import annotations

import os
from pathlib import Path

from .records import Answer, CallContext, Candidate

OUT_DIR_ENV = "FOREMANKIT_OUT_DIR"
DEFAULT_OUT_DIR_NAME = ".agent-out"


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
    return relative_path(path)


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
