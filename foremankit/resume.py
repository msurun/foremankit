"""Resuming a cut-off answer: finds the saved partial text and proves it belongs to the task being sent again.

The `.agent-out/<id>.md` file holds the partial answer and `<id>.request.json` the fingerprint of the prompt and the
source that wrote it. A resume must come from the SAME task (same prompt hash) and goes to the SAME source: another
model cannot be trusted to continue someone else's code."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .outputs import SIDECAR_SUFFIX, read_answer_text
from .records import Answer, CallContext, RouterError

RESUME_SUFFIX = "-r"      # an id resumed once more becomes <id>-r2, <id>-r2-r3 ... so no file is overwritten


@dataclass(frozen=True)
class Resume:
    """A cut-off answer to be finished: its text, the source that wrote it and how often it was continued already."""
    original_id: str
    chain_entry: str
    partial: str
    continuations: int

    def seed(self) -> Answer:
        """The partial answer as the starting point of `complete_if_cut` (no tokens are billed for it again)."""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit": 0, "reasoning_tokens": 0}
        return Answer(self.partial, "length", usage, continuations=self.continuations)


def prompt_fingerprint(messages: list[dict[str, str]]) -> str:
    """SHA-256 of the complete prompt (system text, task, context files): a changed file or task gives another hash."""
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def resumed_id(original_id: str, existing: set[str]) -> str:
    """A free id for the finished answer: <id>-r<n>."""
    number = 1
    while f"{original_id}{RESUME_SUFFIX}{number}" in existing:
        number += 1
    return f"{original_id}{RESUME_SUFFIX}{number}"


def load_resume(out_dir: Path, answer_id: str, ctx: CallContext) -> Resume:
    """The saved partial answer `answer_id`; raises RouterError when it cannot be resumed with this task."""
    if not answer_id or Path(answer_id).name != answer_id:
        raise RouterError("continue_from takes the id of an .agent-out answer (no path)")
    sidecar_path = out_dir / f"{answer_id}{SIDECAR_SUFFIX}"
    if not sidecar_path.is_file():
        raise RouterError(f"cannot continue {answer_id}: {sidecar_path.name} not found in {out_dir}")
    record = json.loads(sidecar_path.read_text(encoding="utf-8"))
    _require_same_task(record, ctx, answer_id)
    text_path = out_dir / f"{answer_id}.md"
    if not text_path.is_file():
        raise RouterError(f"cannot continue {answer_id}: the answer file is missing")
    return Resume(answer_id, record["chain_entry"], read_answer_text(text_path), int(record.get("continuations", 0)))


def _require_same_task(record: dict, ctx: CallContext, answer_id: str) -> None:
    """Same kind, sensitivity and prompt hash, and an answer that really stopped at the token limit."""
    if record.get("finish_reason") != "length":
        raise RouterError(f"{answer_id} was not cut off (finish_reason {record.get('finish_reason')}): nothing to continue")
    if record.get("kind") != ctx.kind or record.get("sensitivity") != ctx.sensitivity:
        raise RouterError(f"{answer_id} was made for kind={record.get('kind')} sensitivity={record.get('sensitivity')}; "
                          "send the same kind and sensitivity")
    if record.get("prompt_sha256") != ctx.prompt_sha:
        raise RouterError(f"the task or a context file differs from the one that produced {answer_id}: send exactly the "
                          "same task, system_hint and context_files, or start a new delegation")
