"""Join a truncated model answer with its continuation, and check the seam.

A paid model can stop at its token limit (finish_reason=length). The caller asks
the same model to continue, then needs the two pieces joined without duplicated
or lost text. This module detects the open code fence at the cut, builds the
continuation request, joins the two pieces, and reports seam problems.

Usage:
    partial = "Here is the code:\\n```python\\ndef add(a, b):"
    joined = stitch(partial, continuation_request(partial))
    problems = seam_problems(partial, joined)
"""

FENCE_CHARACTERS = ("`", "~")
MIN_FENCE_LENGTH = 3

# Overlap scanning bounds: the shared text is a clipped answer suffix and
# continuation prefix, so it is short but not a single character.
MAX_OVERLAP_CHARS = 400
MIN_OVERLAP_CHARS = 12

# The seam is the last stretch of the partial answer; if it appears twice in a
# row in the join, the model repeated itself instead of continuing.
SEAM_WINDOW_CHARS = 40

# A truncated sentence is only flagged on real prose, not on a short snippet.
MIN_PROSE_LENGTH = 200
TRAILING_PROSE_CHARS = (",", ":")

_CONTINUATION_INSTRUCTION = (
    "Continue exactly where you stopped, with no repetition and no preface."
)
_FENCE_INSTRUCTION = (
    "You are inside an open code block: continue inside it and do not write "
    "the opening fence again."
)


def open_fence(text: str) -> str | None:
    """Return the opening fence line of an unclosed code block, or None.

    A fence line starts (after spaces) with three or more backticks or tildes. Outside a block it opens one and may
    carry an info string ("```python"); inside a block only a bare fence of the same character, at least as long as
    the opener, closes it - every other line, including shorter fences, is plain text. The result is stripped."""
    opener: tuple[str, int, str] | None = None
    for line in text.splitlines():
        fence = _fence_line(line)
        if fence is None:
            continue
        character, count, bare = fence
        if opener is None:
            opener = (character, count, line.strip())
        elif character == opener[0] and count >= opener[1] and bare:
            opener = None
    return opener[2] if opener else None


def continuation_request(partial: str) -> str:
    """Build the user message that asks the model to continue the partial."""
    instructions = [_CONTINUATION_INSTRUCTION]
    if open_fence(partial) is not None:
        instructions.append(_FENCE_INSTRUCTION)
    instructions.append(f'Your last line was: "{_last_line(partial)}"')
    return " ".join(instructions)


def stitch(partial: str, continuation: str) -> str:
    """Join a truncated answer and its continuation without duplication.

    Applies, in order: drop leading blank lines of the continuation; drop a
    re-opened fence line when the partial ends inside a fence; remove the
    overlap between the partial suffix and the continuation prefix.
    """
    continuation = _without_leading_blank_lines(continuation)
    continuation = _without_reopened_fence(partial, continuation)
    continuation = _without_overlap(partial, continuation)
    return partial + continuation   # no whitespace is invented: a cut word and its rest join as they come


def seam_problems(partial: str, joined: str) -> list[str]:
    """List short English sentences describing problems at the seam.

    Checks the open fence, the repeated seam, a mid-word or mid-sentence
    ending in plain prose, and a join shorter than the partial. Empty when fine.
    """
    problems = []
    if open_fence(joined) is not None:
        problems.append("The joined text ends inside an open code block.")
    if _ends_with_repeated_seam(partial, joined):
        problems.append("The seam text is repeated twice in a row.")
    if _ends_mid_sentence_in_prose(joined):
        problems.append("The joined text ends mid-word or mid-sentence.")
    if len(joined) < len(partial):
        problems.append("The joined text is shorter than the partial answer.")
    return problems


def _fence_line(line: str) -> tuple[str, int, bool] | None:
    """Return (fence character, count, bare) for a fence line, else None; bare = nothing after the fence."""
    stripped = line.strip()
    if not stripped or stripped[0] not in FENCE_CHARACTERS:
        return None
    count = len(stripped) - len(stripped.lstrip(stripped[0]))
    if count < MIN_FENCE_LENGTH:
        return None
    return stripped[0], count, not stripped[count:].strip()


def _last_line(text: str) -> str:
    """Return the last non-empty line of the text, stripped, or an empty line."""
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _without_leading_blank_lines(continuation: str) -> str:
    """Drop blank lines at the start of the continuation."""
    lines = continuation.splitlines(keepends=True)
    first_content = 0
    while first_content < len(lines) and not lines[first_content].strip():
        first_content += 1
    return "".join(lines[first_content:])


def _without_reopened_fence(partial: str, continuation: str) -> str:
    """Drop the continuation's first line when it re-opens the partial's fence."""
    fence = open_fence(partial)
    if fence is None or fence != _first_line(continuation).strip():
        return continuation
    lines = continuation.splitlines(keepends=True)
    return "".join(lines[1:])


def _first_line(text: str) -> str:
    """Return the text up to its first line break."""
    return text.split("\n", maxsplit=1)[0]


def _without_overlap(partial: str, continuation: str) -> str:
    """Drop the continuation prefix that repeats the partial's suffix."""
    overlap = _longest_overlap(partial, continuation)
    return continuation[overlap:]


def _longest_overlap(partial: str, continuation: str) -> int:
    """Return the longest remaining-partial prefix length shared as overlap.

    The overlap is capped by MAX_OVERLAP_CHARS and must be at least
    MIN_OVERLAP_CHARS to be reported.
    """
    max_length = min(MAX_OVERLAP_CHARS, len(partial), len(continuation))
    for length in range(max_length, MIN_OVERLAP_CHARS - 1, -1):
        if partial.endswith(continuation[:length]):
            return length
    return 0


def _ends_with_repeated_seam(partial: str, joined: str) -> bool:
    """True when the partial's seam window occurs twice in a row in joined."""
    seam = partial[-SEAM_WINDOW_CHARS:]
    if len(seam) < SEAM_WINDOW_CHARS:
        return False
    return joined.find(seam + seam) != -1


def _ends_mid_sentence_in_prose(joined: str) -> bool:
    """True when joined ends mid-word or mid-sentence in plain prose."""
    if len(joined) <= MIN_PROSE_LENGTH or open_fence(joined) is not None:
        return False
    last_character = joined.rstrip()[-1]
    return last_character.isalnum() or last_character in TRAILING_PROSE_CHARS
