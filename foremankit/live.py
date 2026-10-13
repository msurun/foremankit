"""Live meter for the local model: is it reading the prompt, thinking, answering or calling a tool, and how fast.

The gateway feeds every chunk of a streamed Anthropic answer to a `LiveMeter`. The meter recognises the phase from the
`content_block_start` events (thinking / text / tool_use) and counts `content_block_delta` events as tokens: llama.cpp
sends about one delta per token, so the rate is approximate. The VS Code panel shows the result, which is written to
`gateway_live.json` in the state dir, at most a few times per second. Pure logic: the clock and the writer are injected.
"""
from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable
from typing import Any

EVENT = re.compile(r'"type"\s*:\s*"content_block_delta"|"content_block"\s*:\s*\{\s*"type"\s*:\s*"(thinking|text|tool_use)"')
PHASES = {"thinking": "thinking", "text": "answering", "tool_use": "calling a tool"}
RATE_WINDOW_S = 3.0          # tokens per second are measured over the last few seconds, so a pause does not average away
WRITE_EVERY_S = 0.4          # the panel polls about once a second; more writes would only cost disk
KEEP_TAIL = 256              # unparsed end of the stream kept for an event split across two chunks


class LiveMeter:
    """State of one streamed answer; `write` receives a dict whenever the panel should be told something new."""

    def __init__(self, write: Callable[[dict[str, Any]], None], clock: Callable[[], float] = time.time):
        self.write, self.clock = write, clock
        self.phase, self.tokens = "reading prompt", 0
        self.started = clock()
        self.recent: deque[float] = deque()
        self.buffer = ""
        self.last_write = 0.0
        self.first_token_at: float | None = None

    def begin(self) -> None:
        """The request was sent: until the first token arrives the model is reading the prompt."""
        self.push(force=True)

    def feed(self, chunk: bytes) -> None:
        """Account for one chunk of the response stream."""
        self.buffer += chunk.decode("utf-8", errors="ignore")
        end = 0
        for match in EVENT.finditer(self.buffer):
            end = match.end()
            if match.group(1):
                self.phase = PHASES[match.group(1)]
            else:
                self.count_token()
        self.buffer = self.buffer[end:][-KEEP_TAIL:]
        self.push()

    def count_token(self) -> None:
        """One delta event: a token."""
        now = self.clock()
        self.first_token_at = self.first_token_at or now
        self.tokens += 1
        self.recent.append(now)
        while self.recent and now - self.recent[0] > RATE_WINDOW_S:
            self.recent.popleft()

    def rate(self) -> float:
        """Tokens per second over the recent window (0 until two tokens arrived)."""
        if len(self.recent) < 2:
            return 0.0
        span = self.recent[-1] - self.recent[0]
        return (len(self.recent) - 1) / span if span > 0 else 0.0

    def push(self, force: bool = False) -> None:
        """Tell the panel the current state, but not more often than WRITE_EVERY_S."""
        now = self.clock()
        if not force and now - self.last_write < WRITE_EVERY_S:
            return
        self.last_write = now
        self.write({"active": True, "phase": self.phase, "tokens": self.tokens, "tps": round(self.rate(), 1),
                    "since": self.started, "time": now})

    def end(self) -> None:
        """The stream finished: report the average speed of the whole answer (prompt reading excluded) for the idle display."""
        now = self.clock()
        generating = now - self.first_token_at if self.first_token_at else 0.0
        average = round((self.tokens - 1) / generating, 1) if self.tokens > 1 and generating > 0 else 0.0
        self.write({"active": False, "phase": "idle", "tokens": self.tokens, "tps": average, "since": self.started, "time": now})
