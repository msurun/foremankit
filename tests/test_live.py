"""The live meter: phases from the stream, token rate, split events, and throttled writes."""
from __future__ import annotations

from foremankit.live import LiveMeter


def sse(kind: str) -> bytes:
    if kind == "start-thinking":
        return b'event: content_block_start\ndata: {"type":"content_block_start","content_block":{"type":"thinking","thinking":""}}\n\n'
    if kind == "start-text":
        return b'event: content_block_start\ndata: {"type": "content_block_start", "content_block": {"type": "text", "text": ""}}\n\n'
    return b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"x"}}\n\n'


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make():
    states, clock = [], Clock()
    return LiveMeter(states.append, clock), states, clock


def test_it_reads_the_prompt_until_the_first_token():
    meter, states, _ = make()
    meter.begin()
    assert states[-1]["phase"] == "reading prompt" and states[-1]["active"] is True


def test_phase_follows_the_content_blocks():
    meter, states, clock = make()
    meter.begin()
    meter.feed(sse("start-thinking") + sse("delta"))
    clock.now += 1
    meter.feed(sse("start-text") + sse("delta"))
    assert meter.phase == "answering" and meter.tokens == 2


def test_tokens_per_second_come_from_the_recent_deltas():
    meter, _, clock = make()
    for _ in range(11):
        meter.feed(sse("delta"))
        clock.now += 0.05
    assert 19 < meter.rate() < 21


def test_an_event_split_between_two_chunks_is_counted_once():
    meter, _, _ = make()
    event = sse("delta")
    meter.feed(event[:30])
    meter.feed(event[30:])
    meter.feed(event)
    assert meter.tokens == 2


def test_writes_are_throttled_but_the_end_is_always_reported():
    meter, states, clock = make()
    meter.begin()
    for _ in range(20):
        meter.feed(sse("delta"))
        clock.now += 0.01
    assert len(states) <= 2
    clock.now += 1
    meter.end()
    assert states[-1]["active"] is False and states[-1]["phase"] == "idle" and states[-1]["tokens"] == 20
