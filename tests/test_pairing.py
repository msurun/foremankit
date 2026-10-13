"""Pairing: code handling, sealing, a full host/joiner exchange over loopback, wrong code, expiry and key storage."""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from foremankit import pairing
from foremankit.pairing import PairingError, PairingHost


def free_ports() -> tuple[int, int]:
    """Two free ports (UDP discovery, TCP pairing) so tests never collide with a real pairing."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp, socket.socket() as tcp:
        udp.bind(("127.0.0.1", 0))
        tcp.bind(("127.0.0.1", 0))
        return udp.getsockname()[1], tcp.getsockname()[1]


def start_host(code: str, ports: tuple[int, int], key: str = "lab-key-123"):
    states: list = []
    host = PairingHost(code, key, {"name": "legion", "llama_port": 8080}, lambda s: states.append(s.state), ports=ports)
    thread = threading.Thread(target=host.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not states and time.monotonic() < deadline:   # the host publishes 'waiting' once its sockets are open
        time.sleep(0.01)
    return host, thread, states


def test_codes_are_readable_and_unique():
    codes = {pairing.new_code() for _ in range(50)}
    assert len(codes) == 50
    assert all(len(code) == 9 and code[4] == "-" and not set(code) & set("01OI") for code in codes)


def test_a_typed_code_is_normalized():
    assert pairing.normalize_code("kq7f 2mxd") == "KQ7F2MXD"
    assert pairing.normalize_code("KQ7F-2MXD") == "KQ7F2MXD"


def test_sealed_payload_opens_only_with_the_same_code():
    blob = pairing.seal({"peer_key": "secret"}, "KQ7F-2MXD")
    assert pairing.unseal(blob, "kq7f2mxd") == {"peer_key": "secret"}
    with pytest.raises(PairingError):
        pairing.unseal(blob, "AAAA-AAAA")
    assert b"secret" not in blob


def test_a_damaged_blob_is_rejected():
    blob = bytearray(pairing.seal({"peer_key": "secret"}, "KQ7F-2MXD"))
    blob[-1] ^= 1
    with pytest.raises(PairingError):
        pairing.unseal(bytes(blob), "KQ7F-2MXD")


def test_full_exchange_gives_the_joiner_the_key_and_the_host_the_peer():
    ports = free_ports()
    host, thread, states = start_host("KQ7F-2MXD", ports)
    found = pairing.find_hosts(wait=1.0, port=ports[0])
    assert found and found[0]["name"] == "legion" and found[0]["port"] == ports[1]
    payload = pairing.join("127.0.0.1", ports[1], "kq7f2mxd", {"name": "firefly", "llama_port": 8081})
    thread.join(5)
    assert payload["peer_key"] == "lab-key-123" and payload["name"] == "legion"
    assert host.status.state == "paired" and host.status.peer["name"] == "firefly" and host.status.peer["llama_port"] == 8081
    assert states[0] == "waiting" and states[-1] == "paired"


def test_a_wrong_code_fails_on_the_joiner_and_the_session_is_spent():
    ports = free_ports()
    host, thread, _ = start_host("KQ7F-2MXD", ports)
    with pytest.raises(PairingError):
        pairing.join("127.0.0.1", ports[1], "AAAA-AAAA", {"name": "intruder"})
    thread.join(15)
    assert host.status.state == "failed"


def test_an_expired_session_stops_waiting():
    ports = free_ports()
    clock = iter([0.0, 0.0] + [10_000.0] * 50)
    host = PairingHost("KQ7F-2MXD", "k", {"name": "legion"}, lambda s: None, clock=lambda: next(clock), ports=ports)
    assert host.run().state == "expired"


def test_joining_a_host_that_is_not_there_explains_what_to_check():
    with pytest.raises(PairingError, match="firewall"):
        pairing.join("127.0.0.1", free_ports()[1], "KQ7F-2MXD", {"name": "x"})


def test_store_key_adds_keeps_and_refuses_a_different_key(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n", encoding="utf-8")
    assert pairing.store_key(env, "abc") == "added" and "FOREMANKIT_PEER_KEY=abc" in env.read_text(encoding="utf-8")
    assert pairing.store_key(env, "abc") == "same"
    with pytest.raises(PairingError):
        pairing.store_key(env, "different")
    assert pairing.store_key(env, "different", replace=True) == "replaced"
    assert "OTHER=1" in env.read_text(encoding="utf-8") and env.read_text(encoding="utf-8").count("FOREMANKIT_PEER_KEY=") == 1


def test_finding_hosts_when_nobody_listens_returns_nothing_instead_of_crashing():
    assert pairing.find_hosts(wait=0.5, port=free_ports()[0]) == []
