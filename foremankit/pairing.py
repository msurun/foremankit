"""Pairing of lab computers: find each other on the LAN and hand over the lab key without copying files by hand.

How it works (home network, every computer is yours):
  * The computer that HAS the lab key ("host") shows a short code and listens for two minutes (`PairingHost`).
  * The new computer ("joiner") finds hosts with a UDP broadcast (`find_hosts`), you type the code on it, and it fetches the
    key (`join`). The key travels AES-GCM encrypted with a key derived from the code by scrypt, so a listener on the
    network learns nothing without the code. The host hands out ONE sealed key per session: an intruder who connects first
    wastes the session (the real joiner fails, you see it, you start again) and gets a single blob to brute-force against a
    40-bit code under scrypt, which is not practical. Discovery reveals only a name and a port, never a secret.
  * Both sides then register each other as lab nodes (done by `foremankit.panel`, not here).
This module has the protocol and the sockets; the panel and the files are somewhere else.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import select
import socket
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DISCOVERY_PORT = 8766                  # UDP: "who has a lab key?"
PAIRING_PORT = 8767                    # TCP: the key hand-over (one firewall rule for both ports)
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O/1/I: easy to read aloud and type
CODE_LENGTH = 8                        # 32**8 = 2**40 possibilities
SESSION_SECONDS = 120
CONFIRM_SECONDS = 10
SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}
SALT_BYTES, NONCE_BYTES = 16, 12
MAX_MESSAGE = 64 * 1024
FIND_MAGIC = b"foremankit-find"        # discovery request


class PairingError(Exception):
    """Pairing failed in a way the user can act on (wrong code, nobody answered, session over)."""


def new_code() -> str:
    """A fresh code like KQ7F-2MXD."""
    letters = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
    return f"{letters[:4]}-{letters[4:]}"


def normalize_code(text: str) -> str:
    """What the user typed -> the canonical code (upper case, no dashes or spaces)."""
    return "".join(ch for ch in text.upper() if ch in CODE_ALPHABET)


def derive_key(code: str, salt: bytes) -> bytes:
    """scrypt of the normalized code: slows down guessing from a captured blob."""
    return hashlib.scrypt(normalize_code(code).encode(), salt=salt, **SCRYPT)


def seal(payload: dict[str, Any], code: str) -> bytes:
    """salt + nonce + AES-GCM ciphertext of the JSON payload, keyed by the code."""
    salt, nonce = os.urandom(SALT_BYTES), os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(derive_key(code, salt)).encrypt(nonce, json.dumps(payload).encode(), b"foremankit-pairing")
    return salt + nonce + ciphertext


def unseal(blob: bytes, code: str) -> dict[str, Any]:
    """Decrypt what `seal` made; PairingError for a wrong code or a damaged blob."""
    salt, nonce, ciphertext = blob[:SALT_BYTES], blob[SALT_BYTES:SALT_BYTES + NONCE_BYTES], blob[SALT_BYTES + NONCE_BYTES:]
    try:
        return json.loads(AESGCM(derive_key(code, salt)).decrypt(nonce, ciphertext, b"foremankit-pairing"))
    except (InvalidTag, ValueError):
        raise PairingError("wrong code (or the answer was damaged)") from None


def proof(code: str, blob: bytes) -> str:
    """The joiner's receipt: proves it decrypted the key, i.e. knew the code."""
    return hmac.new(derive_key(code, blob[:SALT_BYTES]), b"paired", hashlib.sha256).hexdigest()


def send_message(sock: socket.socket, message: dict[str, Any] | bytes) -> None:
    """Length-prefixed frame: JSON for dicts, raw bytes for the sealed blob."""
    raw = message if isinstance(message, bytes) else json.dumps(message).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def read_message(sock: socket.socket) -> bytes:
    """One length-prefixed frame (PairingError when the peer sends nonsense or hangs up)."""
    header = read_exactly(sock, 4)
    (length,) = struct.unpack(">I", header)
    if length > MAX_MESSAGE:
        raise PairingError("message too large")
    return read_exactly(sock, length)


def read_exactly(sock: socket.socket, count: int) -> bytes:
    """Exactly `count` bytes from the socket."""
    data = b""
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise PairingError("connection closed")
        data += chunk
    return data


@dataclass
class HostState:
    """What the panel shows while a host waits: waiting -> paired | expired | failed."""
    code: str
    state: str = "waiting"
    peer: dict[str, Any] | None = None
    expires: float = 0.0
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON for the state file the panel polls."""
        return {"code": self.code, "state": self.state, "peer": self.peer, "expires": self.expires, "detail": self.detail}


class PairingHost:
    """Waits for one joiner for SESSION_SECONDS, answering discovery broadcasts, then gives it the lab key once."""

    def __init__(self, code: str, lab_key: str, me: dict[str, Any], publish: Callable[[HostState], None],
                 clock: Callable[[], float] = time.time, ports: tuple[int, int] = (DISCOVERY_PORT, PAIRING_PORT)):
        self.code, self.lab_key, self.me, self.publish, self.clock = code, lab_key, me, publish, clock
        self.discovery_port, self.pairing_port = ports
        self.status = HostState(code=code, expires=clock() + SESSION_SECONDS)

    def run(self) -> HostState:
        """Serve until paired, failed or expired; returns the final state."""
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp.bind(("", self.discovery_port))
        tcp = socket.socket()
        tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp.bind(("", self.pairing_port))
        tcp.listen(1)
        self.publish(self.status)
        try:
            while self.status.state == "waiting" and self.clock() < self.status.expires:
                self.serve_once(udp, tcp)
        finally:
            udp.close()
            tcp.close()
        if self.status.state == "waiting":
            self.status.state = "expired"
        self.publish(self.status)
        return self.status

    def serve_once(self, udp: socket.socket, tcp: socket.socket) -> None:
        """Handle at most one discovery packet or one pairing connection, waiting up to half a second."""
        ready, _, _ = select.select([udp, tcp], [], [], 0.5)
        if udp in ready:
            data, address = udp.recvfrom(256)
            if data == FIND_MAGIC:
                reply = {"name": self.me["name"], "port": self.pairing_port}
                udp.sendto(json.dumps(reply).encode(), address)
        if tcp in ready:
            connection, address = tcp.accept()
            with connection:
                self.hand_over(connection, address[0])

    def hand_over(self, connection: socket.socket, ip: str) -> None:
        """One session = one sealed key; then wait for the joiner's proof that it opened it."""
        connection.settimeout(CONFIRM_SECONDS)
        try:
            hello = json.loads(read_message(connection))
            blob = seal({"peer_key": self.lab_key, "name": self.me["name"], "llama_port": self.me.get("llama_port", 8080)},
                        self.code)
            send_message(connection, blob)
            receipt = json.loads(read_message(connection)).get("proof", "")
        except (PairingError, ValueError, OSError) as problem:
            self.status.state, self.status.detail = "failed", f"the joiner did not finish ({problem}); start again"
            return
        if hmac.compare_digest(receipt, proof(self.code, blob)):
            self.status.state = "paired"
            self.status.peer = {"name": str(hello.get("name", ip)), "ip": ip, "llama_port": int(hello.get("llama_port", 8080))}
        else:
            self.status.state, self.status.detail = "failed", "wrong code entered on the other computer; start again"


def find_hosts(wait: float = 2.0, port: int = DISCOVERY_PORT) -> list[dict[str, Any]]:
    """Broadcast a discovery request and collect the answers: [{"name", "ip", "port"}]."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    found: dict[str, dict[str, Any]] = {}
    try:
        sock.sendto(FIND_MAGIC, ("255.255.255.255", port))
        sock.sendto(FIND_MAGIC, ("127.0.0.1", port))
        deadline = time.monotonic() + wait
        while (left := deadline - time.monotonic()) > 0:
            ready, _, _ = select.select([sock], [], [], left)
            if ready:
                try:
                    data, address = sock.recvfrom(1024)
                except ConnectionResetError:
                    continue   # Windows reports "port unreachable" of our own probe (nobody listens on 127.0.0.1) this way
                try:
                    reply = json.loads(data)
                    found[address[0]] = {"name": str(reply["name"]), "ip": address[0], "port": int(reply["port"])}
                except (ValueError, KeyError, TypeError):
                    continue
    finally:
        sock.close()
    return list(found.values())


def join(host_ip: str, host_port: int, code: str, me: dict[str, Any]) -> dict[str, Any]:
    """Fetch the lab key from `host_ip` with the code; returns the host's payload ({peer_key, name, llama_port}).

    Raises PairingError for a wrong code, a closed session or an unreachable host."""
    try:
        with socket.create_connection((host_ip, host_port), timeout=CONFIRM_SECONDS) as sock:
            send_message(sock, {"name": me["name"], "llama_port": me.get("llama_port", 8080)})
            blob = read_message(sock)
            payload = unseal(blob, code)
            send_message(sock, {"proof": proof(code, blob)})
    except OSError as problem:
        raise PairingError(f"cannot reach {host_ip}:{host_port} ({problem}); is pairing running there and the firewall open?") from None
    return payload


def store_key(env_file: Path, key: str, replace: bool = False, var: str = "FOREMANKIT_PEER_KEY") -> str:
    """Write the lab key into `.env`. Returns "added", "same" or "replaced"; PairingError if a different key is there
    and `replace` is not set (two labs with different keys must not be mixed up silently)."""
    text = env_file.read_text(encoding="utf-8") if env_file.is_file() else ""
    lines = text.splitlines()
    current = next((line.split("=", 1)[1].strip() for line in lines if line.startswith(f"{var}=")), "")
    if current == key:
        return "same"
    if current and not replace:
        raise PairingError("this computer already has a different lab key; use 'replace' only if you mean to join the other lab")
    lines = [line for line in lines if not line.startswith(f"{var}=")] + [f"{var}={key}"]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "replaced" if current else "added"
