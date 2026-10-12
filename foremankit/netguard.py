"""Transport guard for private data: only loopback / private / Tailscale addresses."""
from __future__ import annotations

import ipaddress
import socket
from typing import Callable
from urllib.parse import urlparse

ALLOWED_NETS = [ipaddress.ip_network(network_str) for network_str in (
    "127.0.0.0/8",      # loopback (incl. SSH tunnel to localhost)
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "100.64.0.0/10",    # Tailscale / CGNAT
    "::1/128",
    "fd7a:115c:a1e0::/48",  # Tailscale IPv6
    "fc00::/7",         # IPv6 unique-local (the private range of a home/office LAN; Windows hosts publish these)
    "fe80::/10",        # IPv6 link-local (same network segment only)
)]

Resolver = Callable[[str], list[str]]


class PrivacyError(RuntimeError):
    """Raised when private data would leave the trusted network."""


def default_resolver(host: str) -> list[str]:
    """Resolve a hostname to a sorted list of unique IP address strings."""
    address_infos = socket.getaddrinfo(host, None)
    return sorted({address_info[4][0] for address_info in address_infos})


def _ip_allowed(ip: str) -> bool:
    """Return True if the IP address is in an allowed private network."""
    address = ipaddress.ip_address(ip.split("%")[0])
    if getattr(address, "ipv4_mapped", None):
        address = address.ipv4_mapped
    return any(address in network for network in ALLOWED_NETS)


def _resolve_host(host: str, resolver: Resolver | None = None) -> list[str]:
    """Resolve a hostname to a list of IP addresses, or raise an error."""
    try:
        ips = [str(ipaddress.ip_address(host))]
    except ValueError:
        try:
            ips = (resolver or default_resolver)(host)
        except OSError as exc:
            raise PrivacyError(f"cannot resolve {host} ({exc.__class__.__name__}); refusing") from exc
        if not ips:
            raise PrivacyError(f"{host} resolved to nothing; refusing")
    return ips


def check_host(url: str, resolver: Resolver | None = None) -> tuple[bool, str]:
    """(allowed, reason). Fails closed when the host cannot be resolved."""
    host = urlparse(url).hostname
    if not host:
        return False, f"cannot parse host from {url!r}"
    if host.lower() == "localhost":
        return True, "localhost"
    
    try:
        ips = _resolve_host(host, resolver)
    except PrivacyError as exc:
        return False, str(exc)
        
    public_ips = [ip for ip in ips if not _ip_allowed(ip)]
    if public_ips:
        return False, f"{host} -> public address {', '.join(public_ips)}"
    return True, f"{host} -> {', '.join(ips)}"


def assert_private_ok(url: str, *, allow_public_endpoint: bool = False,
                      resolver: Resolver | None = None, warn: Callable[[str], None] | None = None) -> None:
    """Raise PrivacyError if the URL points to a non-private address."""
    ok, reason = check_host(url, resolver)
    if ok:
        return
    if allow_public_endpoint:
        if warn:
            warn(f"WARNING: private data sent to public endpoint ({reason}) because allow_public_endpoint=true")
        return
    raise PrivacyError(f"private data refused: {reason}. Use localhost or a LAN address.")
