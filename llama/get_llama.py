"""Download llama.cpp's llama-server into llama/LLAMA.CPP, the folder every launcher script expects.

    python get_llama.py                 # the tested build (PINNED_TAG) for this machine's GPU
    python get_llama.py --latest        # the newest release instead (untested here)
    python get_llama.py --force         # download again even when llama-server is already there

The build is chosen from the hardware: NVIDIA driver new enough for CUDA 13.4 or 12.4 -> that CUDA build (plus its
separate cudart runtime zip), any other GPU -> Vulkan, no GPU -> CPU. Every archive is checked against the sha256 that
GitHub publishes for the release asset before it is unpacked. Models (.gguf) are not downloaded: put your own next to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "LLAMA.CPP"
PINNED_TAG = "b11429"                       # the build every profile in profiles.json was measured with
API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
CUDA_BUILDS = ((13.4, "13.4"), (12.4, "12.4"))   # (lowest driver CUDA version, build suffix), newest first
SERVER_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"
CHUNK = 1 << 20


def fetch_json(url: str) -> dict:
    """GET a GitHub API url and parse it."""
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "foremankit"})
    with urllib.request.urlopen(request, timeout=30) as answer:
        return json.load(answer)


def driver_cuda_version() -> float | None:
    """The CUDA version the installed NVIDIA driver supports (header of nvidia-smi), or None without an NVIDIA GPU."""
    try:
        text = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", text)
    return float(match.group(1)) if match else None


def has_gpu() -> bool:
    """Any GPU besides the integrated display adapter name 'Microsoft Basic'? Used only to prefer Vulkan over CPU."""
    if sys.platform != "win32":
        return False
    names = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                           capture_output=True, text=True, check=False).stdout
    return any(line.strip() and "basic" not in line.lower() for line in names.splitlines())


def windows_names(tag: str, cuda: float | None, gpu: bool) -> list[str]:
    """Release asset names to fetch on Windows: the CUDA build with its runtime, else Vulkan, else CPU."""
    for lowest, suffix in CUDA_BUILDS:
        if cuda is not None and cuda >= lowest:
            return [f"llama-{tag}-bin-win-cuda-{suffix}-x64.zip", f"cudart-llama-bin-win-cuda-{suffix}-x64.zip"]
    return [f"llama-{tag}-bin-win-{'vulkan' if gpu else 'cpu'}-x64.zip"]


def other_names(tag: str, gpu: bool) -> list[str]:
    """Release asset names on Linux and macOS (CUDA on Linux needs a matching toolkit: not guessed here)."""
    if sys.platform == "darwin":
        return [f"llama-{tag}-bin-macos-{'arm64' if platform.machine() == 'arm64' else 'x64'}.tar.gz"]
    return [f"llama-{tag}-bin-ubuntu-{'vulkan-' if gpu else ''}x64.tar.gz"]


def choose_assets(tag: str, cuda: float | None, gpu: bool) -> list[str]:
    """Names of the release assets this machine needs."""
    return windows_names(tag, cuda, gpu) if sys.platform == "win32" else other_names(tag, gpu or cuda is not None)


def download(url: str, destination: Path) -> str:
    """Save `url` to `destination` and return its sha256 (computed while writing, so the file is read once)."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as answer, destination.open("wb") as out:
        total, done = int(answer.headers.get("Content-Length", 0)), 0
        while block := answer.read(CHUNK):
            out.write(block)
            digest.update(block)
            done += len(block)
            print(f"\r  {destination.name}: {done / 2**20:.0f} / {total / 2**20:.0f} MiB", end="", flush=True)
    print()
    return digest.hexdigest()


def unpack(archive: Path, folder: Path) -> None:
    """Extract a zip or tar.gz; the archive's own top-level folder (llama-bXXXX/) is flattened away."""
    staging = folder.with_name(folder.name + ".unpack")
    shutil.rmtree(staging, ignore_errors=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(staging, filter="data")
    children = list(staging.iterdir())
    source = children[0] if len(children) == 1 and children[0].is_dir() else staging
    folder.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        shutil.move(str(item), str(folder / item.name))
    shutil.rmtree(staging, ignore_errors=True)


def install(tag: str, force: bool = False) -> Path:
    """Download, verify and unpack llama.cpp `tag` (or 'latest') into llama/LLAMA.CPP; returns the llama-server path."""
    server = TARGET / SERVER_NAME
    if server.exists() and not force:
        print(f"llama-server already present: {server} (use --force to download again)")
        return server
    release = fetch_json(f"{API}/latest" if tag == "latest" else f"{API}/tags/{tag}")
    wanted = choose_assets(release["tag_name"], driver_cuda_version(), has_gpu())
    assets = {asset["name"]: asset for asset in release["assets"]}
    missing = [name for name in wanted if name not in assets]
    if missing:
        raise SystemExit(f"release {release['tag_name']} has no asset {', '.join(missing)}")
    scratch = HERE / ".llama-download"
    scratch.mkdir(exist_ok=True)
    for name in wanted:
        fetch_and_unpack(assets[name], scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"llama.cpp {release['tag_name']} installed in {TARGET}")
    return server


def fetch_and_unpack(asset: dict, scratch: Path) -> None:
    """One release asset: download, compare with the published sha256, unpack. A mismatch aborts and keeps nothing."""
    archive = scratch / asset["name"]
    actual = download(asset["browser_download_url"], archive)
    expected = str(asset.get("digest") or "").removeprefix("sha256:")
    if not expected or actual != expected:
        archive.unlink()
        raise SystemExit(f"sha256 check failed for {asset['name']}: published {expected or 'nothing'}, downloaded {actual}")
    unpack(archive, TARGET)


def main(argv: list[str] | None = None) -> None:
    """Command line entry."""
    parser = argparse.ArgumentParser(description="Download llama.cpp's llama-server for this machine.")
    parser.add_argument("--latest", action="store_true", help="newest release instead of the tested one")
    parser.add_argument("--force", action="store_true", help="download even when llama-server exists")
    args = parser.parse_args(argv)
    install("latest" if args.latest else PINNED_TAG, args.force)


if __name__ == "__main__":
    main()
