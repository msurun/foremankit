"""llama/get_llama.py: which release assets a machine needs, checksum refusal and flattening of the archive folder."""
from __future__ import annotations

import hashlib
import importlib.util
import zipfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("get_llama", Path(__file__).resolve().parents[1] / "llama" / "get_llama.py")
GET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GET)


@pytest.mark.parametrize("cuda, gpu, expected", [
    (13.9, True, ["llama-b1-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"]),
    (12.9, True, ["llama-b1-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]),
    (11.8, True, ["llama-b1-bin-win-vulkan-x64.zip"]),
    (None, True, ["llama-b1-bin-win-vulkan-x64.zip"]),
    (None, False, ["llama-b1-bin-win-cpu-x64.zip"]),
])
def test_windows_build_follows_the_driver_and_the_gpu(cuda, gpu, expected):
    assert GET.windows_names("b1", cuda, gpu) == expected


def make_zip(path: Path, inner: str) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(inner, "binary")


def test_unpack_flattens_the_single_top_level_folder(tmp_path: Path):
    archive = tmp_path / "a.zip"
    make_zip(archive, "llama-b1/llama-server.exe")
    GET.unpack(archive, tmp_path / "LLAMA.CPP")
    assert (tmp_path / "LLAMA.CPP" / "llama-server.exe").read_text() == "binary"


def test_a_checksum_mismatch_deletes_the_archive_and_stops(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(GET, "download", lambda url, destination: (destination.write_bytes(b"x"), "bad")[1])
    asset = {"name": "x.zip", "browser_download_url": "u", "digest": "sha256:" + hashlib.sha256(b"x").hexdigest()}
    with pytest.raises(SystemExit):
        GET.fetch_and_unpack(asset, tmp_path)
    assert not (tmp_path / "x.zip").exists()


def test_a_release_without_published_checksum_is_refused(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(GET, "download", lambda url, destination: (destination.write_bytes(b"x"), "abc")[1])
    with pytest.raises(SystemExit):
        GET.fetch_and_unpack({"name": "x.zip", "browser_download_url": "u"}, tmp_path)


def test_an_existing_server_is_not_downloaded_again(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(GET, "TARGET", tmp_path)
    (tmp_path / GET.SERVER_NAME).write_text("x")
    monkeypatch.setattr(GET, "fetch_json", lambda url: pytest.fail("must not touch the network"))
    assert GET.install("b1") == tmp_path / GET.SERVER_NAME
