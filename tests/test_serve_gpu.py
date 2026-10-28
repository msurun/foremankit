# Pure helpers of the Kaggle GPU serve script (kaggle/gpu/serve_gpu.py): llama-server argv, tunnel URL parsing,
# key, model lookup and the READY banner. The script is not a package, so it is loaded by path.
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "kaggle" / "gpu" / "serve_gpu.py"
spec = importlib.util.spec_from_file_location("serve_gpu", SCRIPT)
serve_gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve_gpu)


def after(argv: list[str], flag: str) -> str:
    """The value that follows a flag in an argv list."""
    return argv[argv.index(flag) + 1]


def test_server_command_defaults():
    """Key, port, context, two slots, flash attention, tool templates and the model stem as alias."""
    argv = serve_gpu.server_command(Path("/b/llama-server"), Path("/m/Qwen-A3B.gguf"), "sk-x", {})
    assert all(isinstance(item, str) for item in argv)
    assert after(argv, "--api-key") == "sk-x" and after(argv, "--port") == "8080"
    assert after(argv, "--ctx-size") == "262144" and after(argv, "--cache-type-v") == "q8_0" and after(argv, "--parallel") == "2"
    assert after(argv, "-fa") == "on" and after(argv, "--host") == "127.0.0.1"
    assert {"--jinja", "--kv-unified", "--cache-reuse"} <= set(argv)
    assert after(argv, "--alias") == "Qwen-A3B"


def test_server_command_env_overrides():
    """Context, slots and alias come from the environment; extra args are appended as separate items."""
    env = {"FOREMANKIT_CTX_SIZE": "32768", "FOREMANKIT_PARALLEL": "3", "FOREMANKIT_ALIAS": "qwen",
           "FOREMANKIT_LLAMA_ARGS": "--temp 0.6 --top-k 20"}
    argv = serve_gpu.server_command(Path("b"), Path("m.gguf"), "k", env)
    assert after(argv, "--ctx-size") == "32768" and after(argv, "--parallel") == "3"
    assert after(argv, "--alias") == "qwen" and argv[-4:] == ["--temp", "0.6", "--top-k", "20"]


def test_tunnel_url():
    """The quick-tunnel address is found in a cloudflared line; the API host and other lines are not."""
    line = "2026-10-28T12:00:00Z INF |  https://glad-calibration-perspective.trycloudflare.com  |"
    assert serve_gpu.tunnel_url(line) == "https://glad-calibration-perspective.trycloudflare.com"
    assert serve_gpu.tunnel_url("requesting https://api.trycloudflare.com/tunnel") is None
    assert serve_gpu.tunnel_url("INF Starting tunnel") is None


def test_new_key():
    """A configured key wins; otherwise a fresh random sk- key (an empty variable counts as unset)."""
    assert serve_gpu.new_key({"FOREMANKIT_SERVE_KEY": "sk-fixed"}) == "sk-fixed"
    first, second = serve_gpu.new_key({}), serve_gpu.new_key({"FOREMANKIT_SERVE_KEY": ""})
    assert first.startswith("sk-") and len(first) > 20 and first != second


def test_find_model(tmp_path: Path):
    """The configured file, a missing one refused, else the first gguf that is not a vision projector."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "mmproj-F16.gguf").write_bytes(b"")
    (tmp_path / "a" / "model.gguf").write_bytes(b"")
    assert serve_gpu.find_model({}, tmp_path) == tmp_path / "a" / "model.gguf"
    assert serve_gpu.find_model({"FOREMANKIT_MODEL_GGUF": str(tmp_path / "a" / "model.gguf")}, tmp_path).name == "model.gguf"
    with pytest.raises(FileNotFoundError):
        serve_gpu.find_model({"FOREMANKIT_MODEL_GGUF": str(tmp_path / "missing.gguf")}, tmp_path)
    assert serve_gpu.find_model({}, tmp_path / "a" / "none") is None


def test_banner():
    """The banner names address, key, alias and the panel; the address is the root, not .../v1."""
    text = serve_gpu.banner("https://x.trycloudflare.com", "sk-k", "qwen")
    assert all(part in text for part in ("https://x.trycloudflare.com", "sk-k", "qwen", "ForemanKit"))
    assert "trycloudflare.com/v1" not in text
