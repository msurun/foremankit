"""A llama-server started with -Share wants the lab key from every client, so this machine's own router sends it too,
but only to a loopback address: a url_env pointing at another host must never receive the key."""
from foremankit.nodes import local_endpoints

KEY = "lab-key"


def test_the_key_goes_to_a_server_on_this_machine():
    cfg = {"endpoints": [{"name": "llamacpp", "url": "http://127.0.0.1:8080"}]}
    assert local_endpoints(cfg, KEY)[0].api_key == KEY


def test_the_key_is_not_sent_to_another_host():
    cfg = {"endpoints": [{"name": "remote", "url": "http://192.168.1.50:8080"}]}
    assert local_endpoints(cfg, KEY)[0].api_key is None


def test_without_a_key_the_endpoint_stays_open():
    cfg = {"endpoints": [{"name": "llamacpp", "url": "http://localhost:8080"}]}
    assert local_endpoints(cfg)[0].api_key is None
