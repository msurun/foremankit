"""Tests for the HTTP layer of the provider clients: how a failed request is reported to the router."""

from datetime import date

import httpx
import pytest

from foremankit import providers


def test_payload_with_a_non_json_value_raises_provider_error_naming_the_provider():
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    request = providers.ChatRequest(
        base_url="https://api.example.test/v1", api_key=None, model_id="model", messages=[], max_tokens=10,
        timeout=5, extra={"reasoning_effort": date(2026, 10, 1)})
    with pytest.raises(providers.ProviderError, match="api.example.test"):
        providers._post_chat(client, request)
