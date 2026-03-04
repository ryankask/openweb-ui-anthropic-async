"""
Integration tests for the async Anthropic pipe using pytest.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from anthropic_async import Pipe


@pytest.fixture
def pipe_without_api_key():
    """Provide a Pipe instance without requiring an API key."""
    return Pipe()


@pytest.mark.integration
async def test_basic_non_streaming(pipe_instance, create_text_body, execute_pipe_func):
    """Test basic non-streaming text completion."""
    body = create_text_body("Say 'Hello' in one word.", stream=False)
    params = {"body": body}

    response = await execute_pipe_func(pipe_instance.pipe, params)

    assert isinstance(response, dict)
    choices = response.get("choices", [])
    assert choices
    message = choices[0].get("message", {})
    content = message.get("content", "")
    assert isinstance(content, str) and content
    assert "hello" in content.lower() or "hi" in content.lower()


@pytest.mark.integration
@pytest.mark.streaming
async def test_basic_streaming(pipe_instance, create_text_body, execute_pipe_func):
    """Test basic streaming text completion."""
    body = create_text_body("Count from 1 to 3.", stream=True)
    params = {"body": body}

    response = await execute_pipe_func(pipe_instance.pipe, params)

    assert isinstance(response, AsyncGenerator)

    chunks: list[dict] = []
    async for chunk in response:
        chunks.append(chunk)

    assert chunks
    assert chunks[-1].get("choices", [{}])[0].get("finish_reason") == "stop"

    streamed_text = "".join(
        chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        for chunk in chunks
    )
    assert streamed_text


@pytest.mark.integration
async def test_system_message(pipe_instance, create_system_body, execute_pipe_func):
    """Test system message handling."""
    body = create_system_body(
        "You are a helpful assistant. Always respond with exactly one word.",
        "What is 2+2?",
    )
    params = {"body": body}

    response = await execute_pipe_func(pipe_instance.pipe, params)

    assert isinstance(response, dict)
    choices = response.get("choices", [])
    assert choices
    message = choices[0].get("message", {})
    content = message.get("content", "")
    assert isinstance(content, str) and content


@pytest.mark.integration
@pytest.mark.images
async def test_image_base64(
    pipe_instance, create_image_body, small_test_image, execute_pipe_func
):
    """Test image processing with base64 data."""
    body = create_image_body("What color is this image?", small_test_image)
    params = {"body": body}

    response = await execute_pipe_func(pipe_instance.pipe, params)

    assert isinstance(response, dict)
    choices = response.get("choices", [])
    assert choices
    message = choices[0].get("message", {})
    content = message.get("content", "")
    assert isinstance(content, str) and content


@pytest.mark.integration
@pytest.mark.images
async def test_image_url(
    pipe_instance, create_image_body, test_image_url, execute_pipe_func
):
    """Test image processing with URL."""
    body = create_image_body("Describe this image briefly.", test_image_url)
    params = {"body": body}

    response = await execute_pipe_func(pipe_instance.pipe, params)

    assert isinstance(response, dict)
    choices = response.get("choices", [])
    assert choices
    message = choices[0].get("message", {})
    content = message.get("content", "")
    assert isinstance(content, str) and content


@pytest.mark.integration
@pytest.mark.error_handling
async def test_invalid_model(pipe_instance, create_text_body, execute_pipe_func):
    """Test error handling with invalid model."""
    body = create_text_body("Hello", model="anthropic.invalid-model", max_tokens=10)
    params = {"body": body}

    response = await execute_pipe_func(pipe_instance.pipe, params)

    # Should return an error message
    assert isinstance(response, str)
    assert "error" in response.lower()


@pytest.mark.integration
@pytest.mark.error_handling
@pytest.mark.images
async def test_large_image_error(
    pipe_instance, create_image_body, large_image_data, execute_pipe_func
):
    """Test error handling with oversized image."""
    body = create_image_body("Describe this image.", large_image_data)
    params = {"body": body}

    # Should raise an exception or return error
    try:
        response = await execute_pipe_func(pipe_instance.pipe, params)
        # If it returns a response, it should be an error message
        assert isinstance(response, str)
        assert "error" in response.lower() or "size" in response.lower()
    except ValueError as e:
        # Exception is also acceptable
        assert "size" in str(e).lower()


@pytest.mark.asyncio
async def test_effort_invalid_value(
    pipe_without_api_key, create_text_body, execute_pipe_func
):
    """Invalid effort values should raise a ValueError."""
    body = create_text_body(
        "Test invalid effort",
        model="anthropic.claude-sonnet-4-6",
        max_tokens=100,
    )
    body["effort"] = "ultra"

    params = {"body": body}

    with pytest.raises(ValueError) as excinfo:
        await execute_pipe_func(pipe_without_api_key.pipe, params)

    assert "effort" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_effort_inserts_output_config(
    pipe_without_api_key, create_text_body, execute_pipe_func, monkeypatch
):
    """effort in body is forwarded as output_config for supported models."""
    body = create_text_body(
        "Test effort payload",
        model="anthropic.claude-sonnet-4-6",
        max_tokens=200,
    )
    body["effort"] = "medium"

    captured_payload: dict | None = None

    async def fake_non_stream(self, url, headers, payload):
        nonlocal captured_payload
        captured_payload = payload
        return "ok"

    monkeypatch.setattr(
        pipe_without_api_key.__class__, "non_stream_response", fake_non_stream
    )

    response = await execute_pipe_func(pipe_without_api_key.pipe, {"body": body})

    assert response == "ok"
    assert captured_payload is not None
    assert captured_payload.get("output_config") == {"effort": "medium"}
    assert captured_payload["max_tokens"] == 200


@pytest.mark.asyncio
async def test_effort_allows_temperature(
    pipe_without_api_key, create_text_body, execute_pipe_func, monkeypatch
):
    """effort does not restrict temperature; both can be set together."""
    body = create_text_body(
        "Test effort with temperature",
        model="anthropic.claude-sonnet-4-6",
        max_tokens=200,
    )
    body["effort"] = "low"
    body["temperature"] = 0.5

    captured_payload: dict | None = None

    async def fake_non_stream(self, url, headers, payload):
        nonlocal captured_payload
        captured_payload = payload
        return "ok"

    monkeypatch.setattr(
        pipe_without_api_key.__class__, "non_stream_response", fake_non_stream
    )

    response = await execute_pipe_func(pipe_without_api_key.pipe, {"body": body})

    assert response == "ok"
    assert captured_payload is not None
    assert captured_payload.get("output_config") == {"effort": "low"}
    assert captured_payload.get("temperature") == 0.5


@pytest.mark.asyncio
async def test_effort_valve_default_applied_for_supported_model(
    pipe_without_api_key, create_text_body, execute_pipe_func, monkeypatch
):
    """Valve DEFAULT_EFFORT is used when effort is not in the request body."""
    body = create_text_body(
        "Test valve default effort",
        model="anthropic.claude-sonnet-4-6",
        max_tokens=200,
    )
    # No "effort" key in body — should fall back to valve default
    body.pop("effort", None)

    captured_payload: dict | None = None

    async def fake_non_stream(self, url, headers, payload):
        nonlocal captured_payload
        captured_payload = payload
        return "ok"

    monkeypatch.setattr(
        pipe_without_api_key.__class__, "non_stream_response", fake_non_stream
    )

    response = await execute_pipe_func(pipe_without_api_key.pipe, {"body": body})

    assert response == "ok"
    assert captured_payload is not None
    assert captured_payload.get("output_config") == {
        "effort": pipe_without_api_key.valves.DEFAULT_EFFORT
    }


@pytest.mark.asyncio
async def test_effort_body_overrides_valve_default(
    pipe_without_api_key, create_text_body, execute_pipe_func, monkeypatch
):
    """Per-request effort in body overrides the Valve DEFAULT_EFFORT."""
    pipe_without_api_key.valves.DEFAULT_EFFORT = "high"

    body = create_text_body(
        "Test body overrides valve",
        model="anthropic.claude-sonnet-4-6",
        max_tokens=200,
    )
    body["effort"] = "low"

    captured_payload: dict | None = None

    async def fake_non_stream(self, url, headers, payload):
        nonlocal captured_payload
        captured_payload = payload
        return "ok"

    monkeypatch.setattr(
        pipe_without_api_key.__class__, "non_stream_response", fake_non_stream
    )

    response = await execute_pipe_func(pipe_without_api_key.pipe, {"body": body})

    assert response == "ok"
    assert captured_payload is not None
    assert captured_payload.get("output_config") == {"effort": "low"}


@pytest.mark.asyncio
async def test_effort_skipped_for_unsupported_model(
    pipe_without_api_key, create_text_body, execute_pipe_func, monkeypatch
):
    """output_config is not sent for models that don't support effort."""
    body = create_text_body(
        "Test unsupported model skips effort",
        model="anthropic.claude-haiku-4-5",
        max_tokens=200,
    )
    # Even if effort is set explicitly, Haiku doesn't support it
    body["effort"] = "medium"

    captured_payload: dict | None = None

    async def fake_non_stream(self, url, headers, payload):
        nonlocal captured_payload
        captured_payload = payload
        return "ok"

    monkeypatch.setattr(
        pipe_without_api_key.__class__, "non_stream_response", fake_non_stream
    )

    response = await execute_pipe_func(pipe_without_api_key.pipe, {"body": body})

    assert response == "ok"
    assert captured_payload is not None
    assert "output_config" not in captured_payload


@pytest.mark.asyncio
async def test_adaptive_thinking_valve_adds_thinking_param(
    pipe_without_api_key, create_text_body, execute_pipe_func, monkeypatch
):
    """When ENABLE_ADAPTIVE_THINKING valve is True, thinking: adaptive is sent for supported models."""
    pipe_without_api_key.valves.ENABLE_ADAPTIVE_THINKING = True

    body = create_text_body(
        "Test adaptive thinking valve",
        model="anthropic.claude-sonnet-4-6",
        max_tokens=200,
    )

    captured_payload: dict | None = None

    async def fake_non_stream(self, url, headers, payload):
        nonlocal captured_payload
        captured_payload = payload
        return "ok"

    monkeypatch.setattr(
        pipe_without_api_key.__class__, "non_stream_response", fake_non_stream
    )

    response = await execute_pipe_func(pipe_without_api_key.pipe, {"body": body})

    assert response == "ok"
    assert captured_payload is not None
    assert captured_payload.get("thinking") == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_adaptive_thinking_body_true_overrides_valve_false(
    pipe_without_api_key, create_text_body, execute_pipe_func, monkeypatch
):
    """adaptive_thinking=true in body enables it even when valve is False."""
    pipe_without_api_key.valves.ENABLE_ADAPTIVE_THINKING = False

    body = create_text_body(
        "Test body enables adaptive thinking",
        model="anthropic.claude-sonnet-4-6",
        max_tokens=200,
    )
    body["adaptive_thinking"] = True

    captured_payload: dict | None = None

    async def fake_non_stream(self, url, headers, payload):
        nonlocal captured_payload
        captured_payload = payload
        return "ok"

    monkeypatch.setattr(
        pipe_without_api_key.__class__, "non_stream_response", fake_non_stream
    )

    response = await execute_pipe_func(pipe_without_api_key.pipe, {"body": body})

    assert response == "ok"
    assert captured_payload is not None
    assert captured_payload.get("thinking") == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_adaptive_thinking_body_false_overrides_valve_true(
    pipe_without_api_key, create_text_body, execute_pipe_func, monkeypatch
):
    """adaptive_thinking=false in body disables it even when valve is True."""
    pipe_without_api_key.valves.ENABLE_ADAPTIVE_THINKING = True

    body = create_text_body(
        "Test body disables adaptive thinking",
        model="anthropic.claude-sonnet-4-6",
        max_tokens=200,
    )
    body["adaptive_thinking"] = False

    captured_payload: dict | None = None

    async def fake_non_stream(self, url, headers, payload):
        nonlocal captured_payload
        captured_payload = payload
        return "ok"

    monkeypatch.setattr(
        pipe_without_api_key.__class__, "non_stream_response", fake_non_stream
    )

    response = await execute_pipe_func(pipe_without_api_key.pipe, {"body": body})

    assert response == "ok"
    assert captured_payload is not None
    assert "thinking" not in captured_payload


@pytest.mark.asyncio
async def test_adaptive_thinking_skipped_for_unsupported_model(
    pipe_without_api_key, create_text_body, execute_pipe_func, monkeypatch
):
    """thinking param is not sent for models that don't support adaptive thinking."""
    pipe_without_api_key.valves.ENABLE_ADAPTIVE_THINKING = True

    body = create_text_body(
        "Test unsupported model skips adaptive thinking",
        model="anthropic.claude-haiku-4-5",
        max_tokens=200,
    )

    captured_payload: dict | None = None

    async def fake_non_stream(self, url, headers, payload):
        nonlocal captured_payload
        captured_payload = payload
        return "ok"

    monkeypatch.setattr(
        pipe_without_api_key.__class__, "non_stream_response", fake_non_stream
    )

    response = await execute_pipe_func(pipe_without_api_key.pipe, {"body": body})

    assert response == "ok"
    assert captured_payload is not None
    assert "thinking" not in captured_payload


@pytest.mark.asyncio
async def test_streaming_surfaces_thinking(pipe_without_api_key, monkeypatch):
    """Streaming responses expose thinking events alongside text."""

    sse_lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"id":"thinking-1","type":"thinking"}}\n\n',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","text":"First thought."},"content_block_id":"thinking-1"}\n\n',
        'data: {"type":"content_block_stop","index":0,"content_block_id":"thinking-1"}\n\n',
        'data: {"type":"content_block_start","index":1,"content_block":{"id":"text-1","type":"text"}}\n\n',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Final answer."},"content_block_id":"text-1"}\n\n',
        'data: {"type":"message_stop"}\n\n',
    ]

    class FakeContent:
        def __init__(self, payloads: list[str]):
            self._payloads = [payload.encode("utf-8") for payload in payloads]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._payloads:
                return self._payloads.pop(0)
            raise StopAsyncIteration

    class FakeResponse:
        status = 200

        def __init__(self, lines: list[str]):
            self._lines = lines
            self._content: FakeContent | None = None

        async def __aenter__(self):
            self._content = FakeContent(self._lines)
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        @property
        def content(self):
            return self._content

    class FakeSession:
        def __init__(self, lines: list[str]):
            self._lines = lines
            self.closed = False

        def post(self, url, headers=None, json=None):
            return FakeResponse(self._lines)

        async def close(self):
            self.closed = True

    captured_session: FakeSession | None = None

    async def fake_get_session(self):
        nonlocal captured_session
        captured_session = FakeSession(sse_lines)
        return captured_session

    monkeypatch.setattr(Pipe, "_get_session", fake_get_session)

    stream = pipe_without_api_key.stream_response("https://example.test", {}, {})

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == 3

    reasoning_chunk = chunks[0]
    text_chunk = chunks[1]
    finish_chunk = chunks[2]

    assert (
        reasoning_chunk.get("choices", [{}])[0]
        .get("delta", {})
        .get("reasoning_content")
        == "First thought."
    )
    assert (
        text_chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        == "Final answer."
    )
    assert finish_chunk.get("choices", [{}])[0].get("finish_reason") == "stop"
    assert captured_session is not None and captured_session.closed


@pytest.mark.integration
@pytest.mark.slow
async def test_concurrent_requests(pipe_instance, create_text_body, execute_pipe_func):
    """Test concurrent request handling."""
    # Create multiple concurrent requests
    tasks = []
    for i in range(3):
        body = create_text_body(f"Say {i + 1}", max_tokens=10)
        params = {"body": body}
        task = execute_pipe_func(pipe_instance.pipe, params)
        tasks.append(task)

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    successful_responses = 0
    for response in responses:
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if isinstance(content, str) and content:
                    successful_responses += 1
        elif isinstance(response, str) and response and "error" not in response.lower():
            successful_responses += 1

    # At least 2 out of 3 should succeed
    assert successful_responses >= 2, (
        f"Only {successful_responses}/3 concurrent requests succeeded"
    )
