"""
Integration tests for the async Anthropic pipe using pytest.
"""

import asyncio
from collections.abc import AsyncGenerator, Iterator

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

    assert isinstance(response, str)
    assert len(response) > 0
    assert "hello" in response.lower() or "hi" in response.lower()


@pytest.mark.integration
@pytest.mark.streaming
async def test_basic_streaming(pipe_instance, create_text_body, execute_pipe_func):
    """Test basic streaming text completion."""
    body = create_text_body("Count from 1 to 3.", stream=True)
    params = {"body": body}

    response = await execute_pipe_func(pipe_instance.pipe, params)

    # Collect streaming chunks
    chunks = []
    if isinstance(response, AsyncGenerator):
        async for chunk in response:
            chunks.append(str(chunk))
    elif isinstance(response, Iterator):
        for chunk in response:
            chunks.append(str(chunk))
    else:
        chunks = [str(response)]

    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0


@pytest.mark.integration
async def test_system_message(pipe_instance, create_system_body, execute_pipe_func):
    """Test system message handling."""
    body = create_system_body(
        "You are a helpful assistant. Always respond with exactly one word.",
        "What is 2+2?",
    )
    params = {"body": body}

    response = await execute_pipe_func(pipe_instance.pipe, params)

    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.integration
@pytest.mark.images
async def test_image_base64(
    pipe_instance, create_image_body, small_test_image, execute_pipe_func
):
    """Test image processing with base64 data."""
    body = create_image_body("What color is this image?", small_test_image)
    params = {"body": body}

    response = await execute_pipe_func(pipe_instance.pipe, params)

    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.integration
@pytest.mark.images
async def test_image_url(
    pipe_instance, create_image_body, test_image_url, execute_pipe_func
):
    """Test image processing with URL."""
    body = create_image_body("Describe this image briefly.", test_image_url)
    params = {"body": body}

    response = await execute_pipe_func(pipe_instance.pipe, params)

    assert isinstance(response, str)
    assert len(response) > 0


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
async def test_thinking_budget_requires_supported_model(
    pipe_without_api_key, create_text_body, execute_pipe_func
):
    """Thinking budget should be rejected for unsupported models."""
    body = create_text_body("Check thinking budget", max_tokens=100)
    body.pop("temperature", None)
    body["thinking_budget"] = 50

    params = {"body": body}

    with pytest.raises(ValueError) as excinfo:
        await execute_pipe_func(pipe_without_api_key.pipe, params)

    assert "not supported" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_thinking_budget_blocks_temperature(
    pipe_without_api_key, create_text_body, execute_pipe_func
):
    """Thinking budget must not allow temperature adjustments."""
    body = create_text_body(
        "Check temperature with thinking",
        model="anthropic.claude-sonnet-4-5",
        max_tokens=100,
    )
    body["thinking_budget"] = 50

    params = {"body": body}

    with pytest.raises(ValueError) as excinfo:
        await execute_pipe_func(pipe_without_api_key.pipe, params)

    assert "temperature" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_thinking_budget_validates_top_p(
    pipe_without_api_key, create_text_body, execute_pipe_func
):
    """Thinking budget should enforce top_p range."""
    body = create_text_body(
        "Check top_p with thinking",
        model="anthropic.claude-sonnet-4-5",
        max_tokens=200,
    )
    body.pop("temperature", None)
    body["thinking_budget"] = 150
    body["top_p"] = 0.9

    params = {"body": body}

    with pytest.raises(ValueError) as excinfo:
        await execute_pipe_func(pipe_without_api_key.pipe, params)

    assert "top_p" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_thinking_budget_inserts_payload(
    pipe_without_api_key, create_text_body, execute_pipe_func, monkeypatch
):
    """Thinking budget should be forwarded in the payload when valid."""
    body = create_text_body(
        "Enable thinking",
        model="anthropic.claude-sonnet-4-5",
        max_tokens=200,
    )
    body.pop("temperature", None)
    body["thinking_budget"] = 100

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
    assert captured_payload.get("thinking") == {
        "type": "enabled",
        "budget_tokens": 100,
    }
    assert "temperature" not in captured_payload
    assert captured_payload["max_tokens"] == 200


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
        if (
            isinstance(response, str)
            and len(response) > 0
            and "error" not in response.lower()
        ):
            successful_responses += 1

    # At least 2 out of 3 should succeed
    assert successful_responses >= 2, (
        f"Only {successful_responses}/3 concurrent requests succeeded"
    )
