"""
Integration tests for the async Anthropic pipe using pytest.
"""

import asyncio
from collections.abc import AsyncGenerator, Iterator

import pytest


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
    assert (
        successful_responses >= 2
    ), f"Only {successful_responses}/3 concurrent requests succeeded"
