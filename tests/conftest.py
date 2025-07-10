"""
Pytest configuration and fixtures for Anthropic async pipe tests.
"""

import inspect
import os

# Import the pipe implementation
import sys
from collections.abc import AsyncGenerator, Generator

import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anthropic_async import Pipe


class MockUser(BaseModel):
    id: str = "test_user"
    name: str = "Test User"


@pytest.fixture(scope="session")
def api_key():
    """Get API key from environment, skip tests if not available."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    return key


@pytest.fixture
def pipe_instance(api_key):
    """Create a configured Pipe instance."""
    pipe = Pipe()
    pipe.valves.ANTHROPIC_API_KEY = api_key
    return pipe


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    return MockUser()


async def execute_pipe(pipe, params):
    """Execute pipe function, handling both sync and async versions."""
    if inspect.iscoroutinefunction(pipe):
        return await pipe(**params)
    else:
        return pipe(**params)


async def get_message_content(res: str | Generator | AsyncGenerator) -> str:
    """Extract message content from various response types."""
    if isinstance(res, str):
        return res
    if isinstance(res, Generator):
        return "".join(map(str, res))
    if isinstance(res, AsyncGenerator):
        return "".join([str(stream) async for stream in res])


@pytest.fixture
def execute_pipe_func():
    """Provide the execute_pipe function as a fixture."""
    return execute_pipe


@pytest.fixture
def get_content_func():
    """Provide the get_message_content function as a fixture."""
    return get_message_content


# Test data fixtures
@pytest.fixture
def small_test_image():
    """Small test image (1x1 pixel red PNG) as base64."""
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


@pytest.fixture
def test_image_url():
    """Small test image URL."""
    return "https://placehold.net/4.png"


@pytest.fixture
def large_image_data():
    """Large image data for testing size limits."""
    large_data = "a" * (6 * 1024 * 1024)  # 6MB of data
    return f"data:image/png;base64,{large_data}"


# Body creation helpers
@pytest.fixture
def create_text_body():
    """Factory for creating text message bodies."""

    def _create_body(
        message: str,
        model: str = "anthropic.claude-3-5-haiku-latest",
        stream: bool = False,
        max_tokens: int = 25,
    ):
        return {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": stream,
        }

    return _create_body


@pytest.fixture
def create_image_body():
    """Factory for creating image message bodies."""

    def _create_body(
        text: str,
        image_data: str,
        model: str = "anthropic.claude-3-5-haiku-latest",
        max_tokens: int = 20,
    ):
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False,
        }

    return _create_body


@pytest.fixture
def create_system_body():
    """Factory for creating system message bodies."""

    def _create_body(
        system: str,
        user_message: str,
        model: str = "anthropic.claude-3-5-haiku-latest",
        max_tokens: int = 20,
    ):
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False,
        }

    return _create_body
