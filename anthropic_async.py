"""
title: Anthropic Manifold Pipe (Async)
authors: justinh-rahb, christian-taillon, Ryan Kaskel
author_url: https://github.com/ryankask
funding_url: https://github.com/open-webui
version: 2.1.0
required_open_webui_version: 0.3.17
license: MIT
"""

import json
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any
from urllib.parse import urlparse

import aiohttp
from pydantic import BaseModel, Field

from open_webui.utils.misc import pop_system_message

logger = logging.getLogger(__name__)


class Pipe:
    API_URL: str = "https://api.anthropic.com/v1/messages"
    API_VERSION: str = "2023-06-01"
    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024  # 5MB per image

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(default="")
        REQUEST_TIMEOUT_TOTAL: float = Field(default=60.0)
        REQUEST_TIMEOUT_CONNECT: float = Field(default=3.05)
        REUSE_SESSION: bool = Field(default=False)

    def __init__(self) -> None:
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"
        self.valves = self.Valves(
            **{"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "")}
        )
        self._session: aiohttp.ClientSession | None = None
        logger.info(
            "Pipe instantiated: type=%s id=%s reuse_session=%s timeouts(total=%.2fs, connect=%.2fs)",
            self.type,
            self.id,
            self.valves.REUSE_SESSION,
            self.valves.REQUEST_TIMEOUT_TOTAL,
            self.valves.REQUEST_TIMEOUT_CONNECT,
        )

    def _get_timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=self.valves.REQUEST_TIMEOUT_TOTAL,
            connect=self.valves.REQUEST_TIMEOUT_CONNECT,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.valves.REUSE_SESSION:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(timeout=self._get_timeout())
                logger.info(
                    "Created shared aiohttp ClientSession (reuse_session=True, timeouts(total=%.2fs, connect=%.2fs))",
                    self.valves.REQUEST_TIMEOUT_TOTAL,
                    self.valves.REQUEST_TIMEOUT_CONNECT,
                )
            return self._session
        # Ephemeral session per request
        session = aiohttp.ClientSession(timeout=self._get_timeout())
        logger.info(
            "Created ephemeral aiohttp ClientSession (reuse_session=False, timeouts(total=%.2fs, connect=%.2fs))",
            self.valves.REQUEST_TIMEOUT_TOTAL,
            self.valves.REQUEST_TIMEOUT_CONNECT,
        )
        return session

    @staticmethod
    def _base64_size_bytes(b64: str) -> int:
        b64 = b64.strip()
        padding = 2 if b64.endswith("==") else (1 if b64.endswith("=") else 0)
        return (len(b64) * 3) // 4 - padding

    def get_anthropic_models(self) -> list[dict]:
        return [
            {"id": "claude-3-5-haiku-latest", "name": "claude-3-5-haiku-latest"},
            {"id": "claude-sonnet-4-20250514", "name": "claude-sonnet-4"},
            {"id": "claude-opus-4-1-20250805", "name": "claude-opus-4-1"},
        ]

    def pipes(self) -> list[dict]:
        return self.get_anthropic_models()

    async def process_image(self, image_data: dict[str, Any]) -> dict[str, Any]:
        """Process image data with size validation."""
        url_value = image_data["image_url"]["url"]
        if url_value.startswith("data:image"):
            mime_type, base64_data = url_value.split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            # Check base64 image size with padding consideration
            image_size = self._base64_size_bytes(base64_data)
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image size exceeds 5MB limit: {image_size / (1024 * 1024):.2f}MB"
                )

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            # For URL images, do a lightweight validation and optional HEAD size check
            parsed = urlparse(url_value)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError("Image URL must be http or https")

            session = await self._get_session()
            try:
                # HEAD may not return content-length; treat missing as unknown and allow
                async with session.head(url_value, allow_redirects=True) as response:
                    cl = response.headers.get("content-length")
                    if cl is not None:
                        try:
                            content_length = int(cl)
                            if content_length > self.MAX_IMAGE_SIZE:
                                raise ValueError(
                                    f"Image at URL exceeds 5MB limit: {content_length / (1024 * 1024):.2f}MB"
                                )
                        except ValueError:
                            # Non-integer content-length; ignore
                            pass
            finally:
                if not self.valves.REUSE_SESSION and not session.closed:
                    await session.close()

            return {
                "type": "image",
                "source": {"type": "url", "url": url_value},
            }

    async def pipe(self, body: dict) -> str | AsyncGenerator | AsyncIterator:
        system_message, messages = pop_system_message(body["messages"])

        processed_messages: list[dict] = []
        total_image_size = 0

        for message in messages:
            processed_content: list[dict] = []
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item.get("type") == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image_url":
                        processed_image = await self.process_image(item)
                        processed_content.append(processed_image)

                        # Track total size for base64 images only (URL sizes unknown here)
                        if processed_image["source"]["type"] == "base64":
                            image_size = self._base64_size_bytes(
                                processed_image["source"]["data"]
                            )
                            total_image_size += image_size
                            if (
                                total_image_size > 100 * 1024 * 1024
                            ):  # 100MB total limit
                                raise ValueError(
                                    "Total size of base64 images exceeds 100 MB limit"
                                )
            else:
                processed_content = [
                    {"type": "text", "text": message.get("content", "")}
                ]

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )

        # Robust model id extraction (strip namespace like "anthropic.")
        model_raw = body.get("model", "")
        model_id = model_raw.split(".", 1)[1] if "." in model_raw else model_raw

        # Extract optional parameters only if explicitly provided
        temperature = body.get("temperature")
        top_p = body.get("top_p")
        top_k = body.get("top_k")

        # Prepare system prompt according to API shape
        system_payload: str | None = None
        if system_message:
            sys_content = system_message.get("content")
            if isinstance(sys_content, str):
                system_payload = sys_content
            elif isinstance(sys_content, list):
                # Extract any text pieces, join with newlines
                texts: list[str] = []
                for itm in sys_content:
                    if isinstance(itm, dict) and itm.get("type") == "text":
                        texts.append(itm.get("text", ""))
                    elif isinstance(itm, str):
                        texts.append(itm)
                if texts:
                    system_payload = "\n".join(filter(None, texts))

        stream = body.get("stream", False)
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": processed_messages,
            "max_tokens": body.get("max_tokens", 4096),
            "stop_sequences": body.get("stop", []),
            "stream": stream,
        }

        # Only include optional parameters if they were explicitly set
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if system_payload:
            payload["system"] = system_payload

        headers = {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        try:
            if stream:
                return self.stream_response(self.API_URL, headers, payload)
            else:
                return await self.non_stream_response(self.API_URL, headers, payload)
        except aiohttp.ClientError as e:
            logger.error("Request failed: %s", e)
            return f"Error: Request failed: {e}"
        except Exception as e:  # noqa: BLE001
            logger.exception("Error in pipe method: %s", e)
            return f"Error: {e}"

    async def stream_response(
        self, url: str, headers: dict, payload: dict
    ) -> AsyncGenerator[str]:
        try:
            session = await self._get_session()
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        raise Exception(
                            f"HTTP Error {response.status}: {response_text}"
                        )

                    async for line in response.content:
                        if not line:
                            continue
                        line = line.decode("utf-8").strip()
                        if not line or not line.startswith("data: "):
                            continue
                        try:
                            data = json.loads(line[6:])
                            dtype = data.get("type")
                            if dtype == "content_block_start":
                                cb = data.get("content_block", {})
                                if cb.get("type") == "text" and cb.get("text"):
                                    yield cb["text"]
                            elif dtype == "content_block_delta":
                                delta = data.get("delta", {})
                                if (
                                    delta.get("type") == "text_delta"
                                    and "text" in delta
                                ):
                                    yield delta["text"]
                            elif dtype == "message_stop":
                                break
                            elif dtype == "message":
                                for content in data.get("content", []):
                                    if content.get("type") == "text" and content.get(
                                        "text"
                                    ):
                                        yield content["text"]
                            # Ignore other event types (ping, message_delta, thinking, tool deltas, etc.)
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse JSON line: %s", line)
                        except KeyError as e:
                            logger.warning(
                                "Unexpected data structure: %s; data=%s", e, data
                            )
            finally:
                if not self.valves.REUSE_SESSION and not session.closed:
                    await session.close()
        except aiohttp.ClientError as e:
            logger.error("Request failed: %s", e)
            yield f"Error: Request failed: {e}"
        except Exception as e:  # noqa: BLE001
            logger.exception("General error in stream_response: %s", e)
            yield f"Error: {e}"

    async def non_stream_response(self, url: str, headers: dict, payload: dict) -> str:
        try:
            session = await self._get_session()
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        raise Exception(
                            f"HTTP Error {response.status}: {response_text}"
                        )

                    res = await response.json()
                    return (
                        res["content"][0]["text"]
                        if "content" in res and res["content"]
                        else ""
                    )
            finally:
                if not self.valves.REUSE_SESSION and not session.closed:
                    await session.close()
        except aiohttp.ClientError as e:
            logger.error("Failed non-stream request: %s", e)
            return f"Error: {e}"
