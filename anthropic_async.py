"""
title: Anthropic Manifold Pipe (Async)
authors: justinh-rahb, christian-taillon, Ryan Kaskel
author_url: https://github.com/ryankask
funding_url: https://github.com/open-webui
version: 2.6.0
required_open_webui_version: 0.6.33
license: MIT
"""

import json
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp
from pydantic import BaseModel, Field

from open_webui.utils.misc import pop_system_message

logger = logging.getLogger(__name__)


class Pipe:
    API_URL: str = "https://api.anthropic.com/v1/messages"
    API_VERSION: str = "2023-06-01"
    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024  # 5MB per image
    THINKING_TOP_P_MIN: float = 0.95

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

    @staticmethod
    def _extract_text_from_content_block(content_block: Any) -> str:
        if not isinstance(content_block, dict):
            return ""
        text = content_block.get("text")
        if isinstance(text, str) and text:
            return text
        thinking = content_block.get("thinking")
        if isinstance(thinking, str) and thinking:
            return thinking
        items = content_block.get("content")
        if isinstance(items, list):
            parts: list[str] = []
            for item in items:
                if isinstance(item, dict):
                    value = (
                        item.get("text") or item.get("thinking") or item.get("value")
                    )
                    if isinstance(value, str):
                        parts.append(value)
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "".join(parts)
        return ""

    @staticmethod
    def _build_stream_chunk(
        model: str,
        *,
        content: str | None = None,
        reasoning: str | None = None,
        finish_reason: str | None = None,
    ) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        if content is not None:
            delta["content"] = content
        if reasoning is not None:
            delta["reasoning_content"] = reasoning

        chunk: dict[str, Any] = {
            "id": f"anthropic-{uuid4()}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                }
            ],
        }

        if finish_reason is not None:
            chunk["choices"][0]["finish_reason"] = finish_reason

        return chunk

    @staticmethod
    def _build_completion_response(
        model: str, *, content: str, reasoning: str | None = None
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning

        return {
            "id": f"anthropic-{uuid4()}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                }
            ],
        }

    def get_anthropic_models(self) -> list[dict]:
        return [
            {"id": "claude-haiku-4-5", "name": "claude-haiku-4-5"},
            {"id": "claude-sonnet-4-5", "name": "claude-sonnet-4-5"},
            {"id": "claude-opus-4-5", "name": "claude-opus-4-5"},
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

    async def pipe(
        self, body: dict
    ) -> str | dict[str, Any] | AsyncGenerator | AsyncIterator:
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
        top_p_raw = body.get("top_p")
        top_k = body.get("top_k")
        thinking_budget_raw = body.get("thinking_budget")

        top_p: float | None = None
        if top_p_raw is not None:
            try:
                top_p = float(top_p_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("top_p must be a number") from exc

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

        max_tokens_raw = body.get("max_tokens", 4096)
        try:
            max_tokens = int(max_tokens_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_tokens must be an integer") from exc
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        stream = body.get("stream", False)
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": processed_messages,
            "max_tokens": max_tokens,
            "stop_sequences": body.get("stop", []),
            "stream": stream,
        }

        if thinking_budget_raw is not None:
            try:
                thinking_budget = int(thinking_budget_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("thinking_budget must be an integer") from exc
            if thinking_budget <= 0:
                raise ValueError("thinking_budget must be greater than 0")
            if temperature is not None:
                raise ValueError(
                    "temperature cannot be set when thinking mode is enabled"
                )
            if top_k is not None:
                raise ValueError("top_k cannot be set when thinking mode is enabled")
            if top_p is not None and not (self.THINKING_TOP_P_MIN <= top_p <= 1):
                raise ValueError(
                    "When thinking mode is enabled, top_p must be between 0.95 and 1.0"
                )
            if thinking_budget >= max_tokens:
                raise ValueError("thinking_budget must be less than max_tokens")

            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
        else:
            if temperature is not None:
                payload["temperature"] = temperature
            if top_k is not None:
                payload["top_k"] = top_k

        if top_p is not None:
            payload["top_p"] = top_p
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
    ) -> AsyncGenerator[dict[str, Any]]:
        model = str(payload.get("model", ""))
        finished = False
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
                        line_text = line.decode("utf-8").strip()
                        if not line_text or not line_text.startswith("data: "):
                            continue

                        try:
                            data = json.loads(line_text[6:])
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse JSON line: %s", line_text)
                            continue

                        if not isinstance(data, dict):
                            continue

                        dtype = data.get("type")
                        if dtype == "content_block_start":
                            content_block = data.get("content_block") or {}
                            block_type = content_block.get("type")
                            extracted = self._extract_text_from_content_block(
                                content_block
                            )
                            if block_type == "thinking" and extracted:
                                yield self._build_stream_chunk(
                                    model, reasoning=extracted
                                )
                            elif block_type == "text" and extracted:
                                yield self._build_stream_chunk(model, content=extracted)
                        elif dtype == "content_block_delta":
                            delta = data.get("delta") or {}
                            delta_type = delta.get("type")
                            if delta_type == "thinking_delta":
                                text_piece = delta.get("thinking") or delta.get("text")
                                if isinstance(text_piece, str) and text_piece:
                                    yield self._build_stream_chunk(
                                        model, reasoning=text_piece
                                    )
                            elif delta_type == "text_delta":
                                text_piece = delta.get("text")
                                if isinstance(text_piece, str) and text_piece:
                                    yield self._build_stream_chunk(
                                        model, content=text_piece
                                    )
                        elif dtype == "message":
                            for block in data.get("content", []):
                                if not isinstance(block, dict):
                                    continue
                                block_type = block.get("type")
                                extracted = self._extract_text_from_content_block(block)
                                if not extracted:
                                    continue
                                if block_type == "thinking":
                                    yield self._build_stream_chunk(
                                        model, reasoning=extracted
                                    )
                                elif block_type == "text":
                                    yield self._build_stream_chunk(
                                        model, content=extracted
                                    )
                        elif dtype == "message_stop":
                            yield self._build_stream_chunk(model, finish_reason="stop")
                            finished = True
                            break
                        # Ignore other event types (ping, message_delta, tool deltas, etc.)
            finally:
                if not self.valves.REUSE_SESSION and not session.closed:
                    await session.close()

            if not finished:
                yield self._build_stream_chunk(model, finish_reason="stop")
        except aiohttp.ClientError as e:
            logger.error("Request failed: %s", e)
            yield self._build_stream_chunk(
                model,
                content=f"Error: Request failed: {e}",
                finish_reason="stop",
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("General error in stream_response: %s", e)
            yield self._build_stream_chunk(
                model, content=f"Error: {e}", finish_reason="stop"
            )

    async def non_stream_response(
        self, url: str, headers: dict, payload: dict
    ) -> dict[str, Any]:
        model = str(payload.get("model", ""))
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
                    content_blocks = (
                        res.get("content") if isinstance(res, dict) else None
                    )
                    reasoning_fragments: list[str] = []
                    text_fragments: list[str] = []

                    if isinstance(content_blocks, list):
                        for block in content_blocks:
                            if not isinstance(block, dict):
                                continue
                            block_type = block.get("type")
                            extracted = self._extract_text_from_content_block(block)
                            if not extracted:
                                continue
                            if block_type == "thinking":
                                reasoning_fragments.append(extracted)
                            elif block_type == "text":
                                text_fragments.append(extracted)

                    if not text_fragments and isinstance(res, dict):
                        # Fallback to the first text field if blocks omitted it
                        first_block = res.get("content", [{}])[0]
                        if isinstance(first_block, dict):
                            text_value = first_block.get("text")
                            if isinstance(text_value, str) and text_value:
                                text_fragments.append(text_value)

                    content_text = "".join(text_fragments)
                    reasoning_text = "".join(reasoning_fragments)

                    return self._build_completion_response(
                        model,
                        content=content_text,
                        reasoning=reasoning_text if reasoning_text else None,
                    )
            finally:
                if not self.valves.REUSE_SESSION and not session.closed:
                    await session.close()
        except aiohttp.ClientError as e:
            logger.error("Failed non-stream request: %s", e)
            return {
                "error": {
                    "message": f"Error: {e}",
                }
            }
