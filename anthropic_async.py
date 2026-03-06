"""
title: Anthropic Manifold Pipe (Async)
authors: justinh-rahb, christian-taillon, Ryan Kaskel
author_url: https://github.com/ryankask
funding_url: https://github.com/open-webui
version: 4.0.0
required_open_webui_version: 0.6.33
license: MIT
"""

import asyncio
import codecs
import importlib
import json
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from typing import Any, cast
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp
from pydantic import BaseModel, Field

try:
    StarletteStreamingResponse = importlib.import_module(
        "starlette.responses"
    ).StreamingResponse
except ImportError:  # pragma: no cover - Open WebUI provides Starlette at runtime.
    StarletteStreamingResponse = None

from open_webui.utils.misc import pop_system_message

logger = logging.getLogger(__name__)


class Pipe:
    API_URL: str = "https://api.anthropic.com/v1/messages"
    API_VERSION: str = "2023-06-01"
    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024  # 5MB per image
    STREAM_BUFFER_CHARS: int = 8
    STREAM_BUFFER_MAX_DELAY_SECONDS: float = 0.01
    STREAM_FLUSH_PUNCTUATION: frozenset[str] = frozenset({"\n", ".", "!", "?"})
    VALID_EFFORT_LEVELS: frozenset[str] = frozenset({"low", "medium", "high", "max"})
    EFFORT_SUPPORTED_MODELS: frozenset[str] = frozenset(
        {"claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5"}
    )

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(default="")
        REQUEST_TIMEOUT_TOTAL: float = Field(default=60.0)
        REQUEST_TIMEOUT_CONNECT: float = Field(default=3.05)
        REUSE_SESSION: bool = Field(default=False)
        DEFAULT_EFFORT: str = Field(
            default="medium",
            description="Default effort level for supported models (low, medium, high, max).",
        )
        ENABLE_ADAPTIVE_THINKING: bool = Field(
            default=False,
            description="Send thinking: {type: adaptive} for supported models (Opus 4.6, Sonnet 4.6). Can be overridden per-request via adaptive_thinking in the body.",
        )

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
    def _format_error_text(error: Any) -> str:
        return f"Error: {error}"

    @classmethod
    def _build_error_response(cls, error: Any) -> dict[str, Any]:
        return {"error": {"message": cls._format_error_text(error)}}

    @staticmethod
    def _extract_system_payload(system_message: dict[str, Any] | None) -> str | None:
        if not system_message:
            return None

        sys_content = system_message.get("content")
        if isinstance(sys_content, str):
            return sys_content or None

        if not isinstance(sys_content, list):
            return None

        texts: list[str] = []
        for item in sys_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
            elif isinstance(item, str) and item:
                texts.append(item)

        if texts:
            return "\n".join(texts)

        return None

    @staticmethod
    def _normalize_stop_sequences(stop: Any) -> list[str]:
        if stop is None:
            return []

        if isinstance(stop, str):
            return [stop]

        if isinstance(stop, list):
            normalized: list[str] = []
            for item in stop:
                if not isinstance(item, str):
                    raise ValueError("stop must be a string or list of strings")
                normalized.append(item)
            return normalized

        raise ValueError("stop must be a string or list of strings")

    async def _process_message_content(
        self, content: Any
    ) -> tuple[list[dict[str, Any]], int]:
        processed_content: list[dict[str, Any]] = []
        total_image_size = 0

        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError("message content list items must be objects")

                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text")
                    if not isinstance(text, str):
                        raise ValueError(
                            "text content items must include a string text field"
                        )
                    processed_content.append({"type": "text", "text": text})
                elif item_type == "image_url":
                    processed_image = await self.process_image(item)
                    processed_content.append(processed_image)

                    if processed_image["source"]["type"] == "base64":
                        image_size = self._base64_size_bytes(
                            processed_image["source"]["data"]
                        )
                        total_image_size += image_size
                else:
                    raise ValueError(f"Unsupported content item type: {item_type}")

            return processed_content, total_image_size

        if content is None:
            return [{"type": "text", "text": ""}], 0

        if not isinstance(content, str):
            raise ValueError(
                "message content must be a string or list of content items"
            )

        return [{"type": "text", "text": content}], 0

    @staticmethod
    def _build_stream_chunk(
        chunk_id: str,
        model: str,
        *,
        content: str | None = None,
        reasoning: str | None = None,
        finish_reason: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        if role is not None:
            delta["role"] = role
        if content is not None:
            delta["content"] = content
        if reasoning is not None:
            delta["reasoning_content"] = reasoning

        chunk: dict[str, Any] = {
            "id": chunk_id,
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
    def _encode_sse_payload(payload: dict[str, Any] | str) -> bytes:
        if isinstance(payload, str):
            return f"data: {payload}\n\n".encode()
        return f"data: {json.dumps(payload)}\n\n".encode()

    @classmethod
    def _map_finish_reason(cls, stop_reason: str | None) -> str:
        if stop_reason == "max_tokens":
            return "length"
        if stop_reason == "tool_use":
            return "tool_calls"
        return "stop"

    @staticmethod
    async def _iter_content_bytes(content: Any) -> AsyncIterator[bytes]:
        iter_any = getattr(content, "iter_any", None)
        if callable(iter_any):
            async for chunk in cast(AsyncIterator[bytes], iter_any()):
                if chunk:
                    yield chunk
            return

        async for chunk in cast(AsyncIterator[bytes], content):
            if chunk:
                yield chunk

    @classmethod
    async def _iter_sse_events(
        cls, content: Any
    ) -> AsyncIterator[tuple[str | None, dict[str, Any]]]:
        decoder = codecs.getincrementaldecoder("utf-8")()
        buffer = ""

        async for chunk in cls._iter_content_bytes(content):
            buffer += decoder.decode(chunk)
            normalized = buffer.replace("\r\n", "\n").replace("\r", "\n")
            events = normalized.split("\n\n")
            buffer = events.pop()

            for raw_event in events:
                parsed = cls._parse_sse_event(raw_event)
                if parsed is not None:
                    yield parsed

        buffer += decoder.decode(b"", final=True)
        normalized = buffer.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return

        parsed = cls._parse_sse_event(normalized)
        if parsed is not None:
            yield parsed

    @staticmethod
    def _parse_sse_event(raw_event: str) -> tuple[str | None, dict[str, Any]] | None:
        event_name: str | None = None
        data_lines: list[str] = []

        for line in raw_event.split("\n"):
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip() or None
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if not data_lines:
            return None

        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            return event_name, {"type": "done"}

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Failed to parse SSE payload: %s", payload)
            return None

        if not isinstance(data, dict):
            return None

        return event_name, data

    @staticmethod
    def _build_completion_response(
        model: str,
        *,
        content: str,
        reasoning: str | None = None,
        finish_reason: str = "stop",
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
                    "finish_reason": finish_reason,
                }
            ],
        }

    def get_anthropic_models(self) -> list[dict]:
        return [
            {"id": "claude-haiku-4-5", "name": "claude-haiku-4-5"},
            {"id": "claude-sonnet-4-6", "name": "claude-sonnet-4-6"},
            {"id": "claude-opus-4-6", "name": "claude-opus-4-6"},
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
                # HEAD may fail or omit content-length; treat size as unknown and allow.
                try:
                    async with session.head(
                        url_value, allow_redirects=True
                    ) as response:
                        cl = response.headers.get("content-length")
                        if cl is not None:
                            try:
                                content_length = int(cl)
                            except ValueError:
                                content_length = None

                            if (
                                content_length is not None
                                and content_length > self.MAX_IMAGE_SIZE
                            ):
                                raise ValueError(
                                    f"Image at URL exceeds 5MB limit: {content_length / (1024 * 1024):.2f}MB"
                                )
                except (aiohttp.ClientError, TimeoutError) as exc:
                    logger.warning(
                        "Skipping remote image HEAD validation for %s: %s",
                        url_value,
                        exc,
                    )
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
            processed_content, image_size = await self._process_message_content(
                message.get("content")
            )
            total_image_size += image_size
            if total_image_size > 100 * 1024 * 1024:
                raise ValueError("Total size of base64 images exceeds 100 MB limit")

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
        effort = body.get("effort")
        adaptive_thinking_body = body.get("adaptive_thinking")  # None | bool

        top_p: float | None = None
        if top_p_raw is not None:
            try:
                top_p = float(top_p_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("top_p must be a number") from exc

        # Prepare system prompt according to API shape
        system_payload = self._extract_system_payload(system_message)

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
            "stop_sequences": self._normalize_stop_sequences(body.get("stop")),
            "stream": stream,
        }

        if model_id in self.EFFORT_SUPPORTED_MODELS:
            effective_effort = (
                effort if effort is not None else self.valves.DEFAULT_EFFORT
            )
            if effective_effort not in self.VALID_EFFORT_LEVELS:
                raise ValueError(
                    f"effort must be one of: {', '.join(sorted(self.VALID_EFFORT_LEVELS))}"
                )
            payload["output_config"] = {"effort": effective_effort}

            use_adaptive = (
                adaptive_thinking_body
                if adaptive_thinking_body is not None
                else self.valves.ENABLE_ADAPTIVE_THINKING
            )
            if use_adaptive:
                payload["thinking"] = {"type": "adaptive"}

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
            return self._format_error_text(f"Request failed: {e}")
        except Exception as e:  # noqa: BLE001
            logger.exception("Error in pipe method: %s", e)
            return self._format_error_text(e)

    def stream_response(
        self, url: str, headers: dict, payload: dict
    ) -> Any | AsyncGenerator[dict[str, Any]]:
        model = str(payload.get("model", ""))
        chunk_id = f"anthropic-{uuid4()}"

        async def chunk_stream() -> AsyncGenerator[dict[str, Any]]:
            pending_kind: str | None = None
            pending_text = ""
            finish_reason = "stop"
            finished = False

            async def flush_pending() -> AsyncGenerator[dict[str, Any]]:
                nonlocal pending_kind
                nonlocal pending_text
                if not pending_kind or not pending_text:
                    return

                if pending_kind == "reasoning":
                    yield self._build_stream_chunk(
                        chunk_id, model, reasoning=pending_text
                    )
                else:
                    yield self._build_stream_chunk(
                        chunk_id, model, content=pending_text
                    )

                pending_kind = None
                pending_text = ""

            def should_flush(text: str) -> bool:
                return len(text) >= self.STREAM_BUFFER_CHARS or text.endswith(
                    tuple(self.STREAM_FLUSH_PUNCTUATION)
                )

            def queue_piece(kind: str, text: str) -> AsyncGenerator[dict[str, Any]]:
                nonlocal pending_kind
                nonlocal pending_text

                async def _queue() -> AsyncGenerator[dict[str, Any]]:
                    nonlocal pending_kind
                    nonlocal pending_text
                    if pending_kind and pending_kind != kind:
                        async for chunk in flush_pending():
                            yield chunk

                    pending_kind = kind
                    pending_text += text

                    if should_flush(pending_text):
                        async for chunk in flush_pending():
                            yield chunk

                return _queue()

            try:
                session = await self._get_session()
                try:
                    async with session.post(
                        url, headers=headers, json=payload
                    ) as response:
                        if response.status != 200:
                            response_text = await response.text()
                            raise Exception(
                                f"HTTP Error {response.status}: {response_text}"
                            )

                        yield self._build_stream_chunk(
                            chunk_id, model, role="assistant"
                        )

                        event_stream = cast(
                            AsyncGenerator[tuple[str | None, dict[str, Any]]],
                            self._iter_sse_events(response.content),
                        )
                        next_event_task: (
                            asyncio.Task[tuple[str | None, dict[str, Any]]] | None
                        ) = None
                        try:
                            while True:
                                try:
                                    if next_event_task is None:
                                        next_event_task = asyncio.create_task(
                                            anext(event_stream)
                                        )

                                    if pending_text:
                                        done, _ = await asyncio.wait(
                                            {next_event_task},
                                            timeout=self.STREAM_BUFFER_MAX_DELAY_SECONDS,
                                        )
                                        if not done:
                                            async for chunk in flush_pending():
                                                yield chunk
                                            continue

                                        event_name, data = next_event_task.result()
                                        next_event_task = None
                                    else:
                                        event_name, data = await next_event_task
                                        next_event_task = None
                                except StopAsyncIteration:
                                    break

                                dtype = data.get("type") or event_name

                                if dtype == "content_block_start":
                                    content_block = data.get("content_block") or {}
                                    block_type = content_block.get("type")
                                    extracted = self._extract_text_from_content_block(
                                        content_block
                                    )
                                    if block_type == "thinking" and extracted:
                                        async for chunk in queue_piece(
                                            "reasoning", extracted
                                        ):
                                            yield chunk
                                    elif block_type == "text" and extracted:
                                        async for chunk in queue_piece(
                                            "content", extracted
                                        ):
                                            yield chunk
                                elif dtype == "content_block_delta":
                                    delta = data.get("delta") or {}
                                    delta_type = delta.get("type")
                                    if delta_type == "thinking_delta":
                                        text_piece = delta.get("thinking") or delta.get(
                                            "text"
                                        )
                                        if isinstance(text_piece, str) and text_piece:
                                            async for chunk in queue_piece(
                                                "reasoning", text_piece
                                            ):
                                                yield chunk
                                    elif delta_type == "text_delta":
                                        text_piece = delta.get("text")
                                        if isinstance(text_piece, str) and text_piece:
                                            async for chunk in queue_piece(
                                                "content", text_piece
                                            ):
                                                yield chunk
                                    elif delta_type == "input_json_delta":
                                        async for chunk in flush_pending():
                                            yield chunk
                                    elif delta_type == "signature_delta":
                                        continue
                                elif dtype == "content_block_stop":
                                    async for chunk in flush_pending():
                                        yield chunk
                                elif dtype == "message_delta":
                                    async for chunk in flush_pending():
                                        yield chunk
                                    finish_reason = self._map_finish_reason(
                                        (data.get("delta") or {}).get("stop_reason")
                                    )
                                elif dtype == "message":
                                    for block in data.get("content", []):
                                        if not isinstance(block, dict):
                                            continue
                                        block_type = block.get("type")
                                        extracted = (
                                            self._extract_text_from_content_block(block)
                                        )
                                        if not extracted:
                                            continue
                                        if block_type == "thinking":
                                            async for chunk in queue_piece(
                                                "reasoning", extracted
                                            ):
                                                yield chunk
                                        elif block_type == "text":
                                            async for chunk in queue_piece(
                                                "content", extracted
                                            ):
                                                yield chunk
                                elif dtype == "error":
                                    async for chunk in flush_pending():
                                        yield chunk
                                    error = data.get("error") or {}
                                    message = (
                                        error.get("message") or "Anthropic stream error"
                                    )
                                    yield self._build_stream_chunk(
                                        chunk_id,
                                        model,
                                        content=f"Error: {message}",
                                        finish_reason="stop",
                                    )
                                    finished = True
                                    break
                                elif dtype == "message_stop" or dtype == "done":
                                    async for chunk in flush_pending():
                                        yield chunk
                                    yield self._build_stream_chunk(
                                        chunk_id,
                                        model,
                                        finish_reason=finish_reason,
                                    )
                                    finished = True
                                    break
                                # Ignore ping and unknown event types.
                        finally:
                            if (
                                next_event_task is not None
                                and not next_event_task.done()
                            ):
                                next_event_task.cancel()
                                with suppress(asyncio.CancelledError):
                                    await next_event_task
                            await event_stream.aclose()
                finally:
                    if not self.valves.REUSE_SESSION and not session.closed:
                        await session.close()

                if not finished:
                    async for chunk in flush_pending():
                        yield chunk
                    yield self._build_stream_chunk(
                        chunk_id, model, finish_reason=finish_reason
                    )
            except aiohttp.ClientError as e:
                logger.error("Request failed: %s", e)
                yield self._build_stream_chunk(
                    chunk_id,
                    model,
                    content=self._format_error_text(f"Request failed: {e}"),
                    finish_reason="stop",
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("General error in stream_response: %s", e)
                yield self._build_stream_chunk(
                    chunk_id,
                    model,
                    content=self._format_error_text(e),
                    finish_reason="stop",
                )

        if StarletteStreamingResponse is None:
            return chunk_stream()

        async def sse_stream() -> AsyncGenerator[bytes]:
            async for chunk in chunk_stream():
                yield self._encode_sse_payload(chunk)
            yield self._encode_sse_payload("[DONE]")

        return StarletteStreamingResponse(sse_stream(), media_type="text/event-stream")

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
                    finish_reason = self._map_finish_reason(res.get("stop_reason"))

                    return self._build_completion_response(
                        model,
                        content=content_text,
                        reasoning=reasoning_text if reasoning_text else None,
                        finish_reason=finish_reason,
                    )
            finally:
                if not self.valves.REUSE_SESSION and not session.closed:
                    await session.close()
        except aiohttp.ClientError as e:
            logger.error("Failed non-stream request: %s", e)
            return self._build_error_response(e)
