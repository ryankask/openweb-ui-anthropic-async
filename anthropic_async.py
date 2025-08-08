"""
title: Anthropic Manifold Pipe (Async)
authors: justinh-rahb, christian-taillon, Ryan Kaskel
author_url: https://github.com/ryankask
funding_url: https://github.com/open-webui
version: 2.0.0
required_open_webui_version: 0.3.17
license: MIT
"""

import json
import os
from collections.abc import AsyncGenerator, AsyncIterator

import aiohttp
from pydantic import BaseModel, Field

from open_webui.utils.misc import pop_system_message


class Pipe:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"
        self.valves = self.Valves(
            **{"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "")}
        )
        self.MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image

    def get_anthropic_models(self):
        return [
            {"id": "claude-3-5-haiku-latest", "name": "claude-3-5-haiku-latest"},
            {"id": "claude-sonnet-4-20250514", "name": "claude-sonnet-4"},
            {"id": "claude-opus-4-1-20250805", "name": "claude-opus-4-1"},
        ]

    def pipes(self) -> list[dict]:
        return self.get_anthropic_models()

    async def process_image(self, image_data):
        """Process image data with size validation."""
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            # Check base64 image size
            image_size = len(base64_data) * 3 / 4  # Convert base64 size to bytes
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
            # For URL images, perform size check after fetching
            url = image_data["image_url"]["url"]
            async with aiohttp.ClientSession() as session:
                async with session.head(url, allow_redirects=True) as response:
                    content_length = int(response.headers.get("content-length", 0))

                    if content_length > self.MAX_IMAGE_SIZE:
                        raise ValueError(
                            f"Image at URL exceeds 5MB limit: {content_length / (1024 * 1024):.2f}MB"
                        )

            return {
                "type": "image",
                "source": {"type": "url", "url": url},
            }

    async def pipe(self, body: dict) -> str | AsyncGenerator | AsyncIterator:
        system_message, messages = pop_system_message(body["messages"])

        processed_messages = []
        total_image_size = 0

        for message in messages:
            processed_content = []
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        processed_image = await self.process_image(item)
                        processed_content.append(processed_image)

                        # Track total size for base64 images
                        if processed_image["source"]["type"] == "base64":
                            image_size = len(processed_image["source"]["data"]) * 3 / 4
                            total_image_size += image_size
                            if (
                                total_image_size > 100 * 1024 * 1024
                            ):  # 100MB total limit
                                raise ValueError(
                                    "Total size of images exceeds 100 MB limit"
                                )
            else:
                processed_content = [
                    {"type": "text", "text": message.get("content", "")}
                ]

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )

        model_id = body["model"][body["model"].find(".") + 1 :]

        # Extract optional parameters only if explicitly provided
        temperature = body.get("temperature")
        top_p = body.get("top_p")
        top_k = body.get("top_k")

        payload = {
            "model": model_id,
            "messages": processed_messages,
            "max_tokens": body.get("max_tokens", 4096),
            "stop_sequences": body.get("stop", []),
            **({"system": str(system_message)} if system_message else {}),
            "stream": body.get("stream", False),
        }

        # Only include optional parameters if they were explicitly set
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k

        headers = {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        url = "https://api.anthropic.com/v1/messages"

        try:
            if body.get("stream", False):
                return self.stream_response(url, headers, payload)
            else:
                return await self.non_stream_response(url, headers, payload)
        except aiohttp.ClientError as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    async def stream_response(self, url, headers, payload):
        try:
            timeout = aiohttp.ClientTimeout(total=60, connect=3.05)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        raise Exception(
                            f"HTTP Error {response.status}: {response_text}"
                        )

                    async for line in response.content:
                        if line:
                            line = line.decode("utf-8").strip()
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if data["type"] == "content_block_start":
                                        yield data["content_block"]["text"]
                                    elif data["type"] == "content_block_delta":
                                        yield data["delta"]["text"]
                                    elif data["type"] == "message_stop":
                                        break
                                    elif data["type"] == "message":
                                        for content in data.get("content", []):
                                            if content["type"] == "text":
                                                yield content["text"]

                                except json.JSONDecodeError:
                                    print(f"Failed to parse JSON: {line}")
                                except KeyError as e:
                                    print(f"Unexpected data structure: {e}")
                                    print(f"Full data: {data}")
        except aiohttp.ClientError as e:
            print(f"Request failed: {e}")
            yield f"Error: Request failed: {e}"
        except Exception as e:
            print(f"General error in stream_response method: {e}")
            yield f"Error: {e}"

    async def non_stream_response(self, url, headers, payload):
        try:
            timeout = aiohttp.ClientTimeout(total=60, connect=3.05)
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
        except aiohttp.ClientError as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"
