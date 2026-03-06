# OpenWebUI Anthropic Async Pipe

An async implementation of the Anthropic Claude integration for OpenWebUI, converted from the synchronous version to use `aiohttp` for better performance and scalability.

## Features

- ✅ **Async/Await Support** - Full async implementation using aiohttp
- ✅ **Streaming Responses** - Anthropic SSE is translated into OpenAI-style chunks for Open WebUI
- ✅ **Image Processing** - Support for both base64 and URL images with size validation
- ✅ **Multiple Models** - Support for Claude Sonnet 4 and Opus 4
- ✅ **Error Handling** - Comprehensive error handling and validation
- ✅ **OpenWebUI Compatible** - Drop-in replacement for the sync version

## Function Approach

The `Pipe` implementation is organized around a small number of responsibilities so the Open WebUI function interface stays compatible while Anthropic-specific details are handled internally.

### Request normalization

- OpenAI-style request bodies are converted into Anthropic `messages` payloads.
- System messages are extracted and joined into Anthropic's top-level `system` field.
- `model` values like `anthropic.claude-sonnet-4-6` are normalized to Anthropic model ids.
- OpenAI-style `stop` input is normalized into Anthropic `stop_sequences`.
- Supported model-specific options such as `output_config.effort` and adaptive `thinking` are added only when appropriate.

### Message and image handling

- String content is wrapped as Anthropic text blocks.
- List content is validated item-by-item and currently supports `text` and `image_url` blocks.
- Base64 images are size-checked locally against Anthropic's limits.
- Remote image URLs get a best-effort `HEAD` check for `content-length`; oversized images are rejected, but transient network failures during that preflight do not block the request.

### Streaming strategy

- Anthropic streams Server-Sent Events, not OpenAI chat chunks directly.
- The function incrementally decodes bytes, reconstructs SSE events across arbitrary chunk boundaries, and parses `event:` / `data:` frames safely.
- Anthropic events such as `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`, and `error` are translated into OpenAI-compatible `chat.completion.chunk` payloads.
- When Starlette is available, the function returns a direct `StreamingResponse` with SSE payloads so Open WebUI can forward the stream with less extra wrapping.
- Small adjacent text deltas are micro-batched with a very short timeout to reduce visible stutter without adding much latency.
- Anthropic `stop_reason` values are mapped to OpenAI-style `finish_reason` values such as `stop`, `length`, and `tool_calls`.

### Non-stream responses

- Anthropic content blocks are folded back into a single OpenAI-style `chat.completion` response.
- Thinking blocks are exposed as `reasoning_content` when present.
- The Anthropic `stop_reason` is preserved through OpenAI-style `finish_reason` mapping.

### Error handling

- `aiohttp` transport errors are caught separately from general exceptions.
- Stream errors are surfaced as terminal stream chunks.
- Non-stream errors are returned in a structured `{"error": {"message": ...}}` shape.

## Installation

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

The async pipe can be used as a drop-in replacement for the synchronous version in OpenWebUI. The pipe function signature is compatible with OpenWebUI's async function calling pattern.

```python
from anthropic_async import Pipe

# Initialize the pipe
pipe = Pipe()

# Use in async context
async def example():
    body = {
        "model": "anthropic.claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": False
    }
    
    response = await pipe.pipe(body)
    print(response)
```

## Testing

Comprehensive integration tests using pytest test all functionality against the real Anthropic API.

### Setup
```bash
# Install dependencies including test dependencies
uv sync

# Set API key
export ANTHROPIC_API_KEY=your_api_key_here
```

### Run Tests
```bash
# Run all tests
just test

# Run specific test categories
just test-streaming    # Only streaming tests
just test-images       # Only image tests
just test-errors       # Only error handling tests
just test-slow         # Only slow tests

# Run tests by name pattern
just test-name "image_url"

# Or use pytest directly
uv run pytest tests/ -v
```

### Test Coverage
- ✅ Non-streaming and streaming responses
- ✅ Image processing (base64 and URLs) 
- ✅ System message handling
- ✅ Multiple Claude models
- ✅ Error handling and validation
- ✅ Concurrent request handling
- ✅ Parametrized test cases

## Key Differences from Sync Version

1. **Async Methods**: All HTTP operations use `aiohttp` instead of `requests`
2. **Image Processing**: URL image validation is now async
3. **Streaming**: Parses Anthropic SSE and emits OpenAI-style chunks, optionally through Starlette `StreamingResponse`
4. **Error Handling**: Updated for `aiohttp.ClientError` exceptions
5. **Performance**: Better handling of concurrent requests

## Development

### Available Commands

This project uses [just](https://github.com/casey/just) for task automation:

```bash
# Show all available commands
just

# Run tests
just test
just test-streaming
just test-images

# Code quality
just format        # Format with black
just lint-ruff     # Lint with ruff
just lint-fix      # Lint and auto-fix

# Development
just run           # Run the main module
just install       # Install dependencies
just update        # Update dependencies
```

This project uses [opencode](https://opencode.ai) as a copilot for development assistance.
