# OpenWebUI Anthropic Async Pipe

An async implementation of the Anthropic Claude integration for OpenWebUI, converted from the synchronous version to use `aiohttp` for better performance and scalability.

## Features

- ✅ **Async/Await Support** - Full async implementation using aiohttp
- ✅ **Streaming Responses** - Real-time text generation with async generators
- ✅ **Image Processing** - Support for both base64 and URL images with size validation
- ✅ **Multiple Models** - Support for Claude Sonnet 4 and Opus 4
- ✅ **Error Handling** - Comprehensive error handling and validation
- ✅ **OpenWebUI Compatible** - Drop-in replacement for the sync version

## Installation

1. Install dependencies:
   ```bash
   pip install aiohttp pydantic
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
# Install test dependencies
pip install -e .[test]

# Set API key
export ANTHROPIC_API_KEY=your_api_key_here
```

### Run Tests
```bash
# Run all tests
python run_tests.py

# Or use pytest directly
pytest tests/ -v

# Run specific test categories
pytest tests/ -m streaming -v    # Only streaming tests
pytest tests/ -m images -v       # Only image tests
pytest tests/ -m "not slow" -v   # Skip slow tests
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
3. **Streaming**: Uses async generators for streaming responses
4. **Error Handling**: Updated for `aiohttp.ClientError` exceptions
5. **Performance**: Better handling of concurrent requests

## Development

This project uses [opencode](https://opencode.ai) as a copilot for development assistance.
