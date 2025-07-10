# Integration Tests (pytest)

This directory contains comprehensive integration tests for the async Anthropic pipe using pytest.

## Test Coverage

The tests use pytest markers to organize functionality:

### Core Features
- ✅ **`@pytest.mark.integration`** - All integration tests
- ✅ **`@pytest.mark.streaming`** - Streaming response tests
- ✅ **`@pytest.mark.images`** - Image processing tests
- ✅ **`@pytest.mark.error_handling`** - Error condition tests
- ✅ **`@pytest.mark.slow`** - Slower tests (concurrent requests)

## Running Tests

### Prerequisites
1. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

2. Install test dependencies:
   ```bash
   pip install -e .[test]
   # or manually:
   pip install pytest pytest-asyncio pytest-mock aiohttp pydantic
   ```

### Run All Tests
```bash
# Using the test runner
python run_tests.py

# Or directly with pytest
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Only streaming tests
pytest tests/ -m streaming -v

# Only image tests
pytest tests/ -m images -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# Only integration tests (all of them)
pytest tests/ -m integration -v
```

### Run Specific Tests
```bash
# Single test
pytest tests/test_integration.py::test_basic_non_streaming -v

# Test pattern
pytest tests/ -k "streaming" -v
```

## Pytest Features Used

### Fixtures (`conftest.py`)
- **`api_key`** - Session-scoped API key validation
- **`pipe_instance`** - Configured Pipe instance
- **`create_text_body`** - Factory for text message bodies
- **`create_image_body`** - Factory for image message bodies
- **`small_test_image`** - 1x1 pixel test image
- **`execute_pipe_func`** - OpenWebUI-compatible pipe execution

### Markers
- **`@pytest.mark.integration`** - Integration tests requiring API
- **`@pytest.mark.streaming`** - Streaming functionality
- **`@pytest.mark.images`** - Image processing
- **`@pytest.mark.error_handling`** - Error conditions
- **`@pytest.mark.slow`** - Tests that take longer
- **`@pytest.mark.parametrize`** - Parameterized test cases

### Async Support
- Uses `pytest-asyncio` for async test support
- Automatic async mode configured in `pytest.ini`
- Proper handling of async generators and iterators

## Test Structure

```
tests/
├── conftest.py           # Fixtures and configuration
├── test_integration.py   # Main integration tests
└── README.md            # This file
```

## Cost Optimization

- **Small token limits** (20-50 tokens) to minimize API costs
- **Tiny test images** (1x1 pixel PNG) for image processing
- **Short prompts** for basic functionality validation
- **Parametrized tests** to cover multiple scenarios efficiently

## Expected Output

```bash
$ pytest tests/ -v
========================= test session starts =========================
tests/test_integration.py::test_basic_non_streaming PASSED    [ 10%]
tests/test_integration.py::test_basic_streaming PASSED        [ 20%]
tests/test_integration.py::test_system_message PASSED         [ 30%]
tests/test_integration.py::test_image_base64 PASSED           [ 40%]
tests/test_integration.py::test_image_url PASSED              [ 50%]
tests/test_integration.py::test_multiple_models PASSED        [ 60%]
tests/test_integration.py::test_invalid_model PASSED          [ 70%]
tests/test_integration.py::test_invalid_api_key PASSED        [ 80%]
tests/test_integration.py::test_large_image_error PASSED      [ 90%]
tests/test_integration.py::test_concurrent_requests PASSED    [100%]
========================= 10 passed in 15.23s =========================
```

## Troubleshooting

### Common Issues
1. **API Key Error**: `pytest tests/ --tb=short` for cleaner error output
2. **Import Error**: Ensure you're running from project root
3. **Async Issues**: Check `pytest-asyncio` is installed
4. **Rate Limiting**: Use `pytest tests/ -x` to stop on first failure