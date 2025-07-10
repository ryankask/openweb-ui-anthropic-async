# AGENTS.md - Development Guidelines

## Build/Test/Lint Commands
- Test: `pytest` or `python -m pytest`
- Lint: `ruff check .` (fix with `ruff check . --fix`)
- Format: `black .`
- Type check: `python -m py_compile <file>` for basic syntax checking
- Run single file: `python <filename>.py`

## Code Style Guidelines
- **Python Version**: Requires Python >=3.13 (see pyproject.toml)
- **Imports**: Use standard library first, then third-party (requests, pydantic), then local imports
- **Type Hints**: Use typing module annotations (List, Union, Generator, Iterator, etc.)
- **Error Handling**: Use try/except blocks with specific exception types, log errors with descriptive messages
- **Classes**: Use Pydantic BaseModel for data validation, implement __init__ methods for setup
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Docstrings**: Use triple quotes for function documentation when needed
- **Constants**: Define at class level (e.g., MAX_IMAGE_SIZE = 5 * 1024 * 1024)

## Architecture Patterns
- **OpenWebUI Integration**: This is an OpenWebUI pipe/function for Anthropic API integration
- **Async Support**: Use async/await for I/O operations, support both sync and async function calls
- **Streaming**: Support both streaming and non-streaming responses via generators/iterators
- **Validation**: Use Pydantic models for configuration (Valves pattern)
- **Error Propagation**: Return error messages as strings or yield them in streaming responses

##
