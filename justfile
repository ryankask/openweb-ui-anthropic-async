# Justfile for anthropic-async project

# Show available commands
default:
    @just --list

# Run all tests
test:
    uv run pytest tests/ -v

# Run specific test by name
test-name name:
    uv run pytest tests/ -k "{{name}}" -v

# Run tests by category/marker
test-category category:
    uv run pytest tests/ -m "{{category}}" -v

# Run streaming tests
test-streaming:
    uv run pytest tests/ -m "streaming" -v

# Run image tests
test-images:
    uv run pytest tests/ -m "images" -v

# Run error handling tests
test-errors:
    uv run pytest tests/ -m "error_handling" -v

# Run slow tests
test-slow:
    uv run pytest tests/ -m "slow" -v

# Check syntax of a specific file
lint file:
    uv run python -m py_compile {{file}}

# Check syntax of main files
lint-all:
    uv run python -m py_compile anthropic_async.py
    uv run python -m py_compile existing-code/anthropic_sync.py
    uv run python -m py_compile existing-code/openwebui_functions.py

# Run the main async module
run:
    uv run python anthropic_async.py

# Run the sync version
run-sync:
    uv run python existing-code/anthropic_sync.py

# Install dependencies
install:
    uv sync

# Update dependencies
update:
    uv lock --upgrade