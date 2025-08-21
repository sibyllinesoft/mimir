# Suggested Commands for Mimir Development

## Environment Setup
```bash
# Initial setup (only once)
python setup.py

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Sync dependencies with uv
uv sync --dev
```

## Running the Application
```bash
# Start MCP server (main interface)
uv run mimir-server

# Start UI server (optional management interface)
uv run mimir-ui
```

## Testing Commands
```bash
# Run unit tests only
uv run pytest tests/unit/ -v

# Run integration tests
uv run pytest tests/integration/ -v

# Run all tests with coverage
uv run pytest --cov=src/repoindex

# Skip slow tests
uv run pytest -m "not slow"

# Run specific test file
uv run pytest tests/unit/test_schemas.py -v
```

## Code Quality Commands
```bash
# Format code with black
uv run black src/ tests/

# Lint code with ruff
uv run ruff check src/ tests/

# Type checking with mypy
uv run mypy src/
```

## Git Commands (for the system)
```bash
# Standard git commands work on Linux
git status
git add .
git commit -m "message"
git push
git diff
git log
```

## System Utils (Linux)
```bash
ls -la            # List files with details
cd <directory>    # Change directory
grep -r "pattern" # Recursive search
find . -name "*.py" # Find files by pattern
```

## Development Workflow
1. Make changes to code
2. Run formatter: `uv run black src/ tests/`
3. Run linter: `uv run ruff check src/ tests/`
4. Run type checker: `uv run mypy src/`
5. Run tests: `uv run pytest tests/unit/ -v`
6. Commit changes with git