# Task Completion Checklist for Mimir

When completing any development task, ensure the following steps are performed:

## 1. Code Quality Checks
```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type checking
uv run mypy src/
```

## 2. Testing
```bash
# Run unit tests
uv run pytest tests/unit/ -v

# Run integration tests if changes affect multiple components
uv run pytest tests/integration/ -v

# Check test coverage
uv run pytest --cov=src/repoindex tests/
```

## 3. Documentation Updates
- Update docstrings for new/modified functions
- Update README.md if adding new features
- Update ARCHITECTURE.md for architectural changes
- Update API documentation if changing MCP tools

## 4. Pre-commit Verification
- All tests pass
- No linting errors
- No type checking errors
- Code is properly formatted
- Documentation is updated

## 5. Common Issues to Check
- Async functions properly marked
- Type hints on all functions
- Error handling implemented
- Logging added for debugging
- No hardcoded paths or values
- Configuration options exposed where appropriate

## 6. Performance Considerations
- Streaming for large files
- Concurrent processing where beneficial
- Proper resource cleanup
- Memory efficient data structures

## 7. Security & Privacy
- No external service calls without configuration
- Local-only processing maintained
- No telemetry or data collection
- Proper file permission handling