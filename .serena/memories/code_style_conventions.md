# Code Style and Conventions for Mimir

## Python Version
- Target Python 3.11+
- Use modern Python features and type hints

## Code Style
- **Formatter**: Black with line-length=100
- **Linter**: Ruff with comprehensive rule set (E, W, F, I, B, C4, UP, N)
- **Import sorting**: Handled by Ruff (isort rules)
- **Naming**: PEP 8 naming conventions enforced

## Type Hints
- **Required**: All functions must have type hints (enforced by mypy)
- **Strict mypy configuration**:
  - disallow_untyped_defs = true
  - disallow_incomplete_defs = true
  - no_implicit_optional = true
  - strict_equality = true
- Use `Optional[T]` for nullable types
- Use `list[T]` instead of `List[T]` (Python 3.11+ syntax)

## Docstrings
- Follow Google-style docstrings
- Include parameter descriptions and return types
- Example:
```python
def function_name(param1: str, param2: int) -> dict:
    """Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
```

## Async/Await
- Use async/await consistently for I/O operations
- Mark test functions with `@pytest.mark.asyncio`
- Use `asyncio.create_task()` for concurrent execution

## Project Structure
- Keep related functionality in the same module
- Use dataclasses for data models
- Pydantic for validation and external APIs
- Separate concerns: adapters, pipeline stages, schemas

## Testing Conventions
- Test files named `test_*.py`
- Use pytest markers: @pytest.mark.slow, @pytest.mark.integration, @pytest.mark.unit
- Fixtures defined in conftest.py
- Aim for comprehensive test coverage

## Error Handling
- Use specific exception types
- Provide actionable error messages
- Log errors with structured logging
- Graceful degradation when possible

## File Organization
- One class per file for major components
- Group related utilities in single modules
- Keep imports organized (stdlib, third-party, local)