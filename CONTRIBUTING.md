# Contributing to Mimir

Thank you for your interest in contributing to Mimir! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd mimir
   ```

2. **Set up the development environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

3. **Run tests to verify setup:**
   ```bash
   python -m pytest tests/
   ```

## Code Style and Quality

- **Formatting:** We use Black for code formatting (when available)
- **Linting:** Follow PEP 8 guidelines
- **Type hints:** Use type hints for all public functions
- **Docstrings:** Document all public functions and classes

## Testing

- **Unit tests:** Add unit tests for all new functionality
- **Integration tests:** Add integration tests for complex features
- **Test coverage:** Maintain test coverage above 80%
- **Test commands:**
  ```bash
  python -m pytest tests/unit/
  python -m pytest tests/integration/
  python -m pytest tests/performance/
  ```

## Commit Messages

Use clear, descriptive commit messages:
```
feat(mcp): add monitored server with Skald integration
fix(pipeline): resolve memory leak in indexing process
docs(readme): update installation instructions
chore(deps): update dependencies to latest versions
```

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write code following our style guidelines
   - Add appropriate tests
   - Update documentation if needed

3. **Test your changes:**
   ```bash
   python -m pytest
   ```

4. **Submit a pull request:**
   - Provide a clear description of changes
   - Link any related issues
   - Ensure all CI checks pass

## Architecture Guidelines

- **Modular design:** Keep components loosely coupled
- **Security first:** Follow security best practices
- **Performance:** Consider performance implications
- **Documentation:** Document architectural decisions

## Monitoring Integration

When adding features that interact with the monitoring system:

- Use appropriate Skald decorators for function monitoring
- Add NATS trace emission for significant operations
- Update monitoring configuration if needed
- Test monitoring integration

## Security Considerations

- **Never commit secrets:** Use environment variables or secure vaults
- **Input validation:** Validate all external inputs
- **Dependencies:** Keep dependencies up to date
- **Security review:** Request security review for sensitive changes

## Getting Help

- **Documentation:** Check existing documentation first
- **Issues:** Search existing issues before creating new ones
- **Discussions:** Use GitHub Discussions for questions
- **Code review:** Ask for help during code review process

## Code Review Checklist

Before submitting a pull request, ensure:

- [ ] Code follows style guidelines
- [ ] Tests are added and passing
- [ ] Documentation is updated
- [ ] Security implications are considered
- [ ] Performance impact is evaluated
- [ ] Breaking changes are documented
- [ ] Monitoring integration is tested (if applicable)

## Release Process

1. Features are developed in feature branches
2. Pull requests are reviewed and merged to main
3. Releases are tagged and deployed following semantic versioning
4. Release notes are generated automatically

Thank you for contributing to Mimir!