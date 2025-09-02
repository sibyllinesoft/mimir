# Pull Request Template - Mimir Deep Code Research System

## 📋 Description

Please provide a clear and concise description of what this PR does.

### Type of Change
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Maintenance (dependency updates, code cleanup, etc.)
- [ ] ⚡ Performance improvement
- [ ] 🔒 Security enhancement
- [ ] 🧪 Test improvement

## 🔗 Related Issues

Closes #(issue number)
Fixes #(issue number)
Related to #(issue number)

## 🚀 Changes Made

Please describe the changes made in this PR:

- [ ] Change 1
- [ ] Change 2
- [ ] Change 3

## 🧪 Testing

### Testing Performed
- [ ] Unit tests pass (`pytest tests/unit/`)
- [ ] Integration tests pass (`pytest tests/integration/`)
- [ ] Performance tests pass (`pytest tests/benchmarks/`)
- [ ] Manual testing completed
- [ ] End-to-end testing completed

### Test Coverage
- [ ] New code is covered by tests
- [ ] Coverage percentage maintained or improved
- [ ] Critical paths are thoroughly tested

### Testing Details
Describe any specific testing scenarios, edge cases, or performance considerations:

```
# Example test commands run
pytest tests/unit/test_new_feature.py -v
pytest tests/integration/test_pipeline_integration.py -v
```

## 📊 Performance Impact

- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance impact assessed and acceptable
- [ ] Performance benchmarks run and documented

**Performance Notes:**
<!-- Describe any performance implications, benchmark results, or optimization details -->

## 🔒 Security Considerations

- [ ] No security implications
- [ ] Security review completed
- [ ] Security tests pass
- [ ] No sensitive data exposed
- [ ] Authentication/authorization properly handled

**Security Notes:**
<!-- Describe any security considerations, mitigations, or reviews performed -->

## 📚 Documentation

- [ ] Code is self-documenting with clear variable/function names
- [ ] Comments added for complex logic
- [ ] Docstrings updated for public APIs
- [ ] README updated (if applicable)
- [ ] API documentation updated (if applicable)
- [ ] Architecture documentation updated (if applicable)

## 🔧 Deployment Considerations

- [ ] No deployment changes required
- [ ] Environment variables added/changed (documented)
- [ ] Database migrations required (documented)
- [ ] Configuration changes required (documented)
- [ ] Infrastructure changes required (documented)
- [ ] Backward compatibility maintained

**Deployment Notes:**
<!-- Describe any deployment steps, configuration changes, or migration requirements -->

## ✅ Checklist

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Code is well-commented and self-documenting
- [ ] No debug code, console.logs, or temporary code left
- [ ] Error handling is appropriate and comprehensive
- [ ] Code is DRY (Don't Repeat Yourself)

### CI/CD Pipeline
- [ ] All CI checks pass (linting, type checking, tests)
- [ ] Security scans pass
- [ ] Container builds successfully
- [ ] No new vulnerabilities introduced

### Review Readiness
- [ ] PR is ready for review
- [ ] PR description is clear and complete
- [ ] Commits are atomic and well-described
- [ ] Branch is up to date with target branch

## 🎯 Definition of Done

- [ ] Feature/fix works as expected
- [ ] All tests pass (unit, integration, e2e)
- [ ] Code review completed and approved
- [ ] Security considerations addressed
- [ ] Performance impact assessed
- [ ] Documentation updated
- [ ] CI/CD pipeline passes
- [ ] Ready for deployment

## 📱 Screenshots/Examples

<!-- If applicable, add screenshots, code examples, or output samples to help reviewers understand the changes -->

## 🤝 Reviewer Guidelines

### Focus Areas for Review
- [ ] **Code Quality**: Is the code clean, readable, and maintainable?
- [ ] **Security**: Are there any security vulnerabilities or concerns?
- [ ] **Performance**: Will this impact system performance?
- [ ] **Testing**: Is the testing comprehensive and appropriate?
- [ ] **Architecture**: Does this align with system architecture and patterns?

### Testing the Changes
To test this PR locally:

```bash
# 1. Checkout the branch
git checkout [branch-name]

# 2. Install dependencies
uv sync --extra dev --extra test

# 3. Run tests
pytest tests/ -v

# 4. Run the application
python -m repoindex.mcp.server
```

## 📝 Additional Notes

<!-- Any additional information that reviewers should know about this PR -->

---

**Thank you for contributing to Mimir Deep Code Research System! 🎉**

<!-- 
Please ensure all items in the checklist are completed before requesting review.
If any items are not applicable, mark them as complete and note why in the PR description.
-->