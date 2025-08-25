# Ollama Integration Testing Guide

This document explains how to run integration tests that verify the Mimir MCP server works correctly with Ollama for local LLM processing.

## Overview

The Ollama integration tests verify that:
- Mimir MCP server can connect to Ollama
- Repository indexing works with Ollama embeddings
- Vector search uses Ollama embeddings effectively  
- Question answering integrates with Ollama chat models
- Bundle export includes Ollama metadata

## Prerequisites

### 1. Install Ollama

```bash
# On macOS
brew install ollama

# On Linux
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows
# Download from https://ollama.ai/download
```

### 2. Start Ollama Server

```bash
ollama serve
```

The server will start on `http://localhost:11434` by default.

### 3. Install Required Models

The tests use lightweight models suitable for development:

```bash
# Embedding model (required for vector search)
ollama pull nomic-embed-text

# Chat model (required for question answering)  
ollama pull llama3.1:8b
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install httpx  # For Ollama API calls
```

## Running the Tests

### Option 1: Automated Test Runner (Recommended)

Use the provided test runner script:

```bash
# Full setup and test (downloads models automatically)
python scripts/test_ollama_integration.py --setup --verbose

# Quick tests only (skip slow end-to-end tests)
python scripts/test_ollama_integration.py --quick

# Test specific models
python scripts/test_ollama_integration.py --models "nomic-embed-text,all-minilm"
```

### Option 2: Direct pytest

Run tests directly with pytest:

```bash
# Run all Ollama integration tests
pytest tests/integration/test_mcp_ollama_integration.py -v -m integration

# Run only fast tests
pytest tests/integration/test_mcp_ollama_integration.py -v -m "integration and not slow"

# Run specific test
pytest tests/integration/test_mcp_ollama_integration.py::TestMCPOllamaIntegration::test_ollama_availability -v
```

### Option 3: Individual Test Components

Test individual components:

```bash
# Test just Ollama connectivity
python tests/integration/test_mcp_ollama_integration.py

# Test environment setup
python -c "from tests.integration.test_ollama_config import check_test_environment; print(check_test_environment())"
```

## Test Configuration

### Environment Variables

Customize test behavior with environment variables:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_EMBEDDING_MODEL="nomic-embed-text" 
export OLLAMA_CHAT_MODEL="llama3.1"
export OLLAMA_TIMEOUT="30.0"
export OLLAMA_MAX_RETRIES="3"
```

### Configuration File

See `tests/integration/test_ollama_config.py` for detailed configuration options.

## Test Structure

### Core Test Classes

- **`OllamaTestHelper`**: Utilities for Ollama server interaction
- **`TestMCPOllamaIntegration`**: Main test class with comprehensive integration tests

### Test Categories

1. **Connectivity Tests**
   - Ollama server availability
   - Model availability and download
   - API response validation

2. **Indexing Tests**  
   - Repository indexing with Ollama embeddings
   - File processing and chunking
   - Metadata generation

3. **Search Tests**
   - Vector search with Ollama embeddings
   - Hybrid search (vector + symbol + graph)
   - Query embedding generation

4. **Q&A Tests**
   - Question answering with Ollama chat models
   - Context retrieval and ranking
   - Response quality validation

5. **Bundle Tests**
   - Export with Ollama metadata
   - Bundle structure validation
   - Cross-system compatibility

### Performance Tests

- **Fast Tests** (< 30 seconds): Connectivity, basic operations
- **Standard Tests** (< 5 minutes): Full indexing, search, Q&A
- **Slow Tests** (5-15 minutes): End-to-end workflows, large repositories

## Expected Test Results

### Successful Test Run

```
✅ Ollama server is available and responsive
✅ Generated embedding with 768 dimensions
✅ Successfully indexed repository with Ollama configuration
   Index ID: abc123...
   Files indexed: 3
✅ Ollama-based search returned 5 results
✅ Combined Ollama + symbol search returned 3 results
✅ Ollama-based question answered successfully
   Answer length: 245 characters
   Answer preview: The DataProcessor class is a key component...
✅ Bundle exported with Ollama metadata:
   Model: nomic-embed-text
   Dimensions: 768
   Chunks: 15
✅ End-to-end Ollama workflow completed successfully!
```

## Troubleshooting

### Common Issues

#### 1. Ollama Server Not Running
```
ERROR: Ollama connection failed: Connection refused
```

**Solution**: Start Ollama server
```bash
ollama serve
```

#### 2. Model Not Available
```
ERROR: nomic-embed-text model not available in Ollama
```

**Solution**: Pull required models
```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

#### 3. Timeout Issues
```
ERROR: Model pull failed: timeout
```

**Solution**: Increase timeout or use smaller models
```bash
export OLLAMA_TIMEOUT="120.0"
# Or use lighter models
export OLLAMA_EMBEDDING_MODEL="all-minilm"
```

#### 4. Permission Issues
```
ERROR: Failed to create test directory
```

**Solution**: Check write permissions
```bash
mkdir -p /tmp/mimir_test
chmod 755 /tmp/mimir_test
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python scripts/test_ollama_integration.py --verbose
```

### Manual Verification

Test Ollama manually:

```bash
# Test embedding generation
curl -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed-text", "prompt": "test text"}'

# Test chat completion
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1", "prompt": "Explain machine learning", "stream": false}'
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Ollama Integration Tests

on: [push, pull_request]

jobs:
  test-ollama:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Ollama
        run: |
          curl -fsSL https://ollama.ai/install.sh | sh
          ollama serve &
          sleep 10
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install httpx
      
      - name: Run Ollama integration tests
        run: |
          python scripts/test_ollama_integration.py --setup --quick
```

## Performance Benchmarks

Expected performance on typical hardware:

| Operation | Time | Notes |
|-----------|------|--------|
| Ollama startup | 5-10s | First time only |
| Model download | 30s-5min | Depends on model size |
| Repository indexing | 10-60s | ~50 files |
| Vector search | 100-500ms | Single query |
| Question answering | 1-10s | Depends on model |

## Model Recommendations

### For Development
- **Embedding**: `nomic-embed-text` (768 dims, fast)
- **Chat**: `llama3.1:8b` (good quality/speed balance)

### For Production
- **Embedding**: `bge-large` (1024 dims, highest quality)  
- **Chat**: `llama3.1:70b` (best quality, requires more RAM)

### For Resource-Constrained Environments
- **Embedding**: `all-minilm` (384 dims, very fast)
- **Chat**: `llama3.1:1b` (fastest, basic quality)

## Support

For issues with the Ollama integration tests:

1. Check this README for common solutions
2. Review test logs for specific error messages
3. Verify Ollama server status: `curl http://localhost:11434/api/version`
4. Test individual components before running full suite
5. Use `--verbose` flag for detailed debugging output