# 🚀 Claude Desktop + Mimir Integration Guide

**Transform Claude into a powerful code analysis assistant with research-backed AI techniques**

> Turn your codebase into an intelligent knowledge system that Claude can understand, search, and reason about using cutting-edge RAPTOR and HyDE algorithms.

---

## ⚡ 30-Second Setup

### Step 1: Set Up Mimir
```bash
# Clone and set up
git clone https://github.com/your-org/mimir.git && cd mimir
python setup.py
```

### Step 2: Add to Claude Desktop
Open your Claude Desktop configuration file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/claude-desktop/claude_desktop_config.json`

Add this configuration:
```json
{
  "mcpServers": {
    "mimir": {
      "command": "uv",
      "args": ["run", "python", "-m", "repoindex.mcp.server"],
      "cwd": "/path/to/your/mimir",
      "env": {
        "MIMIR_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Step 3: Restart & Enjoy
Restart Claude Desktop and start using natural language to explore your code!

```
"Index my TypeScript project and find authentication patterns"
"How does error handling work in this codebase?"  
"What's the most complex function and how can we refactor it?"
```

---

## 🎯 What You Get

### 5 Powerful MCP Tools

**🔍 `ensure_repo_index`** - Index any repository with research-grade analysis
```
Claude will: Analyze AST → Extract symbols → Generate embeddings → Build RAPTOR tree
```

**🔎 `search_repo`** - Multi-modal search with semantic understanding
```  
Claude will: Transform query → Search vectors + symbols + graphs → Rank results
```

**🧠 `ask_index`** - Complex code reasoning with multi-hop analysis
```
Claude will: Navigate symbol graphs → Gather evidence → Provide detailed explanations
```

**📦 `get_repo_bundle`** - Export compressed knowledge for sharing
```
Claude will: Package complete analysis → Create portable bundle → Enable collaboration
```

**⚡ `cancel`** - Stop long-running operations
```
Claude will: Gracefully cancel → Clean up resources → Preserve partial progress
```

### 4 Live Resources

- **📊 Real-time status** - Watch indexing progress live
- **📋 Index metadata** - Complete repository analysis details  
- **📝 Activity logs** - Human-readable operation history
- **📁 Bundle access** - Direct access to compressed artifacts

## Configuration Options

### Basic Configuration

```json
{
  "mcpServers": {
    "mimir-repoindex": {
      "command": "mimir-server",
      "args": [],
      "env": {
        "MIMIR_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Advanced Configuration

```json
{
  "mcpServers": {
    "mimir-repoindex": {
      "command": "mimir-server",
      "args": ["--storage-dir", "/custom/storage/path"],
      "env": {
        "MIMIR_LOG_LEVEL": "DEBUG",
        "MIMIR_DATA_DIR": "/custom/data/directory",
        "MIMIR_CACHE_SIZE": "1000",
        "MIMIR_MAX_WORKERS": "4",
        "GOOGLE_API_KEY": "your-gemini-api-key-here"
      }
    }
  }
}
```

### Configuration with Virtual Environment

If you installed Mimir in a virtual environment:

```json
{
  "mcpServers": {
    "mimir-repoindex": {
      "command": "/path/to/venv/bin/mimir-server",
      "args": [],
      "env": {
        "MIMIR_LOG_LEVEL": "INFO",
        "PATH": "/path/to/venv/bin:${PATH}"
      }
    }
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MIMIR_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `MIMIR_DATA_DIR` | `~/.cache/mimir` | Directory for storing indexes and cache |
| `MIMIR_MAX_WORKERS` | `4` | Maximum concurrent indexing operations |
| `MIMIR_CACHE_SIZE` | `500` | Maximum number of cached search results |
| `GOOGLE_API_KEY` | None | Gemini API key for enhanced AI features |
| `MIMIR_ENABLE_UI` | `false` | Enable web UI server |
| `MIMIR_UI_PORT` | `8080` | Web UI port (if enabled) |

## Available MCP Tools

Once configured, Mimir provides these tools in Claude:

### 1. `ensure_repo_index`
Index a repository for search and analysis.

**Parameters:**
- `path` (required): Path to repository root
- `rev` (optional): Git revision (defaults to HEAD)  
- `language` (optional): Primary language (defaults to 'ts')
- `index_opts` (optional): Additional indexing options

### 2. `search_repo`
Search repository using hybrid vector + symbol + graph approach.

**Parameters:**
- `index_id` (required): Index identifier from `ensure_repo_index`
- `query` (required): Search query
- `k` (optional): Number of results (default: 20, max: 100)
- `features` (optional): Search modalities to enable
- `context_lines` (optional): Context lines around matches (default: 5)

### 3. `ask_index`  
Ask complex questions using multi-hop symbol graph reasoning.

**Parameters:**
- `index_id` (required): Index identifier
- `question` (required): Question about the codebase
- `context_lines` (optional): Context lines for evidence (default: 5)

### 4. `get_repo_bundle`
Retrieve complete index bundle for a repository.

**Parameters:**
- `index_id` (required): Index identifier

### 5. `cancel`
Cancel an ongoing indexing operation.

**Parameters:**
- `index_id` (required): Index identifier to cancel

## Available MCP Resources

Mimir also provides these resources for real-time monitoring:

- `repo://status` - Global system status
- `mimir://indexes/{index_id}/status.json` - Pipeline status
- `mimir://indexes/{index_id}/manifest.json` - Index metadata  
- `mimir://indexes/{index_id}/log.md` - Progress log
- `mimir://indexes/{index_id}/bundle.tar.zst` - Compressed bundle

---

## 🎬 Real-World Usage Examples

### 🔍 Example 1: Legacy Code Detective

**You:** *"I inherited this React codebase. Help me understand how user authentication works."*

**Claude with Mimir:**
```
🔄 Indexing your React project...
✅ Found 342 files, analyzing with RAPTOR...
✅ Extracted 1,847 symbols and dependencies
✅ Generated semantic embeddings for 3,234 code chunks

🔍 Searching for authentication patterns...

📄 Found 12 authentication-related components:

🔑 src/auth/AuthProvider.tsx (98% relevance)
   Context provider managing authentication state
   
🛡️ src/hooks/useAuth.ts (95% relevance)  
   Custom hook with login/logout/token management
   
🚪 src/components/LoginForm.tsx (92% relevance)
   Login form with validation and error handling

🧠 Multi-hop analysis reveals:
• AuthProvider wraps the entire app in src/App.tsx
• Login tokens are stored in localStorage via tokenStorage.ts  
• Protected routes use RequireAuth component in src/router/
• API calls include auth headers via axios interceptor
```

### 🐛 Example 2: Bug Investigation

**You:** *"Users report payment processing sometimes fails. Help me trace the payment flow and find potential issues."*

**Claude with Mimir:**
```
🔍 Analyzing payment flow with graph traversal...

💳 Payment Processing Flow:
┌─ Entry Point: PaymentForm.handleSubmit()
├─ Validation: validatePaymentData() → 3 validation rules
├─ API Call: processPayment() → POST /api/payments
├─ Backend: PaymentController.processPayment()
├─ Service: PaymentService.createCharge()
└─ External: StripeAdapter.createPaymentIntent()

⚠️  Potential Issues Detected:
• Missing error handling in PaymentService.createCharge() (line 45)
• Race condition possible in PaymentController (concurrent requests)
• Network timeout not configured for Stripe API calls
• Payment status not persisted before external API call

📊 Code Evidence: 8 files, 23 functions analyzed
🔗 Call graph shows 3 async operations without proper error boundaries
```

### 🏗️ Example 3: Architecture Analysis

**You:** *"Help me understand if this microservices architecture follows good patterns."*

**Claude with Mimir:**
```
🏛️ Architecture Analysis Complete

📊 Service Dependencies (RAPTOR hierarchical view):
```
Level 1: Core Services
├─ user-service (highly coupled - 47 dependencies)
├─ auth-service (well-isolated - 3 dependencies)  
├─ payment-service (moderate coupling - 12 dependencies)
└─ notification-service (loosely coupled - 2 dependencies)

Level 2: Shared Libraries
├─ common-types (used by all services ✅)
├─ database-client (used by 3/4 services ⚠️)
└─ logger (inconsistent usage ❌)

🎯 Pattern Analysis:
✅ Good: Services communicate via message queues
✅ Good: Shared types prevent contract drift
⚠️  Concern: user-service has too many responsibilities
❌ Issue: Direct database coupling in payment-service
❌ Issue: Inconsistent error handling patterns

💡 Recommendations:
1. Split user-service into user-profile + user-auth
2. Add database abstraction layer for payment-service  
3. Standardize on common error handling middleware
```
```

## Troubleshooting

### Connection Issues

1. **Mimir not found**: Ensure mimir-server is in your PATH
   ```bash
   which mimir-server
   mimir-server --version
   ```

2. **Permission errors**: Check data directory permissions
   ```bash
   mkdir -p ~/.cache/mimir
   chmod 755 ~/.cache/mimir
   ```

3. **Import errors**: Reinstall mimir
   ```bash
   pip install --force-reinstall mimir
   ```

### Performance Issues

1. **Slow indexing**: Reduce concurrent workers
   ```json
   "env": {
     "MIMIR_MAX_WORKERS": "2"
   }
   ```

2. **Memory usage**: Limit cache size
   ```json
   "env": {
     "MIMIR_CACHE_SIZE": "100"
   }
   ```

### Debugging

Enable debug logging:

```json
{
  "mcpServers": {
    "mimir-repoindex": {
      "command": "mimir-server",
      "args": [],
      "env": {
        "MIMIR_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

Check logs in:
- **macOS/Linux**: `~/.cache/mimir/logs/`
- **Windows**: `%USERPROFILE%\.cache\mimir\logs\`

## Advanced Usage

### Custom Storage Location

```json
{
  "mcpServers": {
    "mimir-repoindex": {
      "command": "mimir-server", 
      "args": ["--storage-dir", "/custom/path"],
      "env": {
        "MIMIR_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Multiple Mimir Servers

You can configure multiple Mimir servers for different purposes:

```json
{
  "mcpServers": {
    "mimir-work": {
      "command": "mimir-server",
      "args": ["--storage-dir", "~/work-repos"],
      "env": {
        "MIMIR_LOG_LEVEL": "INFO"
      }
    },
    "mimir-personal": {
      "command": "mimir-server", 
      "args": ["--storage-dir", "~/personal-repos"],
      "env": {
        "MIMIR_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### With AI Enhancement

For enhanced AI features, add your Gemini API key:

```json
{
  "mcpServers": {
    "mimir-repoindex": {
      "command": "mimir-server",
      "args": [],
      "env": {
        "MIMIR_LOG_LEVEL": "INFO",
        "GOOGLE_API_KEY": "your-actual-api-key-here"
      }
    }
  }
}
```

## Security Considerations

1. **API Keys**: Store API keys securely, not in shared configurations
2. **File Permissions**: Ensure data directories have appropriate permissions
3. **Network Access**: Mimir runs locally and doesn't require network access except for AI features
4. **Repository Access**: Mimir only reads repositories, never modifies them

## Support

- **Documentation**: See the full documentation in the repository
- **Issues**: Report bugs at the GitHub repository
- **Configuration**: Refer to `ARCHITECTURE.md` for advanced configuration options

## Next Steps

After successful configuration:

1. **Try indexing a small repository** first to verify setup
2. **Experiment with different search queries** to understand capabilities  
3. **Use the ask_index tool** for complex code analysis questions
4. **Check the monitoring resources** to understand system performance
5. **Read ARCHITECTURE.md** for advanced features and configuration