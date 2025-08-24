Of course. Here is the requested `TODO.md` document, which synthesizes the detailed feedback into a structured action plan.

---

# TODO.md

## Big Picture

The primary goal of these tasks is to refine the Mimir project's already excellent foundation into a more robust, extensible, and enterprise-grade system. This is not a rewrite, but a strategic evolution focused on hardening security, improving architectural purity, and formalizing development practices. By addressing these items, we will enhance modularity for future features (like new pipeline stages or AI models), increase maintainability through clearer separation of concerns and centralized configuration, and ensure the system is secure by default, ready for production deployments and easier collaboration.

## Hints & Tips

Approach these changes incrementally, using dedicated feature branches for each major task in the workflow below. Always prioritize the critical security fix first. The project has a solid test suite; ensure that every code change is accompanied by corresponding unit or integration tests to prevent regressions. Some architectural changes, like separating the `QueryEngine`, might be considered breaking changes to the internal API; be mindful of this and consider semantic versioning if external consumers exist. When centralizing configuration, migrate settings piece by piece rather than in one large, disruptive change. The XML workflow is a meticulous guide, but feel free to adapt it if a better implementation path becomes clear during development.

## Refinement Workflow

```xml
<workflow name="Mimir Refinement & Hardening Plan">
    <phase id="1" title="Critical Security Hardening">
        <task title="Fix Hardcoded Cryptographic Salt" priority="critical">
            <details>
                This is the most critical task. A hardcoded salt negates the security benefits of password hashing and must be fixed immediately.
            </details>
            <steps>
                <step file="src/repoindex/security/crypto.py">
                    Modify `SecretsManager._initialize_encryption` and `_save_secrets`. Instead of a hardcoded salt, generate a new, unique, 16-byte random salt for each secret using `secrets.token_bytes(16)`.
                </step>
                <step file="src/repoindex/security/crypto.py">
                    Store the unique salt prepended to the encrypted data in the secrets file. The salt is not secret and can be stored in plaintext.
                </step>
                <step file="src/repoindex/security/crypto.py">
                    Modify `SecretsManager._load_secrets` to first read the 16-byte salt from the file, then use that salt along with the password to derive the key for decrypting the remaining data.
                </step>
                <step file="scripts/migration/migrate_secrets.py">
                    Create a one-time migration script for existing users. This script should prompt for the old password, decrypt all secrets using the old hardcoded salt method, and then re-encrypt them using the new method with unique salts.
                </step>
                <step file="tests/security/test_crypto.py">
                    Update tests to verify that salts are unique and that the new encryption/decryption process works correctly. Add a test for the migration script.
                </step>
                <step file="SECURITY.md">
                    Update documentation to reflect the new, more secure key derivation process and mention the migration path.
                </step>
            </steps>
        </task>
    </phase>

    <phase id="2" title="High-Impact Architectural Refinement">
        <task title="Separate Indexing from Querying Concerns" priority="high">
            <details>
                The `IndexingPipeline` currently handles both building and querying indexes. Separating these responsibilities will lead to a cleaner, more scalable architecture.
            </details>
            <steps>
                <step file="src/repoindex/query_engine.py">
                    Create a new file and a `QueryEngine` class. This class will be responsible for loading a completed index from disk.
                </step>
                <step file="src/repoindex/pipeline/run.py">
                    Move the `search()` and `ask()` methods from the `IndexingPipeline` class into the new `QueryEngine` class. Refactor them to load the necessary artifacts (vector index, serena graph, etc.) from a specified index directory.
                </step>
                <step file="src/repoindex/pipeline/run.py">
                    Simplify the `IndexingPipeline` class to focus solely on the six-stage process of building the index artifacts and writing them to disk.
                </step>
                <step file="src/repoindex/mcp/server.py">
                    Refactor the `MCPServer`. It should now use the `IndexingPipeline` to *create* indexes and a separate instance of the `QueryEngine` to serve `search_repo` and `ask_index` requests against completed indexes.
                </step>
                <step file="tests/">
                    Update all relevant tests. Create new test files for `QueryEngine` and adjust `IndexingPipeline` tests to only cover the build process.
                </step>
                <step file="ARCHITECTURE.md">
                    Update the architecture document to reflect this new separation of concerns.
                </step>
            </steps>
        </task>
    </phase>

    <phase id="3" title="Medium-Impact Improvements (Modularity & Maintainability)">
        <task title="Centralize Configuration Management" priority="medium">
            <details>
                Configuration is spread across environment variables, CLI args, and hardcoded defaults. Centralizing this will improve clarity and ease of management.
            </details>
            <steps>
                <step file="src/repoindex/config.py">
                    Create a new file to house the global configuration.
                </step>
                <step file="src/repoindex/config.py">
                    Define a `Settings` class using Pydantic's `BaseSettings`. This class will define all configuration variables and their default values, automatically loading overrides from environment variables or a `.env` file.
                </step>
                <step>
                    Gradually refactor the codebase (starting with `security/config.py` and `main_secure.py`) to import a single instance of this `Settings` object instead of using `os.getenv` or custom config loading logic directly.
                </step>
                <step file="CONFIGURATION.md">
                    Create a new `CONFIGURATION.md` file that documents every setting, its environment variable, and its purpose.
                </step>
            </steps>
        </task>
        <task title="Formalize Dependency Management Workflow" priority="medium">
            <details>
                Automate the synchronization of abstract and pinned dependencies to ensure reproducible environments.
            </details>
            <steps>
                <step file="pyproject.toml">
                    Ensure all direct dependencies are listed here with compatible version specifiers (e.g., `fastapi>=0.116.0,<0.117.0`).
                </step>
                <step>
                    Generate a pinned `requirements.lock.txt` for development using the command: `uv pip compile pyproject.toml --all-extras -o requirements.lock.txt`.
                </step>
                <step file="README.md, DEVELOPMENT.md">
                    Update documentation to instruct developers to use `uv sync requirements.lock.txt` for installation.
                </step>
                <step file="Dockerfile">
                    Update the Dockerfile build process. The first step should be `uv pip compile pyproject.toml -o requirements.txt` (without dev extras), followed by `uv pip install -r requirements.txt`. This ensures production builds are also pinned and reproducible.
                </step>
            </steps>
        </task>
        <task title="Refactor MCP Server Code Duplication (DRY)" priority="low">
            <details>
                The tool registration logic is duplicated between the standard and secure MCP servers. This should be refactored.
            </details>
            <steps>
                <step file="src/repoindex/mcp/server.py">
                    Define the tool handlers (`_ensure_repo_index`, `_search_repo`, etc.) as standard methods in the base `MCPServer`.
                </step>
                <step file="src/repoindex/mcp/secure_server.py">
                    In `SecureMCPServer`, instead of re-implementing `call_tool`, programmatically wrap the base server's tool handler methods with the `security_middleware`.
                </step>
                <step file="tests/">
                    Run the full integration test suite for both the standard and secure servers to ensure that all tools continue to function correctly after the refactor.
                </step>
            </steps>
        </task>
    </phase>

    <phase id="4" title="Extensibility & Polish">
        <task title="Create Abstract Pipeline Stage" priority="low">
            <details>
                Formalize the concept of a pipeline stage to make the pipeline more modular and extensible.
            </details>
            <steps>
                <step file="src/repoindex/pipeline/stage.py">
                    Create a new file defining an abstract base class `PipelineStage(ABC)` with an abstract `execute` method.
                </step>
                <step>
                    Refactor each of the current `_stage_*` functions in `run.py` into its own class (e.g., `AcquireStage`, `RepoMapperStage`) that inherits from `PipelineStage`.
                </step>
                <step file="src/repoindex/pipeline/run.py">
                    Modify the `IndexingPipeline` to accept a list of `PipelineStage` objects in its constructor and iterate through them in its `_execute_pipeline` method.
                </step>
            </steps>
        </task>
        <task title="Create Abstract LLM Adapter" priority="low">
            <details>
                Decouple the `ask_index` functionality from Gemini to allow for other AI models in the future.
            </details>
            <steps>
                <step file="src/repoindex/pipeline/llm_adapter.py">
                    Create a new file with an abstract base class `LLMAdapter(ABC)` that defines the interface for AI providers (e.g., `synthesize_answer`).
                </step>
                <step file="src/repoindex/pipeline/gemini.py">
                    Refactor `GeminiAdapter` to inherit from and implement the `LLMAdapter` interface.
                </step>
                <step file="src/repoindex/pipeline/ask_index.py">
                    Modify `SymbolGraphNavigator` to accept an instance of `LLMAdapter` in its constructor, using it for the synthesis step.
                </step>
            </steps>
        </task>
        <task title="Add Architectural Diagrams" priority="low">
            <details>
                Visual diagrams will greatly improve the clarity of the documentation.
            </details>
            <steps>
                <step>
                    Create a high-level data flow diagram using Mermaid.js syntax that shows the six pipeline stages and the data passed between them.
                </step>
                <step file="ARCHITECTURE.md">
                    Embed the Mermaid diagram into the `ARCHITECTURE.md` file.
                </step>
            </steps>
        </task>
        <task title="Handle Platform-Specific Code" priority="low">
            <details>
                The `resource` module used for sandboxing is Unix-only and will break on native Windows.
            </details>
            <steps>
                 <step file="src/repoindex/security/sandbox.py">
                    In `ResourceLimiter.set_process_limits`, add a check for `os.name != 'posix'`. If true, log a warning that resource limiting is disabled and return early.
                 </step>
                 <step file="README.md">
                    Add a note in the prerequisites section that full security features (like sandboxing) are only available on Unix-like systems (Linux, macOS, WSL2).
                 </step>
            </steps>
        </task>
    </phase>
</workflow>
```