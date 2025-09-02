# Mimir Distribution System

This directory contains scripts for building, releasing, and validating distributions of the Mimir Deep Code Research System.

## Overview

The distribution system provides:

- **Multiple Distribution Formats**: Python wheel, source distribution, Docker images, standalone executables
- **Automated Building**: Single command builds all formats with proper validation
- **Release Management**: Version bumping, tagging, and publishing automation
- **Cross-Platform Support**: Works on Linux, macOS, and Windows
- **User-Friendly Installation**: Scripts for end users with multiple installation methods
- **Validation**: Comprehensive testing of all distribution formats

## Scripts

### Build Scripts

#### `build.py`
Main build automation script that creates all distribution formats.

```bash
# Build all formats
python scripts/build.py

# Build specific formats
python scripts/build.py --formats wheel sdist
python scripts/build.py --formats docker
python scripts/build.py --formats executable

# Skip environment validation (for CI)
python scripts/build.py --skip-validation
```

**Features:**
- Python wheel distribution
- Source distribution (sdist)
- Docker images with multi-stage builds
- Standalone executables via PyInstaller
- Checksums and build metadata
- Comprehensive error handling

### Release Management

#### `release.py`
Automated release process with version management.

```bash
# Release with version bump
python scripts/release.py minor --changes "Add new MCP features and Docker improvements"

# Dry run (show what would be done)
python scripts/release.py patch --changes "Bug fixes" --dry-run

# Skip GitHub release creation
python scripts/release.py major --changes "Breaking API changes" --skip-github

# Publish to PyPI
python scripts/release.py patch --changes "Hotfix" --publish-pypi

# Use test PyPI
python scripts/release.py patch --changes "Test release" --publish-pypi --pypi-repository testpypi
```

**Features:**
- Semantic version bumping (major.minor.patch)
- Automatic changelog updates
- Git tagging and pushing
- GitHub release creation with artifacts
- Optional PyPI publishing
- Comprehensive validation

### Installation Scripts

#### `install.sh` (Unix/Linux/macOS)
User-friendly installation script with multiple methods.

```bash
# Default installation from PyPI
./scripts/install.sh

# Install from wheel file
./scripts/install.sh --method wheel --path dist/repoindex-1.0.0-py3-none-any.whl

# Install from source with development dependencies
./scripts/install.sh --method source --path . --dev

# Install standalone executable
./scripts/install.sh --method standalone --path mimir-server

# Docker setup
./scripts/install.sh --method docker

# Show help
./scripts/install.sh --help
```

#### `install.bat` (Windows)
Windows equivalent of the Unix installation script.

```cmd
REM Basic installation
scripts\install.bat

REM Install from wheel
scripts\install.bat /method wheel /path repoindex-1.0.0-py3-none-any.whl

REM Install with development dependencies
scripts\install.bat /method source /path . /dev

REM Docker setup
scripts\install.bat /method docker
```

**Features:**
- Multiple installation methods (pip, wheel, source, standalone, Docker)
- Automatic dependency checking
- Environment validation
- User PATH management
- Desktop integration (Linux)
- Comprehensive error handling and user guidance

### Validation

#### `validate-distribution.py`
Comprehensive testing of all distribution formats.

```bash
# Validate all distributions
python scripts/validate-distribution.py

# Validate specific formats
python scripts/validate-distribution.py --tests wheel docker

# Build and validate
python scripts/validate-distribution.py --build-first

# Available tests: wheel, docker, sdist, executable, scripts, checksums
```

**Features:**
- Python wheel installation testing
- Docker image validation
- Source distribution build testing
- Standalone executable functionality testing
- Installation script syntax validation
- Checksum verification
- Detailed reporting

## Configuration Files

### `pyinstaller-config.spec`
PyInstaller configuration for building standalone executables with all dependencies included.

### `MANIFEST.in`
Source distribution manifest specifying which files to include/exclude.

### `.dockerignore`
Docker build context exclusions to optimize image size and security.

### `docker-compose.install.yml`
Complete Docker Compose setup for easy containerized deployment with optional UI and monitoring.

### `.env.example`
Comprehensive environment variable configuration template.

## Distribution Artifacts

When built, the system creates the following artifacts in the `dist/` directory:

```
dist/
├── repoindex-1.0.0-py3-none-any.whl          # Python wheel
├── repoindex-1.0.0.tar.gz                    # Source distribution
├── mimir-server_latest.tar.zip                # Docker image (compressed)
├── mimir-server-linux-x86_64.zip             # Standalone executable (Linux)
├── mimir-server-windows-x86_64.zip           # Standalone executable (Windows)
├── checksums.json                            # SHA256 checksums
├── SHA256SUMS                                 # Traditional checksum format
└── build-info.json                           # Build metadata
```

## Usage Workflows

### For Developers

```bash
# 1. Make changes to code
git add .
git commit -m "Add new features"

# 2. Build all distributions
python scripts/build.py

# 3. Validate distributions
python scripts/validate-distribution.py

# 4. Release (if validation passes)
python scripts/release.py minor --changes "Add new features with improved performance"
```

### For CI/CD

```bash
# In GitHub Actions or similar
python scripts/build.py --skip-validation
python scripts/validate-distribution.py
python scripts/release.py patch --changes "$RELEASE_NOTES" --skip-github
```

### For End Users

```bash
# Simple installation
bash <(curl -sSL https://raw.githubusercontent.com/your-username/mimir/main/scripts/install.sh)

# Or download and customize
wget https://github.com/your-username/mimir/releases/latest/download/install.sh
chmod +x install.sh
./install.sh --method docker  # Use Docker instead of pip
```

## Environment Variables

Key environment variables for the distribution system:

```bash
# Build configuration
BUILD_TARGET=production
BUILD_FORMATS=wheel,sdist,docker,executable

# Release configuration
GITHUB_TOKEN=your-github-token
PYPI_TOKEN=your-pypi-token

# Docker configuration
DOCKER_REGISTRY=your-registry.com
DOCKER_TAG_LATEST=true
```

## Dependencies

### Build Dependencies
- Python 3.11+
- Docker (for image builds)
- PyInstaller (for executables)
- build, twine (for Python packaging)
- git (for version management)

### System Dependencies
- Linux: build-essential, libffi-dev
- macOS: Xcode command line tools
- Windows: Visual Studio Build Tools

## Security Considerations

### Build Security
- All build artifacts are checksummed
- Docker images use multi-stage builds with minimal base images
- Standalone executables are built in isolated environments
- No secrets are embedded in distributions

### Installation Security
- Scripts validate checksums before installation
- HTTPS-only downloads
- User confirmation for system changes
- Minimal privilege requirements

### Release Security
- Signed Git tags for releases
- GitHub release verification
- PyPI publishing with API tokens (not passwords)
- Automated security scanning of dependencies

## Troubleshooting

### Common Build Issues

1. **PyInstaller fails on imports**
   - Check `pyinstaller-config.spec` for missing hidden imports
   - Add missing modules to the `hiddenimports` list

2. **Docker build fails**
   - Ensure Docker daemon is running
   - Check `.dockerignore` for excluded required files
   - Verify base image availability

3. **Wheel build fails**
   - Ensure all dependencies are properly declared in `pyproject.toml`
   - Check for missing C extension dependencies

### Common Installation Issues

1. **Permission errors**
   ```bash
   # Fix file permissions
   chmod +x scripts/install.sh
   
   # Install to user directory
   ./scripts/install.sh --method pip --user
   ```

2. **Docker issues**
   ```bash
   # Restart Docker
   sudo systemctl restart docker
   
   # Clean Docker system
   docker system prune -f
   ```

3. **Python version mismatch**
   ```bash
   # Use specific Python version
   python3.11 scripts/build.py
   ```

## Contributing

### Adding New Distribution Formats

1. Add build logic to `build.py`
2. Add validation logic to `validate-distribution.py`
3. Update installation scripts to support new format
4. Add documentation and examples

### Testing Changes

```bash
# Test build system
python scripts/build.py --formats wheel
python scripts/validate-distribution.py --tests wheel

# Test installation
./scripts/install.sh --method wheel --path dist/repoindex-1.0.0-py3-none-any.whl
```

### Release Process

1. Test all changes locally
2. Update documentation
3. Run full validation suite
4. Create pull request
5. After merge, create release using `release.py`

## Support

- **Issues**: Report problems at https://github.com/your-username/mimir/issues
- **Discussions**: Ask questions at https://github.com/your-username/mimir/discussions
- **Documentation**: Full docs at https://your-username.github.io/mimir/

The distribution system is designed to be robust, user-friendly, and maintainable. It provides multiple deployment options to suit different use cases while maintaining security and reliability standards.