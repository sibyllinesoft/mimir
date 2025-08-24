@echo off
REM Mimir Installation Script for Windows
REM Provides multiple installation methods with user choice

setlocal enabledelayedexpansion

REM Configuration
set "PROJECT_NAME=mimir"
set "REPO_URL=https://github.com/your-username/mimir"
set "PYTHON_MIN_VERSION=3.11"

REM Initialize variables
set "METHOD=pip"
set "PATH_ARG="
set "INSTALL_DEV=false"
set "PYTHON_CMD="

REM Color codes for output (if supported)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Function to display colored output
:log_info
echo [INFO] %~1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:log_warning
echo [WARNING] %~1
goto :eof

:log_error
echo [ERROR] %~1
goto :eof

REM Function to check if command exists
:command_exists
where %1 >nul 2>&1
goto :eof

REM Function to compare versions
:version_ge
REM Simple version comparison for Windows
for /f "tokens=1,2,3 delims=." %%a in ("%1") do set "v1_major=%%a" & set "v1_minor=%%b" & set "v1_patch=%%c"
for /f "tokens=1,2,3 delims=." %%a in ("%2") do set "v2_major=%%a" & set "v2_minor=%%b" & set "v2_patch=%%c"

if !v1_major! GTR !v2_major! exit /b 0
if !v1_major! LSS !v2_major! exit /b 1
if !v1_minor! GTR !v2_minor! exit /b 0
if !v1_minor! LSS !v2_minor! exit /b 1
if !v1_patch! GEQ !v2_patch! exit /b 0
exit /b 1

REM Function to check Python version
:check_python
call :log_info "Checking for Python >= %PYTHON_MIN_VERSION%..."

REM Try different Python commands
for %%p in (python python3 py) do (
    call :command_exists %%p
    if !errorlevel! equ 0 (
        for /f "tokens=2" %%v in ('%%p --version 2^>^&1') do (
            call :version_ge %%v %PYTHON_MIN_VERSION%
            if !errorlevel! equ 0 (
                set "PYTHON_CMD=%%p"
                call :log_success "Found compatible Python: %%p (%%v)"
                goto :python_found
            ) else (
                call :log_warning "Found Python %%p (%%v) but requires >= %PYTHON_MIN_VERSION%"
            )
        )
    )
)

call :log_error "Python >= %PYTHON_MIN_VERSION% not found"
call :log_info "Please install Python >= %PYTHON_MIN_VERSION% and try again"
call :log_info "Visit: https://www.python.org/downloads/"
exit /b 1

:python_found
goto :eof

REM Function to check system dependencies
:check_dependencies
call :log_info "Checking system dependencies..."

call :command_exists git
if !errorlevel! neq 0 (
    call :log_error "Git not found"
    call :log_info "Please install Git for Windows: https://git-scm.com/download/win"
    exit /b 1
)

call :log_success "All system dependencies found"
goto :eof

REM Function to install from PyPI
:install_from_pypi
call :log_info "Installing from PyPI..."

REM Upgrade pip
%PYTHON_CMD% -m pip install --upgrade pip
if !errorlevel! neq 0 (
    call :log_error "Failed to upgrade pip"
    exit /b 1
)

REM Install package
if "%INSTALL_DEV%"=="true" (
    %PYTHON_CMD% -m pip install "repoindex[dev,ui,test]"
) else (
    %PYTHON_CMD% -m pip install "repoindex[ui]"
)

if !errorlevel! neq 0 (
    call :log_error "Installation from PyPI failed"
    exit /b 1
)

call :log_success "Installed from PyPI"
goto :eof

REM Function to install from wheel
:install_from_wheel
call :log_info "Installing from wheel file..."

if not exist "%PATH_ARG%" (
    call :log_error "Wheel file not found: %PATH_ARG%"
    exit /b 1
)

REM Upgrade pip
%PYTHON_CMD% -m pip install --upgrade pip
if !errorlevel! neq 0 (
    call :log_error "Failed to upgrade pip"
    exit /b 1
)

REM Install wheel
%PYTHON_CMD% -m pip install "%PATH_ARG%"
if !errorlevel! neq 0 (
    call :log_error "Installation from wheel failed"
    exit /b 1
)

call :log_success "Installed from wheel: %PATH_ARG%"
goto :eof

REM Function to install from source
:install_from_source
call :log_info "Installing from source..."

if not exist "%PATH_ARG%" (
    call :log_error "Source directory not found: %PATH_ARG%"
    exit /b 1
)

pushd "%PATH_ARG%"

REM Upgrade pip and install build tools
%PYTHON_CMD% -m pip install --upgrade pip build
if !errorlevel! neq 0 (
    call :log_error "Failed to install build tools"
    popd
    exit /b 1
)

REM Install package
if "%INSTALL_DEV%"=="true" (
    %PYTHON_CMD% -m pip install -e ".[dev,ui,test]"
) else (
    %PYTHON_CMD% -m pip install -e ".[ui]"
)

if !errorlevel! neq 0 (
    call :log_error "Installation from source failed"
    popd
    exit /b 1
)

popd
call :log_success "Installed from source: %PATH_ARG%"
goto :eof

REM Function to install standalone executable
:install_standalone
call :log_info "Installing standalone executable..."

if not exist "%PATH_ARG%" (
    call :log_error "Executable not found: %PATH_ARG%"
    exit /b 1
)

REM Create install directory
set "INSTALL_DIR=%USERPROFILE%\.local\bin"
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

REM Copy executable
copy "%PATH_ARG%" "%INSTALL_DIR%\mimir-server.exe"
if !errorlevel! neq 0 (
    call :log_error "Failed to copy executable"
    exit /b 1
)

REM Add to PATH (user environment)
call :log_info "Adding %INSTALL_DIR% to user PATH..."
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "USER_PATH=%%b"
if not defined USER_PATH set "USER_PATH="

REM Check if already in PATH
echo !USER_PATH! | findstr /i "%INSTALL_DIR%" >nul
if !errorlevel! neq 0 (
    if defined USER_PATH (
        setx PATH "!USER_PATH!;%INSTALL_DIR%"
    ) else (
        setx PATH "%INSTALL_DIR%"
    )
    call :log_info "Added to PATH. Restart command prompt to use mimir-server"
) else (
    call :log_info "Directory already in PATH"
)

call :log_success "Installed standalone executable to: %INSTALL_DIR%\mimir-server.exe"
goto :eof

REM Function to install using Docker
:install_docker
call :log_info "Setting up Docker installation..."

call :command_exists docker
if !errorlevel! neq 0 (
    call :log_error "Docker not found"
    call :log_info "Please install Docker Desktop: https://docs.docker.com/desktop/windows/"
    exit /b 1
)

REM Create docker-compose file
set "COMPOSE_FILE=docker-compose.mimir.yml"

(
echo version: '3.8'
echo.
echo services:
echo   mimir-server:
echo     image: mimir-server:latest
echo     container_name: mimir-server
echo     restart: unless-stopped
echo     ports:
echo       - "8000:8000"
echo     volumes:
echo       - mimir-data:/app/data
echo       - mimir-cache:/app/cache
echo       - mimir-logs:/app/logs
echo     environment:
echo       - MIMIR_LOG_LEVEL=INFO
echo       - MIMIR_MAX_WORKERS=4
echo     healthcheck:
echo       test: ["CMD", "python", "-c", "import asyncio; from src.repoindex.mcp.server import MCPServer; print('Health check passed')"]
echo       interval: 30s
echo       timeout: 10s
echo       retries: 3
echo.
echo volumes:
echo   mimir-data:
echo   mimir-cache:
echo   mimir-logs:
) > "%COMPOSE_FILE%"

call :log_success "Created Docker Compose file: %COMPOSE_FILE%"
call :log_info "To start Mimir: docker-compose -f %COMPOSE_FILE% up -d"
call :log_info "To stop Mimir: docker-compose -f %COMPOSE_FILE% down"
goto :eof

REM Function to verify installation
:verify_installation
call :log_info "Verifying installation..."

if "%1"=="pip" goto :verify_pip
if "%1"=="wheel" goto :verify_pip
if "%1"=="source" goto :verify_pip
if "%1"=="standalone" goto :verify_standalone
if "%1"=="docker" goto :verify_docker
goto :eof

:verify_pip
where mimir-server >nul 2>&1
if !errorlevel! equ 0 (
    call :log_success "mimir-server installed successfully"
) else (
    call :log_error "mimir-server command not found in PATH"
    exit /b 1
)
goto :eof

:verify_standalone
if exist "%USERPROFILE%\.local\bin\mimir-server.exe" (
    call :log_success "Standalone executable installed successfully"
) else (
    call :log_error "Standalone executable not found"
    exit /b 1
)
goto :eof

:verify_docker
docker image inspect mimir-server:latest >nul 2>&1
if !errorlevel! equ 0 (
    call :log_success "Docker image available"
) else (
    call :log_warning "Docker image not found locally"
    call :log_info "You may need to build or pull the Docker image"
)
goto :eof

REM Function to show usage
:show_usage
echo.
echo Mimir Installation Script for Windows
echo.
echo USAGE:
echo     %~nx0 [OPTIONS]
echo.
echo OPTIONS:
echo     /method METHOD      Installation method (pip^|wheel^|source^|standalone^|docker)
echo     /path PATH          Path to wheel file, source directory, or executable
echo     /dev                Install development dependencies
echo     /help               Show this help message
echo.
echo INSTALLATION METHODS:
echo     pip                 Install from PyPI (default)
echo     wheel               Install from wheel file (requires /path)
echo     source              Install from source directory (requires /path)
echo     standalone          Install standalone executable (requires /path)
echo     docker              Create Docker Compose setup
echo.
echo EXAMPLES:
echo     %~nx0                                          # Install from PyPI
echo     %~nx0 /method wheel /path mimir-1.0.0.whl     # Install from wheel
echo     %~nx0 /method source /path .\mimir /dev       # Install from source with dev deps
echo     %~nx0 /method standalone /path mimir-server.exe # Install standalone executable
echo     %~nx0 /method docker                          # Set up Docker installation
echo.
goto :eof

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="/method" (
    set "METHOD=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="/path" (
    set "PATH_ARG=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="/dev" (
    set "INSTALL_DEV=true"
    shift
    goto :parse_args
)
if /i "%~1"=="/help" (
    call :show_usage
    exit /b 0
)
call :log_error "Unknown option: %~1"
call :show_usage
exit /b 1

:args_done
goto :eof

REM Main function
:main
call :parse_args %*

REM Validate method
if "%METHOD%"=="pip" goto :method_valid
if "%METHOD%"=="wheel" goto :method_valid
if "%METHOD%"=="source" goto :method_valid
if "%METHOD%"=="standalone" goto :method_valid
if "%METHOD%"=="docker" goto :method_valid

call :log_error "Invalid installation method: %METHOD%"
call :show_usage
exit /b 1

:method_valid

REM Check if path is required
if "%METHOD%"=="wheel" (
    if not defined PATH_ARG (
        call :log_error "Path is required for wheel installation"
        call :show_usage
        exit /b 1
    )
)
if "%METHOD%"=="source" (
    if not defined PATH_ARG (
        call :log_error "Path is required for source installation"
        call :show_usage
        exit /b 1
    )
)
if "%METHOD%"=="standalone" (
    if not defined PATH_ARG (
        call :log_error "Path is required for standalone installation"
        call :show_usage
        exit /b 1
    )
)

call :log_info "Starting Mimir installation..."
call :log_info "Installation method: %METHOD%"

REM Skip dependency checks for Docker
if not "%METHOD%"=="docker" (
    call :check_dependencies
    if !errorlevel! neq 0 exit /b 1
    
    call :check_python
    if !errorlevel! neq 0 exit /b 1
)

REM Install based on method
if "%METHOD%"=="pip" (
    call :install_from_pypi
    if !errorlevel! neq 0 exit /b 1
)
if "%METHOD%"=="wheel" (
    call :install_from_wheel
    if !errorlevel! neq 0 exit /b 1
)
if "%METHOD%"=="source" (
    call :install_from_source
    if !errorlevel! neq 0 exit /b 1
)
if "%METHOD%"=="standalone" (
    call :install_standalone
    if !errorlevel! neq 0 exit /b 1
)
if "%METHOD%"=="docker" (
    call :install_docker
    if !errorlevel! neq 0 exit /b 1
)

REM Verify installation
call :verify_installation %METHOD%
if !errorlevel! equ 0 (
    call :log_success "Mimir installation completed successfully!"
    
    echo.
    call :log_info "Next steps:"
    if "%METHOD%"=="docker" (
        call :log_info "1. Start Mimir: docker-compose -f docker-compose.mimir.yml up -d"
        call :log_info "2. Check status: docker-compose -f docker-compose.mimir.yml ps"
        call :log_info "3. View logs: docker-compose -f docker-compose.mimir.yml logs -f"
    ) else (
        call :log_info "1. Run 'mimir-server --help' to see available options"
        call :log_info "2. Start the server: mimir-server"
        call :log_info "3. Check the documentation for configuration options"
    )
) else (
    call :log_error "Installation verification failed"
    exit /b 1
)

goto :eof

REM Entry point
call :main %*
exit /b %errorlevel%