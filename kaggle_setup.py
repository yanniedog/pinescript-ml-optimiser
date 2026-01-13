#!/usr/bin/env python3
"""
Kaggle Cloud Setup Script for Pine Script ML Optimizer
This script sets up the environment, installs dependencies, configures ngrok, and launches the application

Python Version Requirements:
- Recommended: Python 3.8 - 3.11
- Python 3.12 is supported but some packages will be skipped automatically
- Python < 3.8 is not recommended (some dependencies may not work)

To set Python version in Kaggle:
1. Go to Settings (gear icon) in the notebook
2. Under Environment, select Python version
3. Recommended: Python 3.10 or 3.11
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
import time

# Logging setup - will be initialized in setup_logging()
_LOG_FILE = None
_LOG_FILE_PATH = None
_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None

# Disable IPython's enhanced traceback to avoid AttributeError issues in Kaggle
try:
    import IPython  # type: ignore[import-untyped]
    # Set IPython to use plain Python traceback
    if hasattr(IPython, 'get_ipython'):
        ipython = IPython.get_ipython()
        if ipython is not None:
            ipython.magic('xmode Plain')
except (ImportError, AttributeError):
    pass

# Configuration
REPO_URL = "https://github.com/yanniedog/pinescript-ml-optimiser"
REPO_DIR = "pinescript-ml-optimiser"
NGROK_AUTHTOKEN = "1tve5X0Aea2r2LKv82JymsjmSD9_4Xp7L4YWxE7RuCSU9bdHZ"
NGROK_PORT = 9107

# Global variable to keep ngrok tunnel alive
_ngrok_tunnel = None


def run_command(cmd, check=True, shell=False, capture_output=False):
    """Run a shell command and return the result."""
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()
    
    print(f"  Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        # When check=False, we still want to capture output to see errors
        capture = capture_output or not check
        result = subprocess.run(
            cmd,
            check=check,
            shell=shell,
            capture_output=capture,
            text=True
        )
        if capture_output:
            return result.stdout.strip()
        # If we captured output but check=False, show errors if any
        if capture and not check and result.returncode != 0:
            if result.stderr:
                # Only show stderr if it's not too verbose (avoid spam)
                stderr_lines = result.stderr.strip().split('\n')
                if len(stderr_lines) <= 10:
                    print(f"  ⚠ Command returned non-zero exit code: {result.returncode}")
                else:
                    print(f"  ⚠ Command returned non-zero exit code: {result.returncode}")
                    print(f"  (Error output suppressed - too verbose)")
        return result
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Command failed: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"  stdout: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  stderr: {e.stderr}")
        if check:
            sys.exit(1)
        return None


def step1_clone_repo():
    """Step 1: Clone the repository."""
    global _LOG_FILE_PATH  # Declare global at the start
    
    print("[1/5] Cloning repository...")
    
    repo_path = Path(REPO_DIR)
    if repo_path.exists():
        print(f"  Directory {REPO_DIR} already exists. Removing it...")
        shutil.rmtree(repo_path)
    
    run_command(["git", "clone", REPO_URL, REPO_DIR])
    print("  ✓ Repository cloned successfully")
    
    # Move logfile to repo directory after cloning
    new_log_path = None
    if repo_path.exists():
        print(f"  Moving logfile to repo directory: {repo_path}")
        new_log_path = move_logfile_to_repo(repo_path)
        if new_log_path:
            print(f"  ✓ Logfile moved to: {new_log_path}")
            # Verify the file exists
            if Path(new_log_path).exists():
                print(f"  ✓ Logfile verified at: {new_log_path}")
            else:
                print(f"  ⚠ WARNING: Logfile path reported but file not found: {new_log_path}")
        else:
            print(f"  ⚠ Could not move logfile to repo directory")
            # Check if original logfile still exists
            if _LOG_FILE_PATH and Path(_LOG_FILE_PATH).exists():
                print(f"  ℹ Original logfile still at: {_LOG_FILE_PATH}")
    
    print()
    return new_log_path  # Return the new log path


def step2_install_dependencies():
    """Step 2: Install Python dependencies."""
    print("[2/5] Installing Python dependencies...")
    
    repo_path = Path(REPO_DIR).resolve()  # Use absolute path
    if not repo_path.exists():
        print(f"  ERROR: Repository directory {repo_path} does not exist!")
        sys.exit(1)
    
    original_cwd = Path.cwd()
    os.chdir(repo_path)
    
    try:
        # Try requirements_fixed.txt first, fallback to requirements.txt
        req_fixed = repo_path / "requirements_fixed.txt"
        req_standard = repo_path / "requirements.txt"
        
        # Packages to skip on Python 3.12/Kaggle (not needed or not available)
        skip_packages = ["talib-binary", "pickle5"]
        
        # Check Python version for compatibility fixes
        python_version = sys.version_info
        is_python_312 = python_version.major == 3 and python_version.minor >= 12
        
        # Try to install requirements, handling problematic packages gracefully
        if req_fixed.exists():
            print("  Using requirements_fixed.txt")
            # Create a filtered requirements file from the start
            temp_req = repo_path / "requirements_temp.txt"
            try:
                with open(req_fixed, "r") as f_in, open(temp_req, "w") as f_out:
                    skipped_count = 0
                    for line in f_in:
                        # Skip problematic packages that don't work on Python 3.12/Kaggle
                        line_lower = line.lower().strip()
                        if any(skip in line_lower for skip in skip_packages):
                            if line.strip() and not line.strip().startswith("#"):
                                print(f"    Skipping: {line.strip()}")
                                skipped_count += 1
                            continue
                        f_out.write(line)
                
                if skipped_count > 0:
                    print(f"  Filtered out {skipped_count} package(s) not compatible with this environment")
                
                # For Python 3.12, install compatible versions first to avoid import errors
                if is_python_312:
                    print("  Installing Python 3.12 compatible package versions...")
                    # Upgrade NumPy to a version compatible with Python 3.12 (fixes _center import error)
                    # NumPy 2.x has breaking changes, so we stick with 1.x for compatibility
                    print("    Upgrading NumPy for Python 3.12 compatibility...")
                    run_command(
                        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "numpy>=1.26.0,<2.0.0"],
                        check=False
                    )
                    # Upgrade Optuna to a version compatible with Python 3.12 (fixes _INTEGRATION_IMPORT_ERROR_TEMPLATE error)
                    print("    Upgrading Optuna for Python 3.12 compatibility...")
                    run_command(
                        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "optuna>=3.5.0"],
                        check=False
                    )
                
                # Try installing filtered requirements
                result = run_command(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", str(temp_req)],
                    check=False
                )
                if result and result.returncode == 0:
                    print("  ✓ Dependencies installed successfully")
                else:
                    print("  ⚠ Some dependencies may have failed, but continuing anyway...")
            finally:
                # Clean up temp file
                if temp_req.exists():
                    temp_req.unlink()
        elif req_standard.exists():
            print("  Using requirements.txt")
            result = run_command(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_standard)],
                check=False
            )
            if result and result.returncode == 0:
                print("  ✓ Dependencies installed successfully")
            else:
                print("  ⚠ Some packages may have failed, but continuing...")
        else:
            print("  ERROR: No requirements file found!")
            sys.exit(1)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    print()


def step3_setup_ngrok():
    """Step 3: Setup ngrok."""
    print("[3/5] Setting up ngrok...")
    
    # Install pyngrok if not already installed
    try:
        import pyngrok  # type: ignore[import-untyped]
    except ImportError:
        print("  Installing pyngrok...")
        run_command([sys.executable, "-m", "pip", "install", "-q", "pyngrok>=5.0.0"])
        import pyngrok  # type: ignore[import-untyped]
    
    # Aggressively stop any existing ngrok tunnels and processes
    print("  Stopping any existing ngrok tunnels and processes...")
    try:
        from pyngrok import ngrok  # type: ignore[import-untyped]
        import pyngrok.ngrok  # type: ignore[import-untyped]
        
        # Method 1: Use pyngrok's kill function
        try:
            pyngrok.ngrok.kill()
            print("    ✓ Stopped existing ngrok tunnels (pyngrok)")
        except Exception as e:
            print(f"    ℹ pyngrok.kill() result: {e}")
        
        # Method 2: Kill ngrok processes via system commands
        import psutil
        killed_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'ngrok' in proc.info['name'].lower():
                    proc.kill()
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if killed_count > 0:
            print(f"    ✓ Killed {killed_count} ngrok process(es)")
        
        # Wait a moment for processes to fully terminate
        time.sleep(2)
        
    except ImportError:
        # psutil not available, try without it
        try:
            from pyngrok import ngrok  # type: ignore[import-untyped]
            import pyngrok.ngrok  # type: ignore[import-untyped]
            pyngrok.ngrok.kill()
            time.sleep(2)
        except Exception:
            pass
    except Exception as e:
        print(f"    ℹ Could not stop tunnels: {e}")
    
    # Configure ngrok authtoken
    print("  Configuring ngrok authtoken...")
    try:
        from pyngrok import ngrok  # type: ignore[import-untyped]
        import pyngrok.ngrok  # type: ignore[import-untyped]
        
        # Try to set authtoken - this may trigger ngrok download
        try:
            pyngrok.ngrok.set_auth_token(NGROK_AUTHTOKEN)
            print("    ✓ Ngrok authtoken configured")
        except Exception as token_error:
            # If setting authtoken fails due to download issue, try to download ngrok manually
            error_str = str(token_error).lower()
            if "download" in error_str or "noneType" in error_str or "not callable" in error_str:
                print(f"    ⚠ Ngrok download issue detected: {token_error}")
                print("    ℹ Attempting to download ngrok manually...")
                try:
                    # Try to download ngrok directly
                    import urllib.request
                    ngrok_url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip"
                    ngrok_zip = "/tmp/ngrok.zip"
                    ngrok_bin = "/tmp/ngrok"
                    
                    # Download ngrok
                    urllib.request.urlretrieve(ngrok_url, ngrok_zip)
                    
                    # Extract it
                    import zipfile
                    with zipfile.ZipFile(ngrok_zip, 'r') as zip_ref:
                        zip_ref.extractall("/tmp")
                    
                    # Make it executable
                    os.chmod(ngrok_bin, 0o755)
                    
                    # Set the ngrok path in pyngrok config
                    pyngrok.config.DEFAULT_CONFIG_PATH = ngrok_bin
                    
                    # Set the authtoken
                    pyngrok.ngrok.set_auth_token(NGROK_AUTHTOKEN)
                    print("    ✓ Ngrok downloaded and authtoken configured")
                except Exception as download_error:
                    print(f"    ⚠ Manual download also failed: {download_error}")
                    print(f"    ℹ Continuing without ngrok - application will run locally only")
                    return None
            else:
                # Re-raise if it's a different error
                raise token_error
    except Exception as e:
        print(f"    ⚠ Failed to configure ngrok: {e}")
        print(f"    ℹ Continuing without ngrok - application will run locally only")
        return None
    
    # Check if there's already a tunnel on the desired port
    print(f"  Checking for existing tunnel on port {NGROK_PORT}...")
    try:
        from pyngrok import ngrok  # type: ignore[import-untyped]
        import pyngrok.ngrok  # type: ignore[import-untyped]
        
        # Get all active tunnels
        tunnels = pyngrok.ngrok.get_tunnels()
        existing_tunnel = None
        for tunnel in tunnels:
            if tunnel.config.get('addr') == f'localhost:{NGROK_PORT}' or tunnel.config.get('addr') == str(NGROK_PORT):
                existing_tunnel = tunnel
                break
        
        if existing_tunnel:
            print(f"    ℹ Found existing tunnel on port {NGROK_PORT}")
            public_url = existing_tunnel.public_url
            print(f"    ✓ Reusing existing ngrok tunnel!")
            print(f"    Public URL: {public_url}")
            
            # Keep tunnel alive by writing URL to file
            with open("ngrok_url.txt", "w") as f:
                f.write(public_url)
            
            return public_url
    except Exception as e:
        print(f"    ℹ Could not check for existing tunnels: {e}")
    
    # Start new ngrok tunnel
    print(f"  Starting new ngrok tunnel on port {NGROK_PORT}...")
    try:
        from pyngrok import ngrok  # type: ignore[import-untyped]
        import pyngrok.ngrok  # type: ignore[import-untyped]
        
        # Ensure authtoken is set
        pyngrok.ngrok.set_auth_token(NGROK_AUTHTOKEN)
        
        # Start tunnel - store globally to keep it alive
        global _ngrok_tunnel
        _ngrok_tunnel = ngrok.connect(NGROK_PORT, "http")
        public_url = _ngrok_tunnel.public_url
        
        print(f"    ✓ Ngrok tunnel established!")
        print(f"    Public URL: {public_url}")
        print(f"    The Plotly dashboard will be accessible at: {public_url}")
        
        # Keep tunnel alive by writing URL to file
        with open("ngrok_url.txt", "w") as f:
            f.write(public_url)
        
        return public_url
    except Exception as e:
        error_str = str(e)
        if "ERR_NGROK_108" in error_str or "simultaneous" in error_str.lower():
            print(f"    ⚠ Ngrok error: Multiple sessions detected")
            print(f"    ℹ This usually means another ngrok session is active elsewhere")
            print(f"    ℹ Attempting to continue without ngrok tunnel...")
            print(f"    ℹ You may need to:")
            print(f"       1. Close other ngrok sessions at: https://dashboard.ngrok.com/agents")
            print(f"       2. Wait a few minutes and try again")
            print(f"       3. Or upgrade to a paid ngrok plan")
            print()
            # Continue without ngrok - the app might still work locally
            return None
        else:
            print(f"    ERROR: Failed to start ngrok tunnel: {e}")
            print(f"    ℹ The application may still work, but won't be accessible externally")
            return None


def step4_display_info(public_url=None):
    """Step 4: Display ngrok information."""
    print("[4/5] Ngrok Tunnel Information:")
    
    if public_url:
        print(f"  Public URL: {public_url}")
        print(f"  Port: {NGROK_PORT}")
        print(f"  ✓ Ngrok tunnel is active - dashboard accessible externally")
    else:
        ngrok_url_file = Path("ngrok_url.txt")
        if ngrok_url_file.exists():
            with open(ngrok_url_file, "r") as f:
                url = f.read().strip()
                print(f"  Public URL: {url}")
                print(f"  Port: {NGROK_PORT}")
                print(f"  ✓ Using existing ngrok tunnel")
        else:
            print("  ⚠ Ngrok tunnel not available")
            print(f"  Port: {NGROK_PORT} (local only)")
            print("  ℹ The application will run locally but won't be accessible externally")
            print("  ℹ To fix: Close other ngrok sessions at https://dashboard.ngrok.com/agents")
    
    print()


def step5_launch_app():
    """Step 5: Launch the application."""
    print("[5/5] Launching application...")
    print("=" * 50)
    print("Starting Pine Script ML Optimizer...")
    print("=" * 50)
    print()
    print("Note: The application is now running interactively.")
    print("You can interact with it as you would in a normal terminal.")
    print("Press Ctrl+C to exit.")
    print()
    
    # Change to the repo directory if not already there
    repo_path = Path(REPO_DIR).resolve()
    original_cwd = Path.cwd()
    
    if Path.cwd() != repo_path:
        os.chdir(repo_path)
    
    # Add repo to Python path so we can import modules
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    
    try:
        # Directly import and run instead of using subprocess
        # This works better for interactive input in Kaggle notebooks
        try:
            # Try to import and run the interactive optimizer directly
            from interactive_optimizer import main as run_optimizer
            run_optimizer()
        except ImportError as import_err:
            # If direct import fails, try using launch.py's approach
            print(f"  ℹ Direct import failed, trying launch.py approach: {import_err}")
            from launch import main as launch_main
            launch_main()
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
        os.chdir(original_cwd)
        sys.exit(0)
    except EOFError:
        # EOFError can occur in notebook environments when input() is called
        print("\n\n⚠ Input stream ended (EOF).")
        print("  This can happen in some notebook environments.")
        print("  Try running the script in a different way or check your environment.")
        os.chdir(original_cwd)
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Failed to launch application: {e}")
        import traceback
        # Use format_exc to avoid IPython traceback issues
        try:
            print(traceback.format_exc())
        except Exception:
            # Fallback if traceback formatting fails
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
        os.chdir(original_cwd)
        sys.exit(1)
    finally:
        # Restore original directory
        os.chdir(original_cwd)


def check_python_version():
    """Check Python version and warn if incompatible."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python version: {version_str}")
    
    # Recommended: Python 3.8 - 3.11 (3.12 works but some packages may have issues)
    if version.major != 3:
        print("  ⚠ WARNING: Python 3.x is required!")
        return False
    elif version.minor < 8:
        print("  ⚠ WARNING: Python 3.8+ is recommended (some packages may not work)")
        return False
    elif version.minor >= 12:
        print("  ℹ Python 3.12 detected - some packages will be skipped (expected)")
    else:
        print(f"  ✓ Python {version.major}.{version.minor} is compatible")
    
    print()
    return True


def setup_logging(repo_dir: Path = None):
    """Setup logging to capture all output to a logfile.
    
    Args:
        repo_dir: If provided, create logfile in repo_dir/logs/. Otherwise, use current directory.
    """
    global _LOG_FILE, _LOG_FILE_PATH, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    
    try:
        # Determine log directory
        if repo_dir and repo_dir.exists():
            log_dir = repo_dir / "logs"
        else:
            # Use current directory initially, will move to repo after cloning
            log_dir = Path("logs")
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"kaggle_setup_{timestamp}.log"
        
        # Create a simple file logger
        _LOG_FILE = open(log_path, "w", encoding="utf-8")
        _LOG_FILE_PATH = log_path  # Store the path for later reference
        
        # Store original stdout/stderr
        _ORIGINAL_STDOUT = sys.stdout
        _ORIGINAL_STDERR = sys.stderr
        
        class LogWriter:
            def __init__(self, stream, log_file):
                self.stream = stream
                self.log_file = log_file
            
            def write(self, data):
                if data:
                    self.stream.write(data)
                    if self.log_file:
                        self.log_file.write(data)
                        self.log_file.flush()
            
            def flush(self):
                self.stream.flush()
                if self.log_file:
                    self.log_file.flush()
            
            def isatty(self):
                return self.stream.isatty()
            
            def fileno(self):
                return self.stream.fileno()
        
        sys.stdout = LogWriter(_ORIGINAL_STDOUT, _LOG_FILE)
        sys.stderr = LogWriter(_ORIGINAL_STDERR, _LOG_FILE)
        
        # Write initial message to confirm logfile is working
        _LOG_FILE.write(f"Logfile created at: {log_path}\n")
        _LOG_FILE.write(f"Working directory: {Path.cwd()}\n")
        _LOG_FILE.flush()
        
        return str(log_path)
    except Exception as e:
        # If logging setup fails, continue without logging
        error_msg = f"  ⚠ Could not setup logfile: {e}"
        if _ORIGINAL_STDERR:
            _ORIGINAL_STDERR.write(error_msg + "\n")
        else:
            sys.stderr.write(error_msg + "\n")
        import traceback
        if _ORIGINAL_STDERR:
            traceback.print_exc(file=_ORIGINAL_STDERR)
        else:
            traceback.print_exc()
        return None


def move_logfile_to_repo(repo_dir: Path):
    """Move the current logfile to the repo's logs directory."""
    global _LOG_FILE, _LOG_FILE_PATH
    
    if not _LOG_FILE or not _LOG_FILE_PATH:
        return None
    
    try:
        # Get current logfile path
        current_log_path = Path(_LOG_FILE_PATH)
        
        # Create logs directory in repo
        repo_log_dir = repo_dir / "logs"
        repo_log_dir.mkdir(parents=True, exist_ok=True)
        
        # New logfile path in repo
        new_log_path = repo_log_dir / current_log_path.name
        
        # Close current logfile
        _LOG_FILE.flush()
        _LOG_FILE.close()
        
        # Move the file to repo directory
        if current_log_path.exists():
            if new_log_path.exists():
                # Append to existing file if it exists
                with open(current_log_path, "r", encoding="utf-8") as src:
                    with open(new_log_path, "a", encoding="utf-8") as dst:
                        dst.write(src.read())
                current_log_path.unlink()
            else:
                # Move the file
                current_log_path.rename(new_log_path)
        
        # Reopen logfile in new location (append mode to continue logging)
        _LOG_FILE = open(new_log_path, "a", encoding="utf-8")
        _LOG_FILE_PATH = new_log_path  # Update the stored path
        
        # Update the LogWriter to use the new file
        if hasattr(sys.stdout, 'log_file'):
            sys.stdout.log_file = _LOG_FILE
        if hasattr(sys.stderr, 'log_file'):
            sys.stderr.log_file = _LOG_FILE
        
        return str(new_log_path)
    except Exception as e:
        print(f"  ⚠ Could not move logfile to repo: {e}")
        # Try to reopen in original location
        try:
            if _LOG_FILE_PATH and Path(_LOG_FILE_PATH).exists():
                _LOG_FILE = open(_LOG_FILE_PATH, "a", encoding="utf-8")
                if hasattr(sys.stdout, 'log_file'):
                    sys.stdout.log_file = _LOG_FILE
                if hasattr(sys.stderr, 'log_file'):
                    sys.stderr.log_file = _LOG_FILE
        except Exception:
            pass
        return None


def main():
    """Main setup function."""
    # Declare globals at the start of the function (before any use)
    global _LOG_FILE, _LOG_FILE_PATH
    
    # Setup logging first (before any output) - will be moved to repo after cloning
    log_path = setup_logging()
    
    print("=" * 50)
    print("Pine Script ML Optimizer - Kaggle Setup")
    print("=" * 50)
    if log_path:
        abs_log_path = Path(log_path).resolve()
        print(f"Logfile: {abs_log_path}")
        print(f"  (Absolute path: {abs_log_path})")
        print("  (All output is being saved to this file)")
        print("  (Will be moved to pinescript-ml-optimiser/logs/ after cloning)")
    else:
        print("  ⚠ Logging not available - output will only be shown on screen")
    print()
    
    # Check Python version first
    check_python_version()
    
    try:
        # Clone repo and get updated log path
        new_log_path = step1_clone_repo()
        if new_log_path:
            log_path = new_log_path  # Update log_path to the new location
        
        step2_install_dependencies()
        public_url = step3_setup_ngrok()
        step4_display_info(public_url)
        step5_launch_app()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        # Get current log path from global
        current_log = _LOG_FILE_PATH if _LOG_FILE_PATH else log_path
        if current_log:
            abs_log_path = Path(current_log).resolve()
            print(f"  Logfile saved: {abs_log_path}")
            print(f"  (Absolute path: {abs_log_path})")
            if abs_log_path.exists():
                size = abs_log_path.stat().st_size
                print(f"  (File size: {size} bytes)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: Setup failed: {e}")
        import traceback
        # Use format_exc to avoid IPython traceback issues
        try:
            print(traceback.format_exc())
        except Exception:
            # Fallback if traceback formatting fails
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
        # Get current log path from global
        current_log = _LOG_FILE_PATH if _LOG_FILE_PATH else log_path
        if current_log:
            abs_log_path = Path(current_log).resolve()
            print(f"\n  Logfile saved: {abs_log_path}")
            print(f"  (Absolute path: {abs_log_path})")
            if abs_log_path.exists():
                size = abs_log_path.stat().st_size
                print(f"  (File size: {size} bytes)")
        sys.exit(1)
    finally:
        # Close log file if it was opened and show final location
        # Note: _LOG_FILE and _LOG_FILE_PATH are already declared as global at function start
        if _LOG_FILE:
            try:
                _LOG_FILE.flush()
                _LOG_FILE.close()
            except Exception:
                pass
        
        # Print final logfile location
        if _LOG_FILE_PATH:
            final_log_path = Path(_LOG_FILE_PATH).resolve()
            print("\n" + "=" * 50)
            print("LOGFILE LOCATION:")
            print("=" * 50)
            print(f"  Path: {final_log_path}")
            print(f"  Absolute: {final_log_path}")
            if final_log_path.exists():
                size = final_log_path.stat().st_size
                print(f"  Size: {size} bytes")
                print(f"  ✓ Logfile exists and is accessible")
            else:
                print(f"  ⚠ WARNING: Logfile path reported but file not found!")
            print("=" * 50)


if __name__ == "__main__":
    main()
