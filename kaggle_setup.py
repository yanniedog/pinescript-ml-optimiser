#!/usr/bin/env python3
"""
Kaggle Cloud Setup Script for Pine Script ML Optimizer
This script sets up the environment, installs dependencies, configures ngrok, and launches the application
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
import time

# Configuration
REPO_URL = "https://github.com/yanniedog/pinescript-ml-optimiser"
REPO_DIR = "pinescript-ml-optimiser"
NGROK_AUTHTOKEN = "1tve5X0Aea2r2LKv82JymsjmSD9_4Xp7L4YWxE7RuCSU9bdHZ"
NGROK_PORT = 9107


def run_command(cmd, check=True, shell=False, capture_output=False):
    """Run a shell command and return the result."""
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()
    
    print(f"  Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd,
            check=check,
            shell=shell,
            capture_output=capture_output,
            text=True
        )
        if capture_output:
            return result.stdout.strip()
        return result
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Command failed: {e}")
        if capture_output and e.stdout:
            print(f"  stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"  stderr: {e.stderr}")
        if check:
            sys.exit(1)
        return None


def step1_clone_repo():
    """Step 1: Clone the repository."""
    print("[1/5] Cloning repository...")
    
    repo_path = Path(REPO_DIR)
    if repo_path.exists():
        print(f"  Directory {REPO_DIR} already exists. Removing it...")
        shutil.rmtree(repo_path)
    
    run_command(["git", "clone", REPO_URL, REPO_DIR])
    print("  ✓ Repository cloned successfully")
    print()


def step2_install_dependencies():
    """Step 2: Install Python dependencies."""
    print("[2/5] Installing Python dependencies...")
    
    repo_path = Path(REPO_DIR)
    os.chdir(repo_path)
    
    # Try requirements_fixed.txt first, fallback to requirements.txt
    if (repo_path / "requirements_fixed.txt").exists():
        print("  Using requirements_fixed.txt")
        run_command([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements_fixed.txt"])
    elif (repo_path / "requirements.txt").exists():
        print("  Using requirements.txt")
        run_command([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    else:
        print("  ERROR: No requirements file found!")
        sys.exit(1)
    
    print("  ✓ Dependencies installed successfully")
    print()


def step3_setup_ngrok():
    """Step 3: Setup ngrok."""
    print("[3/5] Setting up ngrok...")
    
    # Install pyngrok if not already installed
    try:
        import pyngrok
    except ImportError:
        print("  Installing pyngrok...")
        run_command([sys.executable, "-m", "pip", "install", "-q", "pyngrok>=5.0.0"])
        import pyngrok
    
    # Stop any existing ngrok tunnels
    print("  Stopping any existing ngrok tunnels...")
    try:
        from pyngrok import ngrok
        import pyngrok.ngrok
        
        try:
            pyngrok.ngrok.kill()
            print("    ✓ Stopped existing ngrok tunnels")
        except Exception as e:
            print(f"    ℹ No existing tunnels to stop: {e}")
    except Exception as e:
        print(f"    ℹ Could not stop tunnels: {e}")
    
    # Configure ngrok authtoken
    print("  Configuring ngrok authtoken...")
    try:
        from pyngrok import ngrok
        import pyngrok.ngrok
        
        pyngrok.ngrok.set_auth_token(NGROK_AUTHTOKEN)
        print("    ✓ Ngrok authtoken configured")
    except Exception as e:
        print(f"    ERROR: Failed to configure ngrok: {e}")
        sys.exit(1)
    
    # Start ngrok tunnel
    print(f"  Starting ngrok tunnel on port {NGROK_PORT}...")
    try:
        from pyngrok import ngrok
        import pyngrok.ngrok
        
        # Ensure authtoken is set
        pyngrok.ngrok.set_auth_token(NGROK_AUTHTOKEN)
        
        # Start tunnel
        tunnel = ngrok.connect(NGROK_PORT, "http")
        public_url = tunnel.public_url
        
        print(f"    ✓ Ngrok tunnel established!")
        print(f"    Public URL: {public_url}")
        print(f"    The Plotly dashboard will be accessible at: {public_url}")
        
        # Keep tunnel alive by writing URL to file
        with open("ngrok_url.txt", "w") as f:
            f.write(public_url)
        
        return public_url
    except Exception as e:
        print(f"    ERROR: Failed to start ngrok tunnel: {e}")
        sys.exit(1)


def step4_display_info(public_url=None):
    """Step 4: Display ngrok information."""
    print("[4/5] Ngrok Tunnel Information:")
    
    if public_url:
        print(f"  Public URL: {public_url}")
    elif Path("ngrok_url.txt").exists():
        with open("ngrok_url.txt", "r") as f:
            url = f.read().strip()
            print(f"  Public URL: {url}")
    
    print(f"  Port: {NGROK_PORT}")
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
    
    # Run launch.py with unbuffered output for real-time interaction
    # Use -u flag for unbuffered output
    os.execv(sys.executable, [sys.executable, "-u", "launch.py"])


def main():
    """Main setup function."""
    print("=" * 50)
    print("Pine Script ML Optimizer - Kaggle Setup")
    print("=" * 50)
    print()
    
    try:
        step1_clone_repo()
        step2_install_dependencies()
        public_url = step3_setup_ngrok()
        step4_display_info(public_url)
        step5_launch_app()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
