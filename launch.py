#!/usr/bin/env python3
"""
Pine Script ML Indicator Optimizer - Launcher
Run this file to start the optimizer. It will install dependencies automatically.

Usage: python launch.py
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install required packages from requirements.txt."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("[ERROR] requirements.txt not found!")
        return False
    
    print("Installing dependencies...")
    print("-" * 50)
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"
        ])
        print("[OK] Dependencies installed successfully")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        return False


def main():
    """Main launcher entry point."""
    # Install requirements first
    if not install_requirements():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Now import and run the interactive optimizer
    try:
        from interactive_optimizer import main as run_optimizer
        run_optimizer()
    except ImportError as e:
        print(f"[ERROR] Failed to import optimizer: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()

