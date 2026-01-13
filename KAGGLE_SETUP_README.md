# Kaggle Cloud Setup Instructions

This directory contains setup scripts to run the Pine Script ML Optimizer on Kaggle cloud computing.

## Python Version Requirements

**Recommended Python Version: 3.8 - 3.11**

The project is designed to work with Python 3.8 through 3.11. Python 3.12 is supported but some packages (like `talib-binary` and `pickle5`) will be automatically skipped.

### How to Set Python Version in Kaggle

1. **In Kaggle Notebooks:**
   - Go to **Settings** (gear icon in the right sidebar)
   - Under **Environment**, find **Python version**
   - Select your desired version (recommended: **3.10** or **3.11**)
   - Click **Save**

2. **In Kaggle Code:**
   - Add this at the top of your notebook to check/verify Python version:
   ```python
   import sys
   print(f"Python version: {sys.version}")
   ```

3. **Alternative: Use a specific Python version in code:**
   - Kaggle notebooks default to Python 3.10, but you can request a different version
   - Note: Kaggle may have limited Python version options available

## Quick Start

### Option 1: Python Script (Recommended for Kaggle)

```bash
python kaggle_setup.py
```

### Option 2: Bash Script

```bash
bash kaggle_setup.sh
```

## What the Script Does

1. **Clones the repository** from GitHub
2. **Installs all dependencies** from requirements files
3. **Stops any existing ngrok endpoints** to avoid conflicts
4. **Initializes ngrok** with the configured authtoken
5. **Launches the application** (`launch.py`) in interactive mode

## Ngrok Configuration

The script automatically:
- Configures ngrok with authtoken: `1tve5X0Aea2r2LKv82JymsjmSD9_4Xp7L4YWxE7RuCSU9bdHZ`
- Creates a tunnel to port 9107 (where the Plotly dashboard runs)
- Displays the public URL for accessing the dashboard

## Usage

After running the setup script, you'll have:
- A fully configured environment
- An active ngrok tunnel exposing the Plotly dashboard
- An interactive terminal session running `launch.py`

You can interact with the application just like you would in a normal terminal environment.

## Notes

- The script uses `requirements_fixed.txt` if available, otherwise falls back to `requirements.txt`
- All output is unbuffered for real-time interaction
- Press `Ctrl+C` to exit the application
- The ngrok URL is saved to `ngrok_url.txt` in the repository directory
