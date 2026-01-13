# Kaggle Cloud Setup Instructions

This directory contains setup scripts to run the Pine Script ML Optimizer on Kaggle cloud computing.

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
