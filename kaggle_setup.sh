#!/bin/bash
# Kaggle Cloud Setup Script for Pine Script ML Optimizer
# This script sets up the environment, installs dependencies, configures ngrok, and launches the application

set -e  # Exit on any error

echo "=========================================="
echo "Pine Script ML Optimizer - Kaggle Setup"
echo "=========================================="
echo ""

# Configuration
REPO_URL="https://github.com/yanniedog/pinescript-ml-optimiser"
REPO_DIR="pinescript-ml-optimiser"
NGROK_AUTHTOKEN="1tve5X0Aea2r2LKv82JymsjmSD9_4Xp7L4YWxE7RuCSU9bdHZ"
NGROK_PORT=9107

# Step 1: Clone the repository
echo "[1/5] Cloning repository..."
if [ -d "$REPO_DIR" ]; then
    echo "  Directory $REPO_DIR already exists. Removing it..."
    rm -rf "$REPO_DIR"
fi

git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
echo "  ✓ Repository cloned successfully"
echo ""

# Step 2: Install Python dependencies
echo "[2/5] Installing Python dependencies..."
# Try requirements_fixed.txt first, fallback to requirements.txt
if [ -f "requirements_fixed.txt" ]; then
    echo "  Using requirements_fixed.txt"
    pip install -q -r requirements_fixed.txt
elif [ -f "requirements.txt" ]; then
    echo "  Using requirements.txt"
    pip install -q -r requirements.txt
else
    echo "  ERROR: No requirements file found!"
    exit 1
fi
echo "  ✓ Dependencies installed successfully"
echo ""

# Step 3: Install ngrok if not already installed
echo "[3/5] Setting up ngrok..."
# Check if pyngrok is installed
python3 -c "import pyngrok" 2>/dev/null || pip install -q pyngrok>=5.0.0

# Stop any existing ngrok tunnels
echo "  Stopping any existing ngrok tunnels..."
python3 << EOF
try:
    from pyngrok import ngrok
    import pyngrok.ngrok
    
    # Kill all existing tunnels
    try:
        pyngrok.ngrok.kill()
        print("    ✓ Stopped existing ngrok tunnels")
    except Exception as e:
        print(f"    ℹ No existing tunnels to stop: {e}")
except ImportError:
    print("    ℹ pyngrok not yet installed, skipping tunnel cleanup")
EOF

# Configure ngrok authtoken
echo "  Configuring ngrok authtoken..."
python3 << EOF
from pyngrok import ngrok
import pyngrok.ngrok

pyngrok.ngrok.set_auth_token("$NGROK_AUTHTOKEN")
print("    ✓ Ngrok authtoken configured")
EOF

# Start ngrok tunnel in background
echo "  Starting ngrok tunnel on port $NGROK_PORT..."
python3 << EOF
from pyngrok import ngrok
import pyngrok.ngrok
import time

# Ensure authtoken is set
pyngrok.ngrok.set_auth_token("$NGROK_AUTHTOKEN")

# Start tunnel
tunnel = ngrok.connect($NGROK_PORT, "http")
public_url = tunnel.public_url

print(f"    ✓ Ngrok tunnel established!")
print(f"    Public URL: {public_url}")
print(f"    The Plotly dashboard will be accessible at: {public_url}")

# Keep tunnel alive by writing URL to file
with open("ngrok_url.txt", "w") as f:
    f.write(public_url)
EOF

echo "  ✓ Ngrok setup complete"
echo ""

# Step 4: Display ngrok URL
if [ -f "ngrok_url.txt" ]; then
    echo "[4/5] Ngrok Tunnel Information:"
    echo "  Public URL: $(cat ngrok_url.txt)"
    echo "  Port: $NGROK_PORT"
    echo ""
fi

# Step 5: Launch the application
echo "[5/5] Launching application..."
echo "=========================================="
echo "Starting Pine Script ML Optimizer..."
echo "=========================================="
echo ""
echo "Note: The application is now running interactively."
echo "You can interact with it as you would in a normal terminal."
echo "Press Ctrl+C to exit."
echo ""

# Run launch.py with unbuffered output for real-time interaction
python3 -u launch.py
