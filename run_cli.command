#!/usr/bin/env bash

# Ensure we are in the directory where the script is located
cd "$(dirname "$0")" || exit 1

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in your PATH."
    echo "Please install Python 3 (e.g., via Homebrew: brew install python) and try again."
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if .venv exists
if [ ! -f ".venv/bin/activate" ]; then
    echo "[INFO] Virtual environment not found. Creating one..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        read -p "Press Enter to exit..."
        exit 1
    fi
    
    echo "[INFO] Activating virtual environment..."
    source .venv/bin/activate
    
    echo "[INFO] Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    
    echo "[INFO] Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies."
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "[INFO] Setup complete."
else
    echo "[INFO] Activating existing virtual environment..."
    source .venv/bin/activate
fi

# Run the application
echo "[INFO] Launching Face Similarity CLI..."
python main.py --cli

echo ""
echo "[INFO] Application finished."
read -p "Press Enter to exit..."
