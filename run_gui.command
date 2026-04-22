#!/usr/bin/env bash

# Ensure we are in the directory where the script is located
cd "$(dirname "$0")" || exit 1

supports_required_version() {
    "$1" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if ((3, 8) <= sys.version_info[:2] <= (3, 12)) else 1)
PY
}

supports_tkinter() {
    "$1" - <<'PY' >/dev/null 2>&1
import tkinter  # noqa: F401
PY
}

pick_python_with_tk() {
    for candidate in python3.11 python3.10 python3.12 python3 /usr/bin/python3; do
        if ! command -v "$candidate" >/dev/null 2>&1; then
            continue
        fi
        local exe
        exe="$(command -v "$candidate")"
        if supports_required_version "$exe" && supports_tkinter "$exe"; then
            echo "$exe"
            return 0
        fi
    done
    return 1
}

if ! PYTHON_BIN="$(pick_python_with_tk)"; then
    echo "[ERROR] Could not find a compatible Python with tkinter support."
    echo "Install Python 3.10/3.11 with Tk support, then rerun this launcher."
    read -p "Press Enter to exit..."
    exit 1
fi
echo "[INFO] Using Python: $PYTHON_BIN"

ensure_venv() {
    local needs_rebuild="false"
    if [ -x ".venv/bin/python" ]; then
        if ! supports_required_version ".venv/bin/python" || ! supports_tkinter ".venv/bin/python"; then
            echo "[WARN] Existing virtual environment is incompatible with GUI requirements."
            needs_rebuild="true"
        fi
    else
        needs_rebuild="true"
    fi

    if [ "$needs_rebuild" = "true" ] && [ -d ".venv" ]; then
        echo "[INFO] Rebuilding virtual environment..."
        rm -rf .venv
    fi

    if [ ! -f ".venv/bin/activate" ]; then
    echo "[INFO] Virtual environment not found. Creating one..."
    "$PYTHON_BIN" -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        read -p "Press Enter to exit..."
        exit 1
    fi
    fi
}

ensure_venv

echo "[INFO] Activating existing virtual environment..."
source .venv/bin/activate

if ! python - <<'PY' >/dev/null 2>&1
import customtkinter  # noqa: F401
import deepface  # noqa: F401
PY
then
    echo "[INFO] Installing dependencies from requirements.txt..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies."
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "[INFO] Setup complete."
fi

# Run the application
echo "[INFO] Launching Face Similarity GUI..."
python main.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Application exited with an error."
    read -p "Press Enter to exit..."
fi
