#!/usr/bin/env bash

# Ensure we are in the directory where the script is located
cd "$(dirname "$0")" || exit 1

supports_required_version() {
    "$1" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if ((3, 8) <= sys.version_info[:2] <= (3, 12)) else 1)
PY
}

pick_python() {
    local candidates="${SIMILARITY_PYTHON_CANDIDATES:-python3.11 python3.10 python3.12 python3 /usr/bin/python3}"
    for candidate in $candidates; do
        if ! command -v "$candidate" >/dev/null 2>&1; then
            continue
        fi
        local exe
        exe="$(command -v "$candidate")"
        if supports_required_version "$exe"; then
            echo "$exe"
            return 0
        fi
    done
    return 1
}

if ! PYTHON_BIN="$(pick_python)"; then
    echo "[ERROR] Could not find a compatible Python version (3.8 - 3.12)."
    echo "Install Python 3.10/3.11 and rerun this launcher."
    read -p "Press Enter to exit..."
    exit 1
fi
echo "[INFO] Using Python: $PYTHON_BIN"

if [ "${SIMILARITY_LAUNCHER_DRY_RUN:-0}" = "1" ]; then
    echo "[INFO] Dry run mode enabled; skipping venv setup and app launch."
    exit 0
fi

ensure_venv() {
    local needs_rebuild="false"
    if [ -x ".venv/bin/python" ]; then
        if ! supports_required_version ".venv/bin/python"; then
            echo "[WARN] Existing virtual environment is incompatible."
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
import deepface  # noqa: F401
import rich  # noqa: F401
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
echo "[INFO] Launching Face Similarity CLI..."
python main.py --cli

echo ""
echo "[INFO] Application finished."
read -p "Press Enter to exit..."
