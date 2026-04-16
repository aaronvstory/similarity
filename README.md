# Face Similarity Application

An enterprise-grade local offline Face Similarity application in Python. This tool uses state-of-the-art machine learning models (DeepFace with RetinaFace for detection and ArcFace for recognition) to securely compare two portrait images and provide a similarity percentage score.

## Features
- **Local & Offline Execution**: No biometric data is sent to the cloud. Everything runs locally on your machine.
- **Dual Interfaces**: 
  - **Modern GUI**: Built with `customtkinter`, featuring a dark-mode, drag-and-drop/clickable upload interface with multithreaded processing to ensure zero freezing.
  - **Pro CLI**: Built with `rich`, featuring an interactive menu for single comparison, automated batch similarity scanning, batch face extraction, and dynamic regex searching.
- **Batch Face Extraction**: Automatically find and crop faces from source images (e.g., driver's licenses) found in recursive folder structures.
- **Accurate Mathematics**: Internally converts DeepFace's raw cosine distance into a human-readable 0-100% percentage grade, matching industry-standard strictness (where a score of >= 80% represents a match).
- **Automated Folder Renaming**: In Batch CLI mode, automatically appends similarity scores into directory names for incredibly fast KYC/Persona reviewing.

## Installation

### Requirements
- Python 3.8+ 

### Quick Start Setup
The project comes with automated cross-platform launchers that create the virtual environment and install all dependencies automatically.

**Windows**: 
Double-click `run_gui.bat` (to open the GUI) or `run_cli.bat` (to open the CLI terminal).

**macOS/Linux**:
First, grant execute permissions to the scripts in your terminal:
```bash
chmod +x *.command
```
Then double click `run_gui.command` or `run_cli.command`.

### Manual Setup
If you prefer setting it up manually:
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py              # Launch GUI
python main.py --cli        # Launch CLI
```

## Batch Processing Usage (CLI)
1. Run the CLI launcher (`run_cli.bat` or `run_cli.command`).
2. The interactive Main Menu will appear. 
3. **Face Extraction**: 
   - Choose **Option 3 (Batch Face Extraction)**.
   - Choose **Option 4 (Settings)** first if your source images are not named "front".
   - Select the root folder. The app will find "front.jpg" (or similar), crop the face, and save it as "extracted.jpg" in the same folder, skipping if it already exists.
4. **Similarity Check**:
   - Choose **Option 2 (Batch Folder Similarity Check)**.
   - Choose **Option 4 (Settings)** first if your images are not named "extracted" and "selfie".
   - Select the root folder. The app will recursively scan every folder. If a folder contains both images, it runs the ArcFace ML models on them.
   - A live progress bar and table will display the results, and the directory will be automatically renamed to include the rounded similarity score.

## Models Used
- **Detector**: `retinaface` (Robust face detection and 5-point alignment)
- **Recognizer**: `ArcFace` (State-of-the-art facial embedding model)
- **Metric**: Cosine Distance

## Contribution
Check `agents.md` and `claude.md` for AI context if you are utilizing LLMs to contribute to this codebase. See `CHANGELOG.md` for recent updates.
