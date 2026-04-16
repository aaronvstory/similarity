# Face Similarity Application

An enterprise-grade local offline Face Similarity application in Python. This tool uses state-of-the-art machine learning models (DeepFace with RetinaFace for detection and ArcFace for recognition) to securely compare two portrait images and provide a similarity percentage score.

## Features
- **Local & Offline Execution**: No biometric data is sent to the cloud. Everything runs locally on your machine.
- **Dual Interfaces**: 
  - **Modern GUI**: Built with `customtkinter`, featuring a dark-mode, drag-and-drop/clickable upload interface with multithreaded processing to ensure zero freezing.
  - **Pro CLI**: Built with `rich`, featuring an interactive menu for single comparison, automated batch folder scanning, and dynamic regex image searching.
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
2. The interactive Main Menu will appear. Choose **Option 3 (Settings)** if your images are not named "extracted" and "selfie". You can input exact names or regular expressions (regex).
3. Choose **Option 2 (Batch Folder Processing)**.
4. Select the root folder that contains all of your individual persona/scan folders.
5. The application will recursively scan every folder. If a folder contains both images, it runs the ArcFace ML models on them.
6. A live progress bar and table will display the results, and the directory will be automatically renamed to include the rounded similarity score.

## Models Used
- **Detector**: `retinaface` (Robust face detection and 5-point alignment)
- **Recognizer**: `ArcFace` (State-of-the-art facial embedding model)
- **Metric**: Cosine Distance

## Contribution
Check `agents.md` and `claude.md` for AI context if you are utilizing LLMs to contribute to this codebase. See `CHANGELOG.md` for recent updates.
