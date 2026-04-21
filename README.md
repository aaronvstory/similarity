# Face Similarity Application

An enterprise-grade local offline Face Similarity application in Python. This tool uses state-of-the-art machine learning models (DeepFace with RetinaFace for detection and ArcFace for recognition) to securely compare two portrait images and provide a similarity percentage score.

## Features
- **Local & Offline Execution**: No biometric data is sent to the cloud. Everything runs locally on your machine.
- **Dual Interfaces**: 
  - **Modern GUI**: Built with `customtkinter`, featuring dark-mode **Similarity** and **Extraction** tabs while keeping long-running ML work on background threads.
  - **Pro CLI**: Built with `rich`, featuring a sectioned interactive menu with dedicated **Similarity** and **Extraction** submenus, plus regex-first keyword search with fuzzy fallback.
- **Batch Face Extraction**: Automatically find and crop faces from source images (e.g., driver's licenses) found in recursive folder structures.
- **Accurate Mathematics**: Internally converts DeepFace's raw cosine distance into a human-readable 0-100% percentage grade, matching industry-standard strictness (where a score of >= 80% represents a match).
- **Automated Folder Renaming**: In Batch CLI mode, folder score tags are overwritten in place (not duplicated) for incredibly fast KYC/Persona reviewing.
- **Prominent Face Selection**: If multiple faces are detected, the largest detected face is automatically used for similarity comparison.

## Project Structure
- Root-only runtime project (`main.py`, `src/`, launch scripts, and `.venv`).
- No nested duplicate app folder is required for normal operation.

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
python main.py --mode similarity --root /path/to/root --yes
python main.py --mode extract --root /path/to/root --yes
python main.py --mode compare --img1 /path/to/extracted.png --img2 /path/to/selfie.png
```

Explicit compare mode requires both `--img1` and `--img2`. CLI configuration overrides such as
`--img1-keyword`, `--img2-keyword`, `--extraction-keyword`, `--padding-ratio`, and
`--existing-file-mode` now also force CLI routing even without `--cli`.

## Batch Processing Usage (CLI)
1. Run the CLI launcher (`run_cli.bat` or `run_cli.command`).
2. The interactive sections menu will appear with **Similarity**, **Extraction**, **Settings**, and **Exit**.
   - From this top-level menu, choose **1** to enter the Similarity submenu or **2** to enter the Extraction submenu.
3. **Face Extraction**: 
   - Choose **Settings** first if your source images are not named "front".
   - Then choose **Extraction** then **Batch Face Extraction**.
   - Select the root folder. The app will find "front.jpg" (or similar), crop the face, and save it as "extracted.jpg" in the same folder.
   - By default (`existing_file_mode = "index"`), if `extracted.jpg` already exists it writes `extracted2.jpg` (then `extracted3.jpg`, etc.). It only skips when Settings sets mode to `skip`.
4. **Similarity Check**:
   - Choose **Settings** first if your images are not named "extracted" and "selfie".
   - Then choose **Similarity** then **Batch Folder Similarity Check**.
   - Select the root folder. The app will recursively scan every folder. If a folder contains both images, it runs the ArcFace ML models on them.
   - A live progress bar and table will display the results, and the directory will be automatically renamed with a single rounded similarity score token.

## CLI Notes
- `apply_runtime_config` rejects invalid `existing_file_mode` values instead of silently defaulting. Valid values are `index`, `skip`, and `overwrite`.
- `padding_ratio` must stay within `0.0` to `1.0`.
- Invalid runtime configuration exits with code `2`, which keeps CLI usage failures distinct from runtime processing errors.
- Keyword matching is regex-first with fuzzy fallback, so a configured keyword such as `selfie` can still match files like `selfie expanded.png`.

## Models Used
- **Detector**: `retinaface` (Robust face detection and 5-point alignment)
- **Recognizer**: `ArcFace` (State-of-the-art facial embedding model)
- **Metric**: Cosine Distance

## Contribution
See `CONTRIBUTING.md` for workflow, architecture, dependency, UX, and testing guidance. Check `agents.md` and `claude.md` for AI context if you are utilizing LLMs to contribute to this codebase. See `CHANGELOG.md` for recent updates.
