# Suggested Commands
## Environment setup
- PowerShell: `python -m venv .venv`
- PowerShell activate: `.\.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt`

## Run the app
- GUI default: `python main.py`
- CLI mode: `python main.py --cli`
- CLI with explicit files: `python main.py --cli --img1 path\to\img1.jpg --img2 path\to\img2.jpg`

## Launcher scripts
- Windows GUI launcher: `.\run_gui.bat`
- Windows CLI launcher: `.\run_cli.bat`
- Face extraction helper launcher: `.\extract_face_launcher.bat`

## Basic repo inspection on Windows
- List files: `Get-ChildItem`
- Recursive search by name: `Get-ChildItem -Recurse`
- Fast text search if ripgrep is installed: `rg pattern src`
- Git status: `git status`
- Git diff: `git diff`

## Verification notes
- No dedicated lint, format, or test config files were found at the repo root.
- Minimum practical verification is to run the relevant entry point you changed (`python main.py` or `python main.py --cli`) inside the virtual environment.
- Because DeepFace model initialization is heavy, allow extra time on the first run.