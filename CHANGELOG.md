# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- **Separated CLI Workflow Menus**: Interactive CLI now starts with dedicated `Similarity` and `Extraction` sections, each with its own submenu and explicit `Back` navigation.
- **Separated GUI Workflow Tabs**: GUI now provides dedicated `Similarity` and `Extraction` tabs with independent action controls and status messaging.

### Changed
- Updated user documentation to reflect sectioned CLI navigation and the tabbed GUI workflow model.

## [1.1.0] - 2026-04-10
### Added
- **Batch Processing CLI**: A major upgrade to the CLI mode allowing users to select a root directory and recursively scan subfolders.
- **Interactive Menu**: The CLI now runs an interactive `rich` menu instead of relying strictly on argparse flags.
- **Dynamic Image Settings**: CLI users can change the regex/keywords used to locate the two face images inside subdirectories (defaulting to "extracted" and "selfie").
- **Automated Folder Renaming**: Batch processing will automatically insert the rounded similarity score into the subfolder's name (e.g., `FAILED PERSONA - Morgan` -> `FAILED PERSONA 81 - Morgan`).
- **Comprehensive Documentation**: Added `README.md`, `CHANGELOG.md`, `agents.md`, and `claude.md`.

### Fixed
- Replaced the mathematical scoring formula in `src/engine.py` to map ArcFace's official 0.68 cosine distance threshold dynamically to an 80% score curve, resolving an issue where the app falsely outputted "59%" for matched photos.

## [1.0.0] - 2026-04-10
### Added
- Initial release.
- FaceEngine using DeepFace, RetinaFace, and ArcFace.
- Modern GUI built with `customtkinter`.
- Professional CLI wrapper using `rich`.
- Cross-platform launchers (`.bat` and `.command`) with automated virtual environment creation and pip dependency resolution.
