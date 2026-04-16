# Style And Conventions
- Keep architectural separation strict: business logic and ML math in `src/engine.py`, GUI-only concerns in `src/gui.py`, CLI-only concerns in `src/cli.py`.
- Prefer small, explicit classes with clear responsibilities: `FaceEngine`, `ModernGUI`, and `ProCLI` are the central types.
- Code style is conventional Python with snake_case methods/functions, PascalCase classes, inline comments only where useful, and docstrings on major public entry points and classes.
- Type hints are present on backend methods and should be preserved or expanded rather than removed.
- GUI design rule: remain dark-mode focused and built with `customtkinter` patterns already used in `ModernGUI`.
- CLI design rule: use `rich` components instead of plain `print()` for terminal UX.
- Performance rule: any GUI path that touches `DeepFace.build_model()` or `DeepFace.verify()` must run in a background daemon thread.
- Dependency rule: do not replace `retina-face` with `retinaface` in `requirements.txt`.
- When making changes, prefer the top-level `src/` application tree unless the task explicitly targets one of the duplicated snapshot directories.