# AI Agent Instructions (agents.md)

If you are an AI assistant interacting with this repository, adhere to the following rules:

1. **Architecture Rule**: The system is split strictly into `engine.py` (business logic and ML math), `gui.py` (CustomTkinter graphical interface), and `cli.py` (Rich-powered terminal interface). Keep concerns separated. Do not put ML logic in the UI files.
2. **Library Constraint**: The ML backend explicitly requires `retina-face` (not `retinaface`). Do not break this in `requirements.txt` as it will cause unresolvable TensorFlow 2.5.0 dependency crashes on Python 3.12.
3. **Execution Rule**: `DeepFace.verify()` and `DeepFace.build_model()` are exceptionally slow on the first run. Any function calling them from `gui.py` MUST be executed inside a background daemon thread to prevent the UI from freezing.
4. **Design Rules**: The GUI must remain dark-mode focused. The CLI must utilize the `rich` library for all terminal output (using `Console`, `Panel`, `Progress`, `Table`, etc.) rather than standard `print()` statements.

Always test dependency installations in isolated virtual environments.
