# Task Completion
- Re-check the architecture boundary before finishing: ML/business logic stays in `src/engine.py`, GUI logic in `src/gui.py`, CLI logic in `src/cli.py`.
- If GUI code was touched, verify any DeepFace model initialization or verification still runs in a background daemon thread.
- If terminal UX was touched, confirm output still uses `rich` patterns instead of plain `print()`.
- If dependencies were touched, confirm `retina-face` is still the detector package in `requirements.txt`.
- Run the most relevant entry point after changes:
  - `python main.py` for GUI changes
  - `python main.py --cli` for CLI changes
- If a fix is too heavy to exercise end-to-end because of model startup cost, state clearly what was and was not verified.
- Summarize any files changed, the behavior change, and verification performed.