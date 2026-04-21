# Contributing

## Workflow
1. Create a feature branch from `main` (recommended prefix: `codex/`).
2. Make focused commits with clear messages.
3. Run relevant validation before opening a PR.
4. Push branch and open a pull request targeting `main`.
5. Address bot/human review feedback in follow-up commits.

## Local Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --cli
```

## Required Architecture
- Keep ML/business logic in `src/engine.py`.
- Keep GUI logic in `src/gui.py`.
- Keep terminal UX and CLI orchestration in `src/cli.py`.
- Do not move model math/detection logic into GUI/CLI modules.

## Dependency Rules
- Use `retina-face` (with hyphen) in `requirements.txt`.
- Do not replace it with `retinaface`.

## Performance and UX Rules
- `DeepFace.build_model()` and `DeepFace.verify()` are heavy on first run.
- Any GUI call path that triggers those operations must stay in a background daemon thread so the UI remains responsive.

## UI Rules
- GUI stays dark-mode focused.
- CLI output should use `rich` components (`Console`, `Panel`, `Progress`, `Table`) instead of plain `print()` for user-facing flow.

## Testing Guidance
- Validate in an isolated virtual environment.
- For feature work, include:
  - happy path behavior,
  - edge cases (missing files, invalid images, no-face cases),
  - regression checks for unchanged flows.
