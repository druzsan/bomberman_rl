# Repository Guidelines

## Project Structure & Module Organization

The game engine lives in the repository root. `main.py` provides the `play` and `replay` CLI commands; `environment.py`, `items.py`, and `agents.py` implement game state, entities, and agent loading. Shared constants are in `settings.py`, while event names are in `events.py`. Add agent implementations under `agent_code/<agent_name>/`; use `agent_code/tpl_agent/` as the callback and training template. Static sprites and fonts belong in `assets/`. Runtime output is written to `logs/`, `replays/`, `results/`, and `screenshots/`. The current automated smoke test is `test.py`.

## Build, Test, and Development Commands

- `uv sync` creates or updates the Python 3.12 environment from `pyproject.toml` and `uv.lock`.
- `uv run python main.py play --n-rounds 1 --no-gui` runs a fast headless match.
- `uv run python main.py play --my-agent <agent_name> --train 1` trains one custom agent against rule-based opponents.
- `uv run python main.py replay replays/<file>` opens a saved replay.
- `uv run python -m unittest test.py` runs the smoke test and verifies that a game log is produced.
- `docker build -t bomberman-rl .` builds the optional dependency-rich development image.

## Coding Style & Naming Conventions

Use four-space indentation and follow PEP 8 conventions. Name modules, functions, and agent directories with `snake_case`, classes with `PascalCase`, and constants with `UPPER_SNAKE_CASE`. Keep callbacks compatible with the interfaces demonstrated in `agent_code/tpl_agent/callbacks.py` and `train.py`. Prefer small, testable helpers and document non-obvious game-state or reward logic. No formatter or linter is configured; keep imports grouped as standard library, third-party, then local modules.

## Testing Guidelines

Tests use Python's `unittest` framework. Add methods named `test_<behavior>` to `unittest.TestCase` classes, and favor deterministic seeds plus `--no-gui` for reliable CI runs. Clean up or isolate generated artifacts when adding tests. There is no stated coverage threshold, but changes to game rules, scenarios, or callbacks should include focused regression tests.

## Commit & Pull Request Guidelines

Recent history uses short, descriptive subjects such as `added game modes` and `Store game state for agents as passed to act`. Keep commits focused and write an imperative summary under roughly 72 characters. Pull requests should explain the behavior change, list verification commands, and link relevant issues. Include screenshots or a short replay for GUI or gameplay changes, and note any new model files, dependencies, or scenario settings.
