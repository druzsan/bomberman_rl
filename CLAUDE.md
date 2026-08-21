# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Bomberman game engine used as a Reinforcement Learning course/competition harness. The engine is fixed; the work happens in `agent_code/<agent_name>/`. Python 3.12, managed by `uv` (see [.tool-versions](.tool-versions)).

## Commands

```bash
uv sync                                                    # create/update .venv from pyproject.toml + uv.lock
uv sync --extra torch                                      # add the optional torch/torchvision extra
uv run python main.py play --n-rounds 1 --no-gui           # fast headless match (4 rule_based_agents)
uv run python main.py play --my-agent <name> --train 1     # train <name> against 3 rule_based_agents
uv run python main.py play --agents a b c d --scenario coin-heaven --seed 42
uv run python main.py replay "replays/<file>.pt"           # replay a saved game (GUI only)
uv run python -m unittest test.py                          # smoke test: plays 1 round, asserts logs/game.log written
uv run python -m unittest test.MainTestCase.test_play      # single test
uv run ruff check .                                        # ruff is a dev dependency; no config in pyproject (defaults)
```

Useful flags: `--save-replay`, `--save-stats` (JSON into `results/`), `--make-video`, `--turn-based`, `--silence-errors` (agent exceptions become an `ERROR` action instead of crashing the game), `--skip-frames`.

Scenarios are defined in [settings.py](settings.py) `SCENARIOS`: `empty`, `coin-heaven`, `loot-crate`, `classic` (the tournament mode). Game rules constants (`BOMB_POWER`, `BOMB_TIMER`, `EXPLOSION_TIMER`, `MAX_STEPS`, `REWARD_KILL`, `REWARD_COIN`, timeouts) also live there.

## Architecture

**Engine (repo root).** [main.py](main.py) parses the CLI and runs `world_controller`, which loops `world.new_round()` then `world.do_step()` until `world.running` is false. [environment.py](environment.py) holds `GenericWorld` → `BombeRLeWorld` (live play) and `GUI`; [replay.py](replay.py) provides `ReplayWorld`. [items.py](items.py) has `Coin`/`Bomb`/`Explosion`, [events.py](events.py) the event-name string constants, [fallbacks.py](fallbacks.py) a pygame stub so headless runs work without a display.

**Step order** ([environment.py:158](environment.py#L158)) is fixed and matters for reward shaping: `poll_and_run_agents` → `collect_coins` → `update_explosions` → `update_bombs` → `evaluate_explosions` → `send_game_events`. So the `new_game_state` an agent sees in `game_events_occurred` is the state *after* the whole world has advanced, and the events list covers everything that happened in that step. Within a step all agents are asked to `act` first, then their actions are applied in a random permutation (recorded in the replay).

**Agent plumbing** ([agents.py](agents.py)). Three layers: `Agent` (game-side player object: position, score, events, think-time budget) → `AgentBackend` → `AgentRunner` (imports and calls the user module). Two backends exist, but `GenericWorld.add_agent` currently hardcodes `SequentialAgentBackend` — everything runs in the main process, so debugging and breakpoints in agent code work directly. `ProcessAgentBackend` is present but unused.

Two non-obvious consequences:

- `SequentialAgentBackend.send_event` **chdirs into `agent_code/<code_name>/` around every callback**, so relative paths in agent code (e.g. `open("my-saved-model.pt")`) resolve inside the agent's own directory, not the repo root.
- `AgentRunner.__init__` validates the callback surface against `AGENT_API` by **argument count**, and raises `NotImplementedError`/`TypeError` at load time if a required callback is missing or has the wrong arity. Extra or missing parameters are a hard failure, not a warning.

**Agent contract.** `agent_code/<name>/callbacks.py` must define `setup(self)` and `act(self, game_state)`. Only when the agent is in training mode is `agent_code/<name>/train.py` imported, and it must define `setup_training(self)`, `game_events_occurred(self, old_game_state, self_action, new_game_state, events)`, and `end_of_round(self, last_game_state, last_action, events)`. `self` is not an instance of a class you wrote — it's a `SimpleNamespace` owned by the engine, pre-populated with `logger` and `train`; anything else you need must be attached during `setup`/`setup_training`. This split exists so a trained agent can be shared without its training code.

`game_state` dict keys: `round`, `step`, `field` (`np.ndarray`, `-1` wall / `0` free / `1` crate), `self` (`(name, score, bombs_left, (x, y))`), `others` (same tuple shape), `bombs` (`((x, y), timer)`), `coins` (`(x, y)`, collectable only), `user_input`, `explosion_map` (float array of remaining danger). Dead agents get `None` instead of a state.

**Think-time budget.** Outside training, an `act` call exceeding `settings.TIMEOUT` (0.5s) is forced to `WAIT` and the overrun is deducted from the next step's budget; in training mode the timeout is infinite.

## Writing a new agent

Copy [agent_code/_template_agent/](agent_code/_template_agent/) to `agent_code/<your_agent_name>/`. It is the strictly typed template: `types.py` declares `GameState`, `Action` and the `AgentContext` protocol locally so the copied directory is self-contained, and `callbacks.py`/`train.py` use relative imports (`from .types import ...`), which is why it ships an `__init__.py`. When you store state on the context (`self.model`, replay buffers, …), add the attribute to `AgentContext` in `types.py`.

[agent_code/tpl_agent/](agent_code/tpl_agent/) is the older untyped upstream template (pickle-based model, no `__init__.py`) — useful as a worked example, but prefer `_template_agent` for new code.

Reference opponents already present: `rule_based_agent` (the standard benchmark and default opponent), `coin_collector_agent`, `random_agent`, `peaceful_agent`, `fail_agent` (deliberately raises, for error-handling tests), `user_agent` (keyboard-controlled).

## Logging and output

`logs/game.log` is the engine log; each agent gets `agent_code/<name>/logs/<agent_name>.log`, rewritten (`mode="w"`) on every run — the wrapper logger records callback durations, the `self.logger` inside agent code writes at DEBUG. Replays land in `replays/`, stats JSON in `results/`, videos/screenshots in `screenshots/`. `replays/`, `results/`, and `dev/` are gitignored.

## Conventions

Four-space indent, PEP 8, `snake_case` modules and agent directories, `UPPER_SNAKE_CASE` constants. Imports grouped stdlib → third-party → local. New agent code should be strictly typed in the style of `_template_agent`. Commit subjects are short and imperative (e.g. `Add agent template with strict typing`).
