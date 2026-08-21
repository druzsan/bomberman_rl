"""A callback-free simulator matching the official Bomberman transition order."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import Action, GameState

COLS = ROWS = 17
MAX_STEPS = 400
BOMB_TIMER = 4
BOMB_POWER = 3
EXPLOSION_TIMER = 2
START_POSITIONS = ((1, 1), (1, 15), (15, 1), (15, 15))
SCENARIOS = {
    "coin-heaven": (0.0, 50),
    "classic": (0.75, 9),
}


@dataclass(slots=True)
class Player:
    player_id: int
    position: tuple[int, int]
    score: int = 0
    can_bomb: bool = True
    alive: bool = True


@dataclass(slots=True)
class Bomb:
    position: tuple[int, int]
    owner: int
    timer: int = BOMB_TIMER


@dataclass(slots=True)
class Explosion:
    coordinates: tuple[tuple[int, int], ...]
    owner: int
    timer: int = EXPLOSION_TIMER
    stage: int = 0


@dataclass(slots=True)
class StepResult:
    events: dict[int, list[str]]
    terminated: dict[int, bool]
    round_over: bool


class BombermanEnv:
    def __init__(self, players: int, scenario: str, seed: int) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}")
        if not 1 <= players <= 4:
            raise ValueError("players must be in [1, 4]")
        self.player_count = players
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.round = 0
        self.field = np.empty((COLS, ROWS), dtype=np.int8)
        self.players: dict[int, Player] = {}
        self.bombs: list[Bomb] = []
        self.explosions: list[Explosion] = []
        self.coins: dict[tuple[int, int], bool] = {}
        self.step_count = 0
        self.running = False

    def reset(self) -> dict[int, GameState]:
        self.round += 1
        # The official world increments the counter before requesting actions.
        self.step_count = 1
        self.bombs = []
        self.explosions = []
        self.field, self.coins = self._build_arena()
        starts = self.rng.permutation(np.asarray(START_POSITIONS))
        self.players = {
            player_id: Player(player_id, tuple(map(int, starts[player_id])))
            for player_id in range(self.player_count)
        }
        self.running = True
        return self.observations()

    def _build_arena(self) -> tuple[np.ndarray, dict[tuple[int, int], bool]]:
        density, coin_count = SCENARIOS[self.scenario]
        arena = np.zeros((COLS, ROWS), dtype=np.int8)
        arena[self.rng.random((COLS, ROWS)) < density] = 1
        arena[:1, :] = arena[-1:, :] = -1
        arena[:, :1] = arena[:, -1:] = -1
        for x in range(COLS):
            for y in range(ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    arena[x, y] = -1
        for x, y in START_POSITIONS:
            for xx, yy in ((x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if arena[xx, yy] == 1:
                    arena[xx, yy] = 0

        positions = np.stack(
            np.meshgrid(np.arange(COLS), np.arange(ROWS), indexing="ij"), -1
        )
        crates = self.rng.permutation(positions[arena == 1])
        free = self.rng.permutation(positions[arena == 0])
        chosen = np.concatenate((crates, free), axis=0)[:coin_count]
        coins = {(int(x), int(y)): arena[x, y] == 0 for x, y in chosen}
        return arena, coins

    def observations(self) -> dict[int, GameState]:
        return {
            player_id: self.observation_for(player_id)
            for player_id, player in self.players.items()
            if player.alive
        }

    def observation_for(self, player_id: int) -> GameState:
        player = self.players[player_id]
        if not player.alive:
            raise ValueError("Dead players have no observation")
        explosion_map = np.zeros_like(self.field, dtype=np.float32)
        for explosion in self.explosions:
            if explosion.stage == 0:
                for position in explosion.coordinates:
                    explosion_map[position] = max(
                        explosion_map[position], explosion.timer - 1
                    )
        return {
            "round": self.round,
            "step": self.step_count,
            "field": self.field.copy(),
            "self": self._player_state(player),
            "others": [
                self._player_state(other)
                for other in self.players.values()
                if other.alive and other.player_id != player_id
            ],
            "bombs": [(bomb.position, bomb.timer) for bomb in self.bombs],
            "coins": [
                position for position, collectable in self.coins.items() if collectable
            ],
            "user_input": None,
            "explosion_map": explosion_map,
        }

    @staticmethod
    def _player_state(player: Player) -> tuple[str, int, bool, tuple[int, int]]:
        return (
            f"player_{player.player_id}",
            player.score,
            player.can_bomb,
            player.position,
        )

    def step(self, actions: dict[int, Action]) -> StepResult:
        if not self.running:
            raise RuntimeError("Call reset before step")
        living_before = {
            player_id for player_id, player in self.players.items() if player.alive
        }
        if actions.keys() != living_before:
            raise ValueError("Exactly one action is required for each living player")
        events = {player_id: [] for player_id in self.players}

        for player_id in self.rng.permutation(tuple(living_before)):
            self._perform_action(
                self.players[int(player_id)],
                actions[int(player_id)],
                events[int(player_id)],
            )
        self._collect_coins(events)
        self._update_explosions()
        self._update_bombs(events)
        self._evaluate_explosions(events)

        living_after = {
            player_id for player_id, player in self.players.items() if player.alive
        }
        round_over = self._time_to_stop()
        if round_over:
            self.running = False
            for player_id in living_after:
                events[player_id].append("SURVIVED_ROUND")
        else:
            # Observations requested after this transition are for the next action.
            self.step_count += 1
        terminated = {
            player_id: player_id not in living_after or round_over
            for player_id in living_before
        }
        return StepResult(events, terminated, round_over)

    def _tile_is_free(self, position: tuple[int, int]) -> bool:
        if self.field[position] != 0:
            return False
        obstacles = {bomb.position for bomb in self.bombs}
        obstacles.update(
            player.position for player in self.players.values() if player.alive
        )
        return position not in obstacles

    def _perform_action(
        self, player: Player, action: Action, events: list[str]
    ) -> None:
        x, y = player.position
        moves = {
            "UP": ((x, y - 1), "MOVED_UP"),
            "RIGHT": ((x + 1, y), "MOVED_RIGHT"),
            "DOWN": ((x, y + 1), "MOVED_DOWN"),
            "LEFT": ((x - 1, y), "MOVED_LEFT"),
        }
        if action in moves and self._tile_is_free(moves[action][0]):
            player.position = moves[action][0]
            events.append(moves[action][1])
        elif action == "BOMB" and player.can_bomb:
            self.bombs.append(Bomb(player.position, player.player_id))
            player.can_bomb = False
            events.append("BOMB_DROPPED")
        elif action == "WAIT":
            events.append("WAITED")
        else:
            events.append("INVALID_ACTION")

    def _collect_coins(self, events: dict[int, list[str]]) -> None:
        for player in self.players.values():
            if player.alive and self.coins.get(player.position, False):
                self.coins[player.position] = False
                player.score += 1
                events[player.player_id].append("COIN_COLLECTED")

    def _update_explosions(self) -> None:
        remaining: list[Explosion] = []
        for explosion in self.explosions:
            explosion.timer -= 1
            if explosion.timer <= 0:
                explosion.stage += 1
                if explosion.stage == 1:
                    explosion.timer = 2
                    self.players[explosion.owner].can_bomb = True
                else:
                    continue
            remaining.append(explosion)
        self.explosions = remaining

    def _blast_coordinates(
        self, position: tuple[int, int]
    ) -> tuple[tuple[int, int], ...]:
        x, y = position
        result = [(x, y)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for distance in range(1, BOMB_POWER + 1):
                candidate = x + dx * distance, y + dy * distance
                if self.field[candidate] == -1:
                    break
                result.append(candidate)
        return tuple(result)

    def _update_bombs(self, events: dict[int, list[str]]) -> None:
        remaining: list[Bomb] = []
        for bomb in self.bombs:
            if bomb.timer <= 0:
                events[bomb.owner].append("BOMB_EXPLODED")
                blast = self._blast_coordinates(bomb.position)
                for position in blast:
                    if self.field[position] == 1:
                        self.field[position] = 0
                        events[bomb.owner].append("CRATE_DESTROYED")
                        if position in self.coins:
                            self.coins[position] = True
                            events[bomb.owner].append("COIN_FOUND")
                self.explosions.append(Explosion(blast, bomb.owner))
            else:
                bomb.timer -= 1
                remaining.append(bomb)
        self.bombs = remaining

    def _evaluate_explosions(self, events: dict[int, list[str]]) -> None:
        hit: set[int] = set()
        for explosion in self.explosions:
            if explosion.stage != 0:
                continue
            for player in self.players.values():
                if player.alive and player.position in explosion.coordinates:
                    hit.add(player.player_id)
                    if player.player_id == explosion.owner:
                        events[player.player_id].append("KILLED_SELF")
                    else:
                        self.players[explosion.owner].score += 5
                        events[explosion.owner].append("KILLED_OPPONENT")
        for player_id in hit:
            self.players[player_id].alive = False
            events[player_id].append("GOT_KILLED")
            for other in self.players.values():
                if other.alive:
                    events[other.player_id].append("OPPONENT_ELIMINATED")

    def _time_to_stop(self) -> bool:
        living = [player for player in self.players.values() if player.alive]
        if not living:
            return True
        no_tasks = (
            len(living) == 1
            and not np.any(self.field == 1)
            and not any(self.coins.values())
            and not self.bombs
            and not self.explosions
        )
        return no_tasks or self.step_count >= MAX_STEPS
