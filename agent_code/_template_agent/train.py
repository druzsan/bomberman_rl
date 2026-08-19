"""Training callbacks for a new agent.

These no-op implementations make the copied template valid in training mode.
Add learning state to ``AgentContext`` in ``types.py`` as the agent grows.
"""

from __future__ import annotations

from .types import Action, AgentContext, GameState


def setup_training(self: AgentContext) -> None:
    """Initialize training-only state after ``callbacks.setup``."""


def game_events_occurred(
    self: AgentContext,
    old_game_state: GameState,
    self_action: Action,
    new_game_state: GameState,
    events: list[str],
) -> None:
    """Process one completed transition and its associated events."""
    self.logger.debug(
        "Encountered events %s after action %s in step %d",
        events,
        self_action,
        new_game_state["step"],
    )


def end_of_round(
    self: AgentContext,
    last_game_state: GameState | None,
    last_action: Action | None,
    events: list[str],
) -> None:
    """Process the final transition and persist training state if needed."""
    self.logger.debug(
        "Round ended after action %s with events %s (state available: %s)",
        last_action,
        events,
        last_game_state is not None,
    )
