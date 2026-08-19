"""Inference callbacks for a new agent.

Copy this directory, rename it, and replace the decision logic in ``act``.
"""

from __future__ import annotations

from .types import Action, AgentContext, GameState


def setup(self: AgentContext) -> None:
    """Initialize persistent agent state once, before the first round."""


def act(self: AgentContext, game_state: GameState) -> Action:
    """Return the action to execute for the current game-state snapshot."""
    self.logger.info("Pick action according to pressed key")
    return game_state["user_input"] or "WAIT"
