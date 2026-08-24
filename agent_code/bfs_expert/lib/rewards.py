"""Custom events, event rewards and potential-based shaping.

Engine rewards are sparse (9 coins and ~0.2 kills over ~350 steps), so every
model needs a denser signal.  Two layers, kept separate so they can be ablated
independently:

* **Potential-based shaping** (:func:`potential`), ``F(s,s') = γ·Φ(s') − Φ(s)``.
  Per Ng et al. (1999) this cannot change the optimal policy, and because a
  potential difference telescopes to zero over a round trip it does not create
  the walk-back-and-forth exploit that a naive "reward for approaching a coin"
  term does.
* **Event rewards** (:data:`DEFAULT_EVENT_REWARDS`), including custom events
  derived from the state pair.  The four bomb-quality events are conditioned on
  the action and are therefore a deliberate deviation from strict potential
  shaping; they carry the highest-value signal in the scheme and get their own
  ablation.

Engine event names are duplicated here as plain strings so a vendored copy of
``lib`` never imports repo-root modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import danger
from .features import OBJ_COIN, OBJ_CRATE, OBJ_OPPONENT, StateInfo

# --- engine events (mirrors events.py) --------------------------------------
MOVED_LEFT = "MOVED_LEFT"
MOVED_RIGHT = "MOVED_RIGHT"
MOVED_UP = "MOVED_UP"
MOVED_DOWN = "MOVED_DOWN"
WAITED = "WAITED"
INVALID_ACTION = "INVALID_ACTION"
BOMB_DROPPED = "BOMB_DROPPED"
BOMB_EXPLODED = "BOMB_EXPLODED"
CRATE_DESTROYED = "CRATE_DESTROYED"
COIN_FOUND = "COIN_FOUND"
COIN_COLLECTED = "COIN_COLLECTED"
KILLED_OPPONENT = "KILLED_OPPONENT"
KILLED_SELF = "KILLED_SELF"
GOT_KILLED = "GOT_KILLED"
OPPONENT_ELIMINATED = "OPPONENT_ELIMINATED"
SURVIVED_ROUND = "SURVIVED_ROUND"

# --- custom events ----------------------------------------------------------
ESCAPED_DANGER = "ESCAPED_DANGER"
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"
STAYED_IN_DANGER = "STAYED_IN_DANGER"
SUICIDAL_BOMB = "SUICIDAL_BOMB"
USELESS_BOMB = "USELESS_BOMB"
GOOD_BOMB = "GOOD_BOMB"
BOMB_NEAR_OPPONENT = "BOMB_NEAR_OPPONENT"
TRAPPED_OPPONENT = "TRAPPED_OPPONENT"
WALKED_INTO_DEAD_END = "WALKED_INTO_DEAD_END"
LOOP = "LOOP"

CUSTOM_EVENTS = (
    ESCAPED_DANGER, MOVED_INTO_DANGER, STAYED_IN_DANGER, SUICIDAL_BOMB,
    USELESS_BOMB, GOOD_BOMB, BOMB_NEAR_OPPONENT, TRAPPED_OPPONENT,
    WALKED_INTO_DEAD_END, LOOP,
)

ENGINE_EVENTS = (
    MOVED_LEFT, MOVED_RIGHT, MOVED_UP, MOVED_DOWN, WAITED, INVALID_ACTION,
    BOMB_DROPPED, BOMB_EXPLODED, CRATE_DESTROYED, COIN_FOUND, COIN_COLLECTED,
    KILLED_OPPONENT, KILLED_SELF, GOT_KILLED, OPPONENT_ELIMINATED, SURVIVED_ROUND,
)

ALL_EVENTS = ENGINE_EVENTS + CUSTOM_EVENTS

DEFAULT_EVENT_REWARDS: dict[str, float] = {
    COIN_COLLECTED: 3.0,
    KILLED_OPPONENT: 15.0,
    CRATE_DESTROYED: 0.4,
    COIN_FOUND: 0.5,
    KILLED_SELF: -20.0,
    GOT_KILLED: -15.0,
    SURVIVED_ROUND: 5.0,
    INVALID_ACTION: -1.0,
    WAITED: -0.05,
    MOVED_UP: -0.02,
    MOVED_DOWN: -0.02,
    MOVED_LEFT: -0.02,
    MOVED_RIGHT: -0.02,
    ESCAPED_DANGER: 1.0,
    MOVED_INTO_DANGER: -1.0,
    STAYED_IN_DANGER: -0.5,
    SUICIDAL_BOMB: -8.0,
    USELESS_BOMB: -1.5,
    GOOD_BOMB: 0.3,          # per crate in the blast
    BOMB_NEAR_OPPONENT: 2.0,
    TRAPPED_OPPONENT: 5.0,
    WALKED_INTO_DEAD_END: -0.5,
    LOOP: -0.5,
}

#: ``KILLED_SELF`` already includes the death penalty, so ``GOT_KILLED`` is not
#: added on top of it (a suicide is worth -20 in total, not -35).
SUPPRESSED_WITH = {KILLED_SELF: (GOT_KILLED,)}


@dataclass
class RewardConfig:
    """All reward knobs in one place so ablations are a config diff."""

    event_rewards: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_EVENT_REWARDS))
    use_events: bool = True
    use_custom_events: bool = True
    use_bomb_quality_events: bool = True
    use_potential: bool = True
    w_coin: float = 0.15
    w_crate: float = 0.05
    w_hunt: float = 0.02
    w_safe: float = 0.5
    distance_cap: int = 20
    gamma: float = 0.95
    reward_scale: float = 1.0
    loop_window: int = 10
    loop_threshold: int = 3
    dead_end_opponent_range: int = 4


BOMB_QUALITY_EVENTS = (SUICIDAL_BOMB, USELESS_BOMB, GOOD_BOMB, BOMB_NEAR_OPPONENT,
                       TRAPPED_OPPONENT)


def potential(info: StateInfo | None, cfg: RewardConfig) -> float:
    """``Φ(s)``: distance to the current objective plus a safety term.

    ``Φ`` of a terminal state is 0 by convention, which is what makes the
    telescoping argument hold for episodic tasks.
    """
    if info is None:
        return 0.0
    phi = 0.0
    if info.obj_dist >= 0:
        d = min(info.obj_dist, cfg.distance_cap)
        if info.objective == OBJ_COIN:
            phi -= cfg.w_coin * d
        elif info.objective == OBJ_CRATE:
            phi -= cfg.w_crate * d
        elif info.objective == OBJ_OPPONENT:
            phi -= cfg.w_hunt * d
    if info.danger_tau < 0:
        phi += cfg.w_safe
    else:
        phi -= cfg.w_safe * (4 - min(info.danger_tau, 4)) / 4.0
    return phi


def custom_events(
    old: StateInfo,
    new: StateInfo | None,
    events: list[str],
    cfg: RewardConfig,
    old_state: dict | None = None,
    visit_counts: dict | None = None,
) -> list[str]:
    """Events derived from the ``(old, new)`` state pair and the engine events.

    ``new`` is ``None`` for the terminal transition of a round in which the
    agent died: the engine never delivers a post-death state.
    """
    out: list[str] = []
    if not cfg.use_custom_events:
        return out

    old_danger = old.danger_tau >= 0
    if new is not None:
        new_danger = new.danger_tau >= 0
        if old_danger and not new_danger:
            out.append(ESCAPED_DANGER)
        elif not old_danger and new_danger:
            out.append(MOVED_INTO_DANGER)
        elif old_danger and new_danger:
            out.append(STAYED_IN_DANGER)
        if new.in_dead_end and 0 <= new.opp_dist <= cfg.dead_end_opponent_range:
            out.append(WALKED_INTO_DEAD_END)
        if visit_counts is not None and visit_counts.get(new.pos, 0) >= cfg.loop_threshold:
            out.append(LOOP)

    if cfg.use_bomb_quality_events and BOMB_DROPPED in events:
        if not old.bomb_safe:
            out.append(SUICIDAL_BOMB)
        elif old.bomb_crates == 0 and old.bomb_hits == 0:
            out.append(USELESS_BOMB)
        if old.bomb_safe and old.bomb_crates >= 1:
            out.extend([GOOD_BOMB] * old.bomb_crates)
        if old.bomb_safe and old.bomb_hits > 0:
            out.append(BOMB_NEAR_OPPONENT)
        if old_state is not None and traps_opponent(old, old_state):
            out.append(TRAPPED_OPPONENT)
    return out


def traps_opponent(old: StateInfo, old_state: dict) -> bool:
    """Does a bomb at our tile leave some opponent with no escape route?"""
    extra = (old.pos,)
    lethal = danger.lethal_bits(old_state["field"], old_state["bombs"],
                                old_state["explosion_map"], extra_bombs=extra)
    blocked = danger.blocked_mask(old_state["field"], old_state["bombs"], extra_bombs=extra)
    for _, _, _, pos in old_state["others"]:
        p = (int(pos[0]), int(pos[1]))
        if lethal[p] == 0:
            continue
        if not danger.survivable(danger.escape_taus(lethal, blocked, p)):
            return True
    return False


def reward_from_events(events: list[str], cfg: RewardConfig) -> float:
    """Sum the configured event rewards, honouring the suppression table."""
    if not cfg.use_events:
        return 0.0
    suppressed: set[str] = set()
    for trigger, victims in SUPPRESSED_WITH.items():
        if trigger in events:
            suppressed.update(victims)
    return sum(cfg.event_rewards.get(ev, 0.0) for ev in events if ev not in suppressed)


def transition_reward(
    old: StateInfo,
    new: StateInfo | None,
    events: list[str],
    cfg: RewardConfig,
    shaping_scale: float = 1.0,
) -> float:
    """Total shaped reward for one transition."""
    r = reward_from_events(events, cfg)
    if cfg.use_potential:
        r += shaping_scale * (cfg.gamma * potential(new, cfg) - potential(old, cfg))
    return r * cfg.reward_scale


def true_reward(events: list[str]) -> float:
    """The tournament score delta: what the agent is actually judged on."""
    return (events.count(COIN_COLLECTED) * 1.0) + (events.count(KILLED_OPPONENT) * 5.0)
