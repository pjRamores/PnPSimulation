"""Reusable action-masking utility for Prospectors & Pirates.

Single source of truth for the game's per-state action validity rules, the
boolean action mask, and the "force an invalid action to a valid one"
enforcement that the environment and the model-backed bot both need.

The utility is intentionally **self-contained** -- it imports only numpy and the
standard library, with no dependency on the environment package. Callers adapt
their own state (the full simulator state, or a lambda ActionRequest parsed by a
bot) into a neutral :class:`MaskState` and then call:

* :func:`is_action_valid` -- ``(action, state) -> (is_valid, reason)``
* :func:`get_action_mask` -- ``state -> np.int8[NUM_ACTION_TYPES]``
* :func:`best_valid_action` -- first valid action by a priority order
* :func:`mask_action` -- ``(action, state) -> action`` (original if valid, else
  an enforced valid replacement)

The action ids below MUST stay in sync with ``env_common.ActionType``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# Action ids -- must match env_common.ActionType.
WAIT = 0
MINE = 1
MOVE_NORTH = 2
MOVE_SOUTH = 3
MOVE_EAST = 4
MOVE_WEST = 5
RECHARGE = 6
RECHARGE_END = 7
ATTACK = 8
JUMP_TO_ASTEROID = 9
SELL = 10
RAISE_SHIELDS = 11
JUMP_TO_TRADING_POST = 12
RESPAWN = 13
PLUNDER = 14
SALVAGE = 15
REPAIR = 16
NEGOTIATE = 17
LOWER_SHIELDS = 18

NUM_ACTION_TYPES = 19

_MOVE_ACTIONS = (MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST)

# Maps each action id to its ``metadata.actionRestrictions`` key. The four MOVE
# directions share the "MOVE" rule, and both JUMP variants share the "JUMP" rule.
# Used to data-drive the allowedWhileRecharging / allowedWithShieldsUp gate.
_ACTION_RESTRICTION_NAME = {
    WAIT: 'WAIT',
    MINE: 'MINE',
    MOVE_NORTH: 'MOVE',
    MOVE_SOUTH: 'MOVE',
    MOVE_EAST: 'MOVE',
    MOVE_WEST: 'MOVE',
    RECHARGE: 'RECHARGE',
    RECHARGE_END: 'RECHARGE_END',
    ATTACK: 'ATTACK',
    JUMP_TO_ASTEROID: 'JUMP',
    SELL: 'SELL',
    RAISE_SHIELDS: 'RAISE_SHIELDS',
    JUMP_TO_TRADING_POST: 'JUMP',
    RESPAWN: 'RESPAWN',
    PLUNDER: 'PLUNDER',
    SALVAGE: 'SALVAGE',
    REPAIR: 'REPAIR',
    NEGOTIATE: 'NEGOTIATE',
    LOWER_SHIELDS: 'LOWER_SHIELDS',
}

# Public alias: callers that reconstruct the observation (env_observation_mixin,
# bot_v6) reuse this single action-id -> restriction-key map so the encoded
# action-restriction features stay in sync with the masker's gate.
ACTION_RESTRICTION_NAME = _ACTION_RESTRICTION_NAME

# Fallback priority orders used by :func:`best_valid_action`.
# When energy is critically low, prioritise RECHARGE to avoid getting stuck.
_LOW_ENERGY_ORDER = (
    RECHARGE, MINE, SELL, WAIT, JUMP_TO_ASTEROID, JUMP_TO_TRADING_POST,
    MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, ATTACK, RAISE_SHIELDS,
)
# Otherwise prefer productive actions over idle WAIT.
_DEFAULT_ORDER = (
    MINE, SELL, JUMP_TO_ASTEROID, JUMP_TO_TRADING_POST,
    MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST,
    RECHARGE, ATTACK, RAISE_SHIELDS, WAIT,
)


@dataclass
class MaskState:
    """Neutral game-state snapshot the masking rules operate on.

    Both the environment (full simulator state) and the model-backed bot
    (lambda ActionRequest) build one of these. ``enemies`` is the already
    resolved list of candidate target ships (the env resolves player vs
    opponent targeting before constructing the state). Entity lists are plain
    ``{'x', 'y', ...}`` dicts; asteroids/wreckage carry ``'nutrinium'`` (and
    asteroids carry ``'mass'``), trading posts carry ``'id'``.
    """

    x: int
    y: int
    energy: int
    health: int
    nutrinium: int
    credits: int
    destroyed: bool
    recharging: bool
    just_recharged: bool
    shield_state: str          # 'POWERED' | 'DRAINING' | 'DOWN'
    shield_value: float
    shield_capacity: float
    shields_up: bool
    modules: object            # iterable / set of module-name strings
    negotiate_post_id: object  # id of the assigned negotiate objective post
    enemies: List[dict]
    asteroids: List[dict]
    trading_posts: List[dict]
    wreckage: List[dict]
    map_width: int
    map_height: int
    max_energy: int
    max_health: int
    energy_costs: dict          # keys: mine, move, attack, shields, jump, plunder, negotiate
    salvage_energy_cost: int
    repair_cost: int            # credits required to repair
    action_restrictions: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure helpers (ported from the env mixins; no env dependency)
# ---------------------------------------------------------------------------
def _distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Euclidean distance between two cells."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _entity_at(x: int, y: int, entities: List[dict]) -> Optional[dict]:
    """First non-destroyed entity exactly at ``(x, y)``, or ``None``."""
    for entity in entities:
        if entity['x'] == x and entity['y'] == y:
            if entity.get('destroyed', False):
                continue
            return entity
    return None


def _nearest_entity(x: int, y: int, entities: List[dict]) -> Optional[dict]:
    """Nearest non-destroyed, non-depleted entity to ``(x, y)``, or ``None``."""
    min_dist = float('inf')
    nearest = None
    for entity in entities:
        if entity.get('destroyed', False):
            continue
        if 'nutrinium' in entity and entity['nutrinium'] <= 0:
            continue
        dist = _distance(x, y, entity['x'], entity['y'])
        if dist < min_dist:
            min_dist = dist
            nearest = entity
    return nearest


def _top_asteroids(x: int, y: int, asteroids: List[dict], count: int = 5) -> List[dict]:
    """Top-N asteroids by the env score ``concentration * nutrinium / (dist + 1)``."""
    scored = []
    max_score = 50.0
    for asteroid in asteroids:
        nutrinium = asteroid.get('nutrinium', 0)
        if nutrinium <= 0:
            continue
        dist = _distance(x, y, asteroid['x'], asteroid['y'])
        mass = max(1, asteroid.get('mass', 1))
        concentration = nutrinium / mass
        raw_score = concentration * nutrinium / (dist + 1)
        scored.append({
            'x': asteroid['x'],
            'y': asteroid['y'],
            'mass': asteroid.get('mass', 0),
            'nutrinium': nutrinium,
            'distance': dist,
            'score': min(1.0, raw_score / max_score),
        })
    scored.sort(key=lambda a: a['score'], reverse=True)
    return scored[:count]


def _entity_shield_state(entity: dict) -> str:
    """Shield state of an enemy entity ('POWERED'|'DRAINING'|'DOWN')."""
    state = entity.get('shield_state')
    if state:
        return str(state).upper()
    shield = entity.get('shield')
    if isinstance(shield, dict):
        return shield.get('state', 'DOWN')
    return 'POWERED' if entity.get('shields_up') else 'DOWN'


def _active_enemies(st: MaskState) -> List[dict]:
    """Non-destroyed candidate target ships."""
    return [t for t in st.enemies if not t.get('destroyed', False)]


def _same_zone_enemies(st: MaskState) -> List[dict]:
    """Active enemy ships sharing the acting ship's tile."""
    return [t for t in _active_enemies(st)
            if t['x'] == st.x and t['y'] == st.y]


def _has_module(st: MaskState, module_name: str) -> bool:
    """Whether the ship has a given equipable module."""
    modules = st.modules
    if not modules:
        return False
    return module_name in modules


def _action_allowed(st: MaskState, action_name: str) -> bool:
    """Honor action_restrictions for the ship's recharging / shields-up state."""
    return _restriction_reason(st, action_name) is None


def _restriction_reason(st: MaskState, action_name: str) -> Optional[str]:
    """Return why ``action_name`` is blocked by ``action_restrictions``, else ``None``.

    Data-drives the per-state gate from ``metadata.actionRestrictions``: an action
    whose ``allowedWhileRecharging`` is false is masked while recharging, and one
    whose ``allowedWithShieldsUp`` is false is masked while shields are POWERED.
    """
    rule = (st.action_restrictions or {}).get(action_name)
    if rule is None:
        return None
    if st.recharging and not rule.get('allowedWhileRecharging', True):
        return f"{action_name} not allowed while recharging"
    if st.shield_state == 'POWERED' and not rule.get('allowedWithShieldsUp', True):
        return f"{action_name} not allowed with shields up"
    return None


# ---------------------------------------------------------------------------
# Validity rules (ported verbatim from EnvMaskingMixin._is_action_valid_for_state)
# ---------------------------------------------------------------------------
def is_action_valid(action: int, st: MaskState) -> Tuple[bool, str]:
    """Validate whether ``action`` is valid given the state ``st``.

    Returns ``(is_valid, reason)`` -- ``reason`` is empty when valid.
    """
    costs = st.energy_costs

    # Rule 1: If DESTROYED, only RESPAWN is valid
    if st.destroyed:
        if action == RESPAWN:
            return True, ""
        return False, "ship is destroyed, only RESPAWN is valid"

    # Rule 2: If NOT destroyed, RESPAWN is invalid
    if action == RESPAWN:
        return False, "can only respawn when destroyed"

    # Rule 3: Data-driven restriction gate (metadata.actionRestrictions). Masks any
    # action disallowed while recharging or with shields POWERED. Mission-/state-based
    # restrictions flow entirely from here -- there is no hardcoded recharging exclusivity.
    restriction_name = _ACTION_RESTRICTION_NAME.get(action)
    if restriction_name is not None:
        blocked = _restriction_reason(st, restriction_name)
        if blocked is not None:
            return False, blocked

    # Rule 4: RECHARGE_END is only valid while recharging
    if action == RECHARGE_END:
        if not st.recharging:
            return False, "not currently recharging"
        return True, ""

    # WAIT - masked only when energy exhausted and not recharging
    if action == WAIT:
        if not st.recharging:
            if st.energy <= 0:
                return False, "energy exhausted, must RECHARGE instead of WAIT"
        return True, ""

    # MINE - requires asteroid at current location with nutrinium
    if action == MINE:
        if st.energy < costs['mine']:
            return False, "insufficient energy to mine"
        asteroid = _entity_at(st.x, st.y, st.asteroids)
        if asteroid is None:
            return False, "no asteroid at current location"
        if asteroid['nutrinium'] <= 0:
            return False, "asteroid has no nutrinium"
        return True, ""

    # MOVE actions - require sufficient energy and stay on map
    if action in _MOVE_ACTIONS:
        if st.energy < costs['move']:
            return False, "insufficient energy to move"
        new_x, new_y = st.x, st.y
        if action == MOVE_NORTH:
            new_y = st.y + 1
        elif action == MOVE_SOUTH:
            new_y = st.y - 1
        elif action == MOVE_EAST:
            new_x = st.x + 1
        elif action == MOVE_WEST:
            new_x = st.x - 1
        if new_x < 0 or new_x >= st.map_width or new_y < 0 or new_y >= st.map_height:
            return False, "would move off map"
        return True, ""

    # RECHARGE - only when not just recharged, not full, and energy low enough
    if action == RECHARGE:
        if st.just_recharged:
            return False, "just finished recharging, do something productive first"
        if st.energy >= st.max_energy:
            return False, "energy already full"
        recharge_threshold = int(st.max_energy * 0.3)
        if st.energy > recharge_threshold:
            return False, f"energy too high to recharge ({st.energy}/{st.max_energy}, threshold {recharge_threshold})"
        return True, ""

    # ATTACK - requires enemy in same zone and sufficient energy
    if action == ATTACK:
        if st.energy < costs['attack']:
            return False, "insufficient energy to attack"
        active_targets = _active_enemies(st)
        if not active_targets:
            return False, "no enemy ships available"
        enemy_in_zone = any(t['x'] == st.x and t['y'] == st.y for t in active_targets)
        if not enemy_in_zone:
            return False, "no enemy in same zone"
        return True, ""

    # JUMP_TO_ASTEROID - requires the JUMP module, asteroids and sufficient energy
    if action == JUMP_TO_ASTEROID:
        if not _has_module(st, 'JUMP'):
            return False, "JUMP module not equipped"
        current_asteroid = _entity_at(st.x, st.y, st.asteroids)
        if current_asteroid is not None and current_asteroid.get('nutrinium', 0) > 0:
            mass = max(current_asteroid.get('mass', 1), 1)
            concentration = current_asteroid['nutrinium'] / mass
            if concentration >= 0.05:
                return False, f"already at asteroid with {concentration:.0%} nutrinium, mine it first"
        top = _top_asteroids(st.x, st.y, st.asteroids, count=1)
        if not top:
            return False, "no asteroids available"
        best = top[0]
        distance = best['distance']
        if distance == 0:
            return False, "best asteroid is at current location (distance 0), mine it or move away"
        energy_cost = int(distance * costs['jump'])
        if st.energy < energy_cost:
            return False, f"insufficient energy (need {energy_cost}, have {st.energy})"
        return True, ""

    # JUMP_TO_TRADING_POST - requires the JUMP module, trading posts, energy, and nutrinium
    if action == JUMP_TO_TRADING_POST:
        if not _has_module(st, 'JUMP'):
            return False, "JUMP module not equipped"
        if st.nutrinium < 10:
            return False, "not enough nutrinium to justify jumping to trading post (need >= 10)"
        current_post = _entity_at(st.x, st.y, st.trading_posts)
        if current_post is not None:
            return False, "already at a trading post, use SELL instead"
        post = _nearest_entity(st.x, st.y, st.trading_posts)
        if post is None:
            return False, "no trading posts available"
        distance = _distance(st.x, st.y, post['x'], post['y'])
        energy_cost = int(distance * costs['jump'])
        if st.energy < energy_cost:
            return False, f"insufficient energy (need {energy_cost}, have {st.energy})"
        return True, ""

    # SELL - requires being at a trading post with nutrinium
    if action == SELL:
        trading_post = _entity_at(st.x, st.y, st.trading_posts)
        if trading_post is None:
            return False, "not at a trading post"
        if st.nutrinium <= 0:
            return False, "no nutrinium to sell"
        return True, ""

    # RAISE_SHIELDS - shields not full, energy, and an enemy threat in zone
    if action == RAISE_SHIELDS:
        if st.shield_state == 'POWERED' and st.shield_value >= st.shield_capacity:
            return False, "shields already fully powered"
        if st.energy < costs['shields']:
            return False, "insufficient energy for shields"
        active_targets = _active_enemies(st)
        if not active_targets:
            return False, "no enemy threat, shields not needed"
        enemy_in_same_zone = any((t['x'] == st.x and t['y'] == st.y) for t in active_targets)
        if not enemy_in_same_zone:
            return False, "no enemy in same zone, no threat"
        return True, ""

    # PLUNDER - energy, and a shields-down enemy with nutrinium in zone
    if action == PLUNDER:
        if st.energy < costs['plunder']:
            return False, "insufficient energy to plunder"
        targets = [t for t in _same_zone_enemies(st)
                   if _entity_shield_state(t) == 'DOWN' and t.get('nutrinium', 0) > 0]
        if not targets:
            return False, "no plunderable (shields-down) target in zone"
        return True, ""

    # SALVAGE - module, energy, and wreckage with nutrinium at current location
    if action == SALVAGE:
        if not _has_module(st, 'SALVAGE'):
            return False, "SALVAGE module not equipped"
        if st.energy < st.salvage_energy_cost:
            return False, "insufficient energy to salvage"
        wreck = next((w for w in st.wreckage
                      if w['x'] == st.x and w['y'] == st.y and w.get('nutrinium', 0) > 0), None)
        if wreck is None:
            return False, "no wreckage to salvage here"
        return True, ""

    # REPAIR - module, trading post, credits, and not already at full health
    if action == REPAIR:
        if not _has_module(st, 'REPAIR'):
            return False, "REPAIR module not equipped"
        if _entity_at(st.x, st.y, st.trading_posts) is None:
            return False, "not at a trading post"
        if st.credits < st.repair_cost:
            return False, "insufficient credits to repair"
        if st.health >= st.max_health:
            return False, "already at full health"
        return True, ""

    # NEGOTIATE - energy and being at the ship's assigned objective trading post
    if action == NEGOTIATE:
        if st.energy < costs['negotiate']:
            return False, "insufficient energy to negotiate"
        post = _entity_at(st.x, st.y, st.trading_posts)
        if post is None or st.negotiate_post_id is None or post.get('id') != st.negotiate_post_id:
            return False, "not at negotiate objective trading post"
        return True, ""

    # LOWER_SHIELDS - valid only when shields are not already DOWN
    if action == LOWER_SHIELDS:
        if st.shield_state == 'DOWN':
            return False, "shields already down"
        return True, ""

    return False, f"unknown action {action}"


def get_action_mask(st: MaskState) -> np.ndarray:
    """Boolean (int8) mask over all action ids: 1 = valid, 0 = invalid."""
    mask = np.zeros(NUM_ACTION_TYPES, dtype=np.int8)
    for action in range(NUM_ACTION_TYPES):
        is_valid, _ = is_action_valid(action, st)
        mask[action] = 1 if is_valid else 0
    return mask


def best_valid_action(st: MaskState, mask: Optional[np.ndarray] = None) -> int:
    """First valid action by priority order (low-energy aware). Falls back to WAIT."""
    if mask is None:
        mask = get_action_mask(st)
    if st.energy <= st.energy_costs.get('move', 5):
        order = _LOW_ENERGY_ORDER
    else:
        order = _DEFAULT_ORDER
    for fb_action in order:
        if mask[fb_action] == 1:
            return int(fb_action)
    return WAIT


def mask_action(action: int, st: MaskState, mask: Optional[np.ndarray] = None) -> int:
    """Return ``action`` if valid for ``st``, else an enforced valid replacement.

    Enforcement mirrors the environment: while recharging force RECHARGE_END (or
    WAIT), when destroyed force RESPAWN, otherwise pick the best valid action.
    """
    is_valid, _ = is_action_valid(action, st)
    if is_valid:
        return int(action)

    if st.recharging:
        if st.energy >= st.max_energy:
            return RECHARGE_END
        if action not in (WAIT, RECHARGE_END):
            return RECHARGE_END
        return WAIT

    if st.destroyed:
        return RESPAWN

    return best_valid_action(st, mask)
