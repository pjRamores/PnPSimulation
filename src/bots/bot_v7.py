"""
Prospectors & Pirates bot (v7).

A deliberately simple "dummy miner" used as a weak benchmark / training opponent.
It follows the same lambda contract as `bot_v2` .. `bot_v6`: a single func:`get_action`
that takes an ActionRequest dict and returns a response dict `{"actionType": str, "payload?": ...}`.

Behaviour:

1. Energy management: recharge when energy is low, and keep waiting until reasonably charged before resuming work.
2. Mine whenever sitting on an asteroid -- regardless of its nutrinium content.
3. Otherwise wander randomly, but only ever steps in a direction that stays on the map (random movement is bounds-validated,
not action-masked).

It never sells, never hauls cargo, never jumps and never fights, so any mined
nutrinium simply accumulates unsold. Directions are emitted in the live-server frame
(N=y+1, S=y-1, E=x+1, W=x-1), matching the environment's MOVE actions.

Invalid economy/combat actions are simply never produced; the environment's action methods would no-op on them anyway,
which is acceptable for this intentionally dumb opponent.
"""

import random

# Tunables
RECHARGE_LOW_ENERGY = 20       # start recharging when energy drops below this
RECHARGE_END_ENERGY = 80       # stop recharging once back to a workable level

_MOVE_DELTAS = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
}

def _to_response(payload):
    """Lightweight lambda response adapter (mirrors bot_v2 .. bot_v6)."""
    if isinstance(payload, dict):
        action = payload
    elif hasattr(payload, "to_dict") and callable(payload.to_dict):
        action = payload.to_dict()
    elif hasattr(payload, "__dict__"):
        action = {
            key: value
            for key, value in vars(payload).items()
            if not key.startswith("_")
        }
    else:
        action = None

    if not isinstance(action, dict):
        return {"actionType": "WAIT"}

    action_type = action.get("actionType")
    if not isinstance(action_type, str) or not action_type:
        return {"actionType": "WAIT"}

    return action


# Geometry / parsing helpers (stateless)
def _entity_at(x, y, entities):
    """First entity exactly at (x, y), or None."""
    for e in entities:
        if e["x"] == x and e["y"] == y:
            return e
    return None

def move(direction):
    return {"actionType": "MOVE", "payload": {"direction": direction}}


class _Context:
    """Parses one ActionRequest into the fields the dummy-miner loop needs."""

    def __init__(self, action_request):
        req = action_request or {}
        me = req.get("me", {}) or {}
        loc = me.get("location", {}) or {}
        self.x = int(loc.get("x", 0))
        self.y = int(loc.get("y", 0))
        self.energy = int(me.get("energy", 0))
        self.nutrinium = int(me.get("nutrinium", 0))
        self.recharging = bool(me.get("recharging", False))

        # Asteroid sensor contacts (flat {x, y, nutrinium, mass} dicts).
        self.asteroids = []
for s in req.get("sensors", []) or []:
    if s.get("type") != "asteroid":
        continue
    s_loc = s.get("location", {}) or {}
    self.asteroids.append({
        "x": int(s_loc.get("x", 0)),
        "y": int(s_loc.get("y", 0)),
        "nutrinium": int(s.get("nutrinium", 0)),
        "mass": int(s.get("mass", 0)),
    })

metadata = (req.get("gameState", {}) or {}).get("metadata", {}) or {}
ship_cfg = metadata.get("shipConfig", {}) or {}
costs = ship_cfg.get("energyCosts", {}) or {}
self.mine_cost = int(costs.get("mine", 10))
# Map bounds keep random movement on the map.
map_cfg = metadata.get("mapConfig", {}) or {}
self.map_w = int(map_cfg.get("width", 125))
self.map_h = int(map_cfg.get("height", 125))

def in_bounds_directions(self):
    """Compass directions whose single step stays within the map."""
    valid = []
    for direction, (dx, dy) in _MOVE_DELTAS.items():
        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx <= self.map_w - 1 and 0 <= ny <= self.map_h - 1:
            valid.append(direction)
    return valid

# -----------------------------------
# Dummy-miner decision
# -----------------------------------
def _dummy_miner_action(ctx):
    """Recharge when low, mine any asteroid under the ship, else wander on-map."""
    # === 1. ENERGY MANAGEMENT ===
    if ctx.recharging:
        if ctx.energy >= RECHARGE_END_ENERGY:
            return {"actionType": "RECHARGE_END"}
        return {"actionType": "WAIT"}

    if ctx.energy < RECHARGE_LOW_ENERGY:
        return {"actionType": "RECHARGE"}

    # === 2. MINE whatever asteroid we're standing on (ignore nutrinium) ===
    if _entity_at(ctx.x, ctx.y, ctx.asteroids) and ctx.energy >= ctx.mine_cost:
        return {"actionType": "MINE"}

    # === 3. WANDER randomly, but only in a direction that stays on the map ===
    directions = ctx.in_bounds_directions()
    if not directions:
        return {"actionType": "WAIT"}
    return _move(random.choice(directions))

# -----------------------------------
# Public lambda contract
# -----------------------------------
def get_heuristic_action(action_request):
    """Decide a dummy-miner action for one ActionRequest (raw dict, pre-adapter)."""
    ctx = _Context(action_request)
    return _dummy_miner_action(ctx)

def get_action(action_request):
    """Public entry point: returns a normalised response dict."""
    return _to_response(get_heuristic_action(action_request))